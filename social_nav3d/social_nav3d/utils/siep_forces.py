"""
SIEP (Stimuli-Induced Equilibrium Point) virtual-force kernels.

In the SIEP framework the robot's desired velocity is the superposition of
virtual forces induced by all stimuli perceived from the environment:

  F_total = F_goal + F_obstacle + F_personal_space + F_velocity_alignment

The direction and magnitude of F_total define the robot's *equilibrium point* –
the velocity the robot is driven towards.  A proportional heading controller
then converts this vector into (v, ω) differential-drive commands.

Three innovations over the plain anisotropic-Gaussian MPC cost approach:

1. **Predictive personal-space forces** – repulsion is integrated over a
   look-ahead horizon so the robot steers away *before* a personal-space
   violation occurs (time-discounted sum of future poses).

2. **Velocity-alignment stimulus** – a gentle force that encourages the robot
   to match the speed/direction of nearby pedestrians walking the same way,
   capturing crowd-flow awareness.

3. **Tanh-saturated goal attraction** – the goal force saturates at the
   desired cruising speed rather than growing unboundedly, giving smoother
   deceleration near the goal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .social import PersonalSpace


# ─────────────────────────────────────────────────────────────────────────────
# Parameter container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SIEPParams:
    """Tunable weights and ranges for each SIEP stimulus."""

    # --- Goal attraction --------------------------------------------------
    k_goal: float = 1.0       # peak force magnitude [m/s] (= desired cruise speed)
    sigma_goal: float = 3.0   # distance at which attraction saturates to ~76 % [m]

    # --- Obstacle repulsion (LiDAR-based) ---------------------------------
    k_obs: float = 2.5        # peak repulsion at zero distance
    obs_influence: float = 2.5  # influence radius [m]

    # --- Predictive personal-space repulsion ------------------------------
    k_ps: float = 2.0         # peak personal-space force magnitude
    ps_horizon: float = 1.5   # prediction look-ahead [s]
    ps_dt: float = 0.1        # prediction integration step [s]

    # --- Velocity-alignment stimulus (crowd-flow following) ---------------
    k_align: float = 0.3      # alignment force scale
    align_radius: float = 3.0  # maximum effective radius [m]
    align_cone_deg: float = 60.0  # directional cone [deg] – only align with
    #   pedestrians whose heading is within this half-angle of the robot's


# ─────────────────────────────────────────────────────────────────────────────
# Individual force kernels
# ─────────────────────────────────────────────────────────────────────────────

def goal_force(
    robot_xy: np.ndarray,
    goal_xy: np.ndarray,
    k: float,
    sigma: float,
) -> np.ndarray:
    """Tanh-saturated attraction toward the goal.

    The magnitude ramps up from zero at the goal and saturates at *k* far
    away, preventing unbounded speed commands in long corridors.
    """
    d = goal_xy - robot_xy
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return np.zeros(2, dtype=float)
    magnitude = k * math.tanh(dist / max(sigma, 1e-6))
    return magnitude * d / dist


def obstacle_force(
    lidar_dists: np.ndarray,
    lidar_angles_world: np.ndarray,
    k: float,
    influence_dist: float,
) -> np.ndarray:
    """Repulsion from obstacles detected by LiDAR.

    Each ray that detects a surface within *influence_dist* contributes a
    force directed *away* from that surface.  Magnitude decays exponentially.
    """
    mask = lidar_dists < influence_dist
    if not np.any(mask):
        return np.zeros(2, dtype=float)

    d_near = lidar_dists[mask]
    a_near = lidar_angles_world[mask]
    # Decay constant chosen so force drops to e^{-2.5} ≈ 8 % at influence_dist.
    # decay = influence_dist * 0.4 means exp(-influence_dist / decay) = exp(-2.5).
    decay = influence_dist * 0.4
    magnitudes = k * np.exp(-d_near / max(decay, 1e-6))
    fx = -float(np.sum(magnitudes * np.cos(a_near)))
    fy = -float(np.sum(magnitudes * np.sin(a_near)))
    return np.array([fx, fy], dtype=float)


def personal_space_force(
    robot_xy: np.ndarray,
    ped_xy: np.ndarray,
    ped_yaw: float,
    ped_vel: np.ndarray,
    ps: PersonalSpace,
    k: float,
    horizon: float,
    dt: float = 0.1,
) -> np.ndarray:
    """Predictive anisotropic personal-space repulsion from one pedestrian.

    Integrates the anisotropic Gaussian cost over predicted pedestrian
    positions up to *horizon* seconds ahead.  Each future step is weighted
    by an exponential discount so that the immediate threat matters most.

    Args:
        robot_xy:  Current robot position (2,).
        ped_xy:    Current pedestrian position (2,).
        ped_yaw:   Pedestrian heading [rad].
        ped_vel:   Pedestrian velocity vector (2,).
        ps:        Personal-space parameters.
        k:         Force scale.
        horizon:   Look-ahead time [s].
        dt:        Integration step [s].
    """
    steps = max(1, int(horizon / max(dt, 1e-6)))
    force = np.zeros(2, dtype=float)

    for t in range(steps):
        ped_pred = ped_xy + ped_vel * (t * dt)
        cost = _aniso_gaussian(robot_xy, ped_pred, ped_yaw, ps)
        d = robot_xy - ped_pred
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            continue
        # Temporal discount: weight current step most heavily; halves every 2 s.
        # exp(-0.5 * t * dt) gives discount factor 0.5 after 2/dt steps (≈ 2 s).
        discount = math.exp(-0.5 * t * dt)
        force += k * cost * discount * (d / dist) * dt

    return force


def velocity_alignment_force(
    robot_xy: np.ndarray,
    robot_vel: np.ndarray,
    ped_xy: np.ndarray,
    ped_vel: np.ndarray,
    k: float,
    radius: float,
    cone_deg: float,
) -> np.ndarray:
    """Velocity-alignment (crowd-flow) stimulus.

    When a pedestrian is nearby *and* moving in roughly the same direction
    the robot is gently encouraged to match that pedestrian's velocity.
    This models the social norm of following crowd flow and smooths the
    robot's passage through streams of pedestrians.

    Args:
        robot_xy:   Current robot position (2,).
        robot_vel:  Current robot velocity vector (2,).
        ped_xy:     Pedestrian position (2,).
        ped_vel:    Pedestrian velocity (2,).
        k:          Force scale.
        radius:     Maximum effective radius [m].
        cone_deg:   Half-angle of the same-direction cone [deg].
    """
    dist = float(np.linalg.norm(ped_xy - robot_xy))
    if dist > radius:
        return np.zeros(2, dtype=float)

    ped_speed = float(np.linalg.norm(ped_vel))
    if ped_speed < 0.1:
        return np.zeros(2, dtype=float)

    # Only align when pedestrian moves in a similar direction
    robot_speed = float(np.linalg.norm(robot_vel))
    if robot_speed > 0.1:
        cos_thresh = math.cos(math.radians(cone_deg))
        cos_angle = float(np.dot(robot_vel / robot_speed, ped_vel / ped_speed))
        if cos_angle < cos_thresh:
            return np.zeros(2, dtype=float)

    # Proximity weight: Gaussian envelope that reaches ~e^{-0.5} ≈ 60 % at
    # half the radius and decays to near-zero at the boundary.  sigma = radius/2
    # ensures the force is negligible beyond the alignment radius.
    sigma_prox = radius * 0.5
    w_prox = math.exp(-(dist ** 2) / (2.0 * sigma_prox ** 2 + 1e-9))
    return k * w_prox * (ped_vel - robot_vel)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aniso_gaussian(
    robot_xy: np.ndarray,
    ped_xy: np.ndarray,
    ped_yaw: float,
    ps: PersonalSpace,
) -> float:
    """Anisotropic Gaussian cost in pedestrian body frame (same formula as
    ``anisotropic_gaussian_cost`` in ``social.py`` but kept local to avoid
    circular imports when both modules are extended independently)."""
    d = robot_xy - ped_xy
    c, s = math.cos(-ped_yaw), math.sin(-ped_yaw)
    dx = c * d[0] - s * d[1]
    dy = s * d[0] + c * d[1]
    sigma_x = ps.sigma_front if dx >= 0 else ps.sigma_back
    sigma_y = ps.sigma_side
    ex = (dx * dx) / (2.0 * sigma_x ** 2 + 1e-9)
    ey = (dy * dy) / (2.0 * sigma_y ** 2 + 1e-9)
    return float(math.exp(-(ex + ey)))
