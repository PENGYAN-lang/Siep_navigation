"""
Interaction-aware human trajectory prediction for Proactive-SIEP.

Two prediction modes
--------------------
1. **ConstantVelocity** (baseline): h_j(tau) = x_j + v_j * tau
2. **InteractionAware** (proactive): uses a Social Force Model to predict how
   each pedestrian responds to the robot's candidate trajectory U.

The interaction-aware model makes H_hat depend on U, enabling the planner to
reason about how its own motion influences pedestrian behaviour — capturing the
"proactive" character of Proactive-SIEP.

Paper equation mapping
----------------------
  H_hat_tau(U) ← predict_all(ped_states, robot_traj_from_U, H, dt)

where the dependency on U enters via ``robot_traj_from_U`` (robot positions
along the horizon) that appear in the SFM reaction terms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SFM-reaction parameters (intentionally lightweight — one-shot forward rollout,
# not a full social-force simulator)
# ─────────────────────────────────────────────────────────────────────────────

# Force scale: how strongly pedestrians react to approaching robot
K_ROBOT_REPULSION: float = 1.2
ROBOT_INFLUENCE_DIST: float = 2.5   # [m]

# Ped-ped repulsion (keeps predicted trajectories realistic)
K_PED_PED: float = 0.6
PED_PED_DIST: float = 1.0           # [m]

# Speed limits for predicted pedestrians
MAX_PED_SPEED: float = 1.5          # [m/s]
PED_SPEED_TAU: float = 0.5          # relaxation time toward preferred velocity


@dataclass
class PedState:
    """Snapshot of one pedestrian for prediction."""
    xy: np.ndarray      # (2,) position
    vel: np.ndarray     # (2,) velocity
    yaw: float          # heading [rad]
    preferred_vel: np.ndarray  # (2,) desired velocity (constant-vel baseline)
    radius: float = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Prediction functions
# ─────────────────────────────────────────────────────────────────────────────

def predict_constant_velocity(
    ped_states: List[PedState],
    H: int,
    dt: float,
) -> List[List[np.ndarray]]:
    """Baseline: linear extrapolation for each pedestrian.

    Returns
    -------
    predictions : List[List[np.ndarray]]
        predictions[j][tau] = xy of pedestrian j at step tau.
    """
    predictions: List[List[np.ndarray]] = []
    for ped in ped_states:
        traj = [ped.xy + ped.vel * (tau * dt) for tau in range(H + 1)]
        predictions.append(traj)
    return predictions


def predict_interaction_aware(
    ped_states: List[PedState],
    robot_traj: List[np.ndarray],   # robot_traj[tau] = (2,) robot xy at step tau
    H: int,
    dt: float,
) -> List[List[np.ndarray]]:
    """Interaction-aware prediction: pedestrians react to the candidate robot traj.

    Each pedestrian uses a simple SFM with three terms:
      1. Self-propulsion toward preferred velocity (relaxation).
      2. Repulsion from robot position (robot influence on pedestrian).
      3. Ped-ped repulsion (prevents predicted trajectories from overlapping).

    The robot's trajectory U is provided as a sequence of xy positions, making
    the predicted pedestrian trajectories H_hat functionally dependent on U.

    Parameters
    ----------
    ped_states : list of PedState (current state)
    robot_traj : list of (2,) arrays, length H+1 (tau=0..H)
    H          : horizon steps
    dt         : time step [s]

    Returns
    -------
    predictions[j][tau] = predicted xy of pedestrian j at step tau
    """
    n = len(ped_states)
    # Working copies of positions and velocities
    pxys = [ped.xy.copy() for ped in ped_states]
    pvels = [ped.vel.copy() for ped in ped_states]

    predictions: List[List[np.ndarray]] = [[] for _ in range(n)]
    for j in range(n):
        predictions[j].append(pxys[j].copy())

    for tau in range(1, H + 1):
        robot_xy = robot_traj[min(tau, len(robot_traj) - 1)]
        new_pxys = [p.copy() for p in pxys]
        new_pvels = [v.copy() for v in pvels]

        for j, ped in enumerate(ped_states):
            # 1. Self-propulsion (relaxation toward preferred velocity)
            F = (ped.preferred_vel - pvels[j]) / PED_SPEED_TAU

            # 2. Robot repulsion
            diff = pxys[j] - robot_xy
            dist = float(np.linalg.norm(diff))
            if 0 < dist < ROBOT_INFLUENCE_DIST:
                decay = ROBOT_INFLUENCE_DIST * 0.4
                mag = K_ROBOT_REPULSION * math.exp(-dist / max(decay, 1e-6))
                F += mag * (diff / dist)

            # 3. Ped-ped repulsion
            for k, other_xy in enumerate(pxys):
                if k == j:
                    continue
                d = pxys[j] - other_xy
                dist_kj = float(np.linalg.norm(d))
                if 0 < dist_kj < PED_PED_DIST:
                    F += K_PED_PED / max(dist_kj, 1e-6) * (d / dist_kj)

            # Euler integration
            new_v = pvels[j] + F * dt
            speed = float(np.linalg.norm(new_v))
            if speed > MAX_PED_SPEED:
                new_v = new_v / speed * MAX_PED_SPEED
            new_pvels[j] = new_v
            new_pxys[j] = pxys[j] + new_v * dt

        pxys = new_pxys
        pvels = new_pvels

        for j in range(n):
            predictions[j].append(pxys[j].copy())

    return predictions


def make_ped_states(
    ped_raw: List[dict],
) -> List[PedState]:
    """Convert raw pedestrian dicts from sim to PedState list.

    Parameters
    ----------
    ped_raw : list of dicts with keys 'xy', 'vel', 'yaw', 'radius'
    """
    states = []
    for d in ped_raw:
        xy = np.asarray(d['xy'], dtype=float)
        vel = np.asarray(d['vel'], dtype=float)
        yaw = float(d.get('yaw', math.atan2(float(vel[1]), float(vel[0]) + 1e-9)))
        radius = float(d.get('radius', 0.3))
        states.append(PedState(
            xy=xy,
            vel=vel,
            yaw=yaw,
            preferred_vel=vel.copy(),  # preferred = current velocity
            radius=radius,
        ))
    return states
