"""
FSM-based realistic pedestrian behaviour module.

Replaces the constant-velocity bounce model with a Finite State Machine (FSM)
that produces behaviorally diverse and reproducible pedestrian motion
suitable for valid evaluation of social navigation algorithms.

States
------
WANDER      → pick random exhibit/waypoint as next target
WALK_TO     → walk toward target with SFM avoidance and speed variation
APPROACH    → decelerate when within APPROACH_DIST of target
VIEWING     → stand still for random dwell time (museum-browsing model)
TURNING     → select a new heading before entering WALK_TO

The FSM captures key real-world pedestrian behaviours:
- stopping at points of interest (exhibits, junctions)
- variable walking speed with Gaussian noise
- natural deceleration near targets
- SFM-style avoidance of other pedestrians and the robot
- seeded reproducibility for controlled experiments

Paper: pedestrian behaviour realism section.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# FSM states
# ─────────────────────────────────────────────────────────────────────────────

class PedFSMState(Enum):
    WANDER   = auto()
    WALK_TO  = auto()
    APPROACH = auto()
    VIEWING  = auto()
    TURNING  = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

APPROACH_DIST: float = 1.2     # [m] switch from WALK_TO to APPROACH
ARRIVED_DIST: float = 0.4      # [m] switch from APPROACH to VIEWING
DWELL_MIN: float = 4.0         # [s] minimum viewing time
DWELL_MAX: float = 14.0        # [s] maximum viewing time
SPEED_NOISE_STD: float = 0.12  # [m/s] per-step Gaussian speed noise
MAX_ACCEL: float = 0.4         # [m/s²] maximum acceleration toward target speed

# Ped-ped SFM avoidance (lightweight, not full SFM)
PED_PED_REPULSION_K: float = 0.8
PED_PED_DIST: float = 0.8      # [m]


# ─────────────────────────────────────────────────────────────────────────────
# FSM Pedestrian dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FSMPedestrian:
    """Extended pedestrian state including FSM fields.

    Attributes
    ----------
    xy          : (2,) current position [m]
    yaw         : current heading [rad]
    vel         : (2,) current velocity [m/s]
    base_speed  : preferred walking speed [m/s] (individual variation)
    target_xy   : current movement target
    fsm_state   : current FSM state
    view_timer  : countdown for VIEWING state [s]
    radius      : collision radius [m]
    """
    xy: np.ndarray
    yaw: float
    vel: np.ndarray
    base_speed: float
    target_xy: Optional[np.ndarray] = None
    fsm_state: PedFSMState = PedFSMState.WANDER
    view_timer: float = 0.0
    radius: float = 0.3
    # Random state per pedestrian (reproducible)
    _rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.default_rng()


def create_fsm_pedestrians(
    n: int,
    world_size: Tuple[float, float],
    speed_range: Tuple[float, float],
    radius: float,
    waypoints: List[np.ndarray],
    seed: int = 0,
    lane_flow: bool = True,
) -> List[FSMPedestrian]:
    """Factory: create N FSMPedestrians with seeded reproducible initial states.

    Parameters
    ----------
    n            : number of pedestrians
    world_size   : (sx, sy) world dimensions [m]
    speed_range  : (vmin, vmax) walking speed range [m/s]
    radius       : pedestrian radius [m]
    waypoints    : list of interest points (exhibits, corners etc.)
    seed         : random seed for reproducibility
    lane_flow    : if True, first n//2 walk rightward and rest leftward
                   (models corridor counter-flow for negotiation scenarios)
    """
    rng = np.random.default_rng(seed)
    sx, sy = world_size
    vmin, vmax = speed_range
    peds = []

    for i in range(n):
        ped_seed = int(rng.integers(0, 2 ** 31))
        ped_rng = np.random.default_rng(ped_seed)

        if lane_flow:
            # Force head-on flows in a corridor band [sy*0.35 .. sy*0.65]
            lane_y = float(ped_rng.uniform(sy * 0.35, sy * 0.65))
            if i < n // 2:
                x = float(ped_rng.uniform(sx * 0.1, sx * 0.25))
                yaw = 0.0
            else:
                x = float(ped_rng.uniform(sx * 0.75, sx * 0.90))
                yaw = math.pi
        else:
            x = float(ped_rng.uniform(1.0, sx - 1.0))
            lane_y = float(ped_rng.uniform(1.0, sy - 1.0))
            yaw = float(ped_rng.uniform(-math.pi, math.pi))

        base_speed = float(ped_rng.uniform(vmin, vmax))
        vel = np.array([math.cos(yaw), math.sin(yaw)]) * base_speed

        peds.append(FSMPedestrian(
            xy=np.array([x, lane_y], dtype=float),
            yaw=yaw,
            vel=vel,
            base_speed=base_speed,
            radius=radius,
            _rng=ped_rng,
        ))

    return peds


# ─────────────────────────────────────────────────────────────────────────────
# FSM step function
# ─────────────────────────────────────────────────────────────────────────────

def step_fsm_pedestrians(
    peds: List[FSMPedestrian],
    waypoints: List[np.ndarray],
    world_size: Tuple[float, float],
    dt: float,
    robot_xy: Optional[np.ndarray] = None,
    robot_radius: float = 0.3,
) -> None:
    """Advance all pedestrians by one FSM step (in-place update).

    Parameters
    ----------
    peds       : list of FSMPedestrian (mutated in-place)
    waypoints  : list of interest points
    world_size : (sx, sy) boundary
    dt         : time step [s]
    robot_xy   : robot position for pedestrian avoidance (optional)
    robot_radius : robot radius for collision avoidance [m]
    """
    sx, sy = world_size
    xmin, ymin, xmax, ymax = 0.5, 0.5, sx - 0.5, sy - 0.5

    for ped in peds:
        _fsm_transition(ped, waypoints, dt)
        _fsm_velocity_update(ped, peds, robot_xy, robot_radius, dt)

        # Integrate position
        ped.xy = ped.xy + ped.vel * dt

        # Boundary reflection
        if ped.xy[0] < xmin or ped.xy[0] > xmax:
            ped.vel[0] *= -1.0
            ped.xy[0] = float(np.clip(ped.xy[0], xmin, xmax))
            ped.yaw = math.atan2(ped.vel[1], ped.vel[0] + 1e-9)
            ped.fsm_state = PedFSMState.WANDER  # re-orient

        if ped.xy[1] < ymin or ped.xy[1] > ymax:
            ped.vel[1] *= -1.0
            ped.xy[1] = float(np.clip(ped.xy[1], ymin, ymax))
            ped.yaw = math.atan2(ped.vel[1], ped.vel[0] + 1e-9)
            ped.fsm_state = PedFSMState.WANDER


# ─────────────────────────────────────────────────────────────────────────────
# Internal FSM logic
# ─────────────────────────────────────────────────────────────────────────────

def _fsm_transition(
    ped: FSMPedestrian,
    waypoints: List[np.ndarray],
    dt: float,
) -> None:
    """Update FSM state based on current conditions."""
    rng = ped._rng

    if ped.fsm_state == PedFSMState.WANDER:
        # Pick a random waypoint or random position as next target
        if waypoints:
            idx = int(rng.integers(0, len(waypoints)))
            ped.target_xy = waypoints[idx].copy()
        else:
            # Random target anywhere in world (will be clipped at boundary)
            sx_approx = max(ped.xy[0] * 2, 10.0)
            sy_approx = max(ped.xy[1] * 2, 10.0)
            ped.target_xy = np.array([
                float(rng.uniform(1.0, sx_approx - 1.0)),
                float(rng.uniform(1.0, sy_approx - 1.0)),
            ], dtype=float)
        ped.fsm_state = PedFSMState.WALK_TO

    elif ped.fsm_state == PedFSMState.WALK_TO:
        if ped.target_xy is None:
            ped.fsm_state = PedFSMState.WANDER
            return
        dist = float(np.linalg.norm(ped.target_xy - ped.xy))
        if dist < APPROACH_DIST:
            ped.fsm_state = PedFSMState.APPROACH

    elif ped.fsm_state == PedFSMState.APPROACH:
        if ped.target_xy is None:
            ped.fsm_state = PedFSMState.WANDER
            return
        dist = float(np.linalg.norm(ped.target_xy - ped.xy))
        if dist < ARRIVED_DIST:
            ped.fsm_state = PedFSMState.VIEWING
            ped.view_timer = float(rng.uniform(DWELL_MIN, DWELL_MAX))

    elif ped.fsm_state == PedFSMState.VIEWING:
        ped.view_timer -= dt
        if ped.view_timer <= 0.0:
            # Small random turn before walking again
            ped.fsm_state = PedFSMState.TURNING

    elif ped.fsm_state == PedFSMState.TURNING:
        # Apply a random yaw rotation (simulates looking around)
        delta_yaw = float(rng.uniform(-math.pi * 0.5, math.pi * 0.5))
        ped.yaw = (ped.yaw + delta_yaw + math.pi) % (2 * math.pi) - math.pi
        ped.fsm_state = PedFSMState.WANDER


def _fsm_velocity_update(
    ped: FSMPedestrian,
    all_peds: List[FSMPedestrian],
    robot_xy: Optional[np.ndarray],
    robot_radius: float,
    dt: float,
) -> None:
    """Compute and apply velocity update based on FSM state + avoidance."""
    rng = ped._rng

    if ped.fsm_state == PedFSMState.VIEWING:
        # Decelerate to zero
        speed = float(np.linalg.norm(ped.vel))
        decel = min(speed, MAX_ACCEL * dt)
        if speed > 1e-6:
            ped.vel = ped.vel * max(0.0, 1.0 - decel / speed)
        return

    if ped.fsm_state == PedFSMState.TURNING:
        ped.vel = np.zeros(2, dtype=float)
        return

    if ped.target_xy is None or ped.fsm_state == PedFSMState.WANDER:
        return

    # ── Desired velocity toward target ───────────────────────────────────
    to_target = ped.target_xy - ped.xy
    dist_target = float(np.linalg.norm(to_target))

    if dist_target < 1e-6:
        desired_speed = 0.0
        desired_dir = np.zeros(2, dtype=float)
    else:
        desired_dir = to_target / dist_target
        if ped.fsm_state == PedFSMState.APPROACH:
            # Decelerate as we approach: speed proportional to distance
            desired_speed = ped.base_speed * min(1.0, dist_target / APPROACH_DIST)
        else:
            # Normal walking speed with Gaussian noise
            noise = float(rng.normal(0.0, SPEED_NOISE_STD))
            desired_speed = max(0.0, ped.base_speed + noise)

    desired_vel = desired_dir * desired_speed

    # ── Ped-ped SFM avoidance ────────────────────────────────────────────
    avoid_F = np.zeros(2, dtype=float)
    for other in all_peds:
        if other is ped:
            continue
        d = ped.xy - other.xy
        dist = float(np.linalg.norm(d))
        if 0 < dist < PED_PED_DIST + ped.radius + other.radius:
            avoid_F += PED_PED_REPULSION_K / max(dist, 1e-6) * (d / dist)

    # ── Robot avoidance ───────────────────────────────────────────────────
    if robot_xy is not None:
        d_robot = ped.xy - robot_xy
        dist_r = float(np.linalg.norm(d_robot))
        robot_inf = 1.5
        if 0 < dist_r < robot_inf + ped.radius + robot_radius:
            decay = robot_inf * 0.4
            mag = 1.5 * math.exp(-dist_r / max(decay, 1e-6))
            avoid_F += mag * (d_robot / dist_r)

    # ── Combine and smooth ───────────────────────────────────────────────
    target_vel = desired_vel + avoid_F

    # Clamp speed
    speed = float(np.linalg.norm(target_vel))
    max_speed = ped.base_speed * 1.5
    if speed > max_speed:
        target_vel = target_vel / speed * max_speed

    # Smooth acceleration (limit change per step)
    dv = target_vel - ped.vel
    dv_mag = float(np.linalg.norm(dv))
    max_dv = MAX_ACCEL * dt
    if dv_mag > max_dv:
        dv = dv / dv_mag * max_dv
    ped.vel = ped.vel + dv

    # Update yaw
    speed_final = float(np.linalg.norm(ped.vel))
    if speed_final > 0.1:
        ped.yaw = float(math.atan2(float(ped.vel[1]), float(ped.vel[0])))
