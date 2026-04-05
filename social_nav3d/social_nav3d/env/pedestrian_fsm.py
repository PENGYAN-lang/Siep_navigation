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
    WANDER       = auto()
    WALK_TO      = auto()
    APPROACH     = auto()
    VIEWING      = auto()
    TURNING      = auto()
    GROUP_FOLLOW = auto()   # group members follow their leader
    PATROL       = auto()   # staff walk a fixed patrol route loop


class PedType(Enum):
    VISITOR = auto()       # museum visitors: slow, long dwell at exhibits
    GROUP   = auto()       # group visitors: move together, follow leader
    STAFF   = auto()       # staff/transit: fast, no dwell, patrol routes


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

# Staff patrol speed — reduced so staff is not much faster than robot
STAFF_SPEED_MIN: float = 0.4
STAFF_SPEED_MAX: float = 0.6

# Group follow parameters
GROUP_MAX_SEPARATION: float = 1.5  # [m] max distance from leader
GROUP_FOLLOW_OFFSET: float = 0.8   # [m] follow distance behind leader

# Visitor dwell times (longer than base)
VISITOR_DWELL_MIN: float = 8.0    # [s]
VISITOR_DWELL_MAX: float = 20.0   # [s]


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
    # Typed behaviour fields
    ped_type: PedType = PedType.VISITOR
    torso_color: list = field(default_factory=lambda: [0.2, 0.6, 0.9, 1.0])  # RGBA
    leader_idx: Optional[int] = None             # for GROUP type: index of leader ped
    patrol_waypoints: Optional[List[np.ndarray]] = field(default=None, repr=False)  # for STAFF
    patrol_idx: int = 0                          # current patrol waypoint index

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


def create_typed_fsm_pedestrians(
    n: int,
    world_size: Tuple[float, float],
    speed_range: Tuple[float, float],
    radius: float,
    waypoints: List[np.ndarray],
    seed: int = 0,
    ped_types: Optional[List[str]] = None,
    wall_aabbs: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[FSMPedestrian]:
    """Factory: create N FSMPedestrians with 3 behavioral types.

    Type distribution (when ped_types=None, auto-generated):
      - 8 visitors (slow, 0.2-0.5 m/s, long dwell 8-20s)
      - 3 group visitors (green, move together)
      - 2 staff (orange, fast patrol, no dwell)

    For n != 13, the distribution scales proportionally.

    Parameters
    ----------
    n         : total pedestrian count
    world_size : (sx, sy) world dimensions [m]
    speed_range : (vmin, vmax) walking speed range [m/s]
    radius    : pedestrian radius [m]
    waypoints : list of interest points (exhibits, corners etc.)
    seed      : random seed for reproducibility
    ped_types : optional list of 'visitor'/'group'/'staff' (length n)
    wall_aabbs : optional list of wall AABBs; when provided, spawn positions
                 that collide with walls are rejected and resampled.
    """
    rng = np.random.default_rng(seed)
    sx, sy = world_size

    def _safe_spawn(rng_: np.random.Generator) -> Tuple[float, float]:
        """Sample a spawn (x, y) that is not inside a wall."""
        for _ in range(50):
            x_ = float(rng_.uniform(1.0, sx - 1.0))
            y_ = float(rng_.uniform(1.0, sy - 1.0))
            if wall_aabbs is None:
                return x_, y_
            if not _check_wall_collision(np.array([x_, y_]), radius, wall_aabbs):
                return x_, y_
        # Fallback: return last sample even if it overlaps (rare edge case)
        return x_, y_

    # ── Determine type list ───────────────────────────────────────────────
    if ped_types is not None:
        type_list = [t.lower() for t in ped_types]
    else:
        if n == 13:
            n_visitor, n_group, n_staff = 8, 3, 2
        else:
            n_staff = max(1, round(n * 2 / 13))
            n_group = max(1, round(n * 3 / 13))
            n_visitor = max(1, n - n_staff - n_group)
            # Adjust to exactly n
            while n_visitor + n_group + n_staff > n:
                n_visitor -= 1
            while n_visitor + n_group + n_staff < n:
                n_visitor += 1
        type_list = (
            ["visitor"] * n_visitor
            + ["group"] * n_group
            + ["staff"] * n_staff
        )

    # ── Color palettes ────────────────────────────────────────────────────
    visitor_colors = [
        [0.2, 0.5, 0.8, 1.0],
        [0.3, 0.4, 0.9, 1.0],
        [0.1, 0.6, 0.7, 1.0],
    ]
    group_colors = [
        [0.2, 0.7, 0.3, 1.0],
        [0.3, 0.8, 0.2, 1.0],
        [0.1, 0.65, 0.35, 1.0],
    ]
    staff_color = [0.9, 0.5, 0.1, 1.0]

    # ── Build patrol routes (for STAFF) ───────────────────────────────────
    def _make_patrol_route(rng_: np.random.Generator) -> List[np.ndarray]:
        """Generate a ~4-waypoint loop through the scene."""
        margin = 1.5
        xs = [margin, sx - margin, sx - margin, margin]
        ys = [margin, margin, sy - margin, sy - margin]
        # Slight random offset so each staff has a different route
        offsets_x = rng_.uniform(-sx * 0.1, sx * 0.1, 4)
        offsets_y = rng_.uniform(-sy * 0.1, sy * 0.1, 4)
        return [
            np.array([
                float(np.clip(xs[k] + offsets_x[k], margin, sx - margin)),
                float(np.clip(ys[k] + offsets_y[k], margin, sy - margin)),
            ], dtype=float)
            for k in range(4)
        ]

    # ── Create pedestrians ────────────────────────────────────────────────
    peds: List[FSMPedestrian] = []
    visitor_ci = 0
    group_ci = 0
    group_leader_idx: Optional[int] = None  # index of first group ped (leader)

    for i, ptype in enumerate(type_list):
        ped_seed = int(rng.integers(0, 2 ** 31))
        ped_rng = np.random.default_rng(ped_seed)

        # Spawn position: uniform across world, rejecting wall-colliding spots
        x, y = _safe_spawn(ped_rng)
        yaw = float(ped_rng.uniform(-math.pi, math.pi))

        if ptype == "visitor":
            base_speed = float(ped_rng.uniform(0.2, 0.5))
            color = list(visitor_colors[visitor_ci % len(visitor_colors)])
            visitor_ci += 1
            vel = np.array([math.cos(yaw), math.sin(yaw)]) * base_speed
            ped = FSMPedestrian(
                xy=np.array([x, y], dtype=float),
                yaw=yaw,
                vel=vel,
                base_speed=base_speed,
                radius=radius,
                _rng=ped_rng,
                ped_type=PedType.VISITOR,
                torso_color=color,
                fsm_state=PedFSMState.WANDER,
            )

        elif ptype == "group":
            base_speed = float(ped_rng.uniform(speed_range[0], speed_range[1]))
            color = list(group_colors[group_ci % len(group_colors)])
            group_ci += 1
            vel = np.array([math.cos(yaw), math.sin(yaw)]) * base_speed

            is_leader = group_leader_idx is None
            if is_leader:
                group_leader_idx = i
                fsm_state = PedFSMState.WALK_TO
                leader_idx = None
            else:
                fsm_state = PedFSMState.GROUP_FOLLOW
                leader_idx = group_leader_idx

            ped = FSMPedestrian(
                xy=np.array([x, y], dtype=float),
                yaw=yaw,
                vel=vel,
                base_speed=base_speed,
                radius=radius,
                _rng=ped_rng,
                ped_type=PedType.GROUP,
                torso_color=color,
                fsm_state=fsm_state,
                leader_idx=leader_idx,
            )

        else:  # staff
            base_speed = float(ped_rng.uniform(STAFF_SPEED_MIN, STAFF_SPEED_MAX))
            patrol_wps = _make_patrol_route(ped_rng)
            # Staff spawn near first patrol waypoint
            x = float(np.clip(patrol_wps[0][0] + ped_rng.uniform(-0.5, 0.5), 0.5, sx - 0.5))
            y = float(np.clip(patrol_wps[0][1] + ped_rng.uniform(-0.5, 0.5), 0.5, sy - 0.5))
            vel = np.array([math.cos(yaw), math.sin(yaw)]) * base_speed
            ped = FSMPedestrian(
                xy=np.array([x, y], dtype=float),
                yaw=yaw,
                vel=vel,
                base_speed=base_speed,
                radius=radius,
                _rng=ped_rng,
                ped_type=PedType.STAFF,
                torso_color=list(staff_color),
                fsm_state=PedFSMState.PATROL,
                patrol_waypoints=patrol_wps,
                patrol_idx=0,
                target_xy=patrol_wps[0].copy(),
            )

        peds.append(ped)

    return peds


# ─────────────────────────────────────────────────────────────────────────────
# FSM step function
# ─────────────────────────────────────────────────────────────────────────────

def _check_wall_collision(
    xy: np.ndarray,
    radius: float,
    wall_aabbs: List[Tuple[float, float, float, float]],
) -> bool:
    """Return True if *xy* (with *radius*) overlaps any wall AABB."""
    px, py = float(xy[0]), float(xy[1])
    for xmin_w, ymin_w, xmax_w, ymax_w in wall_aabbs:
        if (px + radius > xmin_w and px - radius < xmax_w and
                py + radius > ymin_w and py - radius < ymax_w):
            return True
    return False


def step_fsm_pedestrians(
    peds: List[FSMPedestrian],
    waypoints: List[np.ndarray],
    world_size: Tuple[float, float],
    dt: float,
    robot_xy: Optional[np.ndarray] = None,
    robot_radius: float = 0.3,
    wall_aabbs: Optional[List[Tuple[float, float, float, float]]] = None,
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
    wall_aabbs : optional list of ``(xmin, ymin, xmax, ymax)`` wall
                 bounding boxes for internal wall collision detection.
                 When provided, pedestrians will not walk through walls.
    """
    sx, sy = world_size
    xmin, ymin, xmax, ymax = 0.5, 0.5, sx - 0.5, sy - 0.5

    # Update group follower targets (must happen before FSM step)
    for ped in peds:
        if ped.fsm_state == PedFSMState.GROUP_FOLLOW and ped.leader_idx is not None:
            if 0 <= ped.leader_idx < len(peds):
                leader = peds[ped.leader_idx]
                offset_dir = np.array([
                    math.cos(leader.yaw + math.pi),
                    math.sin(leader.yaw + math.pi),
                ])
                ped.target_xy = leader.xy + offset_dir * GROUP_FOLLOW_OFFSET

    for ped in peds:
        _fsm_transition(ped, waypoints, dt)
        _fsm_velocity_update(ped, peds, robot_xy, robot_radius, dt)

        # Integrate position with wall collision sliding
        proposed = ped.xy + ped.vel * dt

        if wall_aabbs is not None and _check_wall_collision(proposed, ped.radius, wall_aabbs):
            # Try sliding: move only in X
            proposed_x = ped.xy + np.array([ped.vel[0] * dt, 0.0])
            # Try sliding: move only in Y
            proposed_y = ped.xy + np.array([0.0, ped.vel[1] * dt])

            can_x = not _check_wall_collision(proposed_x, ped.radius, wall_aabbs)
            can_y = not _check_wall_collision(proposed_y, ped.radius, wall_aabbs)

            if can_x:
                proposed = proposed_x
                ped.vel[1] = 0.0
            elif can_y:
                proposed = proposed_y
                ped.vel[0] = 0.0
            else:
                # Fully blocked — stop and pick new target
                proposed = ped.xy.copy()
                ped.vel[:] = 0.0
                ped.fsm_state = PedFSMState.WANDER

        ped.xy = proposed

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
            if ped.ped_type == PedType.STAFF:
                # Staff don't stop, continue to next patrol waypoint
                ped.fsm_state = PedFSMState.PATROL
            elif ped.ped_type == PedType.GROUP:
                # Group members don't dwell; leader picks a new target, followers keep following
                if ped.leader_idx is None:
                    # This ped is the group leader — turn and pick a new waypoint
                    ped.fsm_state = PedFSMState.TURNING
                else:
                    ped.fsm_state = PedFSMState.GROUP_FOLLOW
            else:
                ped.fsm_state = PedFSMState.VIEWING
                if ped.ped_type == PedType.VISITOR:
                    ped.view_timer = float(rng.uniform(VISITOR_DWELL_MIN, VISITOR_DWELL_MAX))
                else:
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

    elif ped.fsm_state == PedFSMState.PATROL:
        # Staff: advance along patrol waypoints without stopping
        if ped.patrol_waypoints and len(ped.patrol_waypoints) > 0:
            current_wp = ped.patrol_waypoints[ped.patrol_idx]
            dist = float(np.linalg.norm(current_wp - ped.xy))
            if dist < ARRIVED_DIST * 1.5:  # reach next waypoint
                ped.patrol_idx = (ped.patrol_idx + 1) % len(ped.patrol_waypoints)
                ped.target_xy = ped.patrol_waypoints[ped.patrol_idx].copy()
            else:
                ped.target_xy = current_wp.copy()
        # stay in PATROL state (never exits)

    elif ped.fsm_state == PedFSMState.GROUP_FOLLOW:
        # Group followers: target_xy is updated externally in step_fsm_pedestrians
        pass


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

    if ped.fsm_state == PedFSMState.GROUP_FOLLOW:
        if ped.target_xy is not None:
            to_target = ped.target_xy - ped.xy
            dist = float(np.linalg.norm(to_target))
            if dist > GROUP_FOLLOW_OFFSET:
                desired_dir = to_target / dist
                desired_speed = min(ped.base_speed * 1.1, dist * 1.5)
                desired_vel = desired_dir * desired_speed
                # Ped-ped avoidance
                avoid_F = np.zeros(2, dtype=float)
                for other in all_peds:
                    if other is ped:
                        continue
                    d = ped.xy - other.xy
                    d_dist = float(np.linalg.norm(d))
                    if 0 < d_dist < PED_PED_DIST + ped.radius + other.radius:
                        avoid_F += PED_PED_REPULSION_K / max(d_dist, 1e-6) * (d / d_dist)
                if robot_xy is not None:
                    d_robot = ped.xy - robot_xy
                    dist_r = float(np.linalg.norm(d_robot))
                    robot_inf = 1.5
                    if 0 < dist_r < robot_inf + ped.radius + robot_radius:
                        decay = robot_inf * 0.4
                        mag = 1.5 * math.exp(-dist_r / max(decay, 1e-6))
                        avoid_F += mag * (d_robot / dist_r)
                target_vel = desired_vel + avoid_F
                speed_tv = float(np.linalg.norm(target_vel))
                max_speed = ped.base_speed * 1.2
                if speed_tv > max_speed:
                    target_vel = target_vel / speed_tv * max_speed
                dv = target_vel - ped.vel
                dv_mag = float(np.linalg.norm(dv))
                max_dv = MAX_ACCEL * dt
                if dv_mag > max_dv:
                    dv = dv / dv_mag * max_dv
                ped.vel = ped.vel + dv
                speed_final = float(np.linalg.norm(ped.vel))
                if speed_final > 0.1:
                    ped.yaw = float(math.atan2(float(ped.vel[1]), float(ped.vel[0])))
            else:
                # Close enough — decelerate
                speed = float(np.linalg.norm(ped.vel))
                decel = min(speed, MAX_ACCEL * dt)
                if speed > 1e-6:
                    ped.vel = ped.vel * max(0.0, 1.0 - decel / speed)
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
    max_speed = ped.base_speed * 1.2
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
