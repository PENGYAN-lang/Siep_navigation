"""
Context inference module for Proactive-SIEP.

Infers social context z_tau from instantaneous observations:
  - robot pose, pedestrian states, LiDAR geometry

Outputs
-------
z : SocialContext
    Struct containing all context features:
      crowd_density       float   pedestrians within CROWD_RADIUS [m]
      is_corridor         bool    narrow bilateral obstacle channel
      dominant_flow_dir   float   mean heading of co-flow pedestrians [rad]
      has_groups          bool    at least one pedestrian cluster > 1 member
      context_type        str     categorical: OPEN | PASSING | YIELDING |
                                  QUEUE_FLOW | GROUP_INTERACTION

Per-pedestrian context-dependent weights w_j(z) and context force F_ctx(z).

Paper equation mapping
----------------------
  w_j(z_tau)   ←  context_weight(ped_j, z)
  F_ctx(z_tau) ←  context_force(z, robot_vel)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants (tuned to museum / indoor social navigation)
# ─────────────────────────────────────────────────────────────────────────────

CROWD_RADIUS: float = 4.0          # [m] radius for density counting
CORRIDOR_THRESHOLD: float = 2.5    # [m] max free width to classify as corridor
GROUP_DIST_THRESHOLD: float = 1.5  # [m] pedestrians within this radius form group
# Head-on detection: pedestrians moving within 30° of directly opposite direction
HEAD_ON_ANGLE_THRESH: float = math.pi - math.radians(30.0)  # ≈ 150° velocity angle
SAME_DIR_CONE: float = math.radians(60.0)  # [rad] half-cone for co-flow detection


# ─────────────────────────────────────────────────────────────────────────────
# Context type enum
# ─────────────────────────────────────────────────────────────────────────────

class ContextType(Enum):
    """Categorical social context for the current robot situation."""
    OPEN_SPACE        = auto()  # open area, no dominant interaction
    PASSING           = auto()  # pedestrian passing alongside (co-directional)
    YIELDING          = auto()  # pedestrian coming head-on, narrow passage
    QUEUE_FLOW        = auto()  # queue / uni-directional crowd flow
    GROUP_INTERACTION = auto()  # stationary or slow group nearby


@dataclass
class SocialContext:
    """All social context features used in Proactive-SIEP objective.

    Attributes
    ----------
    crowd_density : float
        Number of pedestrians within CROWD_RADIUS.
    is_corridor : bool
        Whether the robot is in a narrow corridor (bilateral confinement).
    dominant_flow_dir : float
        Mean heading of co-directional pedestrians [rad].  NaN if none.
    has_groups : bool
        True if at least one pedestrian cluster (≥2 members) is detected.
    context_type : ContextType
        Categorical classification of the current social situation.
    ped_facing_robot : List[bool]
        For each pedestrian: True if they face toward the robot (perspective-aware).
    group_ids : List[int]
        Cluster index per pedestrian (-1 = singleton).
    """
    crowd_density: float = 0.0
    is_corridor: bool = False
    dominant_flow_dir: float = float('nan')
    has_groups: bool = False
    context_type: ContextType = ContextType.OPEN_SPACE
    ped_facing_robot: List[bool] = field(default_factory=list)
    group_ids: List[int] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Context inference function
# ─────────────────────────────────────────────────────────────────────────────

def infer_context(
    robot_xy: np.ndarray,
    robot_yaw: float,
    ped_xys: List[np.ndarray],
    ped_yaws: List[float],
    ped_vels: List[np.ndarray],
    lidar_dists: Optional[np.ndarray] = None,
    lidar_angles: Optional[np.ndarray] = None,
) -> SocialContext:
    """Infer social context z from current observations.

    Parameters
    ----------
    robot_xy, robot_yaw : robot pose.
    ped_xys, ped_yaws, ped_vels : pedestrian states.
    lidar_dists, lidar_angles : 2-D LiDAR scan (optional).  When provided,
        used to detect corridor geometry.

    Returns
    -------
    SocialContext with all fields populated.
    """
    n_peds = len(ped_xys)
    ctx = SocialContext()

    if n_peds == 0:
        ctx.context_type = ContextType.OPEN_SPACE
        return ctx

    # ── 1. Crowd density ────────────────────────────────────────────────────
    dists_to_robot = np.array([
        float(np.linalg.norm(pxy - robot_xy)) for pxy in ped_xys
    ])
    ctx.crowd_density = float(np.sum(dists_to_robot < CROWD_RADIUS))

    # ── 2. Corridor detection (bilateral obstacle check via LiDAR) ──────────
    ctx.is_corridor = _detect_corridor(lidar_dists, lidar_angles, robot_yaw)

    # ── 3. Pedestrian facing direction (perspective-aware) ───────────────────
    ctx.ped_facing_robot = _pedestrians_facing_robot(robot_xy, ped_xys, ped_yaws)

    # ── 4. Group detection (single-linkage clustering) ───────────────────────
    ctx.group_ids = _cluster_pedestrians(ped_xys)
    max_group = max(ctx.group_ids) if ctx.group_ids else -1
    if max_group >= 0:
        group_sizes = [ctx.group_ids.count(g) for g in range(max_group + 1)]
        ctx.has_groups = any(s >= 2 for s in group_sizes)

    # ── 5. Co-directional and head-on pedestrian analysis ───────────────────
    robot_vel = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=float)
    n_coflow = 0
    n_headon = 0
    flow_dirs: List[float] = []

    for i, (pxy, pyaw, pvel) in enumerate(zip(ped_xys, ped_yaws, ped_vels)):
        pspeed = float(np.linalg.norm(pvel))
        if pspeed < 0.1:
            continue  # stationary: skip for flow analysis
        pdir = pvel / pspeed
        cos_angle = float(np.dot(robot_vel, pdir))
        angle = math.acos(float(np.clip(cos_angle, -1.0, 1.0)))
        if angle < SAME_DIR_CONE:
            n_coflow += 1
            flow_dirs.append(pyaw)
        elif angle > HEAD_ON_ANGLE_THRESH:
            # angle > 150°: pedestrian moving nearly opposite to robot → head-on candidate
            # Additionally check that the pedestrian is in front of the robot
            rel_angle = math.atan2(float(pxy[1] - robot_xy[1]),
                                   float(pxy[0] - robot_xy[0]))
            ang_diff = abs(_wrap_pi(rel_angle - robot_yaw))
            if ang_diff < math.radians(60.0):
                n_headon += 1

    if flow_dirs:
        ctx.dominant_flow_dir = float(np.angle(
            np.mean(np.exp(1j * np.array(flow_dirs)))
        ))

    # ── 6. Categorical context type ─────────────────────────────────────────
    nearby_stopped = sum(
        1 for i, pvel in enumerate(ped_vels)
        if dists_to_robot[i] < CROWD_RADIUS
        and float(np.linalg.norm(pvel)) < 0.2
    )

    if ctx.has_groups and nearby_stopped >= 2:
        ctx.context_type = ContextType.GROUP_INTERACTION
    elif n_headon >= 1 and ctx.is_corridor:
        ctx.context_type = ContextType.YIELDING
    elif n_coflow >= 2:
        ctx.context_type = ContextType.QUEUE_FLOW
    elif n_coflow >= 1 or n_headon >= 1:
        ctx.context_type = ContextType.PASSING
    else:
        ctx.context_type = ContextType.OPEN_SPACE

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Per-pedestrian context weight  w_j(z_tau)
# ─────────────────────────────────────────────────────────────────────────────

def context_weight(
    ped_idx: int,
    ped_xy: np.ndarray,
    ped_yaw: float,
    ped_vel: np.ndarray,
    robot_xy: np.ndarray,
    robot_yaw: float,
    ctx: SocialContext,
) -> float:
    """Compute w_j(z_tau): context-dependent scale on F_human,j.

    Paper equation: w_j(z_tau) modulates the personal-space repulsion force
    from pedestrian j.  Higher weight = stronger avoidance.

    Design principles
    -----------------
    - Pedestrians facing the robot need larger avoidance weight (perspective-
      aware: they may move toward us).
    - In YIELDING context, head-on pedestrian weight is maximised.
    - In GROUP_INTERACTION, group members collectively increase weight.
    - In QUEUE_FLOW, co-directional pedestrian weight is reduced (they flow
      with us).
    - Density scaling: higher crowd density → moderate weight reduction (to
      avoid infeasible planning).
    """
    base_w = 1.0

    # Perspective-aware modulation: pedestrian facing toward robot → +40%
    is_facing = (
        ped_idx < len(ctx.ped_facing_robot) and ctx.ped_facing_robot[ped_idx]
    )
    if is_facing:
        base_w *= 1.4

    # Context-type modulation
    if ctx.context_type == ContextType.YIELDING:
        # Check if this pedestrian is the head-on one
        rel = ped_xy - robot_xy
        rel_ang = math.atan2(float(rel[1]), float(rel[0]))
        ang_diff = abs(_wrap_pi(rel_ang - robot_yaw))
        if ang_diff < math.radians(70.0):
            base_w *= 1.8  # head-on: strong avoidance

    elif ctx.context_type == ContextType.GROUP_INTERACTION:
        gid = ctx.group_ids[ped_idx] if ped_idx < len(ctx.group_ids) else -1
        if gid >= 0:
            base_w *= 1.3  # group members together
        if float(np.linalg.norm(ped_vel)) < 0.2:
            base_w *= 1.2  # stopped pedestrian: don't disturb

    elif ctx.context_type == ContextType.QUEUE_FLOW:
        pspeed = float(np.linalg.norm(ped_vel))
        if pspeed > 0.1:
            cos_a = float(np.dot(
                np.array([math.cos(robot_yaw), math.sin(robot_yaw)]),
                ped_vel / pspeed,
            ))
            if cos_a > math.cos(SAME_DIR_CONE):
                base_w *= 0.7  # co-flow: reduce avoidance

    elif ctx.context_type == ContextType.PASSING:
        # Passing pedestrian: slight increase for safety
        base_w *= 1.1

    # Density scaling (avoid over-reactive planning in crowded spaces)
    if ctx.crowd_density > 4:
        base_w *= 0.85

    return float(np.clip(base_w, 0.2, 3.0))


# ─────────────────────────────────────────────────────────────────────────────
# Context force  F_ctx(z_tau)
# ─────────────────────────────────────────────────────────────────────────────

def context_force(
    robot_xy: np.ndarray,
    robot_yaw: float,
    robot_vel: np.ndarray,
    ctx: SocialContext,
    k_ctx: float = 0.4,
) -> np.ndarray:
    """Compute F_ctx(z_tau): context-level additional force on the robot.

    This captures situation-level forces that are not captured by pairwise
    human interaction forces.

    Paper equation: F_ctx provides an additional stimulus encoding:
    - Flow following: nudge toward dominant flow direction in QUEUE_FLOW.
    - Yield shift: lateral offset force in YIELDING.
    - Group stand-off: gentle repulsion from group centroid.
    """
    F = np.zeros(2, dtype=float)

    if ctx.context_type == ContextType.QUEUE_FLOW:
        # Encourage robot to align with crowd flow direction
        if not math.isnan(ctx.dominant_flow_dir):
            flow_vec = np.array([
                math.cos(ctx.dominant_flow_dir),
                math.sin(ctx.dominant_flow_dir),
            ], dtype=float)
            # Only add flow force if robot isn't already well aligned
            robot_dir = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
            cos_a = float(np.dot(robot_dir, flow_vec))
            if cos_a < 0.9:
                F += k_ctx * flow_vec

    elif ctx.context_type == ContextType.YIELDING:
        # Lateral rightward shift (convention: keep right) using robot's
        # forward-perpendicular to the right.
        right = np.array([math.sin(robot_yaw), -math.cos(robot_yaw)], dtype=float)
        F += k_ctx * right

    # In GROUP_INTERACTION and OPEN_SPACE no additional context force.
    return F


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_corridor(
    lidar_dists: Optional[np.ndarray],
    lidar_angles: Optional[np.ndarray],
    robot_yaw: float,
) -> bool:
    """Return True if LiDAR shows bilateral confinement (corridor-like).

    We check the left and right perpendicular sectors.  If both have close
    obstacles (within CORRIDOR_THRESHOLD), the robot is in a corridor.
    """
    if lidar_dists is None or lidar_angles is None:
        return False

    left_angle = robot_yaw + math.pi / 2.0
    right_angle = robot_yaw - math.pi / 2.0
    cone_half = math.radians(30.0)

    def _min_in_sector(center_ang: float) -> float:
        diffs = np.abs(np.array([
            _wrap_pi(a - center_ang) for a in lidar_angles
        ]))
        mask = diffs < cone_half
        if not np.any(mask):
            return float('inf')
        return float(np.min(lidar_dists[mask]))

    left_min = _min_in_sector(left_angle)
    right_min = _min_in_sector(right_angle)
    return left_min < CORRIDOR_THRESHOLD and right_min < CORRIDOR_THRESHOLD


def _pedestrians_facing_robot(
    robot_xy: np.ndarray,
    ped_xys: List[np.ndarray],
    ped_yaws: List[float],
) -> List[bool]:
    """For each pedestrian, True if their facing direction points toward robot."""
    results = []
    for pxy, pyaw in zip(ped_xys, ped_yaws):
        to_robot = robot_xy - pxy
        dist = float(np.linalg.norm(to_robot))
        if dist < 1e-6:
            results.append(False)
            continue
        to_robot_angle = math.atan2(float(to_robot[1]), float(to_robot[0]))
        ang_diff = abs(_wrap_pi(to_robot_angle - pyaw))
        # "Facing toward" if their heading points within 90° toward robot
        results.append(ang_diff < math.radians(90.0))
    return results


def _cluster_pedestrians(ped_xys: List[np.ndarray]) -> List[int]:
    """Simple single-linkage clustering by distance threshold.

    Returns a group_id per pedestrian: -1 means singleton,
    0..K-1 are cluster IDs (clusters with ≥2 members).
    """
    n = len(ped_xys)
    if n == 0:
        return []

    labels = [-1] * n
    group_counter = 0

    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(ped_xys[i] - ped_xys[j]))
            if d < GROUP_DIST_THRESHOLD:
                # Merge i and j into same group
                if labels[i] == -1 and labels[j] == -1:
                    labels[i] = group_counter
                    labels[j] = group_counter
                    group_counter += 1
                elif labels[i] == -1:
                    labels[i] = labels[j]
                elif labels[j] == -1:
                    labels[j] = labels[i]
                else:
                    # Merge two existing groups
                    old_id = labels[j]
                    new_id = labels[i]
                    labels = [new_id if l == old_id else l for l in labels]

    return labels


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
