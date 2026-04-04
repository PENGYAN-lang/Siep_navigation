"""
Metrics collection for Proactive-SIEP paper experiments.

Computes all metrics required to support paper claims:

  personal_space_violation_rate  – how often robot enters pedestrian PS zone
  min_human_dist_stats           – distribution of minimum human distances
  social_disturbance             – group/queue crossing penalties
  path_length                    – total path length [m]
  time_to_task                   – steps / time to reach goal (or max_steps)
  smoothness                     – jerk and angular acceleration
  exploration_coverage           – grid cell coverage (explore mode)

Output is a flat dict suitable for JSON/CSV serialisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Grid coverage parameters
# ─────────────────────────────────────────────────────────────────────────────
COVERAGE_CELL_SIZE: float = 0.5   # [m] grid cell size for coverage computation


@dataclass
class EpisodeMetrics:
    """Accumulates metrics for one simulation episode."""

    # ── Trajectory record ──────────────────────────────────────────────────
    robot_xys: List[np.ndarray] = field(default_factory=list)
    robot_yaws: List[float] = field(default_factory=list)
    robot_vs: List[float] = field(default_factory=list)
    robot_ws: List[float] = field(default_factory=list)

    # ── Per-step social records ────────────────────────────────────────────
    # min_dist_to_human[t] = minimum distance to any pedestrian at step t
    min_dist_to_human: List[float] = field(default_factory=list)
    # ps_violated[t] = True if any personal space was entered
    ps_violated: List[bool] = field(default_factory=list)
    # group_crossing_events[t] = number of groups crossed through at step t
    group_crossing_events: List[int] = field(default_factory=list)

    # ── Episode outcome ───────────────────────────────────────────────────
    reached_goal: bool = False
    n_steps: int = 0
    dt: float = 0.05

    # ── World metadata ────────────────────────────────────────────────────
    world_size_xy: Tuple[float, float] = (20.0, 20.0)


def record_step(
    metrics: EpisodeMetrics,
    robot_xy: np.ndarray,
    robot_yaw: float,
    v: float,
    w: float,
    ped_xys: List[np.ndarray],
    ped_pss_sigmas: List[float],  # effective sigma_front per pedestrian
    group_ids: Optional[List[int]] = None,
) -> None:
    """Record one simulation step into EpisodeMetrics.

    Parameters
    ----------
    metrics         : metrics accumulator (mutated in-place)
    robot_xy        : robot position (2,)
    robot_yaw       : robot heading [rad]
    v, w            : commanded velocities
    ped_xys         : pedestrian positions
    ped_pss_sigmas  : effective sigma_front per pedestrian (PS boundary proxy)
    group_ids       : group cluster IDs per pedestrian (None = all singletons)
    """
    metrics.robot_xys.append(robot_xy.copy())
    metrics.robot_yaws.append(robot_yaw)
    metrics.robot_vs.append(v)
    metrics.robot_ws.append(w)
    metrics.n_steps += 1

    if not ped_xys:
        metrics.min_dist_to_human.append(float('inf'))
        metrics.ps_violated.append(False)
        metrics.group_crossing_events.append(0)
        return

    dists = np.array([float(np.linalg.norm(robot_xy - pxy)) for pxy in ped_xys])
    min_d = float(np.min(dists))
    metrics.min_dist_to_human.append(min_d)

    # Personal space violation: any pedestrian with distance < sigma_front
    ps_viol = any(
        d < sigma for d, sigma in zip(dists, ped_pss_sigmas)
    )
    metrics.ps_violated.append(ps_viol)

    # Group crossing: count groups whose centroid the robot passed near
    group_cross = 0
    if group_ids is not None:
        seen_groups = set()
        for j, gid in enumerate(group_ids):
            if gid >= 0 and gid not in seen_groups:
                seen_groups.add(gid)
                members = [k for k, g in enumerate(group_ids) if g == gid]
                centroid = np.mean([ped_xys[k] for k in members], axis=0)
                if float(np.linalg.norm(robot_xy - centroid)) < 1.5:
                    group_cross += 1
    metrics.group_crossing_events.append(group_cross)


def compute_metrics(
    metrics: EpisodeMetrics,
    goal_xy: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all metrics from accumulated episode data.

    Returns
    -------
    dict of metric_name → float value  (suitable for CSV/JSON logging)
    """
    result: Dict[str, float] = {}

    n = metrics.n_steps
    dt = metrics.dt

    # ── Personal space violation rate ─────────────────────────────────────
    if n > 0:
        result['ps_violation_rate'] = float(sum(metrics.ps_violated)) / n
    else:
        result['ps_violation_rate'] = 0.0

    # ── Minimum human distance distribution ───────────────────────────────
    finite_dists = [d for d in metrics.min_dist_to_human if math.isfinite(d)]
    if finite_dists:
        arr = np.array(finite_dists)
        result['min_human_dist_mean'] = float(np.mean(arr))
        result['min_human_dist_min'] = float(np.min(arr))
        result['min_human_dist_p10'] = float(np.percentile(arr, 10))
        result['min_human_dist_std'] = float(np.std(arr))
    else:
        result['min_human_dist_mean'] = float('nan')
        result['min_human_dist_min'] = float('nan')
        result['min_human_dist_p10'] = float('nan')
        result['min_human_dist_std'] = float('nan')

    # ── Social disturbance (group crossing events) ────────────────────────
    result['social_disturbance_total'] = float(sum(metrics.group_crossing_events))
    result['social_disturbance_rate'] = (
        result['social_disturbance_total'] / max(n, 1)
    )

    # ── Path length ────────────────────────────────────────────────────────
    if len(metrics.robot_xys) > 1:
        xys = np.array(metrics.robot_xys)
        diffs = np.diff(xys, axis=0)
        result['path_length'] = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        result['path_length'] = 0.0

    # ── Time to task ──────────────────────────────────────────────────────
    result['steps_to_goal'] = float(n)
    result['time_to_goal_s'] = float(n) * dt
    result['reached_goal'] = float(metrics.reached_goal)

    # ── Smoothness: jerk and angular acceleration ─────────────────────────
    if len(metrics.robot_vs) > 2:
        vs = np.array(metrics.robot_vs)
        ws = np.array(metrics.robot_ws)
        # Linear jerk = d²v/dt²
        lin_accel = np.diff(vs) / dt
        lin_jerk = np.diff(lin_accel) / dt
        # Angular jerk = d²ω/dt²
        ang_accel = np.diff(ws) / dt
        ang_jerk = np.diff(ang_accel) / dt

        result['linear_jerk_rms'] = float(np.sqrt(np.mean(lin_jerk ** 2)))
        result['angular_jerk_rms'] = float(np.sqrt(np.mean(ang_jerk ** 2)))
        result['linear_accel_rms'] = float(np.sqrt(np.mean(lin_accel ** 2)))
    else:
        result['linear_jerk_rms'] = 0.0
        result['angular_jerk_rms'] = 0.0
        result['linear_accel_rms'] = 0.0

    # ── Exploration coverage ──────────────────────────────────────────────
    if metrics.robot_xys:
        result['exploration_coverage'] = _compute_coverage(
            metrics.robot_xys, metrics.world_size_xy
        )
    else:
        result['exploration_coverage'] = 0.0

    return result


def _compute_coverage(
    xys: List[np.ndarray],
    world_size: Tuple[float, float],
) -> float:
    """Fraction of world grid cells visited by the robot."""
    sx, sy = world_size
    nx = int(sx / COVERAGE_CELL_SIZE) + 1
    ny = int(sy / COVERAGE_CELL_SIZE) + 1
    visited = set()
    for xy in xys:
        cx = int(float(xy[0]) / COVERAGE_CELL_SIZE)
        cy = int(float(xy[1]) / COVERAGE_CELL_SIZE)
        if 0 <= cx < nx and 0 <= cy < ny:
            visited.add((cx, cy))
    total_cells = nx * ny
    return float(len(visited)) / max(total_cells, 1)
