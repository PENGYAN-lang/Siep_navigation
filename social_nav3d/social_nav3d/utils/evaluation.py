"""
Evaluation metrics for SIEP ablation studies.

Provides lightweight, incremental tracking of key social-navigation metrics
that can be logged per step and summarised at the end of a trial.  These
metrics are the direct ablation signals for comparing:

  * Baseline SIEP (fixed weights)
  * + CA-SIEP  (context-adaptive weights)
  * + UMPS     (uncertainty-modulated personal space)
  * + GF       (group-level force)
  * Full model (all three extensions)

Tracked metrics
---------------
ps_violations : int
    Number of simulation steps in which the robot's centre enters the 0.5 σ
    iso-contour of any pedestrian's personal space (a hard-intrusion event).
ps_severity : float
    Cumulative personal-space Gaussian cost summed over all pedestrian-steps,
    measuring how deeply and frequently the robot intrudes.
path_length : float
    Euclidean arc length of the robot trajectory [m].
steps : int
    Total simulation steps taken.
coverage_area : float
    Estimated covered area in explore mode [m²] – computed as the area of the
    convex hull of recorded visit positions (requires scipy, degrades gracefully
    to NaN if unavailable).

Derived metrics
---------------
nav_efficiency : float
    Straight-line distance from start to end divided by path_length.
    Higher is better (1.0 = perfectly straight path, 0 = no net progress).
mean_ps_cost_per_step : float
    ps_severity / steps.  Lower is better.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .social import PersonalSpace


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _aniso_gaussian_cost(
    robot_xy: np.ndarray,
    ped_xy: np.ndarray,
    ped_yaw: float,
    ps: PersonalSpace,
) -> float:
    """Anisotropic Gaussian cost in [0, 1]; 1.0 = robot at pedestrian centre."""
    d = robot_xy - ped_xy
    c, s = math.cos(-ped_yaw), math.sin(-ped_yaw)
    dx = c * d[0] - s * d[1]
    dy = s * d[0] + c * d[1]
    sigma_x = ps.sigma_front if dx >= 0 else ps.sigma_back
    sigma_y = ps.sigma_side
    ex = (dx * dx) / (2.0 * sigma_x ** 2 + 1e-9)
    ey = (dy * dy) / (2.0 * sigma_y ** 2 + 1e-9)
    return float(math.exp(-(ex + ey)))


# ─────────────────────────────────────────────────────────────────────────────
# Main tracker
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationTracker:
    """Incremental tracker for social-navigation evaluation metrics.

    Args:
        violation_threshold: Anisotropic-Gaussian cost above which a step is
            counted as a personal-space violation.  Default 0.6 corresponds
            roughly to the robot being within ~0.5 σ of the pedestrian centre.
    """

    def __init__(self, violation_threshold: float = 0.6) -> None:
        self.violation_threshold = violation_threshold

        # Accumulators
        self._ps_violations: int = 0
        self._ps_severity: float = 0.0
        self._path_length: float = 0.0
        self._steps: int = 0
        self._prev_xy: Optional[np.ndarray] = None
        self._start_xy: Optional[np.ndarray] = None
        self._end_xy: Optional[np.ndarray] = None
        self._visit_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(
        self,
        robot_xy: np.ndarray,
        peds: List[Tuple[np.ndarray, float, np.ndarray, PersonalSpace]],
    ) -> None:
        """Record one simulation step.

        Args:
            robot_xy: Current robot position (2,).
            peds:     List of (xy, yaw, vel, ps) pedestrian tuples.
        """
        if self._start_xy is None:
            self._start_xy = robot_xy.copy()

        # Path length
        if self._prev_xy is not None:
            self._path_length += float(np.linalg.norm(robot_xy - self._prev_xy))
        self._prev_xy = robot_xy.copy()
        self._end_xy = robot_xy.copy()
        self._visit_history.append(robot_xy.copy())
        self._steps += 1

        # Personal-space metrics
        for (ped_xy, ped_yaw, _, ps) in peds:
            cost = _aniso_gaussian_cost(robot_xy, ped_xy, ped_yaw, ps)
            self._ps_severity += cost
            if cost >= self.violation_threshold:
                self._ps_violations += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict of all tracked and derived metrics.

        Returns:
            Dictionary with keys:
            ``steps``, ``ps_violations``, ``ps_severity``,
            ``path_length``, ``nav_efficiency``,
            ``mean_ps_cost_per_step``, ``coverage_area``.
        """
        steps = max(self._steps, 1)

        nav_eff = 0.0
        if self._start_xy is not None and self._end_xy is not None:
            net_dist = float(np.linalg.norm(self._end_xy - self._start_xy))
            nav_eff = net_dist / max(self._path_length, 1e-6)

        coverage = self._compute_coverage()

        return {
            "steps": self._steps,
            "ps_violations": self._ps_violations,
            "ps_severity": round(self._ps_severity, 3),
            "mean_ps_cost_per_step": round(self._ps_severity / steps, 5),
            "path_length_m": round(self._path_length, 3),
            "nav_efficiency": round(nav_eff, 4),
            "coverage_area_m2": round(coverage, 2) if coverage is not None else None,
        }

    def _compute_coverage(self) -> Optional[float]:
        """Estimate explored area as convex hull area of visit history."""
        if len(self._visit_history) < 3:
            return None
        try:
            from scipy.spatial import ConvexHull
            pts = np.array(self._visit_history, dtype=float)
            hull = ConvexHull(pts)
            return float(hull.volume)  # 'volume' = area in 2-D
        except Exception:
            return None

    def print_summary(self, label: str = "") -> None:
        """Pretty-print the evaluation summary."""
        s = self.summary()
        tag = f"[{label}] " if label else ""
        print(f"\n{tag}── Evaluation Summary ──────────────────────────")
        print(f"  Steps                  : {s['steps']}")
        print(f"  PS violations          : {s['ps_violations']}")
        print(f"  PS severity (total)    : {s['ps_severity']:.3f}")
        print(f"  Mean PS cost / step    : {s['mean_ps_cost_per_step']:.5f}")
        print(f"  Path length            : {s['path_length_m']:.2f} m")
        print(f"  Navigation efficiency  : {s['nav_efficiency']:.4f}")
        if s["coverage_area_m2"] is not None:
            print(f"  Coverage area (hull)   : {s['coverage_area_m2']:.1f} m²")
        print("──────────────────────────────────────────────────")
