from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..utils.geometry import Pose2, integrate_diff_drive
from ..utils.social import PersonalSpace, anisotropic_gaussian_cost


@dataclass
class MPCConfig:
    horizon_s: float
    n_samples: int
    w_goal: float
    w_smooth: float
    w_obstacle: float
    w_social: float
    min_clearance: float


class SamplingMPC:
    """A simple engineering-usable local planner (velocity sampling + rollouts).

    Inputs:
      - current pose
      - goal xy
      - obstacle distances (from lidar rays)
      - pedestrian states (pos, yaw, vel)

    Output:
      - (v, w) control for dt
    """

    def __init__(self, cfg: dict):
        pcfg = cfg['planner']
        self.horizon_s = float(pcfg['horizon_s'])
        self.n_samples = int(pcfg['n_samples'])
        self.w_goal = float(pcfg['w_goal'])
        self.w_smooth = float(pcfg['w_smooth'])
        self.w_obstacle = float(pcfg['w_obstacle'])
        self.w_social = float(pcfg['w_social'])
        self.min_clearance = float(pcfg['min_clearance'])

        self.max_v = float(cfg['robot']['max_v'])
        self.max_w = float(cfg['robot']['max_w'])

        self._prev_u = np.array([0.0, 0.0], dtype=float)

    def sample_controls(self) -> np.ndarray:
        # Bias toward forward motion; include some reverse for escaping.
        vs = np.random.uniform(-0.2 * self.max_v, self.max_v, size=self.n_samples)
        ws = np.random.uniform(-self.max_w, self.max_w, size=self.n_samples)
        return np.stack([vs, ws], axis=1)

    def rollout_cost(
        self,
        pose: Pose2,
        u: np.ndarray,
        dt: float,
        steps: int,
        goal_xy: np.ndarray,
        min_lidar_dist: float,
        peds: List[Tuple[np.ndarray, float, np.ndarray, PersonalSpace]],
    ) -> float:
        v, w = float(u[0]), float(u[1])
        p = pose.copy()
        cost = 0.0
        # If immediate obstacle too close, penalize heavily
        if min_lidar_dist < self.min_clearance:
            cost += self.w_obstacle * (self.min_clearance - min_lidar_dist) ** 2 * 50.0

        for k in range(steps):
            p = integrate_diff_drive(p, v, w, dt)
            # Goal cost
            d = np.linalg.norm(p.xy() - goal_xy)
            cost += self.w_goal * d * dt
            # Social cost (predict pedestrians constant velocity)
            for (ped_xy, ped_yaw, ped_vel, ps) in peds:
                ped_pred = ped_xy + ped_vel * (k * dt)
                cost += self.w_social * anisotropic_gaussian_cost(p.xy(), ped_pred, ped_yaw, ps) * dt

        # Smoothness cost
        cost += self.w_smooth * float(np.sum((u - self._prev_u) ** 2))
        return cost

    def plan(
        self,
        pose: Pose2,
        goal_xy: np.ndarray,
        dt: float,
        min_lidar_dist: float,
        peds: List[Tuple[np.ndarray, float, np.ndarray, PersonalSpace]],
    ) -> Tuple[float, float]:
        steps = max(1, int(self.horizon_s / dt))
        U = self.sample_controls()
        best = None
        best_cost = float('inf')
        for u in U:
            c = self.rollout_cost(pose, u, dt, steps, goal_xy, min_lidar_dist, peds)
            if c < best_cost:
                best_cost = c
                best = u
        assert best is not None
        self._prev_u = best.copy()
        return float(best[0]), float(best[1])
