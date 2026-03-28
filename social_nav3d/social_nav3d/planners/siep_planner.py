"""
SIEP (Stimuli-Induced Equilibrium Point) planner for socially-aware navigation.

The robot's motion dynamics are modelled as virtual forces induced by stimuli
perceived from the surrounding environment.  The net force defines an
*equilibrium point* – a desired velocity vector – that the robot tracks via a
proportional heading controller.

Compared to the sampling-based MPC, this approach is:
  * Deterministic and interpretable – forces are computed analytically.
  * More closely aligned with the SIEP concept from the project description.
  * Naturally extensible to learned force functions or density-aware weights.

Four stimulus channels are combined:
  1. Goal attraction     – tanh-saturated pull toward the navigation goal.
  2. Obstacle repulsion  – exponential push from LiDAR-detected surfaces.
  3. Predictive personal-space repulsion – time-discounted anisotropic
     Gaussian integral over predicted pedestrian positions.
  4. Velocity alignment  – gentle nudge to match crowd-flow direction.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from ..utils.geometry import Pose2, wrap_pi
from ..utils.social import PersonalSpace
from ..utils.siep_forces import (
    SIEPParams,
    goal_force,
    obstacle_force,
    personal_space_force,
    velocity_alignment_force,
)


class SIEPPlanner:
    """Stimuli-Induced Equilibrium Point planner.

    Args:
        cfg: Full config dict (same format used by ``SamplingMPC``).

    Usage::

        planner = SIEPPlanner(cfg)
        v, w = planner.plan(pose, goal_xy, dt, lidar_dists,
                            lidar_angles_world, peds)
    """

    def __init__(self, cfg: dict) -> None:
        pcfg = cfg["planner"]
        self.max_v = float(cfg["robot"]["max_v"])
        self.max_w = float(cfg["robot"]["max_w"])

        # Build SIEPParams from config; fall back to dataclass defaults.
        self.params = SIEPParams(
            k_goal=float(pcfg.get("k_goal", 1.0)),
            sigma_goal=float(pcfg.get("sigma_goal", 3.0)),
            k_obs=float(pcfg.get("k_obs", 2.5)),
            obs_influence=float(pcfg.get("obs_influence", 2.5)),
            k_ps=float(pcfg.get("k_ps", 2.0)),
            ps_horizon=float(pcfg.get("ps_horizon", 1.5)),
            ps_dt=float(pcfg.get("ps_dt", 0.1)),
            k_align=float(pcfg.get("k_align", 0.3)),
            align_radius=float(pcfg.get("align_radius", 3.0)),
            align_cone_deg=float(pcfg.get("align_cone_deg", 60.0)),
        )

        # Heading proportional gain and turn-slowdown factor
        self.k_yaw = float(pcfg.get("k_yaw", 2.5))
        self.heading_slowdown = float(pcfg.get("heading_slowdown", 0.5))

        # Internal state for velocity estimate used in alignment force
        self._prev_v: float = 0.0

    # ------------------------------------------------------------------
    # Core SIEP computation
    # ------------------------------------------------------------------

    def compute_equilibrium(
        self,
        pose: Pose2,
        goal_xy: np.ndarray,
        lidar_dists: np.ndarray,
        lidar_angles_world: np.ndarray,
        peds: List[Tuple[np.ndarray, float, np.ndarray, PersonalSpace]],
        dt: float,
    ) -> np.ndarray:
        """Return the SIEP equilibrium velocity vector (2-D).

        Args:
            pose:               Current robot pose.
            goal_xy:            Goal position (2,).
            lidar_dists:        Per-ray distances in metres (N,).
            lidar_angles_world: World-frame azimuth angles of each ray (N,).
            peds:               List of (xy, yaw, vel, ps) pedestrian tuples.
            dt:                 Simulation time-step [s] (used in prediction).

        Returns:
            2-D force vector representing the desired velocity direction and
            magnitude.
        """
        p = self.params
        robot_xy = pose.xy()
        robot_vel = np.array(
            [self._prev_v * math.cos(pose.yaw),
             self._prev_v * math.sin(pose.yaw)],
            dtype=float,
        )

        # 1. Goal attraction -------------------------------------------------
        F = goal_force(robot_xy, goal_xy, p.k_goal, p.sigma_goal)

        # 2. Obstacle repulsion ----------------------------------------------
        F += obstacle_force(lidar_dists, lidar_angles_world, p.k_obs, p.obs_influence)

        # 3 & 4. Pedestrian stimuli ------------------------------------------
        for (ped_xy, ped_yaw, ped_vel, ps) in peds:
            # Predictive personal-space repulsion
            F += personal_space_force(
                robot_xy, ped_xy, ped_yaw, ped_vel, ps,
                p.k_ps, p.ps_horizon, p.ps_dt,
            )
            # Velocity-alignment (crowd-flow following)
            F += velocity_alignment_force(
                robot_xy, robot_vel, ped_xy, ped_vel,
                p.k_align, p.align_radius, p.align_cone_deg,
            )

        return F

    # ------------------------------------------------------------------
    # Public planning interface (mirrors SamplingMPC.plan signature where
    # possible so run_demo.py can switch planners easily)
    # ------------------------------------------------------------------

    def plan(
        self,
        pose: Pose2,
        goal_xy: np.ndarray,
        dt: float,
        lidar_dists: np.ndarray,
        lidar_angles_world: np.ndarray,
        peds: List[Tuple[np.ndarray, float, np.ndarray, PersonalSpace]],
    ) -> Tuple[float, float]:
        """Compute (v, ω) from the SIEP equilibrium point.

        Args:
            pose:               Current robot pose.
            goal_xy:            Goal position (2,).
            dt:                 Simulation time-step [s].
            lidar_dists:        Per-ray distances (N,).  Rays beyond sensor
                                max-range should carry the max-range value.
            lidar_angles_world: World-frame azimuth of each LiDAR ray (N,).
            peds:               Pedestrian states as (xy, yaw, vel, ps).

        Returns:
            (v, ω) linear and angular velocity commands.
        """
        F = self.compute_equilibrium(
            pose, goal_xy, lidar_dists, lidar_angles_world, peds, dt
        )

        F_mag = float(np.linalg.norm(F))
        if F_mag < 1e-6:
            return 0.0, 0.0

        desired_heading = math.atan2(float(F[1]), float(F[0]))
        desired_speed = min(F_mag, self.max_v)

        # Heading error ± π
        yaw_err = wrap_pi(desired_heading - pose.yaw)

        # Angular control: proportional, clipped
        w = float(np.clip(self.k_yaw * yaw_err, -self.max_w, self.max_w))

        # Linear speed: reduce proportionally when a large heading correction
        # is needed to prevent the robot from sliding off target.
        heading_factor = max(
            0.0,
            1.0 - self.heading_slowdown * abs(yaw_err) / math.pi,
        )
        v = float(np.clip(desired_speed * heading_factor, 0.0, self.max_v))

        self._prev_v = v
        return v, w
