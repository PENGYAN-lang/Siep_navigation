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

Base stimulus channels:
  1. Goal attraction     – tanh-saturated pull toward the navigation goal.
  2. Obstacle repulsion  – exponential push from LiDAR-detected surfaces.
  3. Predictive personal-space repulsion – time-discounted anisotropic
     Gaussian integral over predicted pedestrian positions.
  4. Velocity alignment  – gentle nudge to match crowd-flow direction.

Extended SIEP innovations (ablation-controllable via constructor flags):
  CA  – Context-Adaptive SIEP: force weights dynamically modulated by
        local crowd density, pedestrian facing angle, and corridor geometry.
  UMP – Uncertainty-Modulated Personal Space: σ expands when pedestrian
        trajectory is uncertain (fast speed, large distance).
  GF  – Group-Level Social Force: groups detected online via proximity +
        heading similarity; collective personal space applied to centroid.
  EXP – Frontier Exploration Force: replaces goal attraction when *mode*
        is set to 'explore', enabling goal-free museum roaming.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.geometry import Pose2, wrap_pi
from ..utils.social import PersonalSpace
from ..utils.siep_forces import (
    SIEPParams,
    SocialContext,
    SocialPerceptionLayer,
    context_modulate_weights,
    frontier_exploration_force,
    goal_force,
    group_repulsion_force,
    obstacle_force,
    personal_space_force,
    uncertainty_modulated_ps_force,
    velocity_alignment_force,
    _detect_groups,
)


class SIEPPlanner:
    """Stimuli-Induced Equilibrium Point planner.

    Args:
        cfg:              Full config dict (same format used by ``SamplingMPC``).
        mode:             ``'goal'`` (default) drives to ``goal_xy``;
                          ``'explore'`` uses frontier force for goal-free roaming.
        use_context_adapt: Enable CA-SIEP context-adaptive weight modulation.
        use_uncertainty_ps: Enable uncertainty-modulated personal space.
        use_group_force:   Enable group-level social force.

    Usage::

        planner = SIEPPlanner(cfg, mode='explore', use_context_adapt=True)
        v, w = planner.plan(pose, goal_xy, dt, lidar_dists,
                            lidar_angles_world, peds)
    """

    def __init__(
        self,
        cfg: dict,
        mode: str = "goal",
        use_context_adapt: bool = False,
        use_uncertainty_ps: bool = False,
        use_group_force: bool = False,
    ) -> None:
        pcfg = cfg["planner"]
        self.max_v = float(cfg["robot"]["max_v"])
        self.max_w = float(cfg["robot"]["max_w"])
        self.mode = mode

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
            # CA-SIEP parameters
            use_context_adapt=use_context_adapt,
            context_radius=float(pcfg.get("context_radius", 4.0)),
            density_ps_gain=float(pcfg.get("density_ps_gain", 1.5)),
            density_threshold=float(pcfg.get("density_threshold", 0.2)),
            density_goal_decay=float(pcfg.get("density_goal_decay", 0.6)),
            face_ps_gain=float(pcfg.get("face_ps_gain", 0.5)),
            lambda_min=float(pcfg.get("lambda_min", 0.3)),
            lambda_max=float(pcfg.get("lambda_max", 3.0)),
            # UMPS parameters
            use_uncertainty_ps=use_uncertainty_ps,
            uncertainty_gain=float(pcfg.get("uncertainty_gain", 1.2)),
            max_uncertainty=float(pcfg.get("max_uncertainty", 0.8)),
            # Group-force parameters
            use_group_force=use_group_force,
            group_proximity=float(pcfg.get("group_proximity", 2.0)),
            group_heading_thresh=float(pcfg.get("group_heading_thresh", 0.5)),
            k_group=float(pcfg.get("k_group", 1.5)),
            group_sigma_scale=float(pcfg.get("group_sigma_scale", 1.4)),
            # Exploration parameters
            k_frontier=float(pcfg.get("k_frontier", 1.0)),
            novelty_radius=float(pcfg.get("novelty_radius", 2.0)),
            frontier_random_weight=float(pcfg.get("frontier_random_weight", 0.1)),
        )

        # Heading proportional gain and turn-slowdown factor
        self.k_yaw = float(pcfg.get("k_yaw", 2.5))
        self.heading_slowdown = float(pcfg.get("heading_slowdown", 0.5))

        # Social perception layer (CA-SIEP front-end)
        self._perception = SocialPerceptionLayer(
            context_radius=self.params.context_radius,
            group_proximity=self.params.group_proximity,
            group_heading_thresh=self.params.group_heading_thresh,
        )

        # Internal state
        self._prev_v: float = 0.0
        self._visited: List[np.ndarray] = []    # for explore mode novelty
        self._visit_stride: int = 5             # record every N steps
        self._step_count: int = 0
        self._rng = np.random.default_rng(int(cfg.get('seed', 42)))

        # LiDAR max range (needed for frontier force)
        lidar_cfg = cfg.get("lidar", {})
        self._lidar_max_range = float(lidar_cfg.get("max_range", 8.0))

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

        Applies enabled innovations in order:
        1. Compute social context (if CA-SIEP enabled).
        2. Derive context-adaptive weight multipliers (λ_ps, λ_goal, λ_align).
        3. Accumulate forces:
           • Goal attraction *or* Frontier exploration force.
           • Obstacle repulsion.
           • Per-pedestrian personal-space repulsion (plain or UMPS variant).
           • Per-pedestrian velocity alignment.
           • Group-level collective repulsion (if GF enabled).

        Args:
            pose:               Current robot pose.
            goal_xy:            Goal position (2,) – ignored in explore mode.
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

        # ── Optional: social perception (CA-SIEP front-end) ─────────────────
        lambda_ps = 1.0
        lambda_goal = 1.0
        lambda_align = 1.0

        if p.use_context_adapt:
            ctx: SocialContext = self._perception.perceive(
                robot_xy, pose.yaw, peds, lidar_dists, lidar_angles_world
            )
            lambda_ps, lambda_goal, lambda_align = context_modulate_weights(ctx, p)

        # ── 1. Goal attraction OR frontier exploration ───────────────────────
        if self.mode == "explore":
            F = frontier_exploration_force(
                robot_xy,
                lidar_dists,
                lidar_angles_world,
                self._visited,
                max_range=self._lidar_max_range,
                novelty_radius=p.novelty_radius,
                k_frontier=p.k_frontier,
                random_weight=p.frontier_random_weight,
                rng=self._rng,
            )
        else:
            F = lambda_goal * goal_force(robot_xy, goal_xy, p.k_goal, p.sigma_goal)

        # ── 2. Obstacle repulsion ────────────────────────────────────────────
        F += obstacle_force(lidar_dists, lidar_angles_world, p.k_obs, p.obs_influence)

        # ── 3 & 4. Per-pedestrian stimuli ────────────────────────────────────
        for (ped_xy, ped_yaw, ped_vel, ps) in peds:
            # Personal-space repulsion (plain or UMPS)
            if p.use_uncertainty_ps:
                F += lambda_ps * uncertainty_modulated_ps_force(
                    robot_xy, ped_xy, ped_yaw, ped_vel, ps,
                    p.k_ps, p.ps_horizon, p.ps_dt,
                    uncertainty_gain=p.uncertainty_gain,
                    max_uncertainty=p.max_uncertainty,
                )
            else:
                F += lambda_ps * personal_space_force(
                    robot_xy, ped_xy, ped_yaw, ped_vel, ps,
                    p.k_ps, p.ps_horizon, p.ps_dt,
                )
            # Velocity-alignment (crowd-flow following)
            F += lambda_align * velocity_alignment_force(
                robot_xy, robot_vel, ped_xy, ped_vel,
                p.k_align, p.align_radius, p.align_cone_deg,
            )

        # ── 5. Group-level social force (GF) ─────────────────────────────────
        if p.use_group_force and len(peds) >= 2:
            group_ids, n_groups = _detect_groups(
                peds, p.group_proximity, p.group_heading_thresh
            )
            for gid in range(n_groups):
                members = [peds[i] for i, g in enumerate(group_ids) if g == gid]
                F += group_repulsion_force(
                    robot_xy, members, p.k_group, p.group_sigma_scale
                )

        return F

    # ------------------------------------------------------------------
    # Explore-mode visit tracking
    # ------------------------------------------------------------------

    def _record_visit(self, robot_xy: np.ndarray) -> None:
        """Record the robot's position every N steps for novelty scoring."""
        self._step_count += 1
        if self._step_count % self._visit_stride == 0:
            self._visited.append(robot_xy.copy())

    @property
    def visited_positions(self) -> List[np.ndarray]:
        """Read-only access to visit history (for evaluation / plotting)."""
        return list(self._visited)

    @property
    def coverage_count(self) -> int:
        """Number of distinct recorded visit positions."""
        return len(self._visited)

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
        if self.mode == "explore":
            self._record_visit(pose.xy())

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
