"""
Proactive Uncertainty-aware Context-conditioned SIEP (Proactive-SIEP) planner.

Implements the receding-horizon proactive equilibrium objective:

  U* = argmin_U  sum_{tau=1}^{H} ||F_tot(x_tau, H_hat_tau(U), z_tau)||^2
                 + lambda_s * C_social(U, H_hat(U), z)
                 + lambda_u * C_unc(U)
                 + lambda_d * C_dyn(U)
                 - lambda_e * G_explore(U)

where

  F_tot = F_{goal/frontier}
          + sum_o  F_{obs,o}
          + sum_j  w_j(z_tau) * F_{human,j}(x_tau, h_hat_{j,tau}(U))
          + F_ctx(z_tau)

Key properties
--------------
1. **Proactive** – H_hat(U) depends on U: predicted pedestrian trajectories
   are computed by simulating pedestrian reaction to the candidate robot path.
2. **Context-conditioned** – w_j(z) and F_ctx(z) adapt to social situation.
3. **Uncertainty-aware** – C_unc and uncertainty_scales modulate force magnitude
   based on prediction uncertainty (noise-injected samples).
4. **Safety layer** – optional barrier-function projection clips outputs to
   enforce minimum clearance and dynamics limits.
5. **Exploration** – G_explore replaces goal attraction when explore_mode=True.

Ablation flags (mirror paper ablations)
----------------------------------------
  use_context_adapt   bool  enables w_j(z), F_ctx(z)        (ablation B)
  use_uncertainty_ps  bool  enables C_unc + force inflation  (ablation C)
  use_proactive_pred  bool  enables interaction-aware H_hat  (ablation D)
  use_barrier         bool  enables safety barrier projection (ablation E)
  explore_mode        bool  enables G_explore / F_frontier   (ablation variant)

Paper equation mapping
----------------------
All symbols map directly to functions in objective_terms.py and
context_inference.py.  See module docstrings for detailed mapping.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np

from ..utils.geometry import Pose2, integrate_diff_drive, wrap_pi
from ..utils.social import PersonalSpace

from .context_inference import (
    SocialContext,
    ContextType,
    infer_context,
    context_weight,
    context_force,
)
from .human_prediction import (
    PedState,
    make_ped_states,
    predict_constant_velocity,
    predict_interaction_aware,
)
from .objective_terms import (
    equilibrium_residual,
    social_cost,
    uncertainty_cost,
    dynamics_cost,
    exploration_gain,
    force_total as _ftot,
)

# GPU batch evaluator (optional — falls back gracefully if torch not available)
try:
    from .gpu_batch_eval import BatchSIEPEvaluator, is_gpu_available
    _GPU_EVAL_AVAILABLE = True
except ImportError:
    _GPU_EVAL_AVAILABLE = False
    def is_gpu_available() -> bool:  # type: ignore[misc]
        return False

# Formal CBF safety layer
from .barrier_cbf import BarrierCBF


# ─────────────────────────────────────────────────────────────────────────────
# Noise amplitude for uncertainty estimation (noise-injected rollouts)
# ─────────────────────────────────────────────────────────────────────────────
UNCERTAINTY_NOISE_STD: float = 0.15   # [m/s] velocity noise for unc estimation
UNCERTAINTY_N_SAMPLES: int = 5        # number of noisy rollouts


class ProactiveSIEP:
    """Proactive Uncertainty-aware Context-conditioned SIEP planner.

    Parameters
    ----------
    cfg : dict
        Full simulation config dict.  Required keys: robot, planner.
    use_context_adapt : bool
        If True, use context-conditioned weights w_j(z) and F_ctx(z).
    use_uncertainty_ps : bool
        If True, estimate prediction uncertainty and inflate personal-space sigma.
    use_proactive_pred : bool
        If True, use interaction-aware pedestrian prediction H_hat(U).
    use_barrier : bool
        If True, apply barrier-function safety projection to output commands.
    explore_mode : bool
        If True, replace goal attraction with frontier exploration force.
    """

    def __init__(
        self,
        cfg: dict,
        use_context_adapt: bool = True,
        use_uncertainty_ps: bool = True,
        use_proactive_pred: bool = True,
        use_barrier: bool = True,
        explore_mode: bool = False,
    ) -> None:
        self.cfg = cfg
        self.use_context_adapt = use_context_adapt
        self.use_uncertainty_ps = use_uncertainty_ps
        self.use_proactive_pred = use_proactive_pred
        self.use_barrier = use_barrier
        self.explore_mode = explore_mode

        # Robot limits
        self.max_v = float(cfg['robot']['max_v'])
        self.max_w = float(cfg['robot']['max_w'])
        self.robot_radius = float(cfg['robot']['radius'])

        # Planner config (with defaults)
        pcfg = cfg.get('planner', {})
        self.H = int(pcfg.get('horizon_steps', 20))
        self.n_cand = int(pcfg.get('n_candidates', 80))
        self.dt_plan = float(pcfg.get('plan_dt', cfg['sim']['dt']))

        # Objective weights (lambda values)
        self.lambda_s = float(pcfg.get('lambda_s', 1.0))
        self.lambda_u = float(pcfg.get('lambda_u', 0.5))
        self.lambda_d = float(pcfg.get('lambda_d', 0.3))
        self.lambda_e = float(pcfg.get('lambda_e', 0.8))

        # SIEP force parameters
        self.k_goal = float(pcfg.get('k_goal', 1.0))
        self.sigma_goal = float(pcfg.get('sigma_goal', 3.0))
        self.k_obs = float(pcfg.get('k_obs', 2.5))
        self.obs_influence = float(pcfg.get('obs_influence', 2.5))
        self.k_ps = float(pcfg.get('k_ps', 2.0))
        self.k_align = float(pcfg.get('k_align', 0.3))
        self.k_ctx = float(pcfg.get('k_ctx', 0.4))
        self.k_frontier = float(pcfg.get('k_frontier', 1.0))
        self.novelty_radius = float(pcfg.get('novelty_radius', 1.5))

        # Barrier constraint: minimum clearance [m]
        self.min_clearance = float(pcfg.get('min_clearance', 0.35))

        # Heading controller
        self.k_yaw = float(pcfg.get('k_yaw', 2.5))
        self.heading_slowdown = float(pcfg.get('heading_slowdown', 0.5))

        # Force params dict for objective_terms calls
        self._force_params = {
            'k_goal': self.k_goal,
            'sigma_goal': self.sigma_goal,
            'k_obs': self.k_obs,
            'obs_influence': self.obs_influence,
            'k_ps': self.k_ps,
            'k_frontier': self.k_frontier,
            'novelty_radius': self.novelty_radius,
        }

        # State
        self._prev_v: float = 0.0
        self._prev_w: float = 0.0
        self._visited_xys: List[np.ndarray] = []
        # Grid-based visited-cell set for O(1) novelty lookup
        self._visited_cells: set = set()
        self._novelty_grid_step: float = max(self.novelty_radius, 1e-3)
        # Momentum: track consecutive low-speed steps for kick mechanism
        self._low_speed_steps: int = 0

        # Reproducible sampling (seeded per plan call)
        self._rng = np.random.default_rng(int(cfg.get('seed', 42)))

        # ── GPU batch evaluator (optional, uses PyTorch CUDA when available) ──
        # The GPU evaluator handles per-candidate uncertainty cost (C_unc),
        # fixing the unc_cost_val=0.0 limitation of the CPU path.
        # Paper: GPU acceleration enables n_candidates=512, S=32 MC samples.
        use_gpu = bool(pcfg.get('use_gpu', True))  # default: try GPU
        self._gpu_evaluator: Optional['BatchSIEPEvaluator'] = None
        if use_gpu and _GPU_EVAL_AVAILABLE:
            try:
                from .gpu_batch_eval import BatchSIEPEvaluator
                self._gpu_evaluator = BatchSIEPEvaluator(cfg)
                print(f'[ProactiveSIEP] GPU batch evaluator enabled on '
                      f'{self._gpu_evaluator.device}')
            except Exception as exc:
                print(f'[ProactiveSIEP] GPU evaluator init failed ({exc}), '
                      f'falling back to CPU path.')
                self._gpu_evaluator = None

        # ── Formal CBF safety layer ───────────────────────────────────────────
        # Replaces simple clipping with h(x)+α·ḣ(x,u)≥0 QP projection.
        # Paper: Barrier-SIEP-MPC safety layer (Section IV.C).
        use_cbf_humans = bool(pcfg.get('use_cbf_humans', True))
        self._cbf: Optional[BarrierCBF] = None
        if self.use_barrier:
            self._cbf = BarrierCBF(cfg, use_cbf_humans=use_cbf_humans)

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def plan(
        self,
        pose: Pose2,
        goal_xy: Optional[np.ndarray],
        dt: float,
        lidar_dists: np.ndarray,
        lidar_angles_world: np.ndarray,
        ped_raw: List[dict],
    ) -> Tuple[float, float]:
        """Compute (v, ω) by minimising the Proactive-SIEP objective.

        Parameters
        ----------
        pose               : current robot pose.
        goal_xy            : goal position (None in explore mode).
        dt                 : simulation time step [s].
        lidar_dists        : 1-D array of LiDAR distances [m].
        lidar_angles_world : 1-D array of world-frame ray angles [rad].
        ped_raw            : list of pedestrian dicts from sim.

        Returns
        -------
        (v, ω)  linear and angular velocity commands.
        """
        robot_xy = pose.xy()
        self._visited_xys.append(robot_xy.copy())
        # Update visited-cell set (O(1) novelty check in force/gain functions)
        cell = (int(robot_xy[0] // self._novelty_grid_step),
                int(robot_xy[1] // self._novelty_grid_step))
        self._visited_cells.add(cell)

        # ── 1. Context inference (once per planning call) ────────────────────
        ped_states = make_ped_states(ped_raw)
        ped_xys = [p.xy for p in ped_states]
        ped_yaws = [p.yaw for p in ped_states]
        ped_vels = [p.vel for p in ped_states]
        ped_pss = []
        for d in ped_raw:
            ps_val = d.get('ps')
            if isinstance(ps_val, PersonalSpace):
                ped_pss.append(ps_val)
            elif isinstance(ps_val, dict):
                ped_pss.append(PersonalSpace(**ps_val))
            else:
                ped_pss.append(PersonalSpace())

        if self.use_context_adapt:
            ctx = infer_context(
                robot_xy, pose.yaw,
                ped_xys, ped_yaws, ped_vels,
                lidar_dists, lidar_angles_world,
            )
        else:
            ctx = SocialContext()  # neutral context = uniform weights

        n_peds = len(ped_states)

        # ── 2. Pre-compute context weights (once, not per candidate) ─────────
        if self.use_context_adapt:
            cw_base = [
                context_weight(j, ped_states[j].xy, ped_states[j].yaw,
                               ped_states[j].vel, robot_xy, pose.yaw, ctx)
                for j in range(n_peds)
            ]
        else:
            cw_base = [1.0] * n_peds

        # ── 3. Pre-compute uncertainty scales (once, using straight-ahead traj) ─
        # Use a straight-ahead reference trajectory to estimate uncertainty scales.
        # This is a fast approximation — per-candidate uncertainty is too expensive.
        if self.use_uncertainty_ps and ped_states:
            ref_traj, _ = self._rollout_robot(
                pose,
                np.tile([self._prev_v, 0.0], (self.H, 1)).reshape(self.H, 2),
            )
            _, unc_scales = self._estimate_uncertainty(
                ped_states, ref_traj, ped_pss, self.H, self.dt_plan,
            )
        else:
            unc_scales = [1.0] * n_peds

        # ── 4. Candidate sampling ────────────────────────────────────────────
        candidates = self._sample_candidates()   # (n_cand, H, 2)

        # ── 5. Evaluate objective for each candidate ─────────────────────────
        # GPU path: batch evaluation with per-candidate C_unc (MC-Ensemble).
        # CPU path: sequential loop with pre-computed unc_scales (unc_cost_val
        # approximated via force inflation rather than per-candidate variance).
        if self._gpu_evaluator is not None:
            best_idx, best_cost, _ = self._gpu_evaluator.get_best_candidate(
                candidates, pose, goal_xy,
                lidar_dists, lidar_angles_world,
                ped_raw, ped_pss, cw_base, unc_scales,
                self.explore_mode, self._visited_xys,
            )
            best_u = candidates[best_idx]
        else:
            best_cost = float('inf')
            best_u = candidates[0]

            for u_seq in candidates:
                cost = self._evaluate(
                    u_seq, pose, goal_xy,
                    lidar_dists, lidar_angles_world,
                    ped_states, ped_pss, ctx, cw_base, unc_scales,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_u = u_seq

        # ── 6. Extract first action ──────────────────────────────────────────
        v_cmd = float(np.clip(best_u[0, 0], -self.max_v, self.max_v))
        w_cmd = float(np.clip(best_u[0, 1], -self.max_w, self.max_w))

        # ── 6b. Momentum kick: escape low-speed traps ────────────────────────
        # If the robot has been nearly stationary for too long, apply a
        # random-direction impulse to break out of local force equilibria.
        if self.explore_mode:
            if abs(v_cmd) < 0.1:
                self._low_speed_steps += 1
            else:
                self._low_speed_steps = 0

            if self._low_speed_steps >= 20:  # ~1 s at typical 20 Hz plan rate (dt=0.05 s)
                # Random kick: pick an open direction from lidar
                if lidar_dists is not None and len(lidar_dists) > 0:
                    best_dir_idx = int(np.argmax(lidar_dists))
                    kick_angle = float(lidar_angles_world[best_dir_idx])
                    # Convert world-frame angle to turning rate
                    angle_err = float(
                        math.atan2(math.sin(kick_angle - pose.yaw),
                                   math.cos(kick_angle - pose.yaw))
                    )
                    w_cmd = float(np.clip(self.k_yaw * angle_err, -self.max_w, self.max_w))
                    v_cmd = self.max_v * 0.5
                self._low_speed_steps = 0

        # ── 7. Barrier projection (safety layer) ────────────────────────────
        if self.use_barrier:
            if self._cbf is not None:
                # Formal CBF QP projection: h(x) + α·ḣ(x,u) ≥ 0
                # Paper: Barrier-SIEP-MPC Section IV.C
                v_cmd, w_cmd = self._cbf.project(
                    pose, v_cmd, w_cmd,
                    lidar_dists, lidar_angles_world,
                    ped_raw,
                )
            else:
                # Fallback: simple clipping (backward compatibility)
                v_cmd, w_cmd = self._barrier_project(
                    pose, v_cmd, w_cmd, lidar_dists, lidar_angles_world,
                )

        self._prev_v = v_cmd
        self._prev_w = w_cmd
        return v_cmd, w_cmd

    # ──────────────────────────────────────────────────────────────────────
    # Candidate sampling
    # ──────────────────────────────────────────────────────────────────────

    def _sample_candidates(self) -> np.ndarray:
        """Sample n_cand control sequences of length H.

        Each candidate is an (H, 2) array of [v, w] commands drawn from
        a uniform grid + Gaussian perturbations around the previous action,
        providing a dense yet diverse coverage of the control space.

        Returns
        -------
        candidates : (n_cand, H, 2) float array
        """
        n = self.n_cand
        H = self.H

        # Uniform grid over v and w
        n_v = int(math.sqrt(n * 0.6)) + 1
        n_w = n // n_v + 1
        vs = np.linspace(0.1, self.max_v, n_v)   # bias positive: no near-zero/negative
        ws = np.linspace(-self.max_w, self.max_w, n_w)
        vv, ww = np.meshgrid(vs, ws)
        grid_vw = np.stack([vv.ravel(), ww.ravel()], axis=-1)[:n]

        # Pad or trim to exactly n
        if len(grid_vw) < n:
            extra = self._rng.uniform(
                low=[0.1, -self.max_w],
                high=[self.max_v, self.max_w],
                size=(n - len(grid_vw), 2),
            )
            grid_vw = np.concatenate([grid_vw, extra], axis=0)
        grid_vw = grid_vw[:n]

        # Broadcast constant velocity/turning rate over horizon H
        # (Constant-curvature arcs — simple but covers the space well)
        candidates = np.tile(grid_vw[:, None, :], (1, H, 1))  # (n, H, 2)

        # Add small temporal perturbations (simulates input shaping)
        noise = self._rng.normal(0, 0.05, size=candidates.shape)
        candidates = candidates + noise
        candidates[:, :, 0] = np.clip(candidates[:, :, 0], 0.0, self.max_v)
        candidates[:, :, 1] = np.clip(candidates[:, :, 1],
                                       -self.max_w, self.max_w)
        return candidates

    # ──────────────────────────────────────────────────────────────────────
    # Objective evaluation
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate(
        self,
        u_seq: np.ndarray,
        pose: Pose2,
        goal_xy: Optional[np.ndarray],
        lidar_dists: np.ndarray,
        lidar_angles: np.ndarray,
        ped_states: List[PedState],
        ped_pss: List[PersonalSpace],
        ctx: SocialContext,
        cw_base: List[float],
        unc_scales: List[float],
    ) -> float:
        """Evaluate J(U) for one candidate control sequence U.

        Parameters ``cw_base`` and ``unc_scales`` are pre-computed in ``plan``
        to avoid redundant computation across candidates.

        Returns
        -------
        float  total objective cost (lower is better)
        """
        H = self.H
        dt = self.dt_plan
        robot_xy = pose.xy()

        # ── Robot trajectory rollout ─────────────────────────────────────
        robot_traj, robot_yaws = self._rollout_robot(pose, u_seq)

        # ── Human trajectory prediction ──────────────────────────────────
        if self.use_proactive_pred and ped_states:
            ped_trajs = predict_interaction_aware(
                ped_states, robot_traj, H, dt
            )
        else:
            ped_trajs = predict_constant_velocity(ped_states, H, dt)

        # Pedestrian yaw trajectories (constant heading for simplicity)
        n_peds = len(ped_states)
        ped_yaws_traj = [[ped.yaw] * (H + 1) for ped in ped_states]

        # ── Context weights (broadcast pre-computed base weights) ─────────
        cw_traj = [[cw_base[j]] * (H + 1) for j in range(n_peds)]

        if self.use_context_adapt:
            F_ctx_traj = [
                context_force(
                    robot_traj[tau],
                    robot_yaws[tau],
                    np.array([math.cos(robot_yaws[tau]),
                               math.sin(robot_yaws[tau])]),
                    ctx, self.k_ctx,
                )
                for tau in range(H + 1)
            ]
        else:
            F_ctx_traj = [np.zeros(2)] * (H + 1)

        # ── Uncertainty scales (pre-computed, broadcast over horizon) ─────
        unc_scales_traj = [[unc_scales[j] for j in range(n_peds)]] * (H + 1)
        # CPU path: compute uncertainty cost from pre-computed unc_scales.
        # This quantifies the expected variance in F_human across pedestrians,
        # weighted by their uncertainty scales from the noise-injected rollouts.
        # Paper: C_unc = sum_j (unc_scale_j - 1)^2 (variance proxy).
        # Note: On the GPU path, per-candidate C_unc is computed via MC-Ensemble
        # (S=32 noisy samples), which is more accurate but requires BatchSIEPEvaluator.
        if self.use_uncertainty_ps and unc_scales:
            unc_cost_val = float(np.sum([(s - 1.0) ** 2 for s in unc_scales]))
        else:
            unc_cost_val = 0.0

        # ── Equilibrium residual  ────────────────────────────────────────
        J_eq = equilibrium_residual(
            robot_traj, robot_yaws, goal_xy,
            lidar_dists, lidar_angles,
            ped_trajs, ped_yaws_traj, ped_pss,
            cw_traj, F_ctx_traj,
            self._force_params,
            explore_mode=self.explore_mode,
            visited_xys=self._visited_xys,
            uncertainty_scales=unc_scales_traj,
            visited_cells=self._visited_cells,
        )

        # ── Social cost ──────────────────────────────────────────────────
        J_social = social_cost(
            robot_traj, ped_trajs, ped_pss, ped_yaws_traj, cw_traj,
            self.k_ps,
        )

        # ── Dynamics cost ────────────────────────────────────────────────
        J_dyn = dynamics_cost(u_seq, dt)

        # ── Exploration gain ─────────────────────────────────────────────
        if self.explore_mode:
            J_explore = exploration_gain(
                robot_traj, self._visited_xys, self.novelty_radius,
                visited_cells=self._visited_cells,
            )
        else:
            J_explore = 0.0

        return (
            J_eq
            + self.lambda_s * J_social
            + self.lambda_u * unc_cost_val
            + self.lambda_d * J_dyn
            - self.lambda_e * J_explore
        )

    # ──────────────────────────────────────────────────────────────────────
    # Robot trajectory rollout (unicycle model)
    # ──────────────────────────────────────────────────────────────────────

    def _rollout_robot(
        self,
        pose: Pose2,
        u_seq: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Rollout unicycle model for u_seq.

        Returns
        -------
        traj  : list of (2,) xy positions, length H+1
        yaws  : list of yaw angles [rad], length H+1
        """
        traj = [pose.xy()]
        yaws = [pose.yaw]
        p = pose.copy()
        dt = self.dt_plan
        for t in range(self.H):
            v = float(u_seq[t, 0])
            w = float(u_seq[t, 1])
            p = integrate_diff_drive(p, v, w, dt)
            traj.append(p.xy())
            yaws.append(p.yaw)
        return traj, yaws

    # ──────────────────────────────────────────────────────────────────────
    # Uncertainty estimation via noise-injected rollouts
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_uncertainty(
        self,
        ped_states: List[PedState],
        robot_traj: List[np.ndarray],
        ped_pss: List[PersonalSpace],
        H: int,
        dt: float,
    ) -> Tuple[float, List[float]]:
        """Generate S noisy pedestrian rollouts to estimate prediction variance.

        Noise is injected into each pedestrian's preferred velocity, producing
        a distribution of trajectories that captures the uncertainty in their
        future motion.  The variance across these rollouts forms C_unc and
        informs the uncertainty scaling of F_human.

        Returns
        -------
        (unc_cost, unc_scales)  where unc_scales[j] ≥ 1.0
        """
        S = UNCERTAINTY_N_SAMPLES
        samples = []
        ped_yaws = [p.yaw for p in ped_states]

        for _ in range(S):
            # Inject velocity noise
            noisy_states = []
            for ps_orig in ped_states:
                noise = self._rng.normal(0, UNCERTAINTY_NOISE_STD, 2)
                noisy_pv = ps_orig.preferred_vel + noise
                noisy = PedState(
                    xy=ps_orig.xy.copy(),
                    vel=ps_orig.vel.copy(),
                    yaw=ps_orig.yaw,
                    preferred_vel=noisy_pv,
                    radius=ps_orig.radius,
                )
                noisy_states.append(noisy)
            traj = predict_interaction_aware(noisy_states, robot_traj, H, dt)
            samples.append(traj)

        return uncertainty_cost(samples, robot_traj, self.k_ps, ped_pss, ped_yaws)

    # ──────────────────────────────────────────────────────────────────────
    # Safety barrier projection
    # ──────────────────────────────────────────────────────────────────────

    def _barrier_project(
        self,
        pose: Pose2,
        v: float,
        w: float,
        lidar_dists: np.ndarray,
        lidar_angles: np.ndarray,
    ) -> Tuple[float, float]:
        """Apply CBF-inspired safety constraint to commanded (v, ω).

        If the nearest obstacle is within a safety margin, the linear velocity
        is reduced proportionally (soft barrier).  Hard stop if within robot
        radius + epsilon.

        This implements the constrained realization layer described in the paper:
        the proactive SIEP output is the nominal behaviour; the barrier projects
        it onto a safe set defined by h(x) = d_min - (r_robot + margin) ≥ 0.

        Paper: C_dyn and barrier projection enforce dynamics/safety constraints.
        """
        min_dist = float(np.min(lidar_dists)) if len(lidar_dists) > 0 else float('inf')
        safe_dist = self.robot_radius + self.min_clearance
        margin = 0.8  # [m] – begin slowing down at this distance (reduced for museum)

        if min_dist < safe_dist:
            # Hard stop: imminent collision
            return 0.0, float(np.clip(w, -self.max_w, self.max_w))

        if min_dist < safe_dist + margin and v > 0:
            # Soft deceleration: linear scale factor
            alpha = (min_dist - safe_dist) / margin
            v = v * max(0.0, alpha)

        # Jerk limiting: limit change in v (smoothness constraint from C_dyn)
        max_dv = 0.5  # [m/s] max change per step (relaxed to allow exploration)
        v = float(np.clip(v, self._prev_v - max_dv, self._prev_v + max_dv))
        v = float(np.clip(v, 0.0, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))
        return v, w
