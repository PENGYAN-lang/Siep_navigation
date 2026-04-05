"""
GPU-accelerated batch trajectory evaluator for the Proactive-SIEP planner.

Implements ``BatchSIEPEvaluator``, which evaluates all candidate control
sequences simultaneously as batched PyTorch tensor operations on CUDA (or CPU
when CUDA is unavailable).

Objective (paper eq. mapping)
------------------------------
For each candidate i with control sequence U_i = {(v_t, w_t)}_{t=1..H}:

  J(U_i) =  C_eq(U_i)                               # equilibrium residual
           + lambda_s * C_social(U_i)                 # social cost
           + lambda_u * C_unc(U_i)                   # uncertainty cost
           - lambda_e * G_explore(U_i)               # exploration gain
           + lambda_d * C_dyn(U_i)                   # dynamics / jerk cost

Unicycle rollout (Eq. unicycle model):
    x_{t+1}     = x_t + v_t * cos(theta_t) * dt
    y_{t+1}     = y_t + v_t * sin(theta_t) * dt
    theta_{t+1} = theta_t + w_t * dt

SIEP forces (detailed in paper):
  F_goal   – tanh-saturated attraction toward goal
  F_obs    – exponential obstacle repulsion from LiDAR rays
  F_human  – anisotropic Gaussian personal-space repulsion per pedestrian

Uncertainty estimation:
  C_unc is estimated via S=32 (GPU) / S=5 (CPU) noise-injected pedestrian
  velocity samples.  For each sample s and candidate i, pedestrian trajectories
  are rolled out under noisy constant velocity; the variance of F_human across
  S samples constitutes C_unc per candidate.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional torch import — module remains importable even without torch.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F  # noqa: N812 – standard alias
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    import torch  # type: ignore[assignment]

from ..utils.geometry import Pose2
from ..utils.social import PersonalSpace


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def is_gpu_available() -> bool:
    """Return True if PyTorch with CUDA support is installed and a GPU exists."""
    return TORCH_AVAILABLE and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Internal helper: wrap angles to [-pi, pi] (vectorised)
# ---------------------------------------------------------------------------

def _wrap_pi(x: "torch.Tensor") -> "torch.Tensor":
    """Wrap a tensor of angles to the interval (-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# BatchSIEPEvaluator
# ---------------------------------------------------------------------------

class BatchSIEPEvaluator:
    """GPU-accelerated batch evaluator for the Proactive-SIEP objective.

    All ``evaluate_batch`` operations are performed as batched PyTorch tensor
    operations.  When CUDA is available the tensors live on the GPU; otherwise
    they fall back to CPU (useful for testing and CPU-only deployments).

    Parameters
    ----------
    cfg : dict
        Full simulation config dict with ``planner`` and ``sim`` sub-dicts.
    device : str | torch.device | None
        Explicit device override.  ``None`` → auto-detect CUDA.

    Examples
    --------
    >>> evaluator = BatchSIEPEvaluator(cfg)
    >>> best_idx, best_cost, costs = evaluator.get_best_candidate(
    ...     candidates, pose, goal_xy, lidar_dists, lidar_angles,
    ...     ped_states, ped_pss, cw_base, unc_scales,
    ...     explore_mode=False, visited_xys=[])
    """

    def __init__(self, cfg: dict, device: Optional[str] = None) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for BatchSIEPEvaluator. "
                "Install it with: pip install torch"
            )

        # Keep full config for optional params (e.g. k_frontier)
        self.cfg = cfg

        # ── Device selection ──────────────────────────────────────────────
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # ── Horizon / time step ───────────────────────────────────────────
        pcfg = cfg.get("planner", {})
        sim_dt = float(cfg["sim"]["dt"])
        self.H: int = int(pcfg.get("horizon_steps", 20))
        self.dt: float = float(pcfg.get("plan_dt", sim_dt))

        # ── Force parameters ──────────────────────────────────────────────
        self.k_goal: float = float(pcfg.get("k_goal", 1.0))
        self.sigma_goal: float = float(pcfg.get("sigma_goal", 3.0))
        self.k_obs: float = float(pcfg.get("k_obs", 2.5))
        self.obs_influence: float = float(pcfg.get("obs_influence", 2.5))
        self.k_ps: float = float(pcfg.get("k_ps", 2.0))

        # ── Objective weights ─────────────────────────────────────────────
        self.lambda_s: float = float(pcfg.get("lambda_s", 1.0))
        self.lambda_u: float = float(pcfg.get("lambda_u", 0.5))
        self.lambda_d: float = float(pcfg.get("lambda_d", 0.3))
        self.lambda_e: float = float(pcfg.get("lambda_e", 0.8))

        # ── Exploration ───────────────────────────────────────────────────
        self.novelty_radius: float = float(pcfg.get("novelty_radius", 1.5))
        self.k_frontier: float = float(pcfg.get("k_frontier", 1.0))
        # Grid step matches the novelty radius (same as ProactiveSIEP)
        self._novelty_grid_step: float = max(self.novelty_radius, 1e-3)

        # ── Uncertainty estimation ────────────────────────────────────────
        #    Default: 32 on GPU, 5 on CPU (matches ProactiveSIEP behaviour)
        default_S = 32 if self.device.type == "cuda" else 5
        self.S: int = int(pcfg.get("uncertainty_n_samples", default_S))
        # Velocity noise std for MC-ensemble uncertainty estimation [m/s]
        self._unc_noise_std: float = 0.15

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        candidates: np.ndarray,
        pose: "Pose2",
        goal_xy: Optional[np.ndarray],
        lidar_dists: np.ndarray,
        lidar_angles: np.ndarray,
        ped_states: List[dict],
        ped_pss: List["PersonalSpace"],
        cw_base: List[float],
        unc_scales: List[float],
        explore_mode: bool,
        visited_xys: List[np.ndarray],
        visited_cells: Optional[set] = None,
    ) -> "torch.Tensor":
        """Evaluate all candidate control sequences in parallel on GPU/CPU.

        Parameters
        ----------
        candidates : np.ndarray, shape (n_cand, H, 2)
            Control sequences ``[v, w]`` for each candidate.
        pose : Pose2
            Current robot pose with ``.x``, ``.y``, ``.yaw`` fields.
        goal_xy : np.ndarray shape (2,) or None
            Goal position in world coordinates.  ``None`` in explore mode.
        lidar_dists : np.ndarray, shape (N_rays,)
            Per-ray LiDAR distances in metres.
        lidar_angles : np.ndarray, shape (N_rays,)
            World-frame azimuth angles for each LiDAR ray [rad].
        ped_states : list of dict
            Each dict has keys: ``xy`` (np.ndarray (2,)), ``yaw`` (float),
            ``vel`` (np.ndarray (2,)), ``radius`` (float).
        ped_pss : list of PersonalSpace
            Personal-space objects with ``sigma_front``, ``sigma_side``, and
            ``sigma_back`` fields, one per pedestrian.
        cw_base : list of float
            Base context weight per pedestrian.
        unc_scales : list of float ≥ 1.0
            Uncertainty scale per pedestrian (modulates Gaussian sigma).
        explore_mode : bool
            If True, use frontier exploration reward instead of goal attraction.
        visited_xys : list of np.ndarray
            Previously visited robot positions (used by exploration gain).
        visited_cells : set of (int, int) or None
            Grid-based visited-cell set for O(1) novelty lookup.  When
            provided, ``_compute_exploration_gain`` uses this instead of the
            distance-based ``visited_xys[-100:]`` fallback, which loses
            accuracy once the robot has looped through the same region many
            times.

        Returns
        -------
        costs : torch.Tensor, shape (n_cand,), dtype float32
            Total cost per candidate.  Lower is better.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for evaluate_batch.")

        dev = self.device
        dtype = torch.float32

        n_cand, H, _ = candidates.shape

        # ── 1. Move candidates and initial pose to device ─────────────────
        # ctrl: (n_cand, H, 2)
        ctrl = torch.tensor(candidates, dtype=dtype, device=dev)

        x0 = torch.tensor(pose.x, dtype=dtype, device=dev)
        y0 = torch.tensor(pose.y, dtype=dtype, device=dev)
        yaw0 = torch.tensor(pose.yaw, dtype=dtype, device=dev)

        # ── 2. Roll out unicycle trajectories ─────────────────────────────
        # robot_xy : (n_cand, H+1, 2)
        # robot_yaw: (n_cand, H+1)
        robot_xy, robot_yaw = self._rollout_unicycle(ctrl, x0, y0, yaw0)

        # ── 3. Goal / exploration force tensor ────────────────────────────
        if explore_mode:
            # In explore mode, the frontier force vector is identical for every
            # candidate (it is computed from the current pose, not the candidate
            # endpoint), so including it in F_tot gives zero discriminative power.
            # Instead we zero it out here and rely entirely on G_explore
            # (weighted by lambda_e) to guide the robot toward unvisited regions.
            # See Problem 1 in the issue description.
            F_goal_seq = torch.zeros(
                n_cand, H + 1, 2, dtype=robot_xy.dtype, device=robot_xy.device
            )
        else:
            F_goal_seq = self._compute_goal_forces(robot_xy, goal_xy)
            # (n_cand, H+1, 2)

        # ── 4. Obstacle repulsion ─────────────────────────────────────────
        # Static across candidates and time — broadcast
        F_obs = self._compute_obs_force(lidar_dists, lidar_angles, dev, dtype)
        # F_obs: (2,) → will be broadcast to (n_cand, H+1, 2)
        F_obs_expanded = F_obs.view(1, 1, 2).expand(n_cand, H + 1, 2)

        # ── 5. Pedestrian forces + costs ──────────────────────────────────
        n_peds = len(ped_states)

        if n_peds == 0:
            # Zero-pedestrian case: no human forces or costs
            F_human_seq = torch.zeros(n_cand, H + 1, 2, dtype=dtype, device=dev)
            C_social = torch.zeros(n_cand, dtype=dtype, device=dev)
            C_unc = torch.zeros(n_cand, dtype=dtype, device=dev)
        else:
            # Pack pedestrian state tensors
            ped_xy0, ped_vel, ped_yaw0, sig_front, sig_side, sig_back = \
                self._pack_ped_tensors(ped_states, ped_pss, dev, dtype)

            cw_t = torch.tensor(cw_base, dtype=dtype, device=dev)   # (n_peds,)
            unc_t = torch.tensor(unc_scales, dtype=dtype, device=dev)  # (n_peds,)

            # Deterministic pedestrian trajectories (constant velocity)
            # ped_traj: (n_peds, H+1, 2)
            ped_traj = self._predict_peds_const_vel(ped_xy0, ped_vel, H)

            # Pedestrian yaw (constant over horizon)
            # ped_yaw_seq: (n_peds, H+1)
            ped_yaw_seq = ped_yaw0.unsqueeze(1).expand(n_peds, H + 1)

            # F_human per pedestrian: (n_cand, n_peds, H+1, 2)
            F_human_per_ped = self._compute_human_forces(
                robot_xy, ped_traj, ped_yaw_seq,
                sig_front, sig_side, sig_back,
                unc_t,
            )

            # Weighted sum over pedestrians: (n_cand, H+1, 2)
            # cw_t: (n_peds,) → (1, n_peds, 1, 1)
            cw_b = cw_t.view(1, n_peds, 1, 1)
            F_human_seq = (cw_b * F_human_per_ped).sum(dim=1)

            # C_social: sum over horizon of sum_j cw_j * ||F_human_j||^2
            # F_human_per_ped: (n_cand, n_peds, H+1, 2)
            fh_sq = (F_human_per_ped ** 2).sum(dim=-1)  # (n_cand, n_peds, H+1)
            C_social = (cw_b.squeeze(-1) * fh_sq).sum(dim=(1, 2))  # (n_cand,)

            # Uncertainty cost via MC-Ensemble
            C_unc = self._compute_uncertainty_cost(
                robot_xy, ped_xy0, ped_vel, ped_yaw0,
                sig_front, sig_side, sig_back, unc_t,
            )

        # ── 6. Total force and equilibrium residual ───────────────────────
        # F_tot: (n_cand, H+1, 2)
        F_tot = F_goal_seq + F_obs_expanded + F_human_seq

        # C_eq: sum over t=1..H of ||F_tot_t||^2
        # Shape: (n_cand, H+1) → skip t=0
        F_tot_sq = (F_tot[:, 1:, :] ** 2).sum(dim=-1)  # (n_cand, H)
        C_eq = F_tot_sq.sum(dim=1)                       # (n_cand,)

        # ── 7. Dynamics cost (squared jerk in v and w) ────────────────────
        C_dyn = self._compute_dynamics_cost(ctrl)         # (n_cand,)

        # ── 8. Exploration gain ───────────────────────────────────────────
        G_explore = self._compute_exploration_gain(
            robot_xy, visited_xys, dev, dtype, visited_cells=visited_cells
        )  # (n_cand,)

        # ── 9. Composite objective ────────────────────────────────────────
        # J = C_eq + lambda_s*C_social + lambda_u*C_unc
        #         + lambda_d*C_dyn - lambda_e*G_explore
        costs = (
            C_eq
            + self.lambda_s * C_social
            + self.lambda_u * C_unc
            + self.lambda_d * C_dyn
            - self.lambda_e * G_explore
        )
        return costs.float()

    def get_best_candidate(
        self,
        candidates: np.ndarray,
        pose: "Pose2",
        goal_xy: Optional[np.ndarray],
        lidar_dists: np.ndarray,
        lidar_angles: np.ndarray,
        ped_states: List[dict],
        ped_pss: List["PersonalSpace"],
        cw_base: List[float],
        unc_scales: List[float],
        explore_mode: bool,
        visited_xys: List[np.ndarray],
        visited_cells: Optional[set] = None,
    ) -> tuple:
        """Evaluate all candidates and return the one with lowest cost.

        Parameters
        ----------
        (same as ``evaluate_batch``)

        Returns
        -------
        best_idx : int
            Index into ``candidates`` of the lowest-cost sequence.
        best_cost : float
            Scalar cost value for the best candidate.
        costs : torch.Tensor, shape (n_cand,)
            Full cost tensor for all candidates.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for get_best_candidate.")

        costs = self.evaluate_batch(
            candidates, pose, goal_xy,
            lidar_dists, lidar_angles,
            ped_states, ped_pss,
            cw_base, unc_scales,
            explore_mode, visited_xys,
            visited_cells=visited_cells,
        )
        best_idx = int(torch.argmin(costs).item())
        best_cost = float(costs[best_idx].item())
        return best_idx, best_cost, costs

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _rollout_unicycle(
        self,
        ctrl: "torch.Tensor",
        x0: "torch.Tensor",
        y0: "torch.Tensor",
        yaw0: "torch.Tensor",
    ) -> tuple:
        """Roll out unicycle dynamics for all candidates simultaneously.

        Unicycle model (Eq. unicycle):
            x_{t+1}     = x_t + v_t * cos(theta_t) * dt
            y_{t+1}     = y_t + v_t * sin(theta_t) * dt
            theta_{t+1} = theta_t + w_t * dt

        Parameters
        ----------
        ctrl : Tensor, shape (n_cand, H, 2)
            Control sequences ``[v, w]``.
        x0, y0, yaw0 : scalar Tensors
            Initial robot state.

        Returns
        -------
        robot_xy  : Tensor, shape (n_cand, H+1, 2)
        robot_yaw : Tensor, shape (n_cand, H+1)
        """
        n_cand, H, _ = ctrl.shape
        dev, dtype = ctrl.device, ctrl.dtype

        v = ctrl[:, :, 0]  # (n_cand, H)
        w = ctrl[:, :, 1]  # (n_cand, H)

        xs = torch.zeros(n_cand, H + 1, dtype=dtype, device=dev)
        ys = torch.zeros(n_cand, H + 1, dtype=dtype, device=dev)
        yaws = torch.zeros(n_cand, H + 1, dtype=dtype, device=dev)

        xs[:, 0] = x0
        ys[:, 0] = y0
        yaws[:, 0] = yaw0

        dt = self.dt
        for t in range(H):
            yaws[:, t + 1] = yaws[:, t] + w[:, t] * dt
            xs[:, t + 1] = xs[:, t] + v[:, t] * torch.cos(yaws[:, t]) * dt
            ys[:, t + 1] = ys[:, t] + v[:, t] * torch.sin(yaws[:, t]) * dt

        yaws = _wrap_pi(yaws)
        robot_xy = torch.stack([xs, ys], dim=-1)   # (n_cand, H+1, 2)
        return robot_xy, yaws

    def _compute_goal_forces(
        self,
        robot_xy: "torch.Tensor",
        goal_xy: Optional[np.ndarray],
    ) -> "torch.Tensor":
        """Compute tanh-saturated goal-attraction force F_goal.

        Paper: F_{goal} = k_goal * tanh(dist / sigma_goal) * (goal - x) / dist

        Parameters
        ----------
        robot_xy : Tensor, shape (n_cand, H+1, 2)
        goal_xy  : np.ndarray (2,) or None

        Returns
        -------
        Tensor, shape (n_cand, H+1, 2)
        """
        dev, dtype = robot_xy.device, robot_xy.dtype
        if goal_xy is None:
            return torch.zeros_like(robot_xy)

        goal = torch.tensor(goal_xy, dtype=dtype, device=dev)  # (2,)
        # diff: (n_cand, H+1, 2)
        diff = goal.view(1, 1, 2) - robot_xy
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)  # (n_cand, H+1, 1)
        mag = self.k_goal * torch.tanh(dist / max(self.sigma_goal, 1e-6))
        return mag * diff / dist  # (n_cand, H+1, 2)

    def _compute_obs_force(
        self,
        lidar_dists: np.ndarray,
        lidar_angles: np.ndarray,
        dev: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute obstacle repulsion force from LiDAR rays.

        Paper: F_{obs} = -sum_{o in near} k_obs * exp(-d_o / decay) * direction_o

        Only rays within ``obs_influence`` distance contribute.

        Returns
        -------
        Tensor, shape (2,)
        """
        d = torch.tensor(lidar_dists, dtype=dtype, device=dev)
        a = torch.tensor(lidar_angles, dtype=dtype, device=dev)

        mask = d < self.obs_influence
        if not mask.any():
            return torch.zeros(2, dtype=dtype, device=dev)

        d_near = d[mask]
        a_near = a[mask]

        # Decay length = 40% of influence radius (matches objective_terms.py convention)
        decay = self.obs_influence * 0.4
        magnitudes = self.k_obs * torch.exp(-d_near / max(decay, 1e-6))

        fx = -(magnitudes * torch.cos(a_near)).sum()
        fy = -(magnitudes * torch.sin(a_near)).sum()
        F = torch.stack([fx, fy])

        # Clamp total obstacle force magnitude to avoid the equilibrium
        # residual being completely dominated by F_obs in tight spaces.
        # This preserves the direction of the force but limits its magnitude
        # so that G_explore (exploration gain) can still discriminate candidates.
        max_obs_force = 5.0
        F_norm = F.norm().clamp(min=1e-6)
        if F_norm > max_obs_force:
            F = F * (max_obs_force / F_norm)
        return F

    def _pack_ped_tensors(
        self,
        ped_states: List[dict],
        ped_pss: List["PersonalSpace"],
        dev: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple:
        """Convert pedestrian state lists to device tensors.

        Returns
        -------
        ped_xy0   : Tensor (n_peds, 2)
        ped_vel   : Tensor (n_peds, 2)
        ped_yaw0  : Tensor (n_peds,)
        sig_front : Tensor (n_peds,)
        sig_side  : Tensor (n_peds,)
        sig_back  : Tensor (n_peds,)
        """
        ped_xy0 = torch.tensor(
            np.stack([p["xy"] for p in ped_states]), dtype=dtype, device=dev
        )
        ped_vel = torch.tensor(
            np.stack([p["vel"] for p in ped_states]), dtype=dtype, device=dev
        )
        ped_yaw0 = torch.tensor(
            [p["yaw"] for p in ped_states], dtype=dtype, device=dev
        )
        sig_front = torch.tensor(
            [ps.sigma_front for ps in ped_pss], dtype=dtype, device=dev
        )
        sig_side = torch.tensor(
            [ps.sigma_side for ps in ped_pss], dtype=dtype, device=dev
        )
        sig_back = torch.tensor(
            [ps.sigma_back for ps in ped_pss], dtype=dtype, device=dev
        )
        return ped_xy0, ped_vel, ped_yaw0, sig_front, sig_side, sig_back

    def _predict_peds_const_vel(
        self,
        ped_xy0: "torch.Tensor",
        ped_vel: "torch.Tensor",
        H: int,
    ) -> "torch.Tensor":
        """Constant-velocity pedestrian trajectory prediction.

        Parameters
        ----------
        ped_xy0 : Tensor (n_peds, 2)
        ped_vel : Tensor (n_peds, 2)
        H       : int  horizon length

        Returns
        -------
        Tensor (n_peds, H+1, 2)
            ped_traj[j, t] = ped_xy0[j] + ped_vel[j] * t * dt
        """
        dt = self.dt
        t = torch.arange(
            H + 1, dtype=ped_xy0.dtype, device=ped_xy0.device
        ).view(1, H + 1, 1)  # (1, H+1, 1)
        return ped_xy0.unsqueeze(1) + ped_vel.unsqueeze(1) * (t * dt)

    def _compute_human_forces(
        self,
        robot_xy: "torch.Tensor",
        ped_traj: "torch.Tensor",
        ped_yaw_seq: "torch.Tensor",
        sig_front: "torch.Tensor",
        sig_side: "torch.Tensor",
        sig_back: "torch.Tensor",
        unc_scales: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute anisotropic Gaussian personal-space repulsion F_human.

        Paper: F_{human,j}(x_tau, h_hat_{j,tau})
            = k_ps * G_j(x, h_j) * (x - h_j) / ||x - h_j||

        where G_j is an anisotropic Gaussian with sigma modulated by
        ``unc_scales``:

            sigma_x = sigma_front (if robot is in front of ped) * unc_scale
                    = sigma_back  (if robot is behind ped)      * unc_scale
            sigma_y = sigma_side * unc_scale

        Parameters
        ----------
        robot_xy     : Tensor (n_cand, H+1, 2)
        ped_traj     : Tensor (n_peds, H+1, 2)
        ped_yaw_seq  : Tensor (n_peds, H+1)
        sig_front    : Tensor (n_peds,)
        sig_side     : Tensor (n_peds,)
        sig_back     : Tensor (n_peds,)
        unc_scales   : Tensor (n_peds,)

        Returns
        -------
        Tensor, shape (n_cand, n_peds, H+1, 2)
        """
        n_cand, Hp1, _ = robot_xy.shape
        n_peds = ped_traj.shape[0]

        # rel: (n_cand, n_peds, H+1, 2)  — robot position relative to pedestrian
        r = robot_xy.unsqueeze(1)          # (n_cand, 1, H+1, 2)
        p = ped_traj.unsqueeze(0)          # (1, n_peds, H+1, 2)
        rel = r - p                         # (n_cand, n_peds, H+1, 2)

        dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Rotate into pedestrian heading frame to determine front/back
        # ped_yaw_seq: (n_peds, H+1) → (1, n_peds, H+1, 1)
        yaw = ped_yaw_seq.unsqueeze(0).unsqueeze(-1)
        cy = torch.cos(-yaw)
        sy = torch.sin(-yaw)
        dx = cy * rel[..., 0:1] - sy * rel[..., 1:2]  # forward component
        dy = sy * rel[..., 0:1] + cy * rel[..., 1:2]  # lateral component

        # Apply uncertainty scaling to sigma values
        # unc_scales, sig_front, sig_back, sig_side: (n_peds,) → (1, n_peds, 1, 1)
        sf = (sig_front * unc_scales).view(1, n_peds, 1, 1)
        sb = (sig_back * unc_scales).view(1, n_peds, 1, 1)
        ss = (sig_side * unc_scales).view(1, n_peds, 1, 1)

        # Select sigma_x based on whether robot is in front of pedestrian
        # dx > 0 → robot in front of ped → use sigma_front
        sigma_x = torch.where(dx >= 0, sf, sb)   # (n_cand, n_peds, H+1, 1)

        # Exponent of anisotropic Gaussian
        ex = (dx ** 2) / (2.0 * sigma_x ** 2 + 1e-9)
        ey = (dy ** 2) / (2.0 * ss ** 2 + 1e-9)
        gauss = torch.exp(-(ex + ey))  # (n_cand, n_peds, H+1, 1)

        # Repulsive force direction: away from pedestrian
        direction = rel / dist  # (n_cand, n_peds, H+1, 2)
        return self.k_ps * gauss * direction  # (n_cand, n_peds, H+1, 2)

    def _compute_uncertainty_cost(
        self,
        robot_xy: "torch.Tensor",
        ped_xy0: "torch.Tensor",
        ped_vel: "torch.Tensor",
        ped_yaw0: "torch.Tensor",
        sig_front: "torch.Tensor",
        sig_side: "torch.Tensor",
        sig_back: "torch.Tensor",
        unc_scales: "torch.Tensor",
    ) -> "torch.Tensor":
        """Batch MC-Ensemble uncertainty cost C_unc.

        Estimates prediction uncertainty by injecting Gaussian velocity noise
        (std=0.15 m/s) into S pedestrian rollouts and measuring the variance of
        F_human across samples.

        Paper:
            C_unc[i] = mean_tau( sum_j  Var_s[ F_{human,j}(x_{i,tau}, h_{j,s,tau}) ] )

        where s indexes the S noise samples and i indexes the candidate.

        Parameters
        ----------
        robot_xy   : Tensor (n_cand, H+1, 2)   — candidate trajectories
        ped_xy0    : Tensor (n_peds, 2)
        ped_vel    : Tensor (n_peds, 2)
        ped_yaw0   : Tensor (n_peds,)
        sig_front/side/back : Tensor (n_peds,)
        unc_scales : Tensor (n_peds,)

        Returns
        -------
        Tensor, shape (n_cand,)
        """
        S = self.S
        H = self.H
        n_peds = ped_xy0.shape[0]
        n_cand = robot_xy.shape[0]
        dev, dtype = robot_xy.device, robot_xy.dtype

        # Sample S sets of noisy velocities: (S, n_peds, 2)
        noise = torch.randn(S, n_peds, 2, dtype=dtype, device=dev) * self._unc_noise_std
        # noisy_vel: (S, n_peds, 2)
        noisy_vel = ped_vel.unsqueeze(0) + noise

        # Predict S×n_peds trajectories under noisy velocities
        # t: (1, 1, H+1, 1)
        t = torch.arange(H + 1, dtype=dtype, device=dev).view(1, 1, H + 1, 1)
        dt = self.dt
        # ped_traj_samples: (S, n_peds, H+1, 2)
        ped_traj_samples = (
            ped_xy0.unsqueeze(0).unsqueeze(2)        # (1, n_peds, 1, 2)
            + noisy_vel.unsqueeze(2) * (t * dt)      # (S, n_peds, H+1, 2)
        )

        # Pedestrian yaw (constant); replicate across S samples
        # ped_yaw_seq: (n_peds, H+1) → (S, n_peds, H+1)
        ped_yaw_seq = ped_yaw0.unsqueeze(1).expand(n_peds, H + 1)
        ped_yaw_seq_s = ped_yaw_seq.unsqueeze(0).expand(S, n_peds, H + 1)

        # Compute F_human for all (s, n_cand, n_peds, H+1)
        # robot_xy: (n_cand, H+1, 2) → (1, n_cand, 1, H+1, 2)
        r = robot_xy.unsqueeze(0).unsqueeze(2)
        # ped_traj_samples: (S, n_peds, H+1, 2) → (S, 1, n_peds, H+1, 2)
        p = ped_traj_samples.unsqueeze(1)
        # rel: (S, n_cand, n_peds, H+1, 2)
        rel = r - p
        dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Rotate into pedestrian frame
        # ped_yaw_seq_s: (S, n_peds, H+1) → (S, 1, n_peds, H+1, 1)
        yaw = ped_yaw_seq_s.unsqueeze(1).unsqueeze(-1)
        cy = torch.cos(-yaw)
        sy = torch.sin(-yaw)
        dx = cy * rel[..., 0:1] - sy * rel[..., 1:2]
        dy = sy * rel[..., 0:1] + cy * rel[..., 1:2]

        # Sigma values: (n_peds,) → (1, 1, n_peds, 1, 1)
        sf = (sig_front * unc_scales).view(1, 1, n_peds, 1, 1)
        sb = (sig_back * unc_scales).view(1, 1, n_peds, 1, 1)
        ss = (sig_side * unc_scales).view(1, 1, n_peds, 1, 1)

        sigma_x = torch.where(dx >= 0, sf, sb)
        ex = (dx ** 2) / (2.0 * sigma_x ** 2 + 1e-9)
        ey = (dy ** 2) / (2.0 * ss ** 2 + 1e-9)
        gauss = torch.exp(-(ex + ey))  # (S, n_cand, n_peds, H+1, 1)

        direction = rel / dist          # (S, n_cand, n_peds, H+1, 2)
        F_h = self.k_ps * gauss * direction  # (S, n_cand, n_peds, H+1, 2)

        # ||F_human||: (S, n_cand, n_peds, H+1)
        F_h_mag = F_h.norm(dim=-1)

        # Var over S samples: (n_cand, n_peds, H+1)
        var_F = F_h_mag.var(dim=0)

        # C_unc[i] = mean over H of sum_j var_j
        # sum over peds: (n_cand, H+1)
        var_sum = var_F.sum(dim=1)
        # mean over horizon steps (exclude t=0)
        C_unc = var_sum[:, 1:].mean(dim=1)  # (n_cand,)
        return C_unc

    def _compute_dynamics_cost(self, ctrl: "torch.Tensor") -> "torch.Tensor":
        """Dynamics / smoothness cost (squared jerk).

        Paper: C_dyn(U) = sum_t (delta_v_t^2 + delta_w_t^2) / dt^2

        Parameters
        ----------
        ctrl : Tensor, shape (n_cand, H, 2)

        Returns
        -------
        Tensor, shape (n_cand,)
        """
        if ctrl.shape[1] < 2:
            return torch.zeros(ctrl.shape[0], dtype=ctrl.dtype, device=ctrl.device)
        delta = ctrl[:, 1:, :] - ctrl[:, :-1, :]   # (n_cand, H-1, 2)
        accel = delta / max(self.dt, 1e-6)           # scale to jerk units
        return (accel ** 2).sum(dim=(1, 2))          # (n_cand,)

    def _compute_exploration_gain(
        self,
        robot_xy: "torch.Tensor",
        visited_xys: List[np.ndarray],
        dev: "torch.device",
        dtype: "torch.dtype",
        visited_cells: Optional[set] = None,
    ) -> "torch.Tensor":
        """Count novel positions along each candidate trajectory.

        Paper: G_explore(U) = #{tau : ||x_tau - v||_2 > novelty_radius forall v}

        When ``visited_cells`` (a set of grid-cell (int, int) tuples) is
        provided, novelty is checked via O(1) hash-set lookup using the same
        grid resolution as ``ProactiveSIEP._visited_cells``.  This is accurate
        even after thousands of steps because it checks the *complete* visited
        history rather than only the last 100 positions.

        Falls back to the distance-based ``visited_xys[-100:]`` approach when
        ``visited_cells`` is not supplied (backward compatibility).

        Parameters
        ----------
        robot_xy    : Tensor (n_cand, H+1, 2)
        visited_xys : list of np.ndarray (2,)
        visited_cells : set of (int, int) or None

        Returns
        -------
        Tensor, shape (n_cand,)
        """
        n_cand = robot_xy.shape[0]
        H = robot_xy.shape[1] - 1  # H+1 states → H steps after t=0

        if visited_cells is not None:
            # Grid-based O(1) lookup — accurate across the full run history.
            # Encode each (cx, cy) grid cell as a single int64 key so we can
            # use numpy's vectorised isin() without Python loops per cell.
            traj = robot_xy[:, 1:, :]  # (n_cand, H, 2) — skip t=0
            traj_np = traj.detach().cpu().numpy()

            step = self._novelty_grid_step
            cells_x = np.floor(traj_np[:, :, 0] / step).astype(np.int64)  # (n_cand, H)
            cells_y = np.floor(traj_np[:, :, 1] / step).astype(np.int64)  # (n_cand, H)

            # A large-enough offset to make (x, y) → int64 bijective for
            # typical world sizes (up to ±50 000 cells on each axis).
            OFFSET = np.int64(100_000)
            candidate_keys = cells_x * OFFSET + cells_y  # (n_cand, H)

            if visited_cells:
                visited_keys = np.fromiter(
                    (int(cx) * OFFSET + int(cy) for cx, cy in visited_cells),
                    dtype=np.int64,
                    count=len(visited_cells),
                )
                novel = ~np.isin(candidate_keys, visited_keys)  # (n_cand, H)
            else:
                # No visited cells yet — every position is novel
                novel = np.ones((n_cand, H), dtype=bool)

            gain = novel.sum(axis=1).astype(np.float32)  # (n_cand,)
            return torch.tensor(gain, dtype=dtype, device=dev)

        # ── Fallback: distance-based check (last 100 visited positions) ──
        if not visited_xys:
            # Every position is novel — gain = H (all steps after t=0)
            return torch.full(
                (n_cand,), float(H), dtype=dtype, device=dev
            )

        # Use last 100 visited positions for efficiency
        recent = visited_xys[-100:]
        visited_t = torch.tensor(
            np.stack(recent), dtype=dtype, device=dev
        )  # (V, 2)

        # robot_xy[:, 1:, :]: (n_cand, H, 2) — skip t=0
        traj = robot_xy[:, 1:, :]   # (n_cand, H, 2)

        # Pairwise distances: (n_cand, H, V)
        diff = traj.unsqueeze(2) - visited_t.view(1, 1, -1, 2)
        dists = diff.norm(dim=-1)   # (n_cand, H, V)

        # A position is novel if the minimum distance to any visited point
        # exceeds novelty_radius
        min_dist = dists.min(dim=-1).values  # (n_cand, H)
        novel = (min_dist > self.novelty_radius).float()
        return novel.sum(dim=1)  # (n_cand,)

    def _compute_frontier_forces(
        self,
        robot_xy: "torch.Tensor",
        robot_yaw: "torch.Tensor",
        lidar_dists: np.ndarray,
        lidar_angles: np.ndarray,
        visited_xys: List[np.ndarray],
    ) -> "torch.Tensor":
        """Frontier exploration force (batch version of F_frontier).

        Drives robot toward open, previously-unvisited directions.
        The score for each LiDAR ray combines openness (normalised range) and
        novelty (1.0 if unvisited, 0.2 if visited).

        The resulting force is constant across candidates at t=0 (current pose)
        and propagated using the candidate trajectories for subsequent steps.
        For simplicity, uses the same force vector at every horizon step
        (reactive force based on current LiDAR snapshot).

        Parameters
        ----------
        robot_xy  : Tensor (n_cand, H+1, 2)
        robot_yaw : Tensor (n_cand, H+1)
        lidar_dists  : np.ndarray (N_rays,)
        lidar_angles : np.ndarray (N_rays,) world-frame

        Returns
        -------
        Tensor, shape (n_cand, H+1, 2)
        """
        dev, dtype = robot_xy.device, robot_xy.dtype
        d = torch.tensor(lidar_dists, dtype=dtype, device=dev)
        a = torch.tensor(lidar_angles, dtype=dtype, device=dev)

        max_range = d.max().clamp(min=1e-6)
        openness = d / max_range  # (N_rays,)

        # Novelty per ray (probe at 50% of LiDAR distance)
        recent = visited_xys[-50:] if visited_xys else []
        novelty_vals: List[float] = []
        for i in range(len(lidar_dists)):
            if not recent:
                novelty_vals.append(1.0)
                continue
            ang = float(lidar_angles[i])
            dist_i = float(lidar_dists[i])
            # Current robot position (use initial pose — index 0)
            rx = float(robot_xy[0, 0, 0].item())
            ry = float(robot_xy[0, 0, 1].item())
            # Probe at 50% of LiDAR distance (avoids probing past obstacles)
            # clamped to 2× novelty_radius so far-away rays still get evaluated
            probe_dist = min(dist_i * 0.5, self.novelty_radius * 2.0)
            px = rx + probe_dist * math.cos(ang)
            py = ry + probe_dist * math.sin(ang)
            probe = np.array([px, py], dtype=np.float32)
            visited = any(
                float(np.linalg.norm(probe - v)) < self.novelty_radius
                for v in recent
            )
            # Visited directions get a reduced novelty score (0.2) to discourage
            # revisiting while not completely blocking those directions
            novelty_vals.append(0.2 if visited else 1.0)

        novelty = torch.tensor(novelty_vals, dtype=dtype, device=dev)
        scores = openness * novelty  # (N_rays,)

        fx = (scores * torch.cos(a)).sum()
        fy = (scores * torch.sin(a)).sum()
        F_dir = torch.stack([fx, fy])
        F_norm = F_dir.norm().clamp(min=1e-6)
        F_frontier = F_dir / F_norm  # unit vector

        # Broadcast to (n_cand, H+1, 2)
        F_frontier = self.k_frontier * F_frontier
        return F_frontier.view(1, 1, 2).expand(robot_xy.shape[0], robot_xy.shape[1], 2)
