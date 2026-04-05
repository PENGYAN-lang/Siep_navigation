"""
Objective terms for Proactive-SIEP.

Implements all terms in the receding-horizon objective:

  J(U) = sum_{tau=1}^{H} ||F_tot(x_tau, H_hat_tau(U), z_tau)||^2
         + lambda_s * C_social(U, H_hat(U), z)
         + lambda_u * C_unc(U)
         + lambda_d * C_dyn(U)
         - lambda_e * G_explore(U)

where:

  F_tot = F_{goal/frontier}
          + sum_o  F_{obs,o}
          + sum_j  w_j(z_tau) * F_{human,j}(x_tau, h_hat_{j,tau}(U))
          + F_ctx(z_tau)

Paper equation mapping
----------------------
  force_total(...)  ←  F_tot
  F_goal_force      ←  F_{goal/frontier}
  F_obstacle_force  ←  sum_o F_{obs,o}
  F_human_force     ←  F_{human,j}
  F_ctx             ←  F_ctx
  social_cost       ←  C_social
  uncertainty_cost  ←  C_unc
  dynamics_cost     ←  C_dyn
  exploration_gain  ←  G_explore
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..utils.social import PersonalSpace


# ─────────────────────────────────────────────────────────────────────────────
# Individual force terms
# ─────────────────────────────────────────────────────────────────────────────

def F_goal_force(
    robot_xy: np.ndarray,
    goal_xy: np.ndarray,
    k: float,
    sigma: float,
) -> np.ndarray:
    """Tanh-saturated goal attraction.  F_{goal}."""
    d = goal_xy - robot_xy
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return np.zeros(2, dtype=float)
    magnitude = k * math.tanh(dist / max(sigma, 1e-6))
    return magnitude * d / dist


def F_frontier_force(
    robot_xy: np.ndarray,
    robot_yaw: float,
    lidar_dists: np.ndarray,
    lidar_angles: np.ndarray,
    visited_xys: List[np.ndarray],
    k: float,
    novelty_radius: float = 1.5,
    visited_cells: Optional[set] = None,
) -> np.ndarray:
    """Frontier exploration force (replaces F_goal in exploration mode).

    Drives the robot toward open, previously-unvisited directions without
    requiring a global map (mapless reactive exploration).

    The force direction maximises:
        openness(direction) * novelty(direction)
    where openness is the LiDAR distance in that direction (far = open) and
    novelty is 1 - overlap with visited positions in that direction.

    Paper: G_explore stimulus  (contributes to F_{goal/frontier} in F_tot).

    Parameters
    ----------
    visited_cells : optional pre-computed set of ``(int, int)`` grid-cell
        keys (discretised at ``novelty_radius`` resolution).  When provided
        the O(1) set-lookup is used instead of re-computing the set each call.
    """
    n = len(lidar_angles)
    if n == 0:
        return np.zeros(2, dtype=float)

    # Build visited-cell set once if not supplied (grid-based novelty check)
    grid_step = max(novelty_radius, 1e-3)
    if visited_cells is None and visited_xys:
        visited_cells = {
            (int(vxy[0] // grid_step), int(vxy[1] // grid_step))
            for vxy in visited_xys
        }

    scores = np.zeros(n, dtype=float)
    for idx, (dist, ang) in enumerate(zip(lidar_dists, lidar_angles)):
        # Openness: normalised LiDAR range (0..1)
        max_range = float(np.max(lidar_dists)) + 1e-6
        openness = dist / max_range

        # Novelty: check if probe direction falls in a visited grid cell
        probe_dist = min(dist * 0.5, novelty_radius * 2)
        probe_xy = robot_xy + probe_dist * np.array([math.cos(ang), math.sin(ang)])
        if visited_cells:
            cell = (int(probe_xy[0] // grid_step), int(probe_xy[1] // grid_step))
            visited = cell in visited_cells
        else:
            visited = False
        novelty = 0.2 if visited else 1.0

        scores[idx] = openness * novelty

    # Weighted direction sum
    fx = float(np.dot(scores, np.cos(lidar_angles)))
    fy = float(np.dot(scores, np.sin(lidar_angles)))
    F = np.array([fx, fy], dtype=float)
    F_norm = float(np.linalg.norm(F))
    if F_norm < 1e-6:
        return np.zeros(2, dtype=float)
    return k * F / F_norm


def F_obstacle_force(
    lidar_dists: np.ndarray,
    lidar_angles_world: np.ndarray,
    k: float,
    influence_dist: float,
) -> np.ndarray:
    """Exponential obstacle repulsion from LiDAR.  sum_o F_{obs,o}."""
    mask = lidar_dists < influence_dist
    if not np.any(mask):
        return np.zeros(2, dtype=float)
    d_near = lidar_dists[mask]
    a_near = lidar_angles_world[mask]
    decay = influence_dist * 0.4
    magnitudes = k * np.exp(-d_near / max(decay, 1e-6))
    fx = -float(np.sum(magnitudes * np.cos(a_near)))
    fy = -float(np.sum(magnitudes * np.sin(a_near)))
    return np.array([fx, fy], dtype=float)


def F_human_force(
    robot_xy: np.ndarray,
    ped_xy: np.ndarray,
    ped_yaw: float,
    ps: PersonalSpace,
    k: float,
    uncertainty_scale: float = 1.0,
) -> np.ndarray:
    """Anisotropic Gaussian personal-space repulsion from one pedestrian.

    Paper: F_{human,j}(x_tau, h_hat_{j,tau}(U))

    The uncertainty_scale modulates the effective sigma of the Gaussian — larger
    uncertainty (wider predicted occupancy) widens the personal space, producing
    stronger avoidance.  This realises the uncertainty-force modulation
    described in the framework.

    Parameters
    ----------
    uncertainty_scale : float ≥ 1.0
        Multiplicative factor on ps sigma values.  1.0 = nominal, >1 = inflated.
    """
    d = robot_xy - ped_xy
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return np.zeros(2, dtype=float)

    c, s = math.cos(-ped_yaw), math.sin(-ped_yaw)
    dx = c * d[0] - s * d[1]
    dy = s * d[0] + c * d[1]

    # Apply uncertainty scaling to the sigma parameters
    sigma_x = (ps.sigma_front if dx >= 0 else ps.sigma_back) * uncertainty_scale
    sigma_y = ps.sigma_side * uncertainty_scale

    ex = (dx * dx) / (2.0 * sigma_x ** 2 + 1e-9)
    ey = (dy * dy) / (2.0 * sigma_y ** 2 + 1e-9)
    cost = float(math.exp(-(ex + ey)))
    return k * cost * (d / dist)


def force_total(
    robot_xy: np.ndarray,
    robot_yaw: float,
    goal_xy: Optional[np.ndarray],
    lidar_dists: np.ndarray,
    lidar_angles: np.ndarray,
    ped_xys: List[np.ndarray],
    ped_yaws: List[float],
    ped_pss: List[PersonalSpace],
    context_weights: List[float],
    F_ctx_vec: np.ndarray,
    k_goal: float,
    sigma_goal: float,
    k_obs: float,
    obs_influence: float,
    k_ps: float,
    uncertainty_scales: Optional[List[float]] = None,
    # exploration mode inputs
    explore_mode: bool = False,
    visited_xys: Optional[List[np.ndarray]] = None,
    k_frontier: float = 1.0,
    novelty_radius: float = 1.5,
    visited_cells: Optional[set] = None,
) -> np.ndarray:
    """Compute total force vector F_tot.

    F_tot = F_{goal/frontier}
            + sum_o F_{obs,o}
            + sum_j w_j * F_{human,j}
            + F_ctx
    """
    if uncertainty_scales is None:
        uncertainty_scales = [1.0] * len(ped_xys)

    # Goal or frontier force
    if explore_mode and lidar_dists is not None and visited_xys is not None:
        F = F_frontier_force(
            robot_xy, robot_yaw, lidar_dists, lidar_angles,
            visited_xys, k_frontier, novelty_radius,
            visited_cells=visited_cells,
        )
    elif goal_xy is not None:
        F = F_goal_force(robot_xy, goal_xy, k_goal, sigma_goal)
    else:
        F = np.zeros(2, dtype=float)

    # Obstacle force
    F += F_obstacle_force(lidar_dists, lidar_angles, k_obs, obs_influence)

    # Human forces (context-weighted, uncertainty-modulated)
    for j, (pxy, pyaw, pps, w_j, unc_j) in enumerate(
        zip(ped_xys, ped_yaws, ped_pss, context_weights, uncertainty_scales)
    ):
        F += w_j * F_human_force(robot_xy, pxy, pyaw, pps, k_ps, unc_j)

    # Context force
    F += F_ctx_vec

    return F


# ─────────────────────────────────────────────────────────────────────────────
# Composite cost terms
# ─────────────────────────────────────────────────────────────────────────────

def equilibrium_residual(
    robot_traj: List[np.ndarray],
    robot_yaws: List[float],
    goal_xy: Optional[np.ndarray],
    lidar_dists: np.ndarray,
    lidar_angles: np.ndarray,
    ped_trajs: List[List[np.ndarray]],   # ped_trajs[j][tau]
    ped_yaws_traj: List[List[float]],    # ped_yaws_traj[j][tau]
    ped_pss: List[PersonalSpace],
    context_weights_traj: List[List[float]],  # [j][tau]
    F_ctx_traj: List[np.ndarray],             # [tau]
    params: dict,
    explore_mode: bool = False,
    visited_xys: Optional[List[np.ndarray]] = None,
    uncertainty_scales: Optional[List[List[float]]] = None,
    visited_cells: Optional[set] = None,
) -> float:
    """sum_{tau=1}^{H} ||F_tot(x_tau, H_hat_tau, z_tau)||^2.

    Paper: equilibrium residual — the force at each step should be small
    (ideally zero at the true equilibrium).
    """
    H = len(robot_traj) - 1
    total = 0.0

    if uncertainty_scales is None:
        uncertainty_scales = [[1.0] * len(ped_trajs)] * (H + 1)

    for tau in range(1, H + 1):
        r_xy = robot_traj[tau]
        r_yaw = robot_yaws[tau]
        pxys_t = [ped_trajs[j][tau] for j in range(len(ped_trajs))]
        pyaws_t = [ped_yaws_traj[j][tau] for j in range(len(ped_trajs))]
        cw_t = [context_weights_traj[j][tau] for j in range(len(ped_trajs))]
        F_ctx_t = F_ctx_traj[tau] if tau < len(F_ctx_traj) else np.zeros(2)
        unc_t = uncertainty_scales[tau]

        Fv = force_total(
            r_xy, r_yaw, goal_xy,
            lidar_dists, lidar_angles,
            pxys_t, pyaws_t, ped_pss,
            cw_t, F_ctx_t,
            params['k_goal'], params['sigma_goal'],
            params['k_obs'], params['obs_influence'], params['k_ps'],
            uncertainty_scales=unc_t,
            explore_mode=explore_mode,
            visited_xys=visited_xys,
            k_frontier=params.get('k_frontier', 1.0),
            novelty_radius=params.get('novelty_radius', 1.5),
            visited_cells=visited_cells,
        )
        total += float(np.dot(Fv, Fv))

    return total


def social_cost(
    robot_traj: List[np.ndarray],
    ped_trajs: List[List[np.ndarray]],
    ped_pss: List[PersonalSpace],
    ped_yaws_traj: List[List[float]],
    context_weights_traj: List[List[float]],
    k_ps: float,
) -> float:
    """C_social: integral of personal-space violation along horizon.

    Paper: C_social(U, H_hat(U), z) measures how much the planned trajectory
    intrudes into pedestrian personal spaces, weighted by context.
    """
    H = len(robot_traj) - 1
    total = 0.0
    for tau in range(1, H + 1):
        r_xy = robot_traj[tau]
        for j, (ped_traj, pps) in enumerate(zip(ped_trajs, ped_pss)):
            pxy = ped_traj[tau]
            pyaw = ped_yaws_traj[j][tau]
            w_j = context_weights_traj[j][tau]
            f = F_human_force(r_xy, pxy, pyaw, pps, k_ps)
            total += w_j * float(np.dot(f, f))
    return total


def uncertainty_cost(
    ped_trajs_samples: List[List[List[np.ndarray]]],
    robot_traj: List[np.ndarray],
    k_ps: float,
    ped_pss: List[PersonalSpace],
    ped_yaws: List[float],
) -> Tuple[float, List[float]]:
    """C_unc: uncertainty-aware risk from trajectory prediction variance.

    Models prediction uncertainty as variance across S independently-sampled
    pedestrian trajectories.  In practice with deterministic prediction,
    we estimate uncertainty via noise-injected rollouts.

    Also returns per-pedestrian uncertainty scales for force modulation.

    Paper: C_unc(U) = sum_j sum_tau Var[||F_{human,j}(x_tau, h_hat_{j,tau})||]

    Parameters
    ----------
    ped_trajs_samples : list of S samples; each sample is ped_trajs[j][tau]
    robot_traj        : rollout robot positions (H+1,)
    """
    if len(ped_trajs_samples) <= 1:
        n_peds = len(ped_trajs_samples[0]) if ped_trajs_samples else 0
        return 0.0, [1.0] * n_peds

    S = len(ped_trajs_samples)
    n_peds = len(ped_trajs_samples[0])
    H = len(robot_traj) - 1

    unc_cost = 0.0
    uncertainty_scales: List[float] = []

    for j in range(n_peds):
        pps = ped_pss[j]
        pyaw = ped_yaws[j]
        force_mags: List[List[float]] = []  # [s][tau]

        for s in range(S):
            mags = []
            for tau in range(1, H + 1):
                pxy = ped_trajs_samples[s][j][tau]
                r_xy = robot_traj[tau]
                F = F_human_force(r_xy, pxy, pyaw, pps, k_ps)
                mags.append(float(np.linalg.norm(F)))
            force_mags.append(mags)

        force_arr = np.array(force_mags)  # (S, H)
        # Variance over samples at each time step, summed over horizon
        var_j = float(np.sum(np.var(force_arr, axis=0)))
        unc_cost += var_j

        # Uncertainty scale: inflated sigma proportional to std dev
        mean_std = float(np.mean(np.std(force_arr, axis=0)))
        unc_scale = 1.0 + min(mean_std, 1.0)  # cap at 2x inflation
        uncertainty_scales.append(unc_scale)

    return unc_cost, uncertainty_scales


def dynamics_cost(
    control_seq: np.ndarray,   # (H, 2) array of [v, w] per step
    dt: float,
) -> float:
    """C_dyn: smoothness penalty on control sequence.

    Penalises:
      1. Angular jerk (rate of change of angular velocity).
      2. Linear acceleration magnitude.

    Paper: C_dyn(U) = sum_tau (delta_v^2 + delta_w^2) / dt^2
    """
    if len(control_seq) < 2:
        return 0.0
    delta = np.diff(control_seq, axis=0)  # (H-1, 2)
    # Scale by 1/dt² to get jerk units
    accel = delta / max(dt, 1e-6)
    return float(np.sum(accel ** 2))


def exploration_gain(
    robot_traj: List[np.ndarray],
    visited_xys: List[np.ndarray],
    novelty_radius: float = 1.5,
    visited_cells: Optional[set] = None,
) -> float:
    """G_explore(U): reward for visiting novel positions along trajectory.

    Paper: -lambda_e * G_explore penalises low coverage; maximising G_explore
    encourages frontier-seeking behaviour.

    Computed as the number of trajectory positions that are farther than
    novelty_radius from all previously visited positions.

    Parameters
    ----------
    visited_cells : optional pre-computed set of ``(int, int)`` grid-cell keys.
        When provided the O(1) set-lookup is used.
    """
    if not visited_xys:
        return float(len(robot_traj))

    # Build grid-cell set if not supplied
    grid_step = max(novelty_radius, 1e-3)
    if visited_cells is None:
        visited_cells = {
            (int(vxy[0] // grid_step), int(vxy[1] // grid_step))
            for vxy in visited_xys
        }

    gain = 0.0
    for xy in robot_traj[1:]:
        cell = (int(xy[0] // grid_step), int(xy[1] // grid_step))
        if cell not in visited_cells:
            gain += 1.0
    return gain
