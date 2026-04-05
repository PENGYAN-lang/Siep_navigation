"""
SIEP (Stimuli-Induced Equilibrium Point) virtual-force kernels.

In the SIEP framework the robot's desired velocity is the superposition of
virtual forces induced by all stimuli perceived from the environment:

  F_total = F_goal + F_obstacle + F_personal_space + F_velocity_alignment

The direction and magnitude of F_total define the robot's *equilibrium point* –
the velocity the robot is driven towards.  A proportional heading controller
then converts this vector into (v, ω) differential-drive commands.

Base innovations (original three):

1. **Predictive personal-space forces** – repulsion is integrated over a
   look-ahead horizon so the robot steers away *before* a personal-space
   violation occurs (time-discounted sum of future poses).

2. **Velocity-alignment stimulus** – a gentle force that encourages the robot
   to match the speed/direction of nearby pedestrians walking the same way,
   capturing crowd-flow awareness.

3. **Tanh-saturated goal attraction** – the goal force saturates at the
   desired cruising speed rather than growing unboundedly, giving smoother
   deceleration near the goal.

Extended innovations (novel algorithmic contributions):

4. **Context-Adaptive SIEP (CA-SIEP)** – force-channel weights are dynamically
   modulated by a *social context function* λ(ρ, v̄, θ_face, g) computed from
   local crowd density (ρ), mean pedestrian speed (v̄), pedestrian facing
   angle toward the robot (θ_face), and group membership flag (g).  Unlike
   prior SIEP/SFM work that uses fixed constants, CA-SIEP adapts the force
   balance to the current social situation (e.g. denser crowd ⟹ larger k_ps,
   smaller k_goal).  See ``SocialPerceptionLayer`` and
   ``context_modulate_weights()``.

5. **Uncertainty-Modulated Personal Space (UMPS)** – the prediction horizon
   assumes constant-velocity pedestrian motion, yet real pedestrians deviate.
   We quantify per-pedestrian epistemic uncertainty U(speed, dist, horizon)
   and expand σ_front/side/back proportionally:
       σ_eff = σ_base · (1 + κ · U)
   This yields a larger safety bubble for fast or distant pedestrians whose
   future positions are more uncertain.  See
   ``uncertainty_modulated_ps_force()``.

6. **Group-Level Social Force (GF)** – pedestrians walking in close proximity
   with a similar heading form a *group* whose collective personal space is
   significantly larger than any individual's.  Groups are detected online
   via proximity + heading-similarity clustering and treated as a single
   extended agent with an enlarged Gaussian envelope.
   See ``detect_groups()`` and ``group_repulsion_force()``.

7. **Frontier Exploration Force** – replaces F_goal when no preset destination
   exists.  Each LiDAR ray votes for a direction proportional to its openness
   (ray distance / max_range) weighted by a novelty score derived from the
   robot's visit history.  The resulting force drives the robot toward open,
   unvisited regions without a global map.
   See ``frontier_exploration_force()``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .social import PersonalSpace


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (empirically tuned; see inline comments)
# ─────────────────────────────────────────────────────────────────────────────

# Uncertainty estimation (Innovation 5 / UMPS)
_SPEED_UNCERTAINTY_COEFF: float = 0.15
# Multiplied by speed×horizon. At 1.2 m/s over 1.5 s → U_speed ≈ 0.27.
_DIST_UNCERTAINTY_THRESHOLD: float = 2.0   # [m] – ramp starts beyond this
_DIST_UNCERTAINTY_RATE: float = 0.05       # [1/m] – slope of linear ramp

# Uncertainty expansion of σ – side and back expand less than front
# (pedestrians primarily move forward, so front-uncertainty matters most).
_SIDE_UNCERTAINTY_FACTOR: float = 0.7
_BACK_UNCERTAINTY_FACTOR: float = 0.4

# Group repulsion (Innovation 6 / GF)
# How much each metre of intra-group spread adds to collective σ.
_GROUP_SPREAD_BONUS_FACTOR: float = 0.4
# The side/back collective σ bonus is reduced compared to front.
_GROUP_SPREAD_SIDE_SCALE: float = 0.8
_GROUP_SPREAD_BACK_SCALE: float = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Parameter container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SIEPParams:
    """Tunable weights and ranges for each SIEP stimulus.

    Original channels (1-4) are unchanged so existing configs remain valid.
    Extensions 5-8 add new parameters with safe defaults (off or neutral).
    """

    # --- Goal attraction --------------------------------------------------
    k_goal: float = 1.0       # peak force magnitude [m/s] (= desired cruise speed)
    sigma_goal: float = 3.0   # distance at which attraction saturates to ~76 % [m]

    # --- Obstacle repulsion (LiDAR-based) ---------------------------------
    k_obs: float = 2.5        # peak repulsion at zero distance
    obs_influence: float = 2.5  # influence radius [m]

    # --- Predictive personal-space repulsion ------------------------------
    k_ps: float = 2.0         # peak personal-space force magnitude
    ps_horizon: float = 1.5   # prediction look-ahead [s]
    ps_dt: float = 0.1        # prediction integration step [s]

    # --- Velocity-alignment stimulus (crowd-flow following) ---------------
    k_align: float = 0.3      # alignment force scale
    align_radius: float = 3.0  # maximum effective radius [m]
    align_cone_deg: float = 60.0  # directional cone [deg] – only align with
    #   pedestrians whose heading is within this half-angle of the robot's

    # ── Innovation 4: Context-Adaptive SIEP (CA-SIEP) ─────────────────────
    # Weight modulation is: w_i(t) = w_i_base * λ_i(context)
    # where λ_i is a bounded context-sensitive scalar in [lambda_min, lambda_max].
    use_context_adapt: bool = False   # enable CA-SIEP modulation
    context_radius: float = 4.0       # radius for local context perception [m]
    density_ps_gain: float = 1.5      # λ_ps amplification per ped/m² above threshold
    density_threshold: float = 0.2    # crowd density threshold [ped/m²]
    density_goal_decay: float = 0.6   # goal force reduction factor at high density
    face_ps_gain: float = 0.5         # additional λ_ps when pedestrian faces robot
    lambda_min: float = 0.3           # lower bound on any modulation coefficient
    lambda_max: float = 3.0           # upper bound on any modulation coefficient

    # ── Innovation 5: Uncertainty-Modulated Personal Space (UMPS) ─────────
    use_uncertainty_ps: bool = False  # enable UMPS
    uncertainty_gain: float = 1.2     # κ in σ_eff = σ_base*(1 + κ·U)
    max_uncertainty: float = 0.8      # cap on U to avoid infinite expansion

    # ── Innovation 6: Group-Level Social Force (GF) ────────────────────────
    use_group_force: bool = False      # enable group detection + force
    group_proximity: float = 2.0      # max centre-to-centre dist to form group [m]
    group_heading_thresh: float = 0.5  # cos(heading diff) threshold for grouping
    k_group: float = 1.5              # extra scale for group collective force
    group_sigma_scale: float = 1.4    # σ scale factor for group collective space

    # ── Innovation 7: Frontier Exploration Force ───────────────────────────
    k_frontier: float = 1.0           # frontier force magnitude
    novelty_radius: float = 2.0       # visited-cell radius for novelty scoring [m]
    frontier_random_weight: float = 0.1  # random exploration component weight


# ─────────────────────────────────────────────────────────────────────────────
# Individual force kernels
# ─────────────────────────────────────────────────────────────────────────────

def goal_force(
    robot_xy: np.ndarray,
    goal_xy: np.ndarray,
    k: float,
    sigma: float,
) -> np.ndarray:
    """Tanh-saturated attraction toward the goal.

    The magnitude ramps up from zero at the goal and saturates at *k* far
    away, preventing unbounded speed commands in long corridors.
    """
    d = goal_xy - robot_xy
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return np.zeros(2, dtype=float)
    magnitude = k * math.tanh(dist / max(sigma, 1e-6))
    return magnitude * d / dist


def obstacle_force(
    lidar_dists: np.ndarray,
    lidar_angles_world: np.ndarray,
    k: float,
    influence_dist: float,
) -> np.ndarray:
    """Repulsion from obstacles detected by LiDAR.

    Each ray that detects a surface within *influence_dist* contributes a
    force directed *away* from that surface.  Magnitude decays exponentially.
    """
    mask = lidar_dists < influence_dist
    if not np.any(mask):
        return np.zeros(2, dtype=float)

    d_near = lidar_dists[mask]
    a_near = lidar_angles_world[mask]
    # Decay constant chosen so force drops to e^{-2.5} ≈ 8 % at influence_dist.
    # decay = influence_dist * 0.4 means exp(-influence_dist / decay) = exp(-2.5).
    decay = influence_dist * 0.4
    magnitudes = k * np.exp(-d_near / max(decay, 1e-6))
    fx = -float(np.sum(magnitudes * np.cos(a_near)))
    fy = -float(np.sum(magnitudes * np.sin(a_near)))
    return np.array([fx, fy], dtype=float)


def personal_space_force(
    robot_xy: np.ndarray,
    ped_xy: np.ndarray,
    ped_yaw: float,
    ped_vel: np.ndarray,
    ps: PersonalSpace,
    k: float,
    horizon: float,
    dt: float = 0.1,
) -> np.ndarray:
    """Predictive anisotropic personal-space repulsion from one pedestrian.

    Integrates the anisotropic Gaussian cost over predicted pedestrian
    positions up to *horizon* seconds ahead.  Each future step is weighted
    by an exponential discount so that the immediate threat matters most.

    Args:
        robot_xy:  Current robot position (2,).
        ped_xy:    Current pedestrian position (2,).
        ped_yaw:   Pedestrian heading [rad].
        ped_vel:   Pedestrian velocity vector (2,).
        ps:        Personal-space parameters.
        k:         Force scale.
        horizon:   Look-ahead time [s].
        dt:        Integration step [s].
    """
    steps = max(1, int(horizon / max(dt, 1e-6)))
    force = np.zeros(2, dtype=float)

    for t in range(steps):
        ped_pred = ped_xy + ped_vel * (t * dt)
        cost = _aniso_gaussian(robot_xy, ped_pred, ped_yaw, ps)
        d = robot_xy - ped_pred
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            continue
        # Temporal discount: weight current step most heavily; halves every 2 s.
        # exp(-0.5 * t * dt) gives discount factor 0.5 after 2/dt steps (≈ 2 s).
        discount = math.exp(-0.5 * t * dt)
        force += k * cost * discount * (d / dist) * dt

    return force


def velocity_alignment_force(
    robot_xy: np.ndarray,
    robot_vel: np.ndarray,
    ped_xy: np.ndarray,
    ped_vel: np.ndarray,
    k: float,
    radius: float,
    cone_deg: float,
) -> np.ndarray:
    """Velocity-alignment (crowd-flow) stimulus.

    When a pedestrian is nearby *and* moving in roughly the same direction
    the robot is gently encouraged to match that pedestrian's velocity.
    This models the social norm of following crowd flow and smooths the
    robot's passage through streams of pedestrians.

    Args:
        robot_xy:   Current robot position (2,).
        robot_vel:  Current robot velocity vector (2,).
        ped_xy:     Pedestrian position (2,).
        ped_vel:    Pedestrian velocity (2,).
        k:          Force scale.
        radius:     Maximum effective radius [m].
        cone_deg:   Half-angle of the same-direction cone [deg].
    """
    dist = float(np.linalg.norm(ped_xy - robot_xy))
    if dist > radius:
        return np.zeros(2, dtype=float)

    ped_speed = float(np.linalg.norm(ped_vel))
    if ped_speed < 0.1:
        return np.zeros(2, dtype=float)

    # Only align when pedestrian moves in a similar direction
    robot_speed = float(np.linalg.norm(robot_vel))
    if robot_speed > 0.1:
        cos_thresh = math.cos(math.radians(cone_deg))
        cos_angle = float(np.dot(robot_vel / robot_speed, ped_vel / ped_speed))
        if cos_angle < cos_thresh:
            return np.zeros(2, dtype=float)

    # Proximity weight: Gaussian envelope that reaches ~e^{-0.5} ≈ 60 % at
    # half the radius and decays to near-zero at the boundary.  sigma = radius/2
    # ensures the force is negligible beyond the alignment radius.
    sigma_prox = radius * 0.5
    w_prox = math.exp(-(dist ** 2) / (2.0 * sigma_prox ** 2 + 1e-9))
    return k * w_prox * (ped_vel - robot_vel)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aniso_gaussian(
    robot_xy: np.ndarray,
    ped_xy: np.ndarray,
    ped_yaw: float,
    ps: PersonalSpace,
) -> float:
    """Anisotropic Gaussian cost in pedestrian body frame (same formula as
    ``anisotropic_gaussian_cost`` in ``social.py`` but kept local to avoid
    circular imports when both modules are extended independently)."""
    d = robot_xy - ped_xy
    c, s = math.cos(-ped_yaw), math.sin(-ped_yaw)
    dx = c * d[0] - s * d[1]
    dy = s * d[0] + c * d[1]
    sigma_x = ps.sigma_front if dx >= 0 else ps.sigma_back
    sigma_y = ps.sigma_side
    ex = (dx * dx) / (2.0 * sigma_x ** 2 + 1e-9)
    ey = (dy * dy) / (2.0 * sigma_y ** 2 + 1e-9)
    return float(math.exp(-(ex + ey)))


def _detect_corridor(
    lidar_dists: np.ndarray,
    lidar_angles: np.ndarray,
    robot_yaw: float,
    width_thresh: float = 1.5,
) -> bool:
    """Return True if LiDAR indicates the robot is inside a narrow corridor.

    Checks whether the perpendicular clearance (left + right side of the
    robot's heading) is below *width_thresh* on both sides simultaneously.
    """
    n = len(lidar_dists)
    if n == 0:
        return False

    half_pi = math.pi / 2.0
    left_min = float("inf")
    right_min = float("inf")

    for i in range(n):
        relative_angle = float(lidar_angles[i]) - robot_yaw
        # Normalise to (-pi, pi]
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle <= -math.pi:
            relative_angle += 2 * math.pi

        # Rays ≈ 90° left or right of heading → side clearance
        if abs(relative_angle - half_pi) < math.radians(20):
            left_min = min(left_min, float(lidar_dists[i]))
        elif abs(relative_angle + half_pi) < math.radians(20):
            right_min = min(right_min, float(lidar_dists[i]))

    return left_min < width_thresh and right_min < width_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Innovation 4: Context-Adaptive SIEP (CA-SIEP)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SocialContext:
    """Structured social context perceived from the robot's local environment.

    Computed once per planning step by ``SocialPerceptionLayer.perceive()``
    and consumed by ``context_modulate_weights()`` to produce context-adapted
    SIEP force weights.

    Attributes:
        crowd_density:  Pedestrians per square metre within *context_radius*.
        mean_speed:     Average pedestrian speed [m/s] within *context_radius*.
        max_facing_cos: Maximum cos(angle) between any pedestrian's heading
                        and the robot direction (1 = pedestrian facing robot,
                        -1 = pedestrian facing away).  Used for perspective
                        awareness: a pedestrian facing the robot has larger
                        effective personal space demand.
        n_groups:       Number of detected pedestrian groups.
        group_ids:      List of group ID for each pedestrian (-1 = ungrouped).
        in_corridor:    True when LiDAR indicates a narrow passage.
        local_peds:     Indices of pedestrians within *context_radius*.
    """
    crowd_density: float = 0.0
    mean_speed: float = 0.0
    max_facing_cos: float = -1.0
    n_groups: int = 0
    group_ids: List[int] = field(default_factory=list)
    in_corridor: bool = False
    local_peds: List[int] = field(default_factory=list)


class SocialPerceptionLayer:
    """Converts raw sensor observations into a structured ``SocialContext``.

    This is the *perception front-end* of CA-SIEP.  Rather than using raw
    pedestrian positions directly as force inputs, the planner first queries
    this layer to obtain semantic social features that modulate every force
    channel's weight.

    Args:
        context_radius:        Look-ahead radius for local context [m].
        corridor_width_thresh: LiDAR side clearance below which the robot is
                               considered to be in a narrow corridor [m].
        group_proximity:       Max inter-pedestrian distance for grouping [m].
        group_heading_thresh:  Min cosine of heading difference for grouping.
    """

    def __init__(
        self,
        context_radius: float = 4.0,
        corridor_width_thresh: float = 1.5,
        group_proximity: float = 2.0,
        group_heading_thresh: float = 0.5,
    ) -> None:
        self.context_radius = context_radius
        self.corridor_width_thresh = corridor_width_thresh
        self.group_proximity = group_proximity
        self.group_heading_thresh = group_heading_thresh

    def perceive(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
        peds: List[Tuple[np.ndarray, float, np.ndarray, PersonalSpace]],
        lidar_dists: np.ndarray,
        lidar_angles: np.ndarray,
    ) -> SocialContext:
        """Compute a ``SocialContext`` from current observations.

        Args:
            robot_xy:     Robot position (2,).
            robot_yaw:    Robot heading [rad].
            peds:         List of (xy, yaw, vel, ps) pedestrian tuples.
            lidar_dists:  Per-ray LiDAR distances (N,).
            lidar_angles: Per-ray world-frame angles (N,) [rad].

        Returns:
            Populated ``SocialContext`` instance.
        """
        ctx = SocialContext(group_ids=[-1] * len(peds))

        # ── 1. Local crowd density & speed ──────────────────────────────────
        local_indices = []
        speeds = []
        for i, (pxy, pyaw, pvel, _) in enumerate(peds):
            dist = float(np.linalg.norm(pxy - robot_xy))
            if dist <= self.context_radius:
                local_indices.append(i)
                speeds.append(float(np.linalg.norm(pvel)))
        ctx.local_peds = local_indices
        area = math.pi * self.context_radius ** 2
        ctx.crowd_density = len(local_indices) / max(area, 1e-6)
        ctx.mean_speed = float(np.mean(speeds)) if speeds else 0.0

        # ── 2. Perspective awareness (facing angle) ──────────────────────────
        # Compute for each local pedestrian the cosine of (ped_heading vs
        # direction toward robot).  A pedestrian facing the robot has a high
        # facing_cos, indicating stronger social space demand.
        max_face = -1.0
        for i in local_indices:
            pxy, pyaw, _, _ = peds[i]
            ped_dir = np.array([math.cos(pyaw), math.sin(pyaw)], dtype=float)
            to_robot = robot_xy - pxy
            dist = float(np.linalg.norm(to_robot))
            if dist < 1e-6:
                continue
            facing_cos = float(np.dot(ped_dir, to_robot / dist))
            max_face = max(max_face, facing_cos)
        ctx.max_facing_cos = max_face

        # ── 3. Group detection ───────────────────────────────────────────────
        group_ids, n_groups = _detect_groups(
            peds, self.group_proximity, self.group_heading_thresh
        )
        ctx.group_ids = group_ids
        ctx.n_groups = n_groups

        # ── 4. Corridor detection from LiDAR ────────────────────────────────
        ctx.in_corridor = _detect_corridor(
            lidar_dists, lidar_angles, robot_yaw, self.corridor_width_thresh
        )

        return ctx


def context_modulate_weights(
    ctx: SocialContext,
    params: SIEPParams,
) -> Tuple[float, float, float]:
    """Compute context-adaptive weight multipliers λ for key force channels.

    The modulation formula is:

        λ_ps   = clip(1 + gain_density·max(ρ-ρ₀,0) + gain_face·face_term, λ_min, λ_max)
        λ_goal = clip(1 / (1 + decay·max(ρ-ρ₀,0)), λ_min, 1.0)
        λ_align = clip(1 + corridor_boost·in_corridor, λ_min, λ_max)

    where face_term = max(0, max_facing_cos) captures perspective awareness
    (pedestrians facing the robot demand more personal-space deference).

    Args:
        ctx:    Perceived social context.
        params: Base SIEP parameters.

    Returns:
        (λ_ps, λ_goal, λ_align) – three scalar multipliers.
    """
    rho = ctx.crowd_density
    rho_excess = max(0.0, rho - params.density_threshold)

    # Personal-space modulation: grows with crowd density and facing angle
    face_term = max(0.0, ctx.max_facing_cos)
    lambda_ps = 1.0 + params.density_ps_gain * rho_excess + params.face_ps_gain * face_term
    lambda_ps = float(np.clip(lambda_ps, params.lambda_min, params.lambda_max))

    # Goal modulation: reduce speed in dense crowds
    lambda_goal = 1.0 / (1.0 + params.density_goal_decay * rho_excess)
    lambda_goal = float(np.clip(lambda_goal, params.lambda_min, 1.0))

    # Alignment modulation: boost in corridors (follow crowd flow more strongly)
    lambda_align = 1.0 + 0.5 * float(ctx.in_corridor)
    lambda_align = float(np.clip(lambda_align, params.lambda_min, params.lambda_max))

    return lambda_ps, lambda_goal, lambda_align


# ─────────────────────────────────────────────────────────────────────────────
# Innovation 5: Uncertainty-Modulated Personal Space (UMPS)
# ─────────────────────────────────────────────────────────────────────────────

def _pedestrian_uncertainty(
    ped_vel: np.ndarray,
    ped_dist: float,
    ps_horizon: float,
    max_uncertainty: float = 0.8,
) -> float:
    """Estimate epistemic uncertainty in a pedestrian's predicted trajectory.

    Uncertainty has two components:
    1. *Speed uncertainty*: faster pedestrians deviate more from the constant-
       velocity prediction over horizon *ps_horizon*.  We model this as
       speed × horizon × 0.15 (empirically tuned so a 1.2 m/s ped over 1.5 s
       gives U ≈ 0.27 – a 27 % personal-space expansion).
    2. *Distance uncertainty*: distant pedestrians are detected with lower
       reliability.  Modelled as a linear ramp starting at 2 m.

    Args:
        ped_vel:        Pedestrian velocity vector (2,).
        ped_dist:       Current robot-to-pedestrian distance [m].
        ps_horizon:     Personal-space prediction horizon [s].
        max_uncertainty: Cap on returned U.

    Returns:
        U in [0, max_uncertainty].
    """
    speed = float(np.linalg.norm(ped_vel))
    u_speed = speed * ps_horizon * _SPEED_UNCERTAINTY_COEFF
    u_dist = max(0.0, ped_dist - _DIST_UNCERTAINTY_THRESHOLD) * _DIST_UNCERTAINTY_RATE
    return float(min(u_speed + u_dist, max_uncertainty))


def uncertainty_modulated_ps_force(
    robot_xy: np.ndarray,
    ped_xy: np.ndarray,
    ped_yaw: float,
    ped_vel: np.ndarray,
    ps: PersonalSpace,
    k: float,
    horizon: float,
    dt: float = 0.1,
    uncertainty_gain: float = 1.2,
    max_uncertainty: float = 0.8,
) -> np.ndarray:
    """Personal-space repulsion with uncertainty-expanded Gaussian envelope.

    When the pedestrian's future trajectory is uncertain (high speed or large
    distance), σ_front/side/back are expanded proportionally:

        σ_eff = σ_base · (1 + κ · U)

    where U ∈ [0, max_uncertainty] and κ = *uncertainty_gain*.  This makes the
    robot give a wider berth to unpredictable pedestrians.

    Args:
        robot_xy:         Current robot position (2,).
        ped_xy:           Pedestrian position (2,).
        ped_yaw:          Pedestrian heading [rad].
        ped_vel:          Pedestrian velocity (2,).
        ps:               Nominal personal-space parameters.
        k:                Force scale.
        horizon:          Prediction look-ahead [s].
        dt:               Integration step [s].
        uncertainty_gain: κ in the expansion formula.
        max_uncertainty:  Upper cap on U.

    Returns:
        2-D repulsion force vector.
    """
    dist = float(np.linalg.norm(robot_xy - ped_xy))
    U = _pedestrian_uncertainty(ped_vel, dist, horizon, max_uncertainty)
    expand = 1.0 + uncertainty_gain * U
    ps_expanded = PersonalSpace(
        sigma_front=ps.sigma_front * expand,
        sigma_side=ps.sigma_side * (1.0 + _SIDE_UNCERTAINTY_FACTOR * uncertainty_gain * U),
        sigma_back=ps.sigma_back * (1.0 + _BACK_UNCERTAINTY_FACTOR * uncertainty_gain * U),
    )
    return personal_space_force(
        robot_xy, ped_xy, ped_yaw, ped_vel, ps_expanded, k, horizon, dt
    )


# ─────────────────────────────────────────────────────────────────────────────
# Innovation 6: Group-Level Social Force (GF)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_groups(
    peds: List[Tuple[np.ndarray, float, np.ndarray, PersonalSpace]],
    proximity_thresh: float = 2.0,
    heading_thresh: float = 0.5,
) -> Tuple[List[int], int]:
    """Detect pedestrian groups via proximity + heading-similarity clustering.

    Two pedestrians belong to the same group if:
    1. Their centre-to-centre distance is ≤ *proximity_thresh*.
    2. The cosine of the angle between their heading vectors is ≥ *heading_thresh*.

    Uses a simple union-find approach for transitive closure.

    Args:
        peds:              List of (xy, yaw, vel, ps) tuples.
        proximity_thresh:  Maximum inter-pedestrian distance [m].
        heading_thresh:    Minimum cosine of heading angle difference.

    Returns:
        (group_ids, n_groups) where group_ids[i] is the group label for
        pedestrian *i* (-1 means ungrouped, labels start at 0).
    """
    n = len(peds)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        pxy_i, pyaw_i, _, _ = peds[i]
        dir_i = np.array([math.cos(pyaw_i), math.sin(pyaw_i)])
        for j in range(i + 1, n):
            pxy_j, pyaw_j, _, _ = peds[j]
            dist = float(np.linalg.norm(pxy_i - pxy_j))
            if dist > proximity_thresh:
                continue
            dir_j = np.array([math.cos(pyaw_j), math.sin(pyaw_j)])
            if float(np.dot(dir_i, dir_j)) >= heading_thresh:
                union(i, j)

    # Assign compact group IDs; singletons get -1
    root_to_group: Dict[int, int] = {}
    group_ids = [-1] * n
    gid = 0
    for i in range(n):
        r = find(i)
        members = [j for j in range(n) if find(j) == r]
        if len(members) < 2:
            continue
        if r not in root_to_group:
            root_to_group[r] = gid
            gid += 1
        group_ids[i] = root_to_group[r]

    return group_ids, gid


def group_repulsion_force(
    robot_xy: np.ndarray,
    group_peds: List[Tuple[np.ndarray, float, np.ndarray, PersonalSpace]],
    k_group: float = 1.5,
    sigma_scale: float = 1.4,
) -> np.ndarray:
    """Extra repulsion force from a detected pedestrian group.

    A group is treated as a single extended agent located at the group
    centroid, with a personal space whose σ values are scaled by *sigma_scale*
    relative to the mean individual σ of the group members.  This captures
    the social norm of not cutting through a group of people.

    Args:
        robot_xy:    Current robot position (2,).
        group_peds:  List of (xy, yaw, vel, ps) for group members.
        k_group:     Force scale (additional to per-pedestrian forces).
        sigma_scale: σ expansion factor for the collective envelope.

    Returns:
        2-D repulsion force vector.
    """
    if len(group_peds) < 2:
        return np.zeros(2, dtype=float)

    # Group centroid position and mean heading/velocity
    centroid = np.mean([p[0] for p in group_peds], axis=0)
    mean_yaw = float(np.mean([p[1] for p in group_peds]))
    mean_vel = np.mean([p[2] for p in group_peds], axis=0)

    # Collective personal space: average individual σ scaled up
    mean_sf = float(np.mean([p[3].sigma_front for p in group_peds]))
    mean_ss = float(np.mean([p[3].sigma_side for p in group_peds]))
    mean_sb = float(np.mean([p[3].sigma_back for p in group_peds]))

    # Add an inter-member spread bonus so larger / more spread groups
    # generate proportionally bigger collective space.
    spread = float(np.max([np.linalg.norm(p[0] - centroid) for p in group_peds]))
    bonus = _GROUP_SPREAD_BONUS_FACTOR * spread

    group_ps = PersonalSpace(
        sigma_front=(mean_sf + bonus) * sigma_scale,
        sigma_side=(mean_ss + bonus * _GROUP_SPREAD_SIDE_SCALE) * sigma_scale,
        sigma_back=(mean_sb + bonus * _GROUP_SPREAD_BACK_SCALE) * sigma_scale,
    )

    return personal_space_force(
        robot_xy,
        centroid,
        mean_yaw,
        mean_vel,
        group_ps,
        k=k_group,
        horizon=0.8,
        dt=0.1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Innovation 7: Frontier Exploration Force (goal-free exploration)
# ─────────────────────────────────────────────────────────────────────────────

def frontier_exploration_force(
    robot_xy: np.ndarray,
    lidar_dists: np.ndarray,
    lidar_angles: np.ndarray,
    visited_positions: List[np.ndarray],
    max_range: float = 8.0,
    novelty_radius: float = 2.0,
    k_frontier: float = 1.0,
    random_weight: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Force toward open, unvisited regions of the environment.

    Unlike traditional frontier exploration (which requires a global
    occupancy grid), this force is computed *reactively* from the current
    LiDAR scan and the robot's visit history.  For each ray:

        vote = openness(ray) × novelty(direction)

    where:
        openness(ray)   = d_ray / max_range      (far = open)
        novelty(dir)    = 1 - exp(-D_min / r₀)  (D_min = distance to nearest
                                                  visited position in that
                                                  direction; r₀ = novelty_radius)

    The votes are summed as a weighted vector sum to produce the exploration
    force.  A small random component prevents the robot from getting stuck.

    This is the *SIEP formulation of frontier exploration*: no grid, no
    discrete frontier extraction – just a continuous, reactive force field.

    Args:
        robot_xy:          Current robot position (2,).
        lidar_dists:       Per-ray hit distances (N,).
        lidar_angles:      World-frame ray angles (N,) [rad].
        visited_positions: List of previously visited xy positions.
        max_range:         LiDAR maximum range [m].
        novelty_radius:    Spatial scale for novelty decay [m].
        k_frontier:        Overall force magnitude scale.
        random_weight:     Weight of the random exploration component.
        rng:               Optional numpy random Generator (reproducibility).

    Returns:
        2-D exploration force vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_rays = len(lidar_dists)
    if n_rays == 0:
        return np.zeros(2, dtype=float)

    # Openness: normalised ray distance
    openness = lidar_dists / max(max_range, 1e-6)
    openness = np.clip(openness, 0.0, 1.0)

    # Novelty: based on minimum distance from each ray's midpoint to visited cells
    if len(visited_positions) == 0:
        novelty = np.ones(n_rays, dtype=float)
    else:
        visited_arr = np.array(visited_positions, dtype=float)  # (M, 2)
        novelty = np.zeros(n_rays, dtype=float)
        for i, (d_ray, ang) in enumerate(zip(lidar_dists, lidar_angles)):
            probe_dist = min(float(d_ray) * 0.6, max_range * 0.6)
            probe_xy = robot_xy + probe_dist * np.array(
                [math.cos(float(ang)), math.sin(float(ang))], dtype=float
            )
            diffs = visited_arr - probe_xy[np.newaxis, :]
            min_dist = float(np.min(np.linalg.norm(diffs, axis=1)))
            novelty[i] = 1.0 - math.exp(-min_dist / max(novelty_radius, 1e-6))

    # Combined vote per ray: openness × novelty
    votes = openness * novelty  # (N,)
    ray_dirs = np.stack(
        [np.cos(lidar_angles.astype(float)), np.sin(lidar_angles.astype(float))], axis=1
    )  # (N, 2)
    F = np.einsum("i,ij->j", votes, ray_dirs)

    F_mag = float(np.linalg.norm(F))
    if F_mag > 1e-6:
        F = k_frontier * F / F_mag

    # Small random component to avoid local traps
    angle_rand = float(rng.uniform(0, 2 * math.pi))
    F_rand = np.array([math.cos(angle_rand), math.sin(angle_rand)], dtype=float)
    F = (1.0 - random_weight) * F + random_weight * F_rand

    return F
