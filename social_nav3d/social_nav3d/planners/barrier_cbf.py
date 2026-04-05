"""barrier_cbf.py — Control Barrier Function safety layer for SIEP planner.

Solves a 1-D QP projection at each control step to ensure the robot stays
inside the safe set defined by minimum clearance to obstacles and pedestrians.

Paper reference: Barrier-SIEP-MPC safety layer (Section IV.C).
"""
from __future__ import annotations

import math

import numpy as np

from ..utils.geometry import Pose2


class BarrierCBF:
    """
    Control Barrier Function for SIEP safety projection.

    Barrier function: h(x) = d_min(x) - (r_robot + r_safe)
    where d_min(x) is the minimum distance to any obstacle or human.

    Safety constraint: ḣ(x, u) + α · h(x) ≥ 0

    This is solved as a QP:
        min_u  ||u - u_nom||²
        s.t.   ḣ(x, u) + α · h(x) ≥ 0
               |v| ≤ v_max
               |ω| ≤ ω_max
               |v - v_prev| ≤ Δv_max  (jerk limit)

    The QP projects the nominal SIEP output onto the safe set.

    For the obstacle CBF:
      h_obs(x) = d_nearest_obstacle - (r_robot + r_safe)
      ḣ_obs(x, u) = (x_robot - x_nearest)^T · v_robot / ||x_robot - x_nearest||
      where v_robot = [v·cos(θ), v·sin(θ)]

    For the human personal-space CBF (optional, use_cbf_humans=True):
      h_human_j(x) = ||x_robot - x_ped_j|| - (r_robot + σ_front_j)
      ḣ_human_j(x, u) = (x_robot - x_ped_j)^T · v_robot / ||x_robot - x_ped_j||

    The QP is solved analytically:
      - If all h(x) > margin and constraint is not violated: return u_nom
      - If constraint violated: project u along the constraint gradient via
        the closed-form solution of the 1D QP:
        v* = v_nom - max(0, -(ḣ(x, v_nom) + α·h(x))) * cos(θ_nearest) / ||∂h/∂v||²

    Paper: Barrier-SIEP-MPC safety layer (Section IV.C).
    """

    def __init__(self, cfg: dict, use_cbf_humans: bool = True) -> None:
        """Initialise CBF safety layer from config.

        Args:
            cfg: Config dict with keys ``robot`` (max_v, max_w, radius) and
                 ``planner`` (min_clearance, cbf_alpha, cbf_jerk_limit).
            use_cbf_humans: When *True* (default) pedestrian personal-space
                barriers are included in the constraint set.  Set *False* for
                ablation experiments that test obstacles-only safety.
        """
        rcfg = cfg["robot"]
        self.max_v: float = float(rcfg["max_v"])
        self.max_w: float = float(rcfg["max_w"])
        self.r_robot: float = float(rcfg["radius"])

        pcfg = cfg.get("planner", {})
        self.r_safe: float = float(pcfg.get("min_clearance", 0.35))
        self._alpha: float = float(pcfg.get("cbf_alpha", 1.0))
        self._jerk_limit: float = float(pcfg.get("cbf_jerk_limit", 0.3))

        self._use_cbf_humans: bool = use_cbf_humans
        self._prev_v: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        pose: Pose2,
        v_nom: float,
        w_nom: float,
        lidar_dists: np.ndarray,
        lidar_angles: np.ndarray,
        ped_states: list[dict] | None = None,
    ) -> tuple[float, float]:
        """Project nominal control onto the CBF-safe set.

        Implements Eq. (IV.C) from the paper: finds the minimum-norm
        correction Δu to (v_nom, w_nom) that satisfies all active barriers.

        Algorithm:
          1. Compute h_obs = min_lidar_dist − (r_robot + r_safe).
          2. Evaluate ḣ_obs(u_nom) analytically from the angle to the
             nearest obstacle (lidar frame).
          3. Optionally compute h_human_j and ḣ_human_j for every pedestrian.
          4. Identify the most-violated constraint (smallest ḣ + α·h).
          5. If no constraint violated → return u_nom (after jerk + clamp).
          6. If violated → project v along constraint gradient (closed-form
             1-D QP, see :meth:`_project_v`).
          7. Apply jerk limit |Δv| ≤ Δv_max.
          8. Clamp to [−v_max, v_max] × [−ω_max, ω_max].

        Args:
            pose: Current 2-D robot pose (x, y, yaw).
            v_nom: Nominal linear velocity [m/s] from SIEP.
            w_nom: Nominal angular velocity [rad/s] from SIEP.
            lidar_dists: Array of range measurements [m] (robot frame).
            lidar_angles: Array of beam angles [rad] (robot frame, same
                length as *lidar_dists*).
            ped_states: Optional list of pedestrian state dicts, each with:
                ``xy`` (array-like [x, y]) and ``ps`` (PersonalSpace with
                ``sigma_front`` attribute).

        Returns:
            (v_safe, w_safe): Velocity pair guaranteed to satisfy all active
            CBF constraints up to jerk and velocity bounds.
        """
        constraints: list[tuple[float, float]] = []

        # ---- obstacle barrier (Eq. IV.C-1) ----------------------------
        cbf_obs, g_obs = self._obstacle_constraint(v_nom, lidar_dists, lidar_angles)
        constraints.append((cbf_obs, g_obs))

        # ---- human personal-space barriers (Eq. IV.C-2) ---------------
        if self._use_cbf_humans and ped_states:
            for ped in ped_states:
                result = self._human_constraint(pose, v_nom, ped)
                if result is not None:
                    constraints.append(result)

        # ---- find most violated constraint ----------------------------
        cbf_val, g_v = min(constraints, key=lambda c: c[0])

        # ---- project if necessary -------------------------------------
        v_safe = self._project_v(v_nom, cbf_val, g_v)

        # ---- jerk limit -----------------------------------------------
        v_safe = float(np.clip(v_safe, self._prev_v - self._jerk_limit,
                                self._prev_v + self._jerk_limit))

        # ---- velocity bounds ------------------------------------------
        v_safe = float(np.clip(v_safe, -self.max_v, self.max_v))
        w_safe = float(np.clip(w_nom, -self.max_w, self.max_w))

        self._prev_v = v_safe
        return v_safe, w_safe

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _obstacle_constraint(
        self,
        v_nom: float,
        lidar_dists: np.ndarray,
        lidar_angles: np.ndarray,
    ) -> tuple[float, float]:
        """Compute CBF value and ∂ḣ/∂v for the nearest obstacle.

        Barrier (Eq. IV.C-1):
            h_obs = d_min − (r_robot + r_safe)
            ḣ_obs = ∂h/∂v · v  where  ∂h/∂v = −cos(θ_nearest)

        The lidar angle θ_nearest is measured in the robot frame, so the
        component of v_robot along the obstacle direction is v·cos(θ_nearest).
        The sign of ḣ_obs is negative when moving toward the obstacle.

        Args:
            v_nom: Nominal linear velocity.
            lidar_dists: Range array [m].
            lidar_angles: Angle array [rad], robot frame.

        Returns:
            (cbf_value, gradient):  cbf_value = ḣ_obs(v_nom) + α·h_obs,
                                    gradient  = ∂ḣ_obs/∂v.
        """
        idx_min: int = int(np.argmin(lidar_dists))
        d_min: float = float(lidar_dists[idx_min])
        theta_nearest: float = float(lidar_angles[idx_min])

        h_obs: float = d_min - (self.r_robot + self.r_safe)
        # ∂ḣ_obs/∂v = −cos(θ_nearest)  (robot-frame decomposition)
        g_v: float = -math.cos(theta_nearest)
        cbf_val: float = g_v * v_nom + self._alpha * h_obs
        return cbf_val, g_v

    def _human_constraint(
        self,
        pose: Pose2,
        v_nom: float,
        ped: dict,
    ) -> tuple[float, float] | None:
        """Compute CBF value and ∂ḣ/∂v for one pedestrian.

        Barrier (Eq. IV.C-2):
            h_j = ||p_robot − p_j|| − (r_robot + σ_front_j)
            ḣ_j = n_j · v_robot,   n_j = (p_robot − p_j) / ||p_robot − p_j||
                                  v_robot = v·[cos θ, sin θ]
            ∂ḣ_j/∂v = n_j · [cos θ, sin θ]

        Args:
            pose: Current robot pose.
            v_nom: Nominal linear velocity.
            ped: Dict with ``xy`` (pedestrian position) and ``ps``
                 (PersonalSpace with ``sigma_front``).

        Returns:
            (cbf_value, gradient) or *None* if pedestrian is too close to
            compute a valid unit vector (degenerate case).
        """
        xy_robot = np.array([pose.x, pose.y], dtype=float)
        xy_ped = np.asarray(ped["xy"], dtype=float)

        diff = xy_robot - xy_ped
        dist: float = float(np.linalg.norm(diff))
        if dist < 1e-6:
            return None  # degenerate: skip

        sigma_front: float = float(ped["ps"].sigma_front)
        h_human: float = dist - (self.r_robot + sigma_front)

        # n_j: unit vector from pedestrian toward robot
        n_j = diff / dist
        heading = np.array([math.cos(pose.yaw), math.sin(pose.yaw)], dtype=float)
        g_v: float = float(np.dot(n_j, heading))  # ∂ḣ_j/∂v

        cbf_val: float = g_v * v_nom + self._alpha * h_human
        return cbf_val, g_v

    @staticmethod
    def _project_v(v_nom: float, cbf_val: float, g_v: float) -> float:
        """Closed-form 1-D QP projection (Eq. IV.C, projection step).

        Solves:
            min_v  (v − v_nom)²
            s.t.   g_v · v + β ≥ 0

        where β = α·h is already absorbed into *cbf_val* = g_v·v_nom + β.

        The KKT solution is:
            v* = v_nom + max(0, −cbf_val) · g_v / g_v²
               = v_nom − violation · (−g_v) / g_v²       (sign bookkeeping)

        Equivalently (matching docstring Eq.):
            v* = v_nom − max(0, −cbf_val) · (−g_v) / g_v²

        When g_v² is near zero the constraint is v-independent; no projection
        is needed and v_nom is returned unchanged.

        Args:
            v_nom: Nominal velocity.
            cbf_val: Current CBF condition value ḣ(v_nom) + α·h.
            g_v: Gradient ∂ḣ/∂v.

        Returns:
            Projected velocity v*.
        """
        if cbf_val >= 0.0:
            return v_nom  # constraint satisfied — no correction needed

        g_v_sq: float = g_v * g_v
        if g_v_sq < 1e-12:
            return v_nom  # gradient vanishes — cannot project

        violation: float = -cbf_val  # > 0
        # v* = v_nom + violation * g_v / g_v²  (KKT projection)
        return v_nom + violation * g_v / g_v_sq


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def test_barrier_cbf() -> None:  # noqa: WPS430
        """Basic correctness checks for :class:`BarrierCBF`.

        Scenario A — obstacle directly in front at 0.5 m (inside safe zone):
            Nominal v=1.0 m/s should be reduced to satisfy the CBF constraint.

        Scenario B — obstacle far away (5.0 m, well outside safe zone):
            Nominal v=1.0 m/s should pass through unchanged.

        Scenario C — obstacle at 90° (perpendicular, no heading component):
            Forward velocity is unaffected (gradient ≈ 0).

        Scenario D — jerk limit:
            Large velocity step is clamped to Δv_max.
        """
        cfg: dict = {
            "robot": {"max_v": 1.2, "max_w": 1.8, "radius": 0.25},
            "planner": {
                "min_clearance": 0.35,
                "cbf_alpha": 1.0,
                "cbf_jerk_limit": 0.3,
            },
        }

        cbf = BarrierCBF(cfg, use_cbf_humans=True)
        pose = Pose2(x=0.0, y=0.0, yaw=0.0)

        # --- Scenario A: obstacle 0.5 m directly ahead -----------------
        angles_a = np.array([0.0])        # directly in front
        dists_a = np.array([0.50])        # 0.5 m away
        v_safe, w_safe = cbf.project(pose, v_nom=1.0, w_nom=0.0,
                                     lidar_dists=dists_a, lidar_angles=angles_a)
        print(f"[A] v_safe={v_safe:.4f}  (expect < 1.0, obstacle in front)")
        assert v_safe < 1.0, f"Expected v_safe < 1.0, got {v_safe}"

        # --- Scenario B: obstacle 5.0 m ahead — well clear -------------
        cbf._prev_v = 0.0  # reset jerk state
        angles_b = np.array([0.0])
        dists_b = np.array([5.0])
        v_safe_b, _ = cbf.project(pose, v_nom=1.0, w_nom=0.0,
                                   lidar_dists=dists_b, lidar_angles=angles_b)
        print(f"[B] v_safe={v_safe_b:.4f}  (expect 1.0 or jerk-limited, far obstacle)")
        # Allow for jerk clamp from prev_v=0
        assert v_safe_b > 0.0, "Expected positive velocity for clear path"

        # --- Scenario C: obstacle at 90° (perpendicular) ---------------
        cbf._prev_v = 1.0  # pre-warm so jerk doesn't bite
        angles_c = np.array([math.pi / 2])   # 90° to the side
        dists_c = np.array([0.30])            # very close, but to the side
        v_safe_c, _ = cbf.project(pose, v_nom=1.0, w_nom=0.0,
                                   lidar_dists=dists_c, lidar_angles=angles_c)
        print(f"[C] v_safe={v_safe_c:.4f}  (expect ~1.0, obstacle perpendicular)")
        # cos(90°) ≈ 0 → gradient ≈ 0 → no projection applied
        assert abs(v_safe_c - 1.0) < 1e-3, f"Expected v≈1.0 for side obstacle, got {v_safe_c}"

        # --- Scenario D: jerk limit test --------------------------------
        cbf._prev_v = 0.0
        angles_d = np.array([0.0])
        dists_d = np.array([10.0])            # far away, safety not active
        v_safe_d, _ = cbf.project(pose, v_nom=1.0, w_nom=0.0,
                                   lidar_dists=dists_d, lidar_angles=angles_d)
        print(f"[D] v_safe={v_safe_d:.4f}  (expect ≤ Δv_max=0.3, jerk limit)")
        assert v_safe_d <= 0.3 + 1e-9, f"Jerk limit violated: got {v_safe_d}"

        # --- Scenario E: human personal-space barrier -------------------
        cbf2 = BarrierCBF(cfg, use_cbf_humans=True)
        cbf2._prev_v = 1.0  # avoid jerk

        class _FakePS:
            sigma_front = 1.0  # wide personal space

        ped_near = {"xy": np.array([1.2, 0.0]), "ps": _FakePS()}
        angles_e = np.array([math.pi])    # obstacle behind — not in heading
        dists_e = np.array([10.0])
        v_safe_e, _ = cbf2.project(
            pose, v_nom=1.0, w_nom=0.0,
            lidar_dists=dists_e, lidar_angles=angles_e,
            ped_states=[ped_near],
        )
        print(f"[E] v_safe={v_safe_e:.4f}  (expect < 1.0, pedestrian 1.2 m ahead)")
        assert v_safe_e < 1.0, f"Expected reduced speed near pedestrian, got {v_safe_e}"

        print("\nAll BarrierCBF tests passed ✓")

    test_barrier_cbf()
