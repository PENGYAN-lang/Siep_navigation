from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from social_nav3d.env.sim import SocialNavSim
from social_nav3d.planners.sampling_mpc import SamplingMPC
from social_nav3d.planners.siep_planner import SIEPPlanner
from social_nav3d.utils.config import load_config


def _build_scene_description(robot_pose, ped_state: dict, idx: int) -> str:
    """Format a structured pedestrian description string for the intent engine.

    Args:
        robot_pose: Current robot Pose2 object.
        ped_state:  Dict with keys 'xy', 'yaw', 'vel', 'radius'.
        idx:        Pedestrian index (for labelling).

    Returns:
        Single-line structured description suitable for LLMIntentEngine.
    """
    ped_xy = ped_state["xy"]
    dx = ped_xy[0] - robot_pose.x
    dy = ped_xy[1] - robot_pose.y
    dist = math.hypot(dx, dy)

    ped_speed = float(np.linalg.norm(ped_state["vel"]))
    ped_yaw = float(ped_state["yaw"])

    # Angle from pedestrian to robot
    angle_to_robot = math.atan2(robot_pose.y - ped_xy[1],
                                robot_pose.x - ped_xy[0])
    yaw_diff = abs(math.atan2(math.sin(ped_yaw - angle_to_robot),
                              math.cos(ped_yaw - angle_to_robot)))

    if yaw_diff < math.radians(45):
        facing = "toward_robot"
    elif yaw_diff > math.radians(135):
        facing = "away_from_robot"
    else:
        facing = "sideways"

    label = chr(ord("A") + (idx % 26))
    return (
        f"Pedestrian {label}: pos=({ped_xy[0]:.1f}, {ped_xy[1]:.1f}), "
        f"facing={facing}, speed={ped_speed:.2f}, dist={dist:.1f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="social_nav3d/configs/default.yaml")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument(
        "--planner",
        type=str,
        default="mpc",
        choices=["mpc", "siep"],
        help=(
            "Planning algorithm: "
            '"mpc" = sampling-based MPC (default), '
            '"siep" = Stimuli-Induced Equilibrium Point planner.'
        ),
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="goal",
        choices=["goal", "explore"],
        help=(
            "Navigation mode: "
            '"goal" = navigate to fixed goal (default), '
            '"explore" = free frontier exploration (requires --planner siep).'
        ),
    )
    ap.add_argument(
        "--intent-engine",
        type=str,
        default="rule",
        choices=["rule", "llm"],
        dest="intent_engine",
        help=(
            "Intent inference backend: "
            '"rule" = fast rule-based heuristics (default), '
            '"llm"  = local LLM via transformers (requires GPU + model).'
        ),
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # Validate mode / planner combination                                  #
    # ------------------------------------------------------------------ #
    if args.mode == "explore" and args.planner != "siep":
        print("[run_demo] WARNING: --mode explore requires --planner siep. "
              "Switching to SIEP planner automatically.")
        args.planner = "siep"

    cfg = load_config(args.config)
    if args.gui:
        cfg.setdefault("sim", {})
        cfg["sim"]["gui"] = True
    if args.record:
        cfg.setdefault("sim", {})
        cfg["sim"]["record_video"] = True

    sim = SocialNavSim(cfg)
    sim.reset()

    # ------------------------------------------------------------------ #
    # Intent engine                                                         #
    # ------------------------------------------------------------------ #
    intent_engine = None
    if args.planner == "siep":
        from social_nav3d.utils.llm_intent import LLMIntentEngine  # noqa: PLC0415
        use_local = args.intent_engine == "llm"
        intent_engine = LLMIntentEngine(use_local=use_local)
        print(f"[run_demo] Intent engine: {intent_engine.backend}")

    # ------------------------------------------------------------------ #
    # Planner selection                                                    #
    # ------------------------------------------------------------------ #
    use_siep = args.planner == "siep"
    if use_siep:
        exploration_mode = args.mode == "explore"
        planner = SIEPPlanner(cfg, exploration_mode=exploration_mode)
        mode_label = "explore" if exploration_mode else "goal"
        print(f"[run_demo] Using SIEP planner — mode={mode_label}.")
    else:
        planner = SamplingMPC(cfg)
        print("[run_demo] Using Sampling-MPC planner.")

    # ------------------------------------------------------------------ #
    # Intent inference cadence: call LLM/rule engine every N steps         #
    # ------------------------------------------------------------------ #
    INTENT_INTERVAL = 5  # steps between intent updates
    intent_results: list = []
    intent_step_counter = 0

    traj = []
    max_steps = args.steps or sim.max_steps

    for k in range(max_steps):
        state = sim.get_state()
        ped_states = sim.get_pedestrians_state()
        peds = [(d["xy"], d["yaw"], d["vel"], d["ps"]) for d in ped_states]

        if use_siep:
            lidar_dists, lidar_angles = sim.get_lidar_directional_2d()

            # Refresh intent results periodically
            if intent_engine is not None and intent_step_counter % INTENT_INTERVAL == 0:
                intent_results = []
                for idx, ped_state in enumerate(ped_states):
                    desc = _build_scene_description(state, ped_state, idx)
                    intent_results.append(intent_engine.infer_intent(desc))

            intent_step_counter += 1

            v, w = planner.plan(
                state, sim.goal_xy, sim.dt,
                lidar_dists, lidar_angles, peds,
                intent_results=intent_results if intent_results else None,
            )
        else:
            lidar_min = sim.lidar_min_distance()
            v, w = planner.plan(state, sim.goal_xy, sim.dt, lidar_min, peds)

        sim.step(v, w)
        traj.append([state.x, state.y])

        # ── Termination / status ──────────────────────────────────────
        if args.mode == "goal" and sim.reached_goal():
            print(f"[run_demo] Goal reached at step {k}.")
            break

        if k % 50 == 0 and k > 0:
            status_parts = [f"step={k}", f"pos=({state.x:.1f},{state.y:.1f})"]

            if use_siep and args.mode == "explore":
                sx, sy = cfg["world"]["size_xy"]
                coverage = planner.exploration_coverage(
                    (0.0, 0.0, float(sx), float(sy))
                )
                status_parts.append(f"coverage={coverage * 100:.1f}%")

            if intent_results and ped_states:
                nearest_idx = int(
                    np.argmin([
                        np.linalg.norm(d["xy"] - state.xy())
                        for d in ped_states
                    ])
                )
                nearest_intent = intent_results[nearest_idx]["intent"] if intent_results else "n/a"
                status_parts.append(f"nearest_intent={nearest_intent}")

            print("[run_demo] " + "  ".join(status_parts))

    # ------------------------------------------------------------------ #
    # Save trajectory plot                                                  #
    # ------------------------------------------------------------------ #
    out_dir = Path(cfg["sim"].get("out_dir", "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = np.asarray(traj)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sim.plot_topdown(ax)
    ax.plot(traj[:, 0], traj[:, 1], linewidth=1.2, label="robot path")
    ax.scatter([traj[0, 0]], [traj[0, 1]], marker="o", zorder=5, label="start")
    if args.mode == "goal":
        ax.scatter([sim.goal_xy[0]], [sim.goal_xy[1]], marker="*", s=120,
                   zorder=5, label="goal")
    planner_label = "SIEP" if use_siep else "Sampling-MPC"
    mode_str = args.mode
    ax.set_title(f"SocialNav3D – {planner_label} ({mode_str}): trajectory")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "trajectory.png"
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")

    if cfg["sim"].get("record_video", False):
        print(f"Saved: {out_dir / 'run.mp4'}")

    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
