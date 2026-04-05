from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from social_nav3d.env.sim import SocialNavSim
from social_nav3d.planners.sampling_mpc import SamplingMPC
from social_nav3d.planners.siep_planner import SIEPPlanner
from social_nav3d.utils.config import load_config
from social_nav3d.utils.evaluation import EvaluationTracker

# Ablation variant → (use_context_adapt, use_uncertainty_ps, use_group_force)
_ABLATION_FLAGS: dict = {
    "base":       (False, False, False),
    "ca":         (True,  False, False),   # + context-adaptive weights
    "uncertainty":(False, True,  False),   # + uncertainty-modulated PS
    "group":      (False, False, True),    # + group-level force
    "full":       (True,  True,  True),    # all innovations enabled
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SIEP social navigation demo with ablation support."
    )
    ap.add_argument('--config', type=str, default='social_nav3d/configs/default.yaml')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument(
        '--planner',
        type=str,
        default='siep',
        choices=['mpc', 'siep'],
        help=(
            'Planning algorithm: '
            '"mpc" = sampling-based MPC, '
            '"siep" = Stimuli-Induced Equilibrium Point planner (default).'
        ),
    )
    ap.add_argument(
        '--mode',
        type=str,
        default='goal',
        choices=['goal', 'explore'],
        help=(
            'Navigation mode: '
            '"goal" (default) drives to the config goal position; '
            '"explore" uses frontier exploration force for goal-free roaming.'
        ),
    )
    ap.add_argument(
        '--siep-variant',
        type=str,
        default='base',
        choices=list(_ABLATION_FLAGS.keys()),
        help=(
            'SIEP innovation variant for ablation studies: '
            '"base" = fixed weights (original); '
            '"ca" = +context-adaptive weights (CA-SIEP); '
            '"uncertainty" = +uncertainty-modulated personal space (UMPS); '
            '"group" = +group-level social force (GF); '
            '"full" = all innovations enabled.'
        ),
    )
    ap.add_argument(
        '--eval',
        action='store_true',
        help='Print evaluation summary (PS violations, efficiency) at end of run.',
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.gui:
        cfg.setdefault('sim', {})
        cfg['sim']['gui'] = True
    if args.record:
        cfg.setdefault('sim', {})
        cfg['sim']['record_video'] = True

    sim = SocialNavSim(cfg)
    sim.reset()

    # ------------------------------------------------------------------ #
    # Planner selection                                                    #
    # ------------------------------------------------------------------ #
    use_siep = args.planner == 'siep'
    if use_siep:
        ca, ump, gf = _ABLATION_FLAGS[args.siep_variant]
        planner = SIEPPlanner(
            cfg,
            mode=args.mode,
            use_context_adapt=ca,
            use_uncertainty_ps=ump,
            use_group_force=gf,
        )
        variant_tag = args.siep_variant
        print(
            f'[run_demo] SIEP planner | mode={args.mode} | variant={variant_tag}'
            f' | CA={ca} UMP={ump} GF={gf}'
        )
    else:
        planner = SamplingMPC(cfg)
        print('[run_demo] Using Sampling-MPC planner.')

    tracker = EvaluationTracker() if args.eval else None

    traj = []
    for k in range(args.steps or sim.max_steps):
        state = sim.get_state()
        ped_states = sim.get_pedestrians_state()
        peds = [(d['xy'], d['yaw'], d['vel'], d['ps']) for d in ped_states]

        if use_siep:
            lidar_dists, lidar_angles = sim.get_lidar_directional_2d()
            v, w = planner.plan(state, sim.goal_xy, sim.dt,
                                lidar_dists, lidar_angles, peds)
        else:
            lidar_min = sim.lidar_min_distance()
            v, w = planner.plan(state, sim.goal_xy, sim.dt, lidar_min, peds)

        sim.step(v, w)
        traj.append([state.x, state.y])

        if tracker is not None:
            tracker.update(state.xy(), peds)

        # In goal mode stop when goal is reached; explore mode runs to max_steps
        if args.mode == 'goal' and sim.reached_goal():
            print(f'[run_demo] Goal reached at step {k}.')
            break

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #
    out_dir = Path(cfg['sim'].get('out_dir', 'runs'))
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = np.asarray(traj)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    sim.plot_topdown(ax)
    ax.plot(traj[:, 0], traj[:, 1], lw=1.5)
    ax.scatter([traj[0, 0]], [traj[0, 1]], marker='o', zorder=5)

    if args.mode == 'goal':
        ax.scatter([sim.goal_xy[0]], [sim.goal_xy[1]], marker='*', zorder=5)

    if use_siep and args.mode == 'explore' and hasattr(planner, 'visited_positions'):
        visited = planner.visited_positions
        if visited:
            vis_arr = np.array(visited)
            ax.scatter(vis_arr[:, 0], vis_arr[:, 1],
                       s=10, alpha=0.3, color='green', label='visited')

    planner_label = (
        f'SIEP ({args.siep_variant}, {args.mode})' if use_siep else 'Sampling-MPC'
    )
    ax.set_title(f'SocialNav3D – {planner_label}: trajectory')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    out_path = out_dir / 'trajectory.png'
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')

    if cfg['sim'].get('record_video', False):
        print(f"Saved: {out_dir / 'run.mp4'}")

    if tracker is not None:
        variant = f'SIEP-{args.siep_variant}-{args.mode}' if use_siep else 'MPC'
        tracker.print_summary(label=variant)

    sim.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
