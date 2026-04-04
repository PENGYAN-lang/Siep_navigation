from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from social_nav3d.env.sim import SocialNavSim
from social_nav3d.planners.sampling_mpc import SamplingMPC
from social_nav3d.planners.siep_planner import SIEPPlanner
from social_nav3d.planners.proactive_siep import ProactiveSIEP
from social_nav3d.utils.config import load_config
from social_nav3d.experiments.metrics import (
    EpisodeMetrics,
    record_step,
    compute_metrics,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Proactive-SIEP social navigation demo and evaluation runner.'
    )
    ap.add_argument(
        '--config', type=str,
        default='social_nav3d/configs/default.yaml',
        help='Path to YAML config file.',
    )
    ap.add_argument('--gui', action='store_true',
                    help='Enable PyBullet GUI.')
    ap.add_argument('--record', action='store_true',
                    help='Record video.')
    ap.add_argument('--steps', type=int, default=None,
                    help='Override max_steps.')
    ap.add_argument(
        '--planner',
        type=str,
        default='siep',
        choices=['mpc', 'siep', 'proactive'],
        help=(
            'Planning algorithm: '
            '"mpc" = sampling-based MPC, '
            '"siep" = classic SIEP planner, '
            '"proactive" = Proactive-SIEP (full method).'
        ),
    )
    ap.add_argument(
        '--siep-variant', type=str, default='full',
        choices=['base', 'context', 'uncertainty', 'proactive', 'constrained', 'full'],
        help=(
            'Proactive-SIEP ablation variant (only used when --planner proactive): '
            '"base" disables all innovations; "full" enables all.'
        ),
    )
    ap.add_argument(
        '--mode', type=str, default=None,
        choices=['goal', 'explore'],
        help='Navigation mode override: "goal" (A→B) or "explore" (free roaming).',
    )
    ap.add_argument(
        '--eval', action='store_true',
        help='Compute and print paper metrics at the end of the run.',
    )
    ap.add_argument('--seed', type=int, default=None,
                    help='Override config random seed.')
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.gui:
        cfg.setdefault('sim', {})
        cfg['sim']['gui'] = True
    if args.record:
        cfg.setdefault('sim', {})
        cfg['sim']['record_video'] = True
    if args.mode is not None:
        cfg.setdefault('sim', {})
        cfg['sim']['mode'] = args.mode
    if args.seed is not None:
        cfg['seed'] = args.seed

    sim = SocialNavSim(cfg)
    sim.reset()

    # ------------------------------------------------------------------ #
    # Planner selection                                                    #
    # ------------------------------------------------------------------ #
    mode = cfg['sim'].get('mode', 'goal')
    explore_mode = (mode == 'explore')

    if args.planner == 'proactive':
        from social_nav3d.experiments.runner import ABLATION_FLAGS
        flags = ABLATION_FLAGS.get(args.siep_variant, ABLATION_FLAGS['full'])
        planner = ProactiveSIEP(cfg, explore_mode=explore_mode, **flags)
        print(f'[run_demo] Using Proactive-SIEP planner (variant={args.siep_variant}, '
              f'mode={mode}).')
    elif args.planner == 'siep':
        planner = SIEPPlanner(cfg)
        print(f'[run_demo] Using classic SIEP planner (mode={mode}).')
    else:
        planner = SamplingMPC(cfg)
        print(f'[run_demo] Using Sampling-MPC planner (mode={mode}).')

    # ------------------------------------------------------------------ #
    # Metrics accumulator (for --eval)                                    #
    # ------------------------------------------------------------------ #
    ep_metrics = EpisodeMetrics(dt=sim.dt,
                                world_size_xy=tuple(cfg['world']['size_xy']))

    traj = []
    goal_xy = None if explore_mode else sim.goal_xy

    for k in range(args.steps or sim.max_steps):
        state = sim.get_state()
        ped_states = sim.get_pedestrians_state()
        peds = [(d['xy'], d['yaw'], d['vel'], d['ps']) for d in ped_states]

        if args.planner in ('siep', 'proactive'):
            lidar_dists, lidar_angles = sim.get_lidar_directional_2d()
            if args.planner == 'proactive':
                v, w = planner.plan(
                    state, goal_xy, sim.dt,
                    lidar_dists, lidar_angles, ped_states
                )
            else:
                plan_goal = goal_xy if goal_xy is not None else state.xy()
                v, w = planner.plan(
                    state, plan_goal, sim.dt,
                    lidar_dists, lidar_angles, peds
                )
        else:
            lidar_min = sim.lidar_min_distance()
            plan_goal = goal_xy if goal_xy is not None else state.xy()
            v, w = planner.plan(state, plan_goal, sim.dt, lidar_min, peds)

        sim.step(v, w)
        traj.append([state.x, state.y])

        # Collect metrics
        if args.eval:
            ped_xys = [d['xy'] for d in ped_states]
            ped_sigmas = [
                d['ps'].sigma_front if hasattr(d.get('ps'), 'sigma_front') else 1.6
                for d in ped_states
            ]
            record_step(ep_metrics, state.xy(), state.yaw, v, w,
                        ped_xys, ped_sigmas)

        if not explore_mode and sim.reached_goal():
            print(f'[run_demo] Goal reached at step {k}.')
            ep_metrics.reached_goal = True
            break

    # ------------------------------------------------------------------ #
    # Print metrics                                                        #
    # ------------------------------------------------------------------ #
    if args.eval:
        metrics = compute_metrics(ep_metrics, goal_xy=goal_xy)
        print('\n=== Episode Metrics ===')
        for name, val in sorted(metrics.items()):
            print(f'  {name:35s}: {val:.4f}')

    # ------------------------------------------------------------------ #
    # Save trajectory plot                                                 #
    # ------------------------------------------------------------------ #
    out_dir = Path(cfg['sim'].get('out_dir', 'runs'))
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = np.asarray(traj)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    sim.plot_topdown(ax)
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5)
        ax.scatter([traj[0, 0]], [traj[0, 1]], marker='o', label='start', zorder=5)
    if not explore_mode and goal_xy is not None:
        ax.scatter([sim.goal_xy[0]], [sim.goal_xy[1]], marker='*',
                   s=120, label='goal', zorder=5)

    planner_label = args.planner.upper()
    variant_label = (f' [{args.siep_variant}]' if args.planner == 'proactive' else '')
    ax.set_title(f'Proactive-SIEP Demo – {planner_label}{variant_label} ({mode})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / 'trajectory.png'
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')

    if cfg['sim'].get('record_video', False):
        print(f"Saved: {out_dir / 'run.mp4'}")

    sim.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

