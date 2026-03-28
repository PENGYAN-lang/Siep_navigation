from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from social_nav3d.env.sim import SocialNavSim
from social_nav3d.planners.sampling_mpc import SamplingMPC
from social_nav3d.planners.siep_planner import SIEPPlanner
from social_nav3d.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='social_nav3d/configs/default.yaml')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument(
        '--planner',
        type=str,
        default='mpc',
        choices=['mpc', 'siep'],
        help=(
            'Planning algorithm: '
            '"mpc" = sampling-based MPC (default), '
            '"siep" = Stimuli-Induced Equilibrium Point planner.'
        ),
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
        planner = SIEPPlanner(cfg)
        print('[run_demo] Using SIEP planner (Stimuli-Induced Equilibrium Point).')
    else:
        planner = SamplingMPC(cfg)
        print('[run_demo] Using Sampling-MPC planner.')

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
        if sim.reached_goal():
            print(f'[run_demo] Goal reached at step {k}.')
            break

    out_dir = Path(cfg['sim'].get('out_dir', 'runs'))
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = np.asarray(traj)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    sim.plot_topdown(ax)
    ax.plot(traj[:, 0], traj[:, 1])
    ax.scatter([traj[0, 0]], [traj[0, 1]], marker='o')
    ax.scatter([sim.goal_xy[0]], [sim.goal_xy[1]], marker='*')
    planner_label = 'SIEP' if use_siep else 'Sampling-MPC'
    ax.set_title(f'SocialNav3D \u2013 {planner_label}: trajectory')
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
