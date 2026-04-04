"""
Ablation experiment runner for Proactive-SIEP.

Usage (CLI)
-----------
  python -m social_nav3d.experiments.runner \\
      --config social_nav3d/configs/paper/proactive_siep_full.yaml \\
      --ablation all \\
      --episodes 5 \\
      --out-dir runs/paper

Ablations
---------
  base       – base SIEP (no context, no uncertainty, constant-vel prediction)
  context    – + context/perspective weighting
  uncertainty – + uncertainty-aware force modulation
  proactive  – + proactive human-response prediction
  constrained – + barrier/safety constraint layer
  full       – complete Proactive-SIEP method

Output
------
  runs/paper/<ablation>/metrics.csv    per-episode metrics
  runs/paper/<ablation>/summary.json   aggregated statistics
  runs/paper/summary_plot.png          comparison bar chart
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import load_config
from .metrics import EpisodeMetrics, record_step, compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Ablation flag configurations
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_FLAGS: Dict[str, Dict[str, bool]] = {
    'base': {
        'use_context_adapt': False,
        'use_uncertainty_ps': False,
        'use_proactive_pred': False,
        'use_barrier': False,
    },
    'context': {
        'use_context_adapt': True,
        'use_uncertainty_ps': False,
        'use_proactive_pred': False,
        'use_barrier': False,
    },
    'uncertainty': {
        'use_context_adapt': False,
        'use_uncertainty_ps': True,
        'use_proactive_pred': False,
        'use_barrier': False,
    },
    'proactive': {
        'use_context_adapt': True,
        'use_uncertainty_ps': True,
        'use_proactive_pred': True,
        'use_barrier': False,
    },
    'constrained': {
        'use_context_adapt': True,
        'use_uncertainty_ps': True,
        'use_proactive_pred': True,
        'use_barrier': True,
    },
    'full': {
        'use_context_adapt': True,
        'use_uncertainty_ps': True,
        'use_proactive_pred': True,
        'use_barrier': True,
    },
}


def run_episode(
    cfg: dict,
    planner_flags: Dict[str, bool],
    episode_seed: int,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Run one simulation episode and return metrics dict.

    Parameters
    ----------
    cfg            : loaded config dict
    planner_flags  : ablation flags for ProactiveSIEP constructor
    episode_seed   : seed override for this episode
    max_steps      : override config max_steps if provided
    """
    # Import here to avoid circular imports at module level
    from ..env.sim import SocialNavSim
    from ..planners.proactive_siep import ProactiveSIEP

    cfg = dict(cfg)  # shallow copy
    cfg['seed'] = episode_seed

    sim = SocialNavSim(cfg)
    sim.reset()

    explore_mode = cfg.get('sim', {}).get('mode', 'goal') == 'explore'
    planner = ProactiveSIEP(cfg, explore_mode=explore_mode, **planner_flags)

    steps = max_steps or sim.max_steps
    ep = EpisodeMetrics(dt=sim.dt,
                        world_size_xy=tuple(cfg['world']['size_xy']))

    for k in range(steps):
        state = sim.get_state()
        ped_states = sim.get_pedestrians_state()
        lidar_dists, lidar_angles = sim.get_lidar_directional_2d()

        goal_xy = None if explore_mode else sim.goal_xy

        v, w = planner.plan(
            state, goal_xy, sim.dt,
            lidar_dists, lidar_angles, ped_states
        )
        sim.step(v, w)

        # Record metrics
        ped_xys = [d['xy'] for d in ped_states]
        ped_sigmas = [
            d['ps'].sigma_front if hasattr(d.get('ps'), 'sigma_front')
            else 1.6
            for d in ped_states
        ]
        record_step(ep, state.xy(), state.yaw, v, w, ped_xys, ped_sigmas)

        if not explore_mode and sim.reached_goal():
            ep.reached_goal = True
            break

    sim.close()
    return compute_metrics(ep, goal_xy=sim.goal_xy if not explore_mode else None)


def run_ablation(
    ablation_name: str,
    cfg: dict,
    n_episodes: int,
    out_dir: Path,
    max_steps: Optional[int] = None,
    base_seed: int = 42,
) -> Dict[str, float]:
    """Run n_episodes for one ablation configuration, save results.

    Returns aggregated summary statistics.
    """
    flags = ABLATION_FLAGS[ablation_name]
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / 'metrics.csv'
    all_metrics: List[Dict[str, float]] = []

    for ep_idx in range(n_episodes):
        seed = base_seed + ep_idx
        print(f'  [{ablation_name}] episode {ep_idx+1}/{n_episodes} seed={seed}')
        m = run_episode(cfg, flags, episode_seed=seed, max_steps=max_steps)
        m['episode'] = float(ep_idx)
        m['seed'] = float(seed)
        all_metrics.append(m)

    # Write CSV
    if all_metrics:
        keys = list(all_metrics[0].keys())
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)

    # Compute summary statistics
    summary: Dict[str, float] = {}
    numeric_keys = [k for k in all_metrics[0].keys()
                    if k not in ('episode', 'seed')]
    for k in numeric_keys:
        vals = [m[k] for m in all_metrics if math.isfinite(m.get(k, float('nan')))]
        if vals:
            summary[f'{k}_mean'] = float(np.mean(vals))
            summary[f'{k}_std'] = float(np.std(vals))
        else:
            summary[f'{k}_mean'] = float('nan')
            summary[f'{k}_std'] = float('nan')

    summary['ablation'] = ablation_name  # type: ignore[assignment]
    summary['n_episodes'] = float(n_episodes)

    json_path = out_dir / 'summary.json'
    with json_path.open('w') as f:
        json.dump(summary, f, indent=2)

    print(f'  [{ablation_name}] saved → {out_dir}')
    return summary


def run_all_ablations(
    cfg: dict,
    ablations: List[str],
    n_episodes: int,
    out_dir: Path,
    max_steps: Optional[int] = None,
    base_seed: int = 42,
) -> None:
    """Run all specified ablations and generate a comparison plot.

    Saves summary_plot.png in out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict[str, float]] = {}

    for abl in ablations:
        print(f'\n=== Ablation: {abl} ===')
        summary = run_ablation(
            abl, cfg, n_episodes, out_dir / abl,
            max_steps=max_steps, base_seed=base_seed,
        )
        summaries[abl] = summary

    # Save combined summary
    combined_path = out_dir / 'all_ablations_summary.json'
    with combined_path.open('w') as f:
        json.dump(summaries, f, indent=2)

    # Generate bar chart comparison
    _plot_ablation_comparison(summaries, out_dir)
    print(f'\nAll ablations complete. Results in {out_dir}')


def _plot_ablation_comparison(
    summaries: Dict[str, Dict[str, float]],
    out_dir: Path,
) -> None:
    """Generate a multi-panel bar chart comparing ablation metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; skipping plot.')
        return

    metrics_to_plot = [
        ('ps_violation_rate_mean',    'PS Violation Rate',    '↓'),
        ('min_human_dist_mean_mean',  'Min Human Dist (m)',   '↑'),
        ('social_disturbance_rate_mean', 'Social Disturbance Rate', '↓'),
        ('path_length_mean',          'Path Length (m)',      '-'),
        ('linear_jerk_rms_mean',      'Linear Jerk RMS',      '↓'),
        ('exploration_coverage_mean', 'Coverage (%)',         '↑'),
    ]

    ablation_names = list(summaries.keys())
    n_plots = len(metrics_to_plot)
    ncols = 3
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(ablation_names)))

    for idx, (key, label, direction) in enumerate(metrics_to_plot):
        ax = axes[idx]
        vals = [summaries[abl].get(key, float('nan')) for abl in ablation_names]
        errs = [summaries[abl].get(key.replace('_mean', '_std'), 0.0)
                for abl in ablation_names]
        bars = ax.bar(range(len(ablation_names)), vals, yerr=errs,
                      color=colors, capsize=4, alpha=0.85)
        ax.set_xticks(range(len(ablation_names)))
        ax.set_xticklabels(ablation_names, rotation=25, ha='right', fontsize=8)
        ax.set_title(f'{label} ({direction})', fontsize=9)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Proactive-SIEP Ablation Comparison', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plot_path = out_dir / 'summary_plot.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'Plot saved: {plot_path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description='Proactive-SIEP ablation experiment runner'
    )
    ap.add_argument(
        '--config', type=str,
        default='social_nav3d/configs/paper/proactive_siep_full.yaml',
        help='Path to experiment config YAML',
    )
    ap.add_argument(
        '--ablation', type=str, default='all',
        help='Ablation to run: base|context|uncertainty|proactive|constrained|full|all',
    )
    ap.add_argument('--episodes', type=int, default=5,
                    help='Number of episodes per ablation')
    ap.add_argument('--steps', type=int, default=None,
                    help='Override max_steps')
    ap.add_argument('--out-dir', type=str, default='runs/paper',
                    help='Output directory for results')
    ap.add_argument('--seed', type=int, default=42,
                    help='Base random seed')
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)

    if args.ablation == 'all':
        ablations = list(ABLATION_FLAGS.keys())
    else:
        ablations = [a.strip() for a in args.ablation.split(',')]
        for a in ablations:
            if a not in ABLATION_FLAGS:
                print(f'Unknown ablation: {a}.  Options: {list(ABLATION_FLAGS.keys())}')
                return 1

    run_all_ablations(
        cfg, ablations, args.episodes, out_dir,
        max_steps=args.steps, base_seed=args.seed,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
