# Proactive-SIEP: Socially-Aware Navigation Framework

A research-grade implementation of **Proactive Uncertainty-aware Context-conditioned SIEP** (Proactive-SIEP) for socially acceptable robot navigation in human-populated environments.

---

## Method Overview

The planner solves a receding-horizon proactive equilibrium objective:

```
U* = argmin_U  Σ_{τ=1}^{H} ‖F_tot(x_τ, Ĥ_τ(U), z_τ)‖²
               + λ_s · C_social(U, Ĥ(U), z)
               + λ_u · C_unc(U)
               + λ_d · C_dyn(U)
               − λ_e · G_explore(U)
```

where the total force decomposes as:

```
F_tot = F_{goal/frontier}
        + Σ_o F_{obs,o}
        + Σ_j w_j(z_τ) · F_{human,j}(x_τ, ĥ_{j,τ}(U))
        + F_ctx(z_τ)
```

### Term-to-Code Mapping

| Paper symbol | Code location | Description |
|---|---|---|
| `F_{goal}` | `objective_terms.F_goal_force` | Tanh-saturated goal attraction |
| `F_{frontier}` | `objective_terms.F_frontier_force` | Mapless exploration force |
| `F_{obs}` | `objective_terms.F_obstacle_force` | LiDAR-based obstacle repulsion |
| `F_{human,j}` | `objective_terms.F_human_force` | Anisotropic Gaussian PS repulsion |
| `F_ctx(z)` | `context_inference.context_force` | Context-level social force |
| `w_j(z_τ)` | `context_inference.context_weight` | Per-pedestrian context weight |
| `Ĥ(U)` | `human_prediction.predict_interaction_aware` | Proactive pedestrian prediction |
| `C_social` | `objective_terms.social_cost` | Integrated PS violation cost |
| `C_unc` | `objective_terms.uncertainty_cost` | Prediction uncertainty penalty |
| `C_dyn` | `objective_terms.dynamics_cost` | Smoothness / jerk penalty |
| `G_explore` | `objective_terms.exploration_gain` | Frontier novelty reward |

### Key Innovations

1. **Proactive human-response prediction** – `Ĥ(U)` depends on candidate robot trajectory `U` via an SFM reaction model. The robot plans trajectories that account for how pedestrians will respond.

2. **Context-conditioned weighting** – `w_j(z_τ)` and `F_ctx(z_τ)` adapt to detected social context (PASSING / YIELDING / QUEUE_FLOW / GROUP_INTERACTION / OPEN_SPACE), pedestrian facing direction (perspective-aware), and corridor geometry.

3. **Uncertainty-aware force modulation** – Prediction uncertainty (noise-injected rollouts) inflates the personal-space sigma of `F_human`, producing stronger avoidance when predictions are uncertain.

4. **Safety barrier projection** – A CBF-inspired layer projects the proactive SIEP output onto the safe set, enforcing minimum clearance and jerk limits.

5. **Goal-free exploration** – `F_frontier` replaces `F_goal` in explore mode. The robot seeks open, previously-unvisited directions without a global map.

6. **FSM pedestrian behaviour** – Pedestrians use a Finite State Machine (WANDER → WALK_TO → APPROACH → VIEWING → TURNING) with variable speed, social-force avoidance, and exhibit-browsing dwell behaviour.

---

## Repository Structure

```
social_nav3d/
  run_demo.py                        # Main entry point
  social_nav3d/
    env/
      sim.py                         # PyBullet simulation (FSM pedestrians)
      pedestrian_fsm.py              # FSM pedestrian behaviour
    planners/
      proactive_siep.py              # Proactive-SIEP planner (main method)
      context_inference.py           # Social context inference + w_j(z), F_ctx
      human_prediction.py            # Interaction-aware pedestrian prediction
      objective_terms.py             # All force / cost / gain functions
      siep_planner.py                # Classic SIEP planner (baseline)
      sampling_mpc.py                # Sampling-based MPC (baseline)
    experiments/
      metrics.py                     # Paper metrics computation
      runner.py                      # Ablation experiment runner
    configs/
      default.yaml                   # Default simulation config
      paper/
        proactive_siep_full.yaml     # Full Proactive-SIEP config
        ablation_base.yaml           # Ablation: base SIEP
        ablation_context.yaml        # Ablation: + context
        ablation_uncertainty.yaml    # Ablation: + uncertainty
        ablation_proactive.yaml      # Ablation: + proactive prediction
        ablation_constrained.yaml    # Ablation: + barrier constraint
        explore_mode.yaml            # Explore mode config
```

---

## Installation

```bash
pip install -r social_nav3d/requirements.txt
```

---

## Running Experiments

### Baseline (classic SIEP planner)
```bash
cd social_nav3d
python run_demo.py --planner siep --config social_nav3d/configs/default.yaml --eval
```

### Full Proactive-SIEP (goal navigation)
```bash
cd social_nav3d
python run_demo.py \
    --config social_nav3d/configs/paper/proactive_siep_full.yaml \
    --planner proactive \
    --siep-variant full \
    --eval
```

### Exploration mode (free roaming, no fixed goal)
```bash
cd social_nav3d
python run_demo.py \
    --config social_nav3d/configs/paper/explore_mode.yaml \
    --planner proactive \
    --mode explore \
    --eval
```

### Single ablation
```bash
cd social_nav3d
python run_demo.py \
    --config social_nav3d/configs/paper/ablation_context.yaml \
    --planner proactive \
    --siep-variant context \
    --eval
```

### Run all ablations (paper experiment pipeline)
```bash
cd social_nav3d
python -m social_nav3d.experiments.runner \
    --config social_nav3d/configs/paper/proactive_siep_full.yaml \
    --ablation all \
    --episodes 5 \
    --out-dir runs/paper
```

---

## Output Files

After running the ablation pipeline:

```
runs/paper/
  base/
    metrics.csv          # Per-episode metrics
    summary.json         # Mean ± std over episodes
  context/  ...
  full/     ...
  all_ablations_summary.json   # Combined summary
  summary_plot.png             # Comparison bar chart
```

### Metrics

| Metric | Description |
|---|---|
| `ps_violation_rate` | Fraction of steps with any personal-space intrusion |
| `min_human_dist_mean/min/p10` | Distribution of minimum robot-human distance |
| `social_disturbance_rate` | Group/queue crossing events per step |
| `path_length` | Total path length [m] |
| `time_to_goal_s` | Time to reach goal [s] |
| `linear_jerk_rms` | RMS linear jerk (smoothness) |
| `angular_jerk_rms` | RMS angular jerk |
| `exploration_coverage` | Fraction of world cells visited (explore mode) |

---

## Ablation Variants

| Variant | Context | Uncertainty | Proactive Pred | Barrier |
|---|---|---|---|---|
| `base` | ✗ | ✗ | ✗ | ✗ |
| `context` | ✓ | ✗ | ✗ | ✗ |
| `uncertainty` | ✗ | ✓ | ✗ | ✗ |
| `proactive` | ✓ | ✓ | ✓ | ✗ |
| `constrained` | ✓ | ✓ | ✓ | ✓ |
| `full` | ✓ | ✓ | ✓ | ✓ |
