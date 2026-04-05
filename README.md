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
| `C_unc` | `objective_terms.uncertainty_cost` | MC-Ensemble prediction uncertainty penalty |
| `C_dyn` | `objective_terms.dynamics_cost` | Smoothness / jerk penalty |
| `G_explore` | `objective_terms.exploration_gain` | Frontier novelty reward |
| `h(x)` | `barrier_cbf.BarrierCBF` | CBF barrier function for safety |

### Key Innovations

1. **Proactive human-response prediction** – `Ĥ(U)` depends on candidate robot trajectory `U` via an SFM reaction model. The robot plans trajectories that account for how pedestrians will respond.

2. **Context-conditioned weighting** – `w_j(z_τ)` and `F_ctx(z_τ)` adapt to detected social context (PASSING / YIELDING / QUEUE_FLOW / GROUP_INTERACTION / OPEN_SPACE), pedestrian facing direction (perspective-aware), and corridor geometry.

3. **Uncertainty-aware force modulation** – Prediction uncertainty (MC-Ensemble with S=32 noisy rollouts on GPU) inflates the personal-space sigma of `F_human` AND contributes to the `C_unc` term in the objective, producing stronger avoidance when predictions are uncertain.

4. **Formal CBF safety barrier** – A Control Barrier Function `h(x) = d_min - (r_robot + r_safe)` projects the nominal SIEP output onto the safe set via closed-form QP projection: `ḣ(x,u) + α·h(x) ≥ 0`.

5. **Goal-free exploration** – `F_frontier` replaces `F_goal` in explore mode. The robot seeks open, previously-unvisited directions without a global map.

6. **FSM pedestrian behaviour** – Pedestrians use a Finite State Machine with 3 behavioral types: Visitors (slow, long dwell), Group visitors (cluster following), Staff (fast patrol routes).

7. **GPU-accelerated batch evaluation** – PyTorch CUDA evaluates all 512 candidate trajectories simultaneously with 32 MC-Ensemble noise samples for uncertainty. Falls back to CPU NumPy when CUDA is unavailable.

---

## Repository Structure

```
social_nav3d/
  run_demo.py                        # Main entry point
  social_nav3d/
    env/
      sim.py                         # PyBullet simulation (FSM pedestrians, video)
      pedestrian_fsm.py              # FSM pedestrian behaviour (3 types)
      museum_builder.py              # Museum scene construction (Nanjing Museum style)
      video_recorder.py              # 3D multi-view video recorder
    planners/
      proactive_siep.py              # Proactive-SIEP planner (main method)
      gpu_batch_eval.py              # GPU batch trajectory evaluator (PyTorch CUDA)
      barrier_cbf.py                 # Formal CBF safety layer (QP projection)
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
        museum_explore.yaml          # Museum exploration config (13 peds, GPU)
        museum_goal.yaml             # Museum goal-nav config (13 peds, GPU)
        ablation_base.yaml           # Ablation: base SIEP
        ablation_context.yaml        # Ablation: + context
        ablation_uncertainty.yaml    # Ablation: + uncertainty
        ablation_proactive.yaml      # Ablation: + proactive prediction
        ablation_constrained.yaml    # Ablation: + barrier constraint
        explore_mode.yaml            # Explore mode config
```

---

## Installation

### Standard (CPU) Install

```bash
pip install -r social_nav3d/requirements.txt
```

### GPU Install (recommended for H800 / CUDA systems)

PyTorch must be installed with CUDA support matching your system:

```bash
# CPU-only fallback
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x (recommended for H800/A100/V100)
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu118

# Then install the rest
pip install -r social_nav3d/requirements.txt
```

Verify GPU availability:
```python
from social_nav3d.planners.gpu_batch_eval import is_gpu_available
print(is_gpu_available())  # True on CUDA systems
```

---

## Running Experiments

### Museum Exploration Demo (Full Setup)

```bash
cd social_nav3d
# Full museum exploration demo with GPU, 3D video, 13 pedestrians
python run_demo.py --museum --planner proactive --siep-variant full --gpu --record --camera all --eval
```

### Museum Goal Navigation

```bash
cd social_nav3d
python run_demo.py \
    --config social_nav3d/configs/paper/museum_goal.yaml \
    --planner proactive \
    --siep-variant full \
    --gpu --record --camera follow --eval
```

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

### Run all ablations including +CBF (paper experiment pipeline)
```bash
cd social_nav3d
python -m social_nav3d.experiments.runner \
    --config social_nav3d/configs/paper/museum_explore.yaml \
    --ablation all \
    --episodes 10 \
    --gpu \
    --out-dir runs/museum_paper
```

### CLI Flags Reference

| Flag | Description |
|---|---|
| `--gpu` / `--no-gpu` | Force GPU/CPU mode (default: auto-detect) |
| `--camera follow\|overhead\|cinematic\|all` | Video camera mode |
| `--museum` | Shortcut: load `museum_explore.yaml` |
| `--record` | Record MP4 video to `runs/` directory |
| `--siep-variant cbf` | Use formal CBF barrier (instead of simple clipping) |

---

## Museum Scene

The museum scene (`world.type: museum`) is inspired by Nanjing Museum layout:

- **Entrance hall** (bottom center): open 8m × 6m area
- **Main gallery** (center): large 18m × 14m room with 5 exhibit pedestals
- **Left corridor**: narrow 4m × 14m passage
- **Wing room** (right): 8m × 12m room with 3 exhibits
- **Top corridor**: 26m × 5m connecting passage

Pedestrians (13 total) are spawned with 3 behavioral types:
- **8 Visitors** (blue): slow 0.3–0.7 m/s, dwell 8–20s at exhibits
- **3 Group visitors** (green): move together, follow leader
- **2 Staff** (orange): fast 0.8–1.2 m/s, patrol fixed routes

---

## 3D Video Recording

Three camera modes are available with `--record`:

| Mode | Description |
|---|---|
| `follow` | Camera follows robot from behind and above |
| `overhead` | Fixed bird's-eye view of entire scene |
| `cinematic` | Slow orbit around scene center |
| `all` | Output three separate MP4 files (one per mode) |

Videos are saved to `runs/<out_dir>/run.mp4` (or `run_follow.mp4`, etc. for `--camera all`).
Works in headless mode (no display required).

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
  cbf/      ...          # +CBF ablation
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

| Variant | Context | Uncertainty | Proactive Pred | CBF Barrier |
|---|---|---|---|---|
| `base` | ✗ | ✗ | ✗ | ✗ |
| `context` | ✓ | ✗ | ✗ | ✗ |
| `uncertainty` | ✗ | ✓ | ✗ | ✗ |
| `proactive` | ✓ | ✓ | ✓ | ✗ |
| `constrained` | ✓ | ✓ | ✓ | ✓ |
| `full` | ✓ | ✓ | ✓ | ✓ |
| `cbf` | ✓ | ✓ | ✓ | ✓ (formal CBF) |

