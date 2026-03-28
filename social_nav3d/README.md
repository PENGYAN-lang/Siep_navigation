# SocialNav3D (PyBullet)

This is a runnable 3D physics demo (PyBullet) for socially-aware navigation.

Included:
- 3D world (obstacles + ramps)
- differential-drive robot
- 3D LiDAR-style raycasts (multi-elevation)
- dynamic pedestrians + anisotropic personal-space penalty
- sampling MPC-style local planner

## Recommended environment (H800)
- **Python**: 3.10 (3.9–3.11 should work)
- **CUDA**: whatever your instance provides (check `nvidia-smi`)
- **PyTorch** (optional, for future learning modules): pick the build matching your CUDA

### Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Optional: install PyTorch with CUDA (example for cu121)
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```
If your machine is CUDA 12.2/12.3, use the closest supported PyTorch wheel for your setup.

## Run
```bash
python run_demo.py --config social_nav3d/configs/default.yaml
# GUI mode (if available)
python run_demo.py --gui
```
Outputs go to `runs/trajectory.png` (and optionally `runs/run.mp4` if you turn on recording).

## Make it “bigger”
Edit `social_nav3d/configs/default.yaml`:
- increase `pedestrians.count`
- add more `world.obstacles`
- increase `lidar.n_az` (more beams)
- increase `planner.n_samples` / `planner.horizon_s`


### Install (conda)
```bash
conda create -n socialnav3d python=3.10 -y
conda activate socialnav3d
pip install -r requirements.txt
```

### PyTorch (optional; for future learned prediction)
Check CUDA version:
```bash
nvidia-smi
```
Then install a CUDA build that matches your driver/toolkit. Example for CUDA 12.1:
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
```

## Run
Headless (recommended on remote):
```bash
python run_demo.py --config social_nav3d/configs/default.yaml
```
With GUI:
```bash
python run_demo.py --gui
```

Outputs go to `runs/trajectory.png`.
