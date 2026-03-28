"""
make_video.py — 生成 3D PyBullet 视频 demo
支持两种 planner:
  --planner mpc   (默认) 使用 GPUSocialMPC (需要 GPU + PyTorch)
  --planner siep  使用 SIEP 力场规划器 (纯 CPU 即可)

用法:
  python make_video.py --config social_nav3d/configs/default.yaml --planner siep
  python make_video.py --config social_nav3d/configs/default.yaml --planner mpc
"""

import os
import math
import time
import argparse
import numpy as np
import imageio
import yaml
import pybullet as p

from social_nav3d.env.sim import SocialNavSim
from social_nav3d.planners.siep_planner import SIEPPlanner
from social_nav3d.utils.config import load_config


def safe_render(client_id, target_pos, width=1280, height=720):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target_pos,
        distance=9,
        yaw=45,
        pitch=-35,
        roll=0,
        upAxisIndex=2
    )
    proj = p.computeProjectionMatrixFOV(60, width / height, 0.1, 120.0)
    w, h, rgba, _, _ = p.getCameraImage(
        width, height, view, proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id
    )
    rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]
    return rgb


def pose2_to_xytheta(pose2):
    """从 Pose2 对象提取 x, y, theta"""
    for ax, ay, ath in (("x", "y", "theta"), ("x", "y", "yaw")):
        if hasattr(pose2, ax) and hasattr(pose2, ay) and hasattr(pose2, ath):
            return float(getattr(pose2, ax)), float(getattr(pose2, ay)), float(getattr(pose2, ath))
    if hasattr(pose2, "p") and hasattr(pose2.p, "x") and hasattr(pose2.p, "y"):
        th = float(getattr(pose2, "theta", 0.0))
        return float(pose2.p.x), float(pose2.p.y), th
    try:
        arr = list(pose2)
        if len(arr) >= 3:
            return float(arr[0]), float(arr[1]), float(arr[2])
    except Exception:
        pass
    return 10.0, 10.0, 0.0


def get_goal_from_cfg(cfg):
    """从 config 中读取 goal 坐标"""
    candidates = [
        ("goal",), ("task", "goal"), ("sim", "goal"),
        ("world", "goal"), ("env", "goal"), ("scenario", "goal"),
        ("robot", "goal"),
    ]
    for path in candidates:
        d = cfg
        ok = True
        for k in path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                ok = False
                break
        if ok:
            if isinstance(d, (list, tuple)) and len(d) >= 2:
                return float(d[0]), float(d[1])
            if isinstance(d, dict) and "x" in d and "y" in d:
                return float(d["x"]), float(d["y"])
    return 10.0, 10.0


def main():
    parser = argparse.ArgumentParser(description="生成 SIEP / MPC 导航视频 demo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default="runs")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--vmax", type=float, default=1.2)
    parser.add_argument("--wmax", type=float, default=2.0)
    parser.add_argument(
        "--planner", type=str, default="siep", choices=["mpc", "siep"],
        help='选择规划器: "siep" = SIEP力场 (默认), "mpc" = GPU Social MPC'
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- 加载配置 ----
    cfg = load_config(args.config)
    goal_x, goal_y = get_goal_from_cfg(cfg)

    sim = SocialNavSim(cfg)
    sim.reset()

    # ---- 初始化 planner ----
    use_siep = args.planner == "siep"

    if use_siep:
        planner = SIEPPlanner(cfg)
        planner_label = "SIEP"
        print("[planner] 使用 SIEP 力场规划器 (CPU)")
    else:
        import torch
        from social_nav3d.planners.gpu_social_mpc import GPUSocialMPC
        planner = GPUSocialMPC(
            device="cuda",
            horizon=25,
            samples=8192,
            dt=0.1,
            v_limits=(0.0, args.vmax),
            w_limits=(-args.wmax, args.wmax),
            robot_radius=0.35,
            w_goal=2.0,
            w_smooth=0.2,
            w_ps=6.0,
            w_coll=80.0,
            w_obs=200.0,
            w_bound=200.0,
        )
        assert torch.cuda.is_available(), "MPC 模式需要 GPU!"
        torch.cuda.synchronize()
        _ = torch.empty((1024, 1024), device="cuda")
        torch.cuda.synchronize()
        planner_label = "GPU-MPC"
        print("[planner] 使用 GPU Social MPC")

    print(f"[planner] ready — {planner_label}")

    # ---- PyBullet client ----
    client_id = getattr(sim, "client", None)
    if not isinstance(client_id, int):
        raise RuntimeError(f"sim.client is not an int: {type(client_id)}")

    # ---- 主循环 ----
    frames = []
    traj = []
    world = cfg["world"]
    t0 = time.time()

    for step in range(args.steps):
        pose = sim.get_state()
        x, y, th = pose2_to_xytheta(pose)
        frames.append(safe_render(client_id, [x, y, 1.0]))
        traj.append((x, y, 0.0))

        peds_raw = sim.get_pedestrians_state()

        if use_siep:
            # SIEP planner 接口: plan(pose, goal_xy, dt, lidar_dists, lidar_angles, peds)
            lidar_dists, lidar_angles = sim.get_lidar_directional_2d()
            ped_tuples = [(d['xy'], d['yaw'], d['vel'], d['ps']) for d in peds_raw]
            v, w = planner.plan(
                pose, sim.goal_xy, sim.dt,
                lidar_dists, lidar_angles, ped_tuples
            )
        else:
            # GPU MPC 接口: act(pose, peds, goal_xy, world=world)
            v, w = planner.act(pose, peds_raw, (goal_x, goal_y), world=world)

        if step % 50 == 0:
            elapsed = (time.time() - t0) * 1000
            print(f"[{planner_label}] step={step}  elapsed={elapsed:.0f}ms  pos=({x:.1f},{y:.1f})")

        sim.step(float(v), float(w))

        dist = math.hypot(goal_x - x, goal_y - y)
        if dist < 0.3:
            print(f"[{planner_label}] 到达目标! step={step}")
            break

    total_s = time.time() - t0
    print(f"[{planner_label}] 仿真完成, 共 {len(frames)} 帧, 耗时 {total_s:.1f}s")

    # ---- 保存视频 ----
    video_name = f"demo_{planner_label.lower()}.mp4"
    mp4_path = os.path.join(args.outdir, video_name)
    imageio.mimsave(mp4_path, frames, fps=args.fps)
    print(f"[OK] 视频已保存: {mp4_path}")

    # ---- 保存 2D / 3D 轨迹图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        traj = np.array(traj, dtype=np.float32)

        # 2D 俯视图
        fig, ax = plt.subplots(figsize=(7, 7))
        sim.plot_topdown(ax)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label=planner_label)
        ax.scatter([traj[0, 0]], [traj[0, 1]], marker="o", s=80, zorder=5, color="green")
        ax.scatter([goal_x], [goal_y], marker="*", s=200, zorder=5, color="red")
        ax.legend()
        ax.set_title(f"SocialNav3D — {planner_label}: trajectory")
        fig.tight_layout()
        png_name = f"traj_{planner_label.lower()}.png"
        fig.savefig(os.path.join(args.outdir, png_name), dpi=200)
        print(f"[OK] 轨迹图已保存: {os.path.join(args.outdir, png_name)}")

        # 3D 视图
        fig3 = plt.figure(figsize=(7, 7))
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_title(f"{planner_label} Trajectory (3D)")
        fig3.savefig(os.path.join(args.outdir, f"traj_3d_{planner_label.lower()}.png"), dpi=200)
        print(f"[OK] 3D 图已保存")

    except Exception as e:
        print(f"[WARN] 画图跳过: {e}")

    sim.close()


if __name__ == "__main__":
    main()