  import math
import torch
import numpy as np


def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def rollout_diffdrive(x0, y0, yaw0, vw_seq, dt: float):
    """
    x0,y0,yaw0: scalar tensors on device
    vw_seq: (K, H, 2) with v,w
    returns: (K, H+1, 3) -> x,y,yaw
    """
    K, H, _ = vw_seq.shape
    v = vw_seq[..., 0]
    w = vw_seq[..., 1]

    x = torch.zeros((K, H + 1), device=vw_seq.device, dtype=vw_seq.dtype)
    y = torch.zeros((K, H + 1), device=vw_seq.device, dtype=vw_seq.dtype)
    yaw = torch.zeros((K, H + 1), device=vw_seq.device, dtype=vw_seq.dtype)

    x[:, 0] = x0
    y[:, 0] = y0
    yaw[:, 0] = yaw0

    for t in range(H):
        yaw[:, t + 1] = yaw[:, t] + w[:, t] * dt
        x[:, t + 1] = x[:, t] + v[:, t] * torch.cos(yaw[:, t]) * dt
        y[:, t + 1] = y[:, t] + v[:, t] * torch.sin(yaw[:, t]) * dt

    yaw = wrap_pi(yaw)
    traj = torch.stack([x, y, yaw], dim=-1)
    return traj


def ped_predict_constvel(ped_xy, ped_vel, H: int, dt: float):
    """
    ped_xy: (N,2) ; ped_vel: (N,2)
    return: (N, H+1, 2)
    """
    N = ped_xy.shape[0]
    t = torch.arange(H + 1, device=ped_xy.device, dtype=ped_xy.dtype).view(1, H + 1, 1)
    return ped_xy.view(N, 1, 2) + ped_vel.view(N, 1, 2) * (t * dt)


def personal_space_cost(robot_xy, ped_xy, ped_yaw, sig_front, sig_side, sig_back):
    """
    robot_xy: (K, T, 2)
    ped_xy:   (N, T, 2)
    ped_yaw:  (N,) yaw at current time (use constant yaw for anisotropy)
    sig_*:    (N,) personal space params

    output: (K, T) aggregated over pedestrians
    """
    # relative position in world: r - p
    # robot_xy: (K,T,2), ped_xy: (N,T,2) -> broadcast to (K,N,T,2)
    rel = robot_xy[:, None, :, :] - ped_xy[None, :, :, :]  # (K,N,T,2)

    # rotate into pedestrian heading frame
    cy = torch.cos(ped_yaw).view(1, -1, 1, 1)
    sy = torch.sin(ped_yaw).view(1, -1, 1, 1)
    dx = rel[..., 0:1]
    dy = rel[..., 1:2]
    # forward axis: x' =  cos*yaw*dx + sin*yaw*dy
    xh = cy * dx + sy * dy
    yh = -sy * dx + cy * dy

    # anisotropic sigma depends on whether robot is in front/back of pedestrian
    # xh>0 => in front
    sig = torch.where(
        xh >= 0,
        sig_front.view(1, -1, 1, 1),
        sig_back.view(1, -1, 1, 1)
    )
    sig_y = sig_side.view(1, -1, 1, 1)

    # gaussian-like penalty
    q = (xh / sig) ** 2 + (yh / sig_y) ** 2
    # exp(-0.5 q) gives smooth personal-space field
    field = torch.exp(-0.5 * q.squeeze(-1))  # (K,N,T)
    return field.sum(dim=1)  # (K,T)

def aabb_distance_xy(robot_xy, box_center_xy, half_extents_xy):
    """
    robot_xy: (K,T,2)
    box_center_xy: (M,2)
    half_extents_xy: (M,2)
    return: (K,T,M) distance from point to AABB (0 if inside)
    """
    p = robot_xy[:, :, None, :]                    # (K,T,1,2)
    c = box_center_xy[None, None, :, :]            # (1,1,M,2)
    h = half_extents_xy[None, None, :, :]          # (1,1,M,2)
    d = (p - c).abs() - h
    d = torch.clamp(d, min=0.0)
    return torch.sqrt((d ** 2).sum(dim=-1) + 1e-6) # (K,T,M)

def boundary_violation_xy(robot_xy, size_xy):
    """
    robot_xy: (K,T,2)
    size_xy: (2,) => [sx, sy]
    return: (K,) mean outside distance
    """
    sx, sy = float(size_xy[0]), float(size_xy[1])
    x = robot_xy[..., 0]
    y = robot_xy[..., 1]
    out = (
        torch.relu(0.0 - x)
        + torch.relu(x - sx)
        + torch.relu(0.0 - y)
        + torch.relu(y - sy)
    )
    return out.mean(dim=1)


class GPUSocialMPC:
    def __init__(
        self,
        device="cuda",
        horizon=25,
        samples=4096,
        dt=0.1,
        v_limits=(0.0, 1.2),
        w_limits=(-2.0, 2.0),
        robot_radius=0.35,
        w_goal=2.0,
        w_smooth=0.2,
        w_ps=4.0,
        w_obs=200.0,
        w_bound=200.0,
        w_coll=50.0,
        ps_clip=50.0
        ):
        self.device = torch.device(device)
        self.H = int(horizon)
        self.K = int(samples)
        self.dt = float(dt)
        self.vmin, self.vmax = v_limits
        self.wmin, self.wmax = w_limits
        self.robot_radius = float(robot_radius)

        self.w_goal = float(w_goal)
        self.w_smooth = float(w_smooth)
        self.w_ps = float(w_ps)
        self.w_obs = float(w_obs)
        self.w_bound = float(w_bound)
        self.w_coll = float(w_coll)
        self.ps_clip = float(ps_clip)

        # sampling distribution (tunable)
        self.v_mu = 0.8
        self.v_sigma = 0.35
        self.w_mu = 0.0
        self.w_sigma = 0.8

    @torch.no_grad()
    def act(self, robot_pose2, ped_list, goal_xy, world=None):
        """
        robot_pose2: Pose2 with x,y,yaw
        ped_list: list of dicts with keys: xy, vel, yaw, ps, radius
        goal_xy: (gx,gy)
        returns: (v,w)
        """
        x0 = torch.tensor(float(robot_pose2.x), device=self.device)
        y0 = torch.tensor(float(robot_pose2.y), device=self.device)
        yaw0 = torch.tensor(float(robot_pose2.yaw), device=self.device)

        gx = torch.tensor(float(goal_xy[0]), device=self.device)
        gy = torch.tensor(float(goal_xy[1]), device=self.device)

        # sample control sequences on GPU
        v = torch.randn((self.K, self.H), device=self.device) * self.v_sigma + self.v_mu
        w = torch.randn((self.K, self.H), device=self.device) * self.w_sigma + self.w_mu
        v = torch.clamp(v, self.vmin, self.vmax)
        w = torch.clamp(w, self.wmin, self.wmax)
        vw = torch.stack([v, w], dim=-1)  # (K,H,2)

        traj = rollout_diffdrive(x0, y0, yaw0, vw, self.dt)  # (K,H+1,3)
        rxy = traj[:, :, 0:2]  # (K,T,2), T=H+1

        # goal cost: distance to goal at final step + small along the way
        dxg = rxy[:, :, 0] - gx
        dyg = rxy[:, :, 1] - gy
        dist = torch.sqrt(dxg * dxg + dyg * dyg + 1e-6)
        cost_goal = dist[:, -1] + 0.1 * dist.mean(dim=1)

        # smoothness / comfort: penalize angular rate & changes
        cost_smooth = (w.abs().mean(dim=1) + 0.5 * (w[:, 1:] - w[:, :-1]).abs().mean(dim=1))

        # pedestrians: build tensors
        cost_ps = torch.zeros((self.K,), device=self.device)
        cost_coll = torch.zeros((self.K,), device=self.device)
        cost_obs = torch.zeros((self.K,), device=self.device)
        cost_bound = torch.zeros((self.K,), device=self.device)
        # pedestrians: build tensors (fast path)
        if len(ped_list) > 0:
            ped_xy_np  = np.asarray([p["xy"] for p in ped_list], dtype=np.float32)   # (N,2)
            ped_vel_np = np.asarray([p["vel"] for p in ped_list], dtype=np.float32)  # (N,2)
            ped_yaw_np = np.asarray([float(p["yaw"]) for p in ped_list], dtype=np.float32)  # (N,)
            ped_r_np   = np.asarray([float(p.get("radius", 0.3)) for p in ped_list], dtype=np.float32)  # (N,)

            ped_xy  = torch.as_tensor(ped_xy_np,  device=self.device)
            ped_vel = torch.as_tensor(ped_vel_np, device=self.device)
            ped_yaw = torch.as_tensor(ped_yaw_np, device=self.device)
            ped_r   = torch.as_tensor(ped_r_np,   device=self.device)

            sig_front_np = np.asarray([float(p["ps"].sigma_front) for p in ped_list], dtype=np.float32)
            sig_side_np  = np.asarray([float(p["ps"].sigma_side)  for p in ped_list], dtype=np.float32)
            sig_back_np  = np.asarray([float(p["ps"].sigma_back)  for p in ped_list], dtype=np.float32)

            sig_front = torch.as_tensor(sig_front_np, device=self.device)
            sig_side  = torch.as_tensor(sig_side_np,  device=self.device)
            sig_back  = torch.as_tensor(sig_back_np,  device=self.device)

            ped_traj = ped_predict_constvel(ped_xy, ped_vel, self.H, self.dt)  # (N,T,2)

            # personal space field cost
            ps_field = personal_space_cost(rxy, ped_traj, ped_yaw, sig_front, sig_side, sig_back)  # (K,T)
            cost_ps = torch.clamp(ps_field.mean(dim=1), 0.0, self.ps_clip)

            # collision-like hard penalty: distance to pedestrian discs
            # compute min distance over time & peds
            # rxy (K,T,2), ped_traj (N,T,2) -> (K,N,T)
            diff = rxy[:, None, :, :] - ped_traj[None, :, :, :]
            d2 = (diff ** 2).sum(dim=-1)  # (K,N,T)
            d = torch.sqrt(d2 + 1e-6)
            min_d = d.min(dim=2).values  # (K,N) min over time
            # collide if min_d < robot_r + ped_r + margin
            margin = 0.05
            thresh = (self.robot_radius + ped_r + margin).view(1, -1)
            viol = torch.relu(thresh - min_d)  # (K,N)
            cost_coll = (viol.sum(dim=1))  # (K,)
            
        if world is not None:
    # boundary
            if "size_xy" in world:
                cost_bound = boundary_violation_xy(rxy, world["size_xy"])

            # obstacles (AABB)
            obs = world.get("obstacles", [])
            if len(obs) > 0:
                box_c = torch.tensor([o["pos"][:2] for o in obs], device=self.device, dtype=torch.float32)          # (M,2)
                box_h = torch.tensor([o["half_extents"][:2] for o in obs], device=self.device, dtype=torch.float32) # (M,2)
                dist_box = aabb_distance_xy(rxy, box_c, box_h)   # (K,T,M)

                margin = 0.05
                viol_box = torch.relu((self.robot_radius + margin) - dist_box)
                cost_obs = (viol_box ** 2).sum(dim=(1,2))



        total = (
            self.w_goal * cost_goal
            + self.w_smooth * cost_smooth
            + self.w_ps * cost_ps
            + self.w_coll * cost_coll
            + self.w_obs * cost_obs
            + self.w_bound * cost_bound
        )


        best = torch.argmin(total).item()
        v0 = float(v[best, 0].item())
        w0 = float(w[best, 0].item())
        return v0, w0
