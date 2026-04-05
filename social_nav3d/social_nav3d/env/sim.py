from __future__ import annotations
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from ..utils.geometry import Pose2, integrate_diff_drive
from ..utils.social import PersonalSpace
from .pedestrian_fsm import (
    FSMPedestrian,
    create_fsm_pedestrians,
    create_typed_fsm_pedestrians,
    step_fsm_pedestrians,
)
from .video_recorder import VideoRecorder3D, create_recorder


@dataclass
class Pedestrian:
    body_id: int
    radius: float
    xy: np.ndarray
    yaw: float
    vel: np.ndarray
    ps: PersonalSpace


@dataclass
class LidarConfig:
    n_az: int
    n_el: int
    max_range: float
    el_deg: List[float]


class SocialNavSim:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dt = float(cfg['sim']['dt'])
        self.max_steps = int(cfg['sim']['max_steps'])
        self.gui = bool(cfg['sim']['gui'])
        self.record_video = bool(cfg['sim'].get('record_video', False))
        self.out_dir = Path(cfg['sim'].get('out_dir', 'runs'))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Simulation mode: 'goal' (A→B) or 'explore' (free roaming)
        self.mode: str = cfg['sim'].get('mode', 'goal')

        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)

        self.plane_id = p.loadURDF('plane.urdf', physicsClientId=self.client)

        self._build_world()
        self.robot_id = self._build_robot()

        self.lidar = LidarConfig(**cfg['lidar'])

        self.start = np.array(cfg['robot']['start'], dtype=float)
        self.goal = np.array(cfg['robot']['goal'], dtype=float)
        self.goal_xy = self.goal[:2]

        self.robot_limits = {
            'max_v': float(cfg['robot']['max_v']),
            'max_w': float(cfg['robot']['max_w'])
        }
        self.robot_pose = Pose2(float(self.start[0]), float(self.start[1]), float(self.start[2]))
        self.max_v = float(cfg['robot']['max_v'])
        self.max_w = float(cfg['robot']['max_w'])
        self.robot_radius = float(cfg['robot']['radius'])

        # Waypoints / exhibit positions for FSM pedestrian navigation
        self._waypoints: List[np.ndarray] = self._load_waypoints()

        # PyBullet Pedestrian list (physics bodies with visual links)
        self.pedestrians: List[Pedestrian] = []
        # FSM logic layer (parallel to self.pedestrians, same index)
        self._fsm_peds: List[FSMPedestrian] = []

        self._spawn_pedestrians()

        # Visited positions for exploration coverage tracking
        self._visited_xys: List[np.ndarray] = []

        # ── Video recorder (optional) ─────────────────────────────────────
        self._recorder: Optional[VideoRecorder3D] = None
        if self.record_video:
            camera_mode = cfg['sim'].get('camera_mode', 'follow')
            out_path = str(self.out_dir / 'run.mp4')
            self._recorder = create_recorder(
                client=self.client,
                cfg=cfg,
                mode=camera_mode,
                out_path=out_path,
            )

    def close(self):
        if self._recorder is not None:
            self._recorder.save()
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def _build_world(self):
        world_type = self.cfg['world'].get('type', 'simple')

        if world_type == 'museum':
            # Museum scene: use museum_builder for rich multi-room layout
            from .museum_builder import build_museum_world
            museum_data = build_museum_world(self.client, self.cfg)
            self._museum_data = museum_data
            # Use exhibit positions as additional waypoints for pedestrians
            self._museum_exhibit_positions = museum_data.get('exhibit_positions', [])
            return

        self._museum_data = None
        self._museum_exhibit_positions = []

        # Obstacles: simple boxes
        for obs in self.cfg['world'].get('obstacles', []):
            pos = obs['pos']
            hx, hy, hz = obs['half_extents']
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=self.client
            )
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0.6, 0.6, 0.6, 1],
                physicsClientId=self.client
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                physicsClientId=self.client
            )

        # Ramps: tilted boxes
        for r in self.cfg['world'].get('ramps', []):
            pos = r['pos']
            sx, sy, sz = r['size']
            pitch = math.radians(r.get('pitch_deg', 10))
            orn = p.getQuaternionFromEuler([0, pitch, 0])
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[sx / 2, sy / 2, sz / 2], physicsClientId=self.client
            )
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[sx / 2, sy / 2, sz / 2], rgbaColor=[0.3, 0.3, 0.8, 1],
                physicsClientId=self.client
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                baseOrientation=orn,
                physicsClientId=self.client
            )

    def _resolve_urdf_path(self, urdf_rel: str) -> Path:
        """
        Resolve URDF path robustly:
        1) absolute path
        2) relative to social_nav3d/social_nav3d (package root)
        3) relative to current working directory
        """
        path = Path(urdf_rel)
        if path.is_absolute() and path.exists():
            return path

        base_dir = Path(__file__).resolve().parent.parent  # social_nav3d/social_nav3d
        cand = (base_dir / urdf_rel).resolve()
        if cand.exists():
            return cand

        cand2 = (Path.cwd() / urdf_rel).resolve()
        if cand2.exists():
            return cand2

        raise FileNotFoundError(f"[robot] URDF not found: {urdf_rel} (tried {cand} and {cand2})")

    def _lift_robot_to_ground(self, body: int) -> Tuple[float, float]:
        """
        Compute min_z/max_z over base and all links, then lift so min_z is slightly above 0.
        Returns (height, base_z_after_lift).
        """
        num_joints = p.getNumJoints(body, physicsClientId=self.client)

        min_z = 1e9
        max_z = -1e9
        # base = -1, links = [0..num_joints-1]
        for link in [-1] + list(range(num_joints)):
            aabb_min, aabb_max = p.getAABB(body, linkIndex=link, physicsClientId=self.client)
            min_z = min(min_z, aabb_min[2])
            max_z = max(max_z, aabb_max[2])

        # lift so min_z -> 0.01
        lift = -min_z + 0.01
        pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=self.client)
        new_pos = [pos[0], pos[1], pos[2] + lift]
        p.resetBasePositionAndOrientation(body, new_pos, orn, physicsClientId=self.client)

        height = max_z - min_z
        return float(height), float(new_pos[2])

    def _build_robot(self) -> int:
        urdf_rel = self.cfg['robot'].get('urdf', None)
        start = self.cfg['robot']['start']
        yaw = float(start[2])

        # Default: simplified cylinder robot
        if not urdf_rel:
            r = float(self.cfg['robot']['radius'])
            h = float(self.cfg['robot']['height'])
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h, physicsClientId=self.client)
            vis = p.createVisualShape(
                p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=[0.1, 0.2, 0.9, 1],
                physicsClientId=self.client
            )
            body = p.createMultiBody(
                baseMass=20.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[start[0], start[1], h / 2],
                baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                physicsClientId=self.client
            )
            p.changeDynamics(body, -1, lateralFriction=1.0, rollingFriction=0.01, physicsClientId=self.client)
            self.robot_base_z = float(h / 2)
            return body

        # URDF robot
        urdf_path = self._resolve_urdf_path(urdf_rel)

        # Help PyBullet resolve meshes referenced by relative paths inside URDF
        p.setAdditionalSearchPath(str(urdf_path.parent), physicsClientId=self.client)
        # common layout: .../atom01_description/urdf  -> meshes in ../meshes
        meshes_dir = (urdf_path.parent.parent / "meshes").resolve()
        if meshes_dir.exists():
            p.setAdditionalSearchPath(str(meshes_dir), physicsClientId=self.client)

        scale = float(self.cfg['robot'].get('scale', 1.0))
        body = p.loadURDF(
            str(urdf_path),
            basePosition=[start[0], start[1], 1.0],  # temporary height, will be corrected
            baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
            useFixedBase=False,
            globalScaling=scale,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client
        )

        # Lift to ground robustly (prevents feet underground)
        height, base_z = self._lift_robot_to_ground(body)
        self.robot_base_z = base_z

        # Update planner geometry (radius/height) from base AABB (cheap, stable)
        aabb_min, aabb_max = p.getAABB(body, linkIndex=-1, physicsClientId=self.client)
        ext_x = aabb_max[0] - aabb_min[0]
        ext_y = aabb_max[1] - aabb_min[1]
        self.cfg['robot']['height'] = float(height)
        self.cfg['robot']['radius'] = float(max(ext_x, ext_y) / 2.0)

        p.changeDynamics(body, -1, lateralFriction=1.0, rollingFriction=0.01, physicsClientId=self.client)

        # settle a few steps (optional but helps contact)
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.client)

        return body

    def _load_waypoints(self) -> List[np.ndarray]:
        """Load or auto-generate waypoints for FSM pedestrian navigation.

        Waypoints are used as "interest points" (exhibit positions) that
        pedestrians walk toward, dwell at, then depart — modelling museum-style
        visitor behaviour.

        If the config specifies ``pedestrians.waypoints``, those are used.
        Otherwise a grid of waypoints is auto-generated from world obstacles
        (obstacle centres make reasonable exhibit proxies) plus uniformly
        distributed fallback points.
        """
        wps_cfg = self.cfg.get('pedestrians', {}).get('waypoints', None)
        if wps_cfg:
            return [np.array(wp, dtype=float) for wp in wps_cfg]

        # Museum world: use exhibit positions as waypoints
        if hasattr(self, '_museum_exhibit_positions') and self._museum_exhibit_positions:
            return [np.array(pos, dtype=float) for pos in self._museum_exhibit_positions]

        # Auto-generate: use obstacle positions + grid points
        sx, sy = self.cfg['world']['size_xy']
        wps: List[np.ndarray] = []
        for obs in self.cfg['world'].get('obstacles', []):
            wps.append(np.array(obs['pos'][:2], dtype=float))

        # Add a grid of fallback positions
        for gx in np.linspace(sx * 0.2, sx * 0.8, 3):
            for gy in np.linspace(sy * 0.2, sy * 0.8, 3):
                wps.append(np.array([gx, gy], dtype=float))

        return wps

    def _spawn_pedestrians(self):
        """Spawn pedestrians with FSM logic layer.

        Creates:
          - PyBullet multi-body (physics + visual) for each pedestrian.
          - Corresponding FSMPedestrian for behaviour (FSM state machine).

        Uses typed FSM pedestrians (visitor/group/staff) when world type is
        'museum' or pedestrian count is ≥10, to produce diverse behaviors.
        The two lists (self.pedestrians and self._fsm_peds) are kept in sync:
        index i in both lists refers to the same individual.
        """
        n = int(self.cfg['pedestrians']['count'])
        rad = float(self.cfg['pedestrians']['radius'])
        vmin, vmax = self.cfg['pedestrians']['speed_range']
        ps_cfg = self.cfg['pedestrians']['personal_space']
        ps = PersonalSpace(**ps_cfg)
        seed = int(self.cfg.get('seed', 0))

        # ── FSM pedestrian logic (no PyBullet dependency) ────────────────
        sx, sy = self.cfg['world']['size_xy']
        world_type = self.cfg['world'].get('type', 'simple')

        # Use typed pedestrians for museum or when count >= 10
        use_typed = (world_type == 'museum' or n >= 10)
        if use_typed:
            self._fsm_peds = create_typed_fsm_pedestrians(
                n=n,
                world_size=(sx, sy),
                speed_range=(vmin, vmax),
                radius=rad,
                waypoints=self._waypoints,
                seed=seed,
            )
        else:
            self._fsm_peds = create_fsm_pedestrians(
                n=n,
                world_size=(sx, sy),
                speed_range=(vmin, vmax),
                radius=rad,
                waypoints=self._waypoints,
                seed=seed,
                lane_flow=True,
            )

        # Shared collision shape (all pedestrians same radius)
        col = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=rad, height=0.9, physicsClientId=self.client
        )
        base_vis = p.createVisualShape(
            p.GEOM_CAPSULE, radius=rad, length=0.9,
            rgbaColor=[0.0, 0.0, 0.0, 0.0], physicsClientId=self.client
        )

        link_masses = [0.0, 0.0]
        link_collision = [-1, -1]
        link_positions = [[0.0, 0.0, 0.65], [0.0, 0.0, 1.10]]
        link_orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, 0]),
        ]
        link_inertial_pos = [[0, 0, 0], [0, 0, 0]]
        link_inertial_orn = [
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, 0]),
        ]
        link_parent = [0, 0]
        link_joint_type = [p.JOINT_FIXED, p.JOINT_FIXED]
        link_joint_axis = [[0, 0, 1], [0, 0, 1]]

        for fsm_ped in self._fsm_peds:
            x, y = float(fsm_ped.xy[0]), float(fsm_ped.xy[1])
            yaw = float(fsm_ped.yaw)

            # Per-pedestrian torso color (from typed FSM or default blue)
            torso_color = list(getattr(fsm_ped, 'torso_color', [0.2, 0.6, 0.9, 1.0]))

            torso_vis = p.createVisualShape(
                p.GEOM_CYLINDER, radius=rad * 0.9, length=0.55,
                rgbaColor=torso_color, physicsClientId=self.client
            )
            head_vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=rad * 0.75,
                rgbaColor=[0.95, 0.85, 0.7, 1.0], physicsClientId=self.client
            )
            link_visual = [torso_vis, head_vis]

            body = p.createMultiBody(
                baseMass=70.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=base_vis,
                basePosition=[x, y, 0.9 / 2 + rad],
                baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                linkMasses=link_masses,
                linkCollisionShapeIndices=link_collision,
                linkVisualShapeIndices=link_visual,
                linkPositions=link_positions,
                linkOrientations=link_orientations,
                linkInertialFramePositions=link_inertial_pos,
                linkInertialFrameOrientations=link_inertial_orn,
                linkParentIndices=link_parent,
                linkJointTypes=link_joint_type,
                linkJointAxis=link_joint_axis,
                physicsClientId=self.client
            )
            p.changeDynamics(
                body, -1, lateralFriction=1.0, rollingFriction=0.0,
                physicsClientId=self.client
            )
            self.pedestrians.append(Pedestrian(
                body_id=body,
                radius=rad,
                xy=fsm_ped.xy.copy(),
                yaw=yaw,
                vel=fsm_ped.vel.copy(),
                ps=ps,
            ))

    def reset(self) -> Pose2:
        self.robot_pose = Pose2(float(self.start[0]), float(self.start[1]), float(self.start[2]))
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [self.robot_pose.x, self.robot_pose.y, float(self.robot_base_z)],
            p.getQuaternionFromEuler([0, 0, self.robot_pose.yaw]),
            physicsClientId=self.client
        )
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._visited_xys.clear()
        return self.robot_pose.copy()

    def _step_pedestrians(self):
        """Advance pedestrian FSMs and sync PyBullet bodies."""
        sx, sy = self.cfg['world']['size_xy']

        # Step FSM logic (computes new positions in _fsm_peds)
        step_fsm_pedestrians(
            self._fsm_peds,
            waypoints=self._waypoints,
            world_size=(sx, sy),
            dt=self.dt,
            robot_xy=self.robot_pose.xy(),
            robot_radius=self.robot_radius,
        )

        # Sync Pedestrian (physics layer) ← FSMPedestrian (logic layer)
        rad = float(self.cfg['pedestrians']['radius'])
        for ped, fsm_ped in zip(self.pedestrians, self._fsm_peds):
            ped.xy = fsm_ped.xy.copy()
            ped.vel = fsm_ped.vel.copy()
            ped.yaw = fsm_ped.yaw

            p.resetBasePositionAndOrientation(
                ped.body_id,
                [float(ped.xy[0]), float(ped.xy[1]), 0.9 / 2 + ped.radius],
                p.getQuaternionFromEuler([0, 0, ped.yaw]),
                physicsClientId=self.client
            )

    def get_lidar_scan(self) -> Tuple[np.ndarray, np.ndarray]:
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        rx, ry, rz = base_pos
        yaw = p.getEulerFromQuaternion(base_orn)[2]

        origins = []
        targets = []

        az = np.linspace(-math.pi, math.pi, self.lidar.n_az, endpoint=False)
        el = np.radians(np.array(self.lidar.el_deg, dtype=float))

        for e in el:
            for a in az:
                ang = a + yaw
                dx = math.cos(ang) * math.cos(e)
                dy = math.sin(ang) * math.cos(e)
                dz = math.sin(e)
                origins.append([rx, ry, rz + 0.20])
                targets.append([rx + dx * self.lidar.max_range,
                                ry + dy * self.lidar.max_range,
                                rz + dz * self.lidar.max_range])

        results = p.rayTestBatch(origins, targets, physicsClientId=self.client)
        hit_frac = np.array([r[2] for r in results], dtype=float)
        hit_pos = np.array([r[3] for r in results], dtype=float)
        return hit_frac, hit_pos

    def get_lidar_directional_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D planar lidar for SIEP planner.
        Uses the horizontal (el=0 deg) ring from get_lidar_scan, or falls back
        to a dedicated flat raycast if no zero-elevation ring exists.

        Returns
        -------
        dists  : (n_az,) float ndarray  – hit distance in metres (max_range if no hit)
        angles : (n_az,) float ndarray  – ray angles in robot frame (radians), [-pi, pi)
        """
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        rx, ry, rz = base_pos
        robot_yaw = p.getEulerFromQuaternion(base_orn)[2]

        n_az = self.lidar.n_az
        angles = np.linspace(-math.pi, math.pi, n_az, endpoint=False)  # robot-frame angles

        origins = []
        targets = []
        for a in angles:
            ang = a + robot_yaw  # world-frame angle
            dx = math.cos(ang)
            dy = math.sin(ang)
            origins.append([rx, ry, rz + 0.20])
            targets.append([
                rx + dx * self.lidar.max_range,
                ry + dy * self.lidar.max_range,
                rz + 0.20,  # flat horizontal scan
            ])

        results = p.rayTestBatch(origins, targets, physicsClientId=self.client)
        # hit_fraction == 1.0 means no hit
        dists = np.array([r[2] * self.lidar.max_range for r in results], dtype=float)
        return dists, angles

    def apply_control(self, v: float, w: float):
        v = float(np.clip(v, -self.max_v, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))

        self.robot_pose = integrate_diff_drive(self.robot_pose, v, w, self.dt)

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [self.robot_pose.x, self.robot_pose.y, float(self.robot_base_z)],
            p.getQuaternionFromEuler([0, 0, self.robot_pose.yaw]),
            physicsClientId=self.client
        )

    def step(self, v: float, w: float, step_idx: int = 0):
        self._step_pedestrians()
        self.apply_control(v, w)
        p.stepSimulation(physicsClientId=self.client)
        # Track visited positions for exploration coverage
        self._visited_xys.append(self.robot_pose.xy().copy())

        # Capture video frame if recorder is active
        if self._recorder is not None:
            self._recorder.capture_frame(
                robot_xy=self.robot_pose.xy(),
                robot_yaw=self.robot_pose.yaw,
                step_idx=step_idx,
            )

    def get_visited_xys(self) -> List[np.ndarray]:
        """Return visited robot positions (for exploration mode)."""
        return self._visited_xys

    def state(self) -> dict:
        return {
            'pose': self.robot_pose.copy(),
            'goal_xy': self.goal[:2].copy(),
            'pedestrians': [
                {
                    'xy': ped.xy.copy(),
                    'yaw': ped.yaw,
                    'vel': ped.vel.copy(),
                    'radius': ped.radius,
                    'ps': ped.ps,
                } for ped in self.pedestrians
            ],
        }

    def get_state(self):
        return self.robot_pose.copy()

    def get_pedestrians_state(self):
        return [
            {'xy': ped.xy.copy(), 'yaw': ped.yaw, 'vel': ped.vel.copy(), 'ps': ped.ps, 'radius': ped.radius}
            for ped in self.pedestrians
        ]

    def lidar_min_distance(self) -> float:
        _, dists = self.get_lidar_scan()
        if dists.size == 0:
            return float('inf')
        return float(dists.min())

    def reached_goal(self, tol: float = 0.5) -> bool:
        d = np.linalg.norm(self.robot_pose.xy() - self.goal_xy)
        return d <= tol

    def plot_topdown(self, ax):
        sx, sy = self.cfg['world']['size_xy']
        ax.set_xlim(0, sx)
        ax.set_ylim(0, sy)
        ax.set_aspect('equal', 'box')

        for ob in self.cfg['world'].get('obstacles', []):
            pos = ob['pos']
            hx, hy, _ = ob['half_extents']
            ax.add_patch(__import__('matplotlib').patches.Rectangle(
                (pos[0] - hx, pos[1] - hy), 2 * hx, 2 * hy, fill=False
            ))

        for ped in self.pedestrians:
            ax.add_patch(__import__('matplotlib').patches.Circle(
                (float(ped.xy[0]), float(ped.xy[1])), float(ped.radius), fill=False
            ))

        ax.scatter([self.start[0]], [self.start[1]], marker='o')
        ax.scatter([self.goal_xy[0]], [self.goal_xy[1]], marker='*')