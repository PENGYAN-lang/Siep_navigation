from __future__ import annotations
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from ..utils.geometry import Pose2, integrate_diff_drive
from ..utils.social import PersonalSpace


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

        self.pedestrians: List[Pedestrian] = []
        self._spawn_pedestrians()

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def _build_world(self):
        # Obstacles: simple boxes (exhibit pedestals, display cases, etc.)
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

        # Walls: thin tall boxes (perimeter + interior partitions)
        # Walls use a slightly different colour to be visually distinct.
        for wall in self.cfg['world'].get('walls', []):
            pos = wall['pos']
            hx, hy, hz = wall['half_extents']
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=self.client
            )
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0.85, 0.82, 0.78, 1],
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

    # ------------------------------------------------------------------
    # Pedestrian clothing colour palette (varied, realistic)
    # ------------------------------------------------------------------
    _CLOTHING_PALETTE: List[Tuple[float, float, float]] = [
        (0.18, 0.36, 0.60),  # navy blue
        (0.55, 0.27, 0.07),  # brown
        (0.12, 0.52, 0.29),  # forest green
        (0.72, 0.15, 0.15),  # deep red
        (0.45, 0.45, 0.45),  # charcoal
        (0.85, 0.65, 0.12),  # mustard yellow
        (0.30, 0.22, 0.48),  # dark purple
        (0.15, 0.45, 0.55),  # teal
        (0.80, 0.38, 0.10),  # burnt orange
        (0.22, 0.22, 0.22),  # near black
        (0.60, 0.78, 0.85),  # light blue
        (0.50, 0.70, 0.40),  # sage green
    ]

    def _make_humanoid_links(
        self,
        clothing_rgb: Tuple[float, float, float],
        rad: float,
    ) -> Tuple[list, list, list, list, list, list, list, list, list]:
        """Build the link arrays for a multi-link articulated humanoid.

        The humanoid has 6 visual links (all FIXED joints, no collision):
          0 – torso      (box)
          1 – head       (sphere)
          2 – left upper arm  (capsule)
          3 – right upper arm (capsule)
          4 – left upper leg  (capsule)
          5 – right upper leg (capsule)

        Returns:
            Tuple of (masses, collisions, visuals, positions, orientations,
                      inertial_pos, inertial_orn, parents, joint_types, joint_axes)
        """
        cr, cg, cb = clothing_rgb
        skin = (0.95, 0.82, 0.70, 1.0)

        # Visual shapes (no collision shapes for links → cheaper physics)
        torso_v = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[rad * 0.8, rad * 0.55, 0.28],
            rgbaColor=[cr, cg, cb, 1.0], physicsClientId=self.client,
        )
        head_v = p.createVisualShape(
            p.GEOM_SPHERE, radius=rad * 0.72,
            rgbaColor=skin, physicsClientId=self.client,
        )
        arm_v = p.createVisualShape(
            p.GEOM_CAPSULE, radius=rad * 0.25, length=0.28,
            rgbaColor=skin, physicsClientId=self.client,
        )
        leg_v = p.createVisualShape(
            p.GEOM_CAPSULE, radius=rad * 0.30, length=0.38,
            rgbaColor=[cr, cg, cb, 1.0], physicsClientId=self.client,
        )

        # PyBullet linkParentIndices: 0 = base body, 1 = link 0, 2 = link 1, …
        # link 0 = torso  → parentIdx 0 (base)
        # link 1 = head   → parentIdx 1 (link 0 = torso)
        # link 2 = left arm  → parentIdx 1 (torso)
        # link 3 = right arm → parentIdx 1 (torso)
        # link 4 = left leg  → parentIdx 0 (base)
        # link 5 = right leg → parentIdx 0 (base)
        ident_orn = p.getQuaternionFromEuler([0, 0, 0])

        visuals    = [torso_v, head_v, arm_v,       arm_v,        leg_v,         leg_v]
        # Positions are relative to *parent* link frame
        positions  = [
            [0.0,  0.0,  0.65],   # torso centre (above collision capsule base)
            [0.0,  0.0,  0.32],   # head above torso
            [0.0,  0.50, 0.15],   # left arm (Y offset)
            [0.0, -0.50, 0.15],   # right arm
            [0.0,  0.20, -0.65],  # left leg (below base, Y offset)
            [0.0, -0.20, -0.65],  # right leg
        ]
        orientations = [ident_orn] * 6
        masses       = [0.0] * 6
        collisions   = [-1]  * 6
        inertial_pos = [[0, 0, 0]] * 6
        inertial_orn = [ident_orn] * 6
        # parents: torso→base, head→torso, arms→torso, legs→base
        parents      = [0, 1, 1, 1, 0, 0]
        joint_types  = [p.JOINT_FIXED] * 6
        joint_axes   = [[0, 0, 1]] * 6

        return (masses, collisions, visuals, positions, orientations,
                inertial_pos, inertial_orn, parents, joint_types, joint_axes)

    def _spawn_pedestrians(self):
        n = int(self.cfg['pedestrians']['count'])
        rad = float(self.cfg['pedestrians']['radius'])
        vmin, vmax = self.cfg['pedestrians']['speed_range']
        ps_cfg = self.cfg['pedestrians']['personal_space']
        ps = PersonalSpace(**ps_cfg)

        rng = np.random.default_rng(int(self.cfg.get('seed', 0)))

        # Determine spawn region from world size
        sx, sy = self.cfg['world']['size_xy']

        # Physics collision capsule — shared across all pedestrians
        col = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=rad, height=0.9, physicsClientId=self.client
        )
        # Transparent base visual — actual body is rendered via links
        base_vis = p.createVisualShape(
            p.GEOM_CAPSULE, radius=rad, length=0.9,
            rgbaColor=[0.0, 0.0, 0.0, 0.0], physicsClientId=self.client
        )

        palette = self._CLOTHING_PALETTE

        for i in range(n):
            # Spread pedestrians across the available space
            x = float(rng.uniform(1.5, sx - 1.5))
            y = float(rng.uniform(1.5, sy - 1.5))
            yaw = float(rng.uniform(-math.pi, math.pi))
            speed = float(rng.uniform(vmin, vmax))
            vel = np.array([math.cos(yaw), math.sin(yaw)], dtype=float) * speed

            clothing = palette[i % len(palette)]

            (link_masses, link_col, link_vis, link_pos, link_orn,
             link_iner_pos, link_iner_orn, link_parents,
             link_joint_types, link_joint_axes) = self._make_humanoid_links(clothing, rad)

            body = p.createMultiBody(
                baseMass=70.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=base_vis,
                basePosition=[x, y, 0.9 / 2 + rad],
                baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                linkMasses=link_masses,
                linkCollisionShapeIndices=link_col,
                linkVisualShapeIndices=link_vis,
                linkPositions=link_pos,
                linkOrientations=link_orn,
                linkInertialFramePositions=link_iner_pos,
                linkInertialFrameOrientations=link_iner_orn,
                linkParentIndices=link_parents,
                linkJointTypes=link_joint_types,
                linkJointAxis=link_joint_axes,
                physicsClientId=self.client,
            )

            p.changeDynamics(
                body, -1, lateralFriction=1.0, rollingFriction=0.0,
                physicsClientId=self.client,
            )

            self.pedestrians.append(
                Pedestrian(
                    body_id=body,
                    radius=rad,
                    xy=np.array([x, y], dtype=float),
                    yaw=yaw,
                    vel=vel,
                    ps=ps,
                )
            )

    def reset(self) -> Pose2:
        self.robot_pose = Pose2(float(self.start[0]), float(self.start[1]), float(self.start[2]))
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [self.robot_pose.x, self.robot_pose.y, float(self.robot_base_z)],
            p.getQuaternionFromEuler([0, 0, self.robot_pose.yaw]),
            physicsClientId=self.client
        )
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        return self.robot_pose.copy()

    def _step_pedestrians(self):
        sx, sy = self.cfg['world']['size_xy']
        xmin, ymin, xmax, ymax = 0.5, 0.5, sx - 0.5, sy - 0.5

        for ped in self.pedestrians:
            ped.xy = ped.xy + ped.vel * self.dt

            # bounce on bounds
            if ped.xy[0] < xmin or ped.xy[0] > xmax:
                ped.vel[0] *= -1
                ped.xy[0] = np.clip(ped.xy[0], xmin, xmax)
            if ped.xy[1] < ymin or ped.xy[1] > ymax:
                ped.vel[1] *= -1
                ped.xy[1] = np.clip(ped.xy[1], ymin, ymax)

            ped.yaw = math.atan2(ped.vel[1], ped.vel[0] + 1e-9)

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

    def step(self, v: float, w: float):
        self._step_pedestrians()
        self.apply_control(v, w)
        p.stepSimulation(physicsClientId=self.client)

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