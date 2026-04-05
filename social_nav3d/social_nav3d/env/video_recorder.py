"""Multi-view 3D video recorder using PyBullet's built-in camera."""
from __future__ import annotations

import warnings
from enum import Enum
from math import cos, pi, sin
from typing import TYPE_CHECKING, Optional

import numpy as np
import pybullet as p

if TYPE_CHECKING:
    from typing import Any


class CameraMode(Enum):
    FOLLOW = "follow"
    OVERHEAD = "overhead"
    CINEMATIC = "cinematic"


class VideoRecorder3D:
    """Multi-view 3D video recorder using PyBullet's built-in camera.

    Supports three camera modes:
    1. FOLLOW: camera follows robot from behind and above
    2. OVERHEAD: fixed bird's-eye view of entire scene
    3. CINEMATIC: slow orbit around the scene center

    Outputs MP4 video via imageio-ffmpeg.

    Usage
    -----
    recorder = VideoRecorder3D(client=sim.client, cfg=cfg, mode='follow', out_path='runs/video.mp4')
    # in sim.step():
    recorder.capture_frame(robot_xy, robot_yaw, step_idx)
    # after episode:
    recorder.save()
    """

    def __init__(
        self,
        client: int,
        cfg: dict[str, Any],
        mode: str = "follow",
        out_path: str = "runs/video.mp4",
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.mode = CameraMode(mode)
        self.out_path = out_path

        self.width = 1280
        self.height = 720
        self.fps = 30

        size_xy = cfg["world"]["size_xy"]
        if isinstance(size_xy, (list, tuple)):
            if len(size_xy) < 2:
                raise ValueError(
                    f"cfg['world']['size_xy'] must have at least 2 elements [sx, sy], got {size_xy!r}"
                )
            self._center_x = float(size_xy[0]) / 2.0
            self._center_y = float(size_xy[1]) / 2.0
        else:
            self._center_x = float(size_xy) / 2.0
            self._center_y = float(size_xy) / 2.0
        self._max_steps: int = cfg["sim"]["max_steps"]

        self._frames: list[np.ndarray] = []

        # Pre-compute fixed overhead view matrix once
        if self.mode is CameraMode.OVERHEAD:
            self._overhead_view = p.computeViewMatrix(
                cameraEyePosition=[self._center_x, self._center_y, 20.0],
                cameraTargetPosition=[self._center_x, self._center_y, 0.0],
                cameraUpVector=[0, 1, 0],
                physicsClientId=self.client,
            )
        else:
            self._overhead_view = None

    def _compute_view_matrix(
        self, robot_xy: tuple[float, float], robot_yaw: float, step_idx: int
    ) -> Any:
        """Return the appropriate PyBullet view matrix for the current mode."""
        if self.mode is CameraMode.FOLLOW:
            rx, ry = robot_xy
            eye_x = rx - 3.0 * cos(robot_yaw)
            eye_y = ry - 3.0 * sin(robot_yaw)
            return p.computeViewMatrix(
                cameraEyePosition=[eye_x, eye_y, 4.0],
                cameraTargetPosition=[rx, ry, 0.8],
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.client,
            )

        if self.mode is CameraMode.OVERHEAD:
            return self._overhead_view

        # CINEMATIC: orbit 360° over the episode
        angle = 2.0 * pi * step_idx / max(self._max_steps, 1)
        cam_x = self._center_x + 15.0 * cos(angle)
        cam_y = self._center_y + 15.0 * sin(angle)
        return p.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, 8.0],
            cameraTargetPosition=[self._center_x, self._center_y, 1.0],
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.client,
        )

    def capture_frame(
        self,
        robot_xy: tuple[float, float],
        robot_yaw: float,
        step_idx: int = 0,
    ) -> None:
        """Capture a single rendered frame and append it to the frame buffer.

        Only every 2nd simulation step is captured to stay near 30 fps when
        the simulator runs at 50 Hz.
        """
        if step_idx % 2 != 0:
            return

        view_matrix = self._compute_view_matrix(robot_xy, robot_yaw, step_idx)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.width / self.height,
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self.client,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            physicsClientId=self.client,
        )

        rgb = np.array(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        self._frames.append(rgb)

    def save(self) -> None:
        """Write accumulated frames to an MP4 file using imageio."""
        if not self._frames:
            warnings.warn("VideoRecorder3D.save() called with no frames – skipping.", stacklevel=2)
            return

        try:
            import imageio  # noqa: PLC0415
        except ImportError:
            warnings.warn(
                "imageio is not installed; cannot save video. "
                "Install it with: pip install imageio imageio-ffmpeg",
                stacklevel=2,
            )
            return

        writer = imageio.get_writer(self.out_path, fps=self.fps, codec="libx264", quality=8)
        for frame in self._frames:
            writer.append_data(frame)
        writer.close()

        print(f"[VideoRecorder3D] Saved {len(self._frames)} frames → {self.out_path}")


def create_recorder(
    client: int,
    cfg: dict[str, Any],
    mode: str,
    out_path: str,
) -> Optional[VideoRecorder3D]:
    """Create a VideoRecorder3D, or return None if imageio is unavailable.

    Parameters
    ----------
    client:
        PyBullet physics client ID.
    cfg:
        Simulation configuration dictionary.
    mode:
        Camera mode string: ``'follow'``, ``'overhead'``, or ``'cinematic'``.
    out_path:
        Destination MP4 file path.
    """
    try:
        import imageio  # noqa: F401, PLC0415
    except ImportError:
        warnings.warn(
            "imageio not found – video recording disabled. "
            "Install with: pip install imageio imageio-ffmpeg",
            stacklevel=2,
        )
        return None

    return VideoRecorder3D(client=client, cfg=cfg, mode=mode, out_path=out_path)
