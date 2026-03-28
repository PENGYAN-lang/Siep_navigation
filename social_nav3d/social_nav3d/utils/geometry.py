from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


@dataclass
class Pose2:
    x: float
    y: float
    yaw: float

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def copy(self) -> 'Pose2':
        return Pose2(self.x, self.y, self.yaw)


def integrate_diff_drive(p: Pose2, v: float, w: float, dt: float) -> Pose2:
    """Unicycle model integration."""
    yaw2 = wrap_pi(p.yaw + w * dt)
    return Pose2(
        p.x + v * math.cos(p.yaw) * dt,
        p.y + v * math.sin(p.yaw) * dt,
        yaw2,
    )
