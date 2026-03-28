from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PersonalSpace:
    """Anisotropic Gaussian personal space in pedestrian body frame.

    sigma_front/side/back are in meters (std dev). Smaller => stronger penalty near body.
    """

    sigma_front: float = 1.6
    sigma_side: float = 0.9
    sigma_back: float = 0.7


def anisotropic_gaussian_cost(
    robot_xy: np.ndarray,
    ped_xy: np.ndarray,
    ped_yaw: float,
    ps: PersonalSpace,
) -> float:
    """Return a smooth penalty for robot being near pedestrian.

    We express delta in pedestrian frame; use different sigma for front vs back.
    """
    d = robot_xy - ped_xy
    c, s = math.cos(-ped_yaw), math.sin(-ped_yaw)
    dx = c * d[0] - s * d[1]
    dy = s * d[0] + c * d[1]

    sigma_x = ps.sigma_front if dx >= 0 else ps.sigma_back
    sigma_y = ps.sigma_side

    # Gaussian (unnormalized) – cost in (0,1]
    ex = (dx * dx) / (2 * sigma_x * sigma_x + 1e-9)
    ey = (dy * dy) / (2 * sigma_y * sigma_y + 1e-9)
    return float(math.exp(-(ex + ey)))
