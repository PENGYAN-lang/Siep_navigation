"""museum_builder.py – Build a museum-style 3D environment in PyBullet.

Inspired by the layout of Nanjing Museum:
  Entrance hall → main gallery → wing rooms with connecting corridors.

Overall footprint: ~40 m × 35 m.

Room layout
-----------
  entrance_hall : x  8–20,  y  0– 8   (bottom-centre entrance, 12 m × 8 m)
  main_gallery  : x  4–24,  y  8–26   (large central gallery, ~20 m × 18 m)
  left_corridor : x  0– 4,  y  8–26   (wide left passage, 4 m wide)
  wing_room     : x 24–40,  y 10–26   (right wing, 16 m × 16 m)
  top_corridor  : x  4–40,  y 26–35   (upper passage, 36 m × 9 m)
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import pybullet as p

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WALL_HEIGHT: float = 2.0
WALL_HALF_T: float = 0.15          # half-thickness of every wall segment
WALL_COLOR: list[float] = [0.85, 0.82, 0.78, 1.0]   # museum beige
PEDESTAL_COLOR: list[float] = [0.55, 0.45, 0.35, 1.0]  # dark wood brown
PEDESTAL_HALF_EXTENTS: list[float] = [0.25, 0.25, 0.4]  # 0.5 m × 0.5 m × 0.8 m tall
DOORWAY_HALF_GAP: float = 3.0      # half of the 6.0 m doorway opening

# Pre-defined exhibit positions (x, y) – used as pedestrian waypoints too.
# Reduced set for the expanded scene (8 pedestals, more spread out).
EXHIBIT_POSITIONS: list[tuple[float, float]] = [
    # main gallery (4)
    (10.0, 14.0), (16.0, 14.0),
    (10.0, 21.0), (16.0, 21.0),
    # wing room (3)
    (28.0, 14.0), (34.0, 14.0), (30.0, 20.0),
    # entrance hall (1)
    (14.0,  4.0),
]

ROOM_BOUNDS: dict[str, list[float]] = {
    "entrance_hall": [ 8.0,  0.0, 20.0,  8.0],
    "main_gallery":  [ 4.0,  8.0, 24.0, 26.0],
    "left_corridor": [ 0.0,  8.0,  4.0, 26.0],
    "wing_room":     [24.0, 10.0, 40.0, 26.0],
    "top_corridor":  [ 4.0, 26.0, 40.0, 35.0],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_wall(
    client: int,
    cx: float,
    cy: float,
    half_w: float,
    half_d: float,
    height: float = WALL_HEIGHT,
) -> int:
    """Create a static box wall in *client* and return its body ID.

    Parameters
    ----------
    client:
        PyBullet physics client ID.
    cx, cy:
        Centre position in the horizontal plane.
    half_w:
        Half-extent along the X axis.
    half_d:
        Half-extent along the Y axis.
    height:
        Full wall height (box half-extent = height / 2).

    Returns
    -------
    int
        PyBullet body ID of the created wall.
    """
    col_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half_w, half_d, height / 2.0],
        physicsClientId=client,
    )
    vis_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half_w, half_d, height / 2.0],
        rgbaColor=WALL_COLOR,
        physicsClientId=client,
    )
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[cx, cy, height / 2.0],
        physicsClientId=client,
    )
    return body_id


def _create_pedestal(
    client: int,
    cx: float,
    cy: float,
) -> int:
    """Create a static exhibit pedestal and return its body ID.

    Parameters
    ----------
    client:
        PyBullet physics client ID.
    cx, cy:
        Centre position in the horizontal plane.

    Returns
    -------
    int
        PyBullet body ID of the created pedestal.
    """
    he = PEDESTAL_HALF_EXTENTS
    col_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=he,
        physicsClientId=client,
    )
    vis_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=he,
        rgbaColor=PEDESTAL_COLOR,
        physicsClientId=client,
    )
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[cx, cy, he[2]],
        physicsClientId=client,
    )
    return body_id


def _wall_with_doorway(
    client: int,
    *,
    fixed_coord: float,
    span_min: float,
    span_max: float,
    door_centre: float,
    axis: str,
) -> list[int]:
    """Build a wall segment that has a 2 m doorway cut out of it.

    The wall runs along one axis and is split into two halves around the
    doorway opening.

    Parameters
    ----------
    client:
        PyBullet physics client ID.
    fixed_coord:
        The coordinate value of the wall's centre on the *perpendicular* axis.
    span_min, span_max:
        Start and end of the wall along the *parallel* axis.
    door_centre:
        Centre of the doorway gap along the parallel axis.
    axis:
        ``'x'`` → wall runs east–west (parallel to X); ``'y'`` → north–south.

    Returns
    -------
    list[int]
        Body IDs of the two wall pieces (left/right or bottom/top of gap).
    """
    gap_lo = door_centre - DOORWAY_HALF_GAP
    gap_hi = door_centre + DOORWAY_HALF_GAP

    ids: list[int] = []

    for seg_min, seg_max in [(span_min, gap_lo), (gap_hi, span_max)]:
        if seg_max <= seg_min:
            continue  # degenerate segment, skip
        half_span = (seg_max - seg_min) / 2.0
        centre_along = (seg_min + seg_max) / 2.0
        if axis == "x":
            # wall parallel to X → half_w along X, thin along Y
            ids.append(
                _create_wall(client, centre_along, fixed_coord, half_span, WALL_HALF_T)
            )
        else:
            # wall parallel to Y → thin along X, half_w along Y
            ids.append(
                _create_wall(client, fixed_coord, centre_along, WALL_HALF_T, half_span)
            )

    return ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _wall_aabbs_with_doorway(
    fixed_coord: float,
    span_min: float,
    span_max: float,
    door_centre: float,
    axis: str,
) -> list[tuple[float, float, float, float]]:
    """Compute AABBs for a wall-with-doorway without needing PyBullet.

    Mirrors the geometry of :func:`_wall_with_doorway` exactly.

    Returns
    -------
    list of (xmin, ymin, xmax, ymax) AABBs
    """
    gap_lo = door_centre - DOORWAY_HALF_GAP
    gap_hi = door_centre + DOORWAY_HALF_GAP
    aabbs: list[tuple[float, float, float, float]] = []

    for seg_min, seg_max in [(span_min, gap_lo), (gap_hi, span_max)]:
        if seg_max <= seg_min:
            continue
        half_span = (seg_max - seg_min) / 2.0
        centre_along = (seg_min + seg_max) / 2.0
        if axis == "x":
            cx, cy, hw, hd = centre_along, fixed_coord, half_span, WALL_HALF_T
        else:
            cx, cy, hw, hd = fixed_coord, centre_along, WALL_HALF_T, half_span
        aabbs.append((cx - hw, cy - hd, cx + hw, cy + hd))

    return aabbs


def build_wall_aabbs() -> list[tuple[float, float, float, float]]:
    """Return axis-aligned bounding boxes for all museum walls.

    Does **not** require PyBullet — purely geometric.  Mirrors the wall
    layout of :func:`build_museum_world` exactly so that the FSM
    pedestrian collision check stays in sync with the 3-D scene.

    Museum footprint: x ∈ [0, 40], y ∈ [0, 35].

    Returns
    -------
    list of ``(xmin, ymin, xmax, ymax)`` tuples
    """
    aabbs: list[tuple[float, float, float, float]] = []

    def solid(cx: float, cy: float, hw: float, hd: float) -> None:
        aabbs.append((cx - hw, cy - hd, cx + hw, cy + hd))

    # ── Outer perimeter ──────────────────────────────────────────────
    # South y = 0, door at x = 14
    aabbs.extend(_wall_aabbs_with_doorway(0.0, 0.0, 40.0, 14.0, "x"))
    # North y = 35
    solid(20.0, 35.0, 20.0, WALL_HALF_T)
    # West x = 0
    solid(0.0, 17.5, WALL_HALF_T, 17.5)
    # East x = 40
    solid(40.0, 17.5, WALL_HALF_T, 17.5)

    # ── Entrance hall (y = 8 internal wall + side walls) ─────────────
    # North wall of entrance: y = 8, x ∈ [8, 20], door at x = 14
    aabbs.extend(_wall_aabbs_with_doorway(8.0, 8.0, 20.0, 14.0, "x"))
    # West side  x = 8,  y ∈ [0, 8]
    solid(8.0, 4.0, WALL_HALF_T, 4.0)
    # East side  x = 20, y ∈ [0, 8]
    solid(20.0, 4.0, WALL_HALF_T, 4.0)

    # ── Left corridor ↔ main gallery  (x = 4, y ∈ [8, 26]) ──────────
    # Door at y = 17
    aabbs.extend(_wall_aabbs_with_doorway(4.0, 8.0, 26.0, 17.0, "y"))

    # ── Main gallery ↔ wing room  (x = 24, y ∈ [10, 26]) ────────────
    # Door at y = 18
    aabbs.extend(_wall_aabbs_with_doorway(24.0, 10.0, 26.0, 18.0, "y"))
    # Wing room south closing wall: y = 10, x ∈ [24, 40]
    solid(32.0, 10.0, 8.0, WALL_HALF_T)

    # ── Top corridor separator ────────────────────────────────────────
    # Main gallery ceiling: y = 26, x ∈ [4, 24], door at x = 14
    aabbs.extend(_wall_aabbs_with_doorway(26.0, 4.0, 24.0, 14.0, "x"))
    # Wing room ceiling: y = 26, x ∈ [24, 40], door at x = 32
    aabbs.extend(_wall_aabbs_with_doorway(26.0, 24.0, 40.0, 32.0, "x"))

    return aabbs


def build_museum_world(client: int, cfg: dict) -> dict:
    """Build a museum-style environment in PyBullet.

    Inspired by Nanjing Museum layout: entrance hall → main gallery →
    wing rooms with connecting corridors.

    Parameters
    ----------
    client : int
        PyBullet physics client ID.
    cfg : dict
        Config dict with a ``'world'`` key containing at minimum
        ``'size_xy'`` (width, height) of the simulated world.

    Returns
    -------
    dict
        ``wall_ids``          – list of wall body IDs.
        ``exhibit_ids``       – list of exhibit pedestal body IDs.
        ``exhibit_positions`` – list of ``(x, y)`` exhibit centres
                                (also usable as pedestrian waypoints).
        ``room_bounds``       – dict mapping room names to
                                ``[x_min, y_min, x_max, y_max]``.
        ``wall_aabbs``        – list of ``(xmin, ymin, xmax, ymax)``
                                wall bounding boxes for collision checks.
    """
    wall_ids: list[int] = []

    # ------------------------------------------------------------------
    # 1. Outer perimeter walls
    #    Museum footprint: x ∈ [0, 40], y ∈ [0, 35]
    # ------------------------------------------------------------------

    # South wall  y = 0  (full width, entrance gap at x = 14)
    wall_ids.extend(
        _wall_with_doorway(
            client,
            fixed_coord=0.0,
            span_min=0.0,
            span_max=40.0,
            door_centre=14.0,
            axis="x",
        )
    )

    # North wall  y = 35  (full width, no gap)
    wall_ids.append(_create_wall(client, 20.0, 35.0, 20.0, WALL_HALF_T))

    # West wall  x = 0  (full height)
    wall_ids.append(_create_wall(client, 0.0, 17.5, WALL_HALF_T, 17.5))

    # East wall  x = 40  (full height)
    wall_ids.append(_create_wall(client, 40.0, 17.5, WALL_HALF_T, 17.5))

    # ------------------------------------------------------------------
    # 2. Internal wall: entrance hall ↔ main gallery  (y = 8)
    #    Runs x ∈ [8, 20], doorway at x = 14
    # ------------------------------------------------------------------
    wall_ids.extend(
        _wall_with_doorway(
            client,
            fixed_coord=8.0,
            span_min=8.0,
            span_max=20.0,
            door_centre=14.0,
            axis="x",
        )
    )
    # Solid walls closing the sides of the entrance hall pocket
    # west side of entrance hall  x = 8, y ∈ [0, 8]
    wall_ids.append(_create_wall(client, 8.0, 4.0, WALL_HALF_T, 4.0))
    # east side of entrance hall  x = 20, y ∈ [0, 8]
    wall_ids.append(_create_wall(client, 20.0, 4.0, WALL_HALF_T, 4.0))

    # ------------------------------------------------------------------
    # 3. Left corridor ↔ main gallery  (x = 4, y ∈ [8, 26])
    #    Doorway at y = 17
    # ------------------------------------------------------------------
    wall_ids.extend(
        _wall_with_doorway(
            client,
            fixed_coord=4.0,
            span_min=8.0,
            span_max=26.0,
            door_centre=17.0,
            axis="y",
        )
    )

    # ------------------------------------------------------------------
    # 4. Main gallery ↔ wing room  (x = 24, y ∈ [10, 26])
    #    Doorway at y = 18
    # ------------------------------------------------------------------
    wall_ids.extend(
        _wall_with_doorway(
            client,
            fixed_coord=24.0,
            span_min=10.0,
            span_max=26.0,
            door_centre=18.0,
            axis="y",
        )
    )
    # Short south wall closing the wing room pocket below y = 10
    # (x ∈ [24, 40], y = 10)
    wall_ids.append(_create_wall(client, 32.0, 10.0, 8.0, WALL_HALF_T))

    # ------------------------------------------------------------------
    # 5. Main gallery top / top corridor bottom  (y = 26)
    #    Two non-overlapping sections, each with its own doorway:
    #      • x ∈ [4, 24]  – main gallery ceiling,  doorway at x = 14
    #      • x ∈ [24, 40] – wing room ceiling,      doorway at x = 32
    # ------------------------------------------------------------------

    # Main gallery ceiling: x ∈ [4, 24], doorway at x = 14
    wall_ids.extend(
        _wall_with_doorway(
            client,
            fixed_coord=26.0,
            span_min=4.0,
            span_max=24.0,
            door_centre=14.0,
            axis="x",
        )
    )

    # Wing room ceiling: x ∈ [24, 40], doorway at x = 32
    wall_ids.extend(
        _wall_with_doorway(
            client,
            fixed_coord=26.0,
            span_min=24.0,
            span_max=40.0,
            door_centre=32.0,
            axis="x",
        )
    )

    # ------------------------------------------------------------------
    # 6. Exhibit pedestals
    # ------------------------------------------------------------------
    exhibit_ids: list[int] = []
    for ex, ey in EXHIBIT_POSITIONS:
        exhibit_ids.append(_create_pedestal(client, ex, ey))

    return {
        "wall_ids": wall_ids,
        "exhibit_ids": exhibit_ids,
        "exhibit_positions": list(EXHIBIT_POSITIONS),
        "room_bounds": {k: list(v) for k, v in ROOM_BOUNDS.items()},
        "wall_aabbs": build_wall_aabbs(),
    }


def get_museum_spawn_positions(cfg: dict, n: int, seed: int = 42) -> list[list[float]]:
    """Return *n* valid pedestrian spawn positions inside museum rooms.

    Positions are sampled uniformly from the combined area of all rooms,
    excluding a 0.5 m keep-out margin from every wall to avoid spawning
    inside geometry.  Sampling is reproducible via *seed*.

    Parameters
    ----------
    cfg : dict
        Config dict (reserved for future use, e.g. custom margins).
    n : int
        Number of spawn positions to return.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of [x, y]
        Each element is a two-element list ``[x, y]``.

    Raises
    ------
    ValueError
        If the requested number of positions cannot be satisfied after a
        reasonable number of sampling attempts.
    """
    margin: float = 0.5
    rng = np.random.default_rng(seed)

    # Build a list of (shrunken) bounding boxes to sample from, weighted by area.
    boxes: list[tuple[float, float, float, float]] = []
    areas: list[float] = []
    for bounds in ROOM_BOUNDS.values():
        xlo = bounds[0] + margin
        ylo = bounds[1] + margin
        xhi = bounds[2] - margin
        yhi = bounds[3] - margin
        if xhi > xlo and yhi > ylo:
            boxes.append((xlo, ylo, xhi, yhi))
            areas.append((xhi - xlo) * (yhi - ylo))

    total_area = sum(areas)
    weights = [a / total_area for a in areas]

    positions: list[list[float]] = []
    max_attempts = n * 20
    attempts = 0

    while len(positions) < n and attempts < max_attempts:
        # Pick a room proportional to area
        box_idx: int = int(rng.choice(len(boxes), p=weights))
        xlo, ylo, xhi, yhi = boxes[box_idx]
        x = float(rng.uniform(xlo, xhi))
        y = float(rng.uniform(ylo, yhi))
        positions.append([x, y])
        attempts += 1

    if len(positions) < n:
        raise ValueError(
            f"Could not generate {n} spawn positions after {max_attempts} attempts. "
            f"Only {len(positions)} positions found."
        )

    return positions
