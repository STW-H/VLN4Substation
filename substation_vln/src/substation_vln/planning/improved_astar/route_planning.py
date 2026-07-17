"""Helpers for planning a parsed natural-language route."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from substation_vln.planning.common.grid import GridSpec
from substation_vln.planning.improved_astar.goal_pose_region import quantize_heading
from substation_vln.tasks import RoutePlan


PoseState = tuple[int, int, int]


def resolve_route_start(
    start_points: list[dict[str, Any]],
    plan: RoutePlan,
    pose_free: np.ndarray,
    grid: GridSpec,
    yaw_deg: float,
    random_seed: int | None,
) -> tuple[PoseState, dict[str, Any]]:
    heading = quantize_heading(math.radians(yaw_deg), pose_free.shape[0])
    candidates: list[tuple[PoseState, dict[str, Any]]] = []
    for item in start_points:
        col, row = grid.xy_to_grid(np.asarray([item["xy"]], dtype=np.float64))[0]
        state = (int(row), int(col), heading)
        if (
            0 <= state[0] < grid.height
            and 0 <= state[1] < grid.width
            and pose_free[state[2], state[0], state[1]] > 0
        ):
            candidates.append((state, item))

    if plan.start_point is not None:
        matches = [
            pair for pair in candidates
            if pair[1]["start_point_name"] == plan.start_point
        ]
        if len(matches) != 1:
            raise ValueError(
                f"指定起点 {plan.start_point!r} 不存在、名称不唯一或当前航向下发生碰撞"
            )
        return matches[0]
    if not candidates:
        raise ValueError("没有当前航向下可用的机器人起始点")
    rng = np.random.default_rng(random_seed)
    return candidates[int(rng.integers(0, len(candidates)))]


def waypoint_goal_states(
    waypoint: dict[str, Any],
    pose_free: np.ndarray,
    grid: GridSpec,
    tolerance_m: float,
    terminal_distance_weight: float,
) -> dict[PoseState, float]:
    target_xy = np.asarray(waypoint["xy"], dtype=np.float64)
    free_rows, free_cols = np.nonzero(np.any(pose_free > 0, axis=0))
    xs, ys = grid.grid_to_xy(free_cols, free_rows)
    distances = np.hypot(xs - target_xy[0], ys - target_xy[1])
    selected = distances <= float(tolerance_m)
    if not np.any(selected):
        if not len(distances):
            return {}
        selected[np.argmin(distances)] = True

    goals: dict[PoseState, float] = {}
    for row, col, distance in zip(
        free_rows[selected], free_cols[selected], distances[selected], strict=True
    ):
        for heading in np.flatnonzero(pose_free[:, row, col] > 0):
            goals[(int(row), int(col), int(heading))] = (
                float(terminal_distance_weight) * float(distance)
            )
    return goals


def equipment_goal_states(
    equipment_index: int,
    region_data: np.lib.npyio.NpzFile,
    tilt_cost_weight: float,
) -> tuple[dict[PoseState, float], dict[PoseState, int]]:
    indices = np.flatnonzero(
        region_data["goal_equipment_index"] == int(equipment_index)
    )
    goals: dict[PoseState, float] = {}
    lookup: dict[PoseState, int] = {}
    for global_index in indices:
        state = (
            int(region_data["goal_rows"][global_index]),
            int(region_data["goal_cols"][global_index]),
            int(region_data["goal_heading_bins"][global_index]),
        )
        terminal = (
            float(tilt_cost_weight)
            * float(region_data["goal_tilt_costs"][global_index])
        )
        if state not in goals or terminal < goals[state]:
            goals[state] = terminal
            lookup[state] = int(global_index)
    return goals, lookup


def pose_state_payload(
    grid: GridSpec,
    state: PoseState,
    heading_bins: int,
) -> dict[str, Any]:
    x, y = grid.grid_to_xy(np.asarray([state[1]]), np.asarray([state[0]]))
    yaw = state[2] * 2.0 * math.pi / heading_bins
    return {
        "grid_rc_heading": [int(state[0]), int(state[1]), int(state[2])],
        "xy": [float(x[0]), float(y[0])],
        "yaw_rad": float(yaw),
        "yaw_deg": float(math.degrees(yaw)),
    }
