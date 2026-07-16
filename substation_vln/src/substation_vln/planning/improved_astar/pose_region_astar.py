"""A* over grid position and robot heading with a region of terminal poses."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import cv2
import numpy as np


State = tuple[int, int, int]


@dataclass(frozen=True)
class PoseAStarConfig:
    cost_weight: float = 1.0
    heuristic_weight: float = 1.0
    rotation_cost_per_bin: float = 0.35
    lateral_motion_weight: float = 0.25
    tilt_cost_weight: float = 1.0
    min_traversal_cost: float = 1.0e-6
    allow_diagonal: bool = True


@dataclass(frozen=True)
class PoseAStarResult:
    path_states: list[State]
    total_cost: float
    path_cost: float
    terminal_goal_cost: float
    expanded_nodes: int

    @property
    def found(self) -> bool:
        return bool(self.path_states)

def region_astar_path(
    traversable_mask: np.ndarray,
    cost_map: np.ndarray,
    start_rc: tuple[int, int],
    goal_positions: Iterable[tuple[int, int]],
    *,
    cost_weight: float = 1.0,
    resolution_m: float = 1.0,
    goal_terminal_costs: dict[tuple[int, int], float] | None = None,
) -> list[tuple[int, int]]:
    """Fast two-dimensional region-goal A* used to construct a refinement corridor."""
    height, width = traversable_mask.shape
    goals = set(goal_positions)
    if not goals:
        return []
    start = (int(start_rc[0]), int(start_rc[1]))
    if traversable_mask[start] == 0:
        raise ValueError(f"Start position is not traversable: {start}")
    goal_mask = np.zeros((height, width), dtype=np.uint8)
    for row, col in goals:
        goal_mask[row, col] = 1
    distance = cv2.distanceTransform((goal_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    finite_costs = cost_map[np.isfinite(cost_map) & (traversable_mask > 0)]
    min_map_cost = float(finite_costs.min()) if len(finite_costs) else 0.0
    lower_step_cost = float(resolution_m) * (1.0 + cost_weight * max(min_map_cost, 0.0))
    offsets = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    ]
    heap: list[tuple[float, int, tuple[int, int]]] = [(float(distance[start]) * lower_step_cost, 0, start)]
    g_score = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()
    counter = 0
    best_goal: tuple[int, int] | None = None
    best_total = math.inf
    while heap:
        priority, _, current = heapq.heappop(heap)
        if current in closed:
            continue
        if best_goal is not None and priority >= best_total:
            break
        if current in goals:
            terminal = 0.0 if goal_terminal_costs is None else float(goal_terminal_costs.get(current, 0.0))
            total = g_score[current] + terminal
            if total < best_total:
                best_goal = current
                best_total = total
        closed.add(current)
        row, col = current
        for dr, dc, step in offsets:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width or traversable_mask[nr, nc] == 0:
                continue
            if dr and dc and (traversable_mask[row + dr, col] == 0 or traversable_mask[row, col + dc] == 0):
                continue
            traversal = max(float(cost_map[nr, nc]), 0.0)
            tentative = g_score[current] + step * float(resolution_m) * (1.0 + cost_weight * traversal)
            neighbor = (nr, nc)
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            counter += 1
            priority = tentative + float(distance[nr, nc]) * lower_step_cost
            heapq.heappush(heap, (priority, counter, neighbor))
    if best_goal is None:
        return []
    path = [best_goal]
    current = best_goal
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_corridor_mask(shape: tuple[int, int], path_rc: list[tuple[int, int]], radius_cells: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not path_rc:
        return mask
    points = np.asarray([(col, row) for row, col in path_rc], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [points], False, 1, thickness=1, lineType=cv2.LINE_8)
    radius = max(0, int(radius_cells))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel)
    return mask


def _reconstruct(came_from: dict[State, State], state: State) -> list[State]:
    path = [state]
    while state in came_from:
        state = came_from[state]
        path.append(state)
    path.reverse()
    return path


def _goal_distance_cells(shape: tuple[int, int], goals: Iterable[State]) -> np.ndarray:
    goal_mask = np.zeros(shape, dtype=np.uint8)
    for row, col, _ in goals:
        goal_mask[row, col] = 1
    if not np.any(goal_mask):
        raise ValueError("Goal pose set is empty")
    return cv2.distanceTransform((goal_mask == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)


def pose_region_astar_search(
    pose_free_masks: np.ndarray,
    cost_map: np.ndarray,
    start_state: State,
    goal_terminal_costs: dict[State, float],
    config: PoseAStarConfig,
    search_mask: np.ndarray | None = None,
    resolution_m: float = 1.0,
) -> PoseAStarResult:
    bins, height, width = pose_free_masks.shape
    start_row, start_col, start_heading = start_state
    if not (0 <= start_row < height and 0 <= start_col < width and 0 <= start_heading < bins):
        raise ValueError(f"Start state is outside state space: {start_state}")
    if pose_free_masks[start_heading, start_row, start_col] == 0:
        raise ValueError(f"Robot footprint collides at start state: {start_state}")
    if search_mask is not None and search_mask[start_row, start_col] == 0:
        raise ValueError("Start state is outside the pose-refinement corridor")
    if not goal_terminal_costs:
        return PoseAStarResult([], math.inf, math.inf, math.inf, 0)

    goal_distance = _goal_distance_cells((height, width), goal_terminal_costs)
    translations = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if config.allow_diagonal:
        diagonal = math.sqrt(2.0)
        translations += [(-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal)]

    g_score: dict[State, float] = {start_state: 0.0}
    came_from: dict[State, State] = {}
    closed: set[State] = set()
    heap: list[tuple[float, int, State]] = []
    counter = 0
    finite_costs = cost_map[np.isfinite(cost_map)]
    min_map_cost = float(finite_costs.min()) if len(finite_costs) else 0.0
    heuristic_scale = float(resolution_m) * (1.0 + config.cost_weight * max(min_map_cost, 0.0))
    start_h = float(goal_distance[start_row, start_col]) * heuristic_scale
    heapq.heappush(heap, (config.heuristic_weight * start_h, counter, start_state))
    expanded = 0
    best_goal: State | None = None
    best_total = math.inf
    best_terminal = math.inf

    while heap:
        priority, _, current = heapq.heappop(heap)
        if current in closed:
            continue
        if best_goal is not None and priority >= best_total:
            break
        current_g = g_score[current]
        if current in goal_terminal_costs:
            # Goal values are already weighted terminal costs. Keeping the
            # historical argument/result names avoids breaking stored callers.
            terminal = float(goal_terminal_costs[current])
            total = current_g + terminal
            if total < best_total:
                best_goal = current
                best_total = total
                best_terminal = terminal
        closed.add(current)
        expanded += 1
        row, col, heading = current

        for next_heading in ((heading - 1) % bins, (heading + 1) % bins):
            if pose_free_masks[next_heading, row, col] == 0:
                continue
            neighbor = (row, col, next_heading)
            tentative = current_g + config.rotation_cost_per_bin
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            counter += 1
            h = float(goal_distance[row, col]) * heuristic_scale
            heapq.heappush(heap, (tentative + config.heuristic_weight * h, counter, neighbor))

        yaw = heading * (2.0 * math.pi / bins)
        for dr, dc, step in translations:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if search_mask is not None and search_mask[nr, nc] == 0:
                continue
            if pose_free_masks[heading, nr, nc] == 0 or not np.isfinite(cost_map[nr, nc]):
                continue
            if dr != 0 and dc != 0:
                if pose_free_masks[heading, row + dr, col] == 0 or pose_free_masks[heading, row, col + dc] == 0:
                    continue
            movement_yaw = math.atan2(float(-dr), float(dc))
            alignment = abs(math.cos(movement_yaw - yaw))
            lateral_penalty = config.lateral_motion_weight * (1.0 - alignment)
            traversal = max(float(cost_map[nr, nc]), config.min_traversal_cost)
            tentative = current_g + step * float(resolution_m) * (1.0 + config.cost_weight * traversal + lateral_penalty)
            neighbor = (nr, nc, heading)
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            counter += 1
            h = float(goal_distance[nr, nc]) * heuristic_scale
            heapq.heappush(heap, (tentative + config.heuristic_weight * h, counter, neighbor))

    if best_goal is None:
        return PoseAStarResult([], math.inf, math.inf, math.inf, expanded)
    return PoseAStarResult(
        path_states=_reconstruct(came_from, best_goal),
        total_cost=float(best_total),
        path_cost=float(g_score[best_goal]),
        terminal_goal_cost=float(best_terminal),
        expanded_nodes=expanded,
    )
