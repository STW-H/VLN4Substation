"""A* over grid position and robot heading with a region of terminal poses."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import cv2
import numpy as np


State = tuple[int, int, int]
DirectionField = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class PoseAStarConfig:
    cost_weight: float = 1.0
    heuristic_weight: float = 1.0
    rotation_cost_per_bin: float = 0.35
    lateral_motion_weight: float = 0.25
    tilt_cost_weight: float = 1.0
    min_traversal_cost: float = 1.0e-6
    allow_diagonal: bool = True
    preferred_path_direction_reward: float = 0.0
    preferred_path_reverse_penalty: float = 0.0
    path_turn_cost_weight: float = 0.0
    max_path_turn_deg: float = 180.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_path_turn_deg <= 180.0:
            raise ValueError("max_path_turn_deg must be in [0, 180]")
        if self.path_turn_cost_weight < 0.0:
            raise ValueError("path_turn_cost_weight cannot be negative")
        if self.preferred_path_direction_reward < 0.0:
            raise ValueError("preferred_path_direction_reward cannot be negative")
        if self.preferred_path_reverse_penalty < 0.0:
            raise ValueError("preferred_path_reverse_penalty cannot be negative")


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


@dataclass(frozen=True)
class HierarchicalPoseAStarResult:
    pose_result: PoseAStarResult
    coarse_path_rc: list[tuple[int, int]]
    corridor_mask: np.ndarray
    used_corridor_radius_m: float

def region_astar_path(
    traversable_mask: np.ndarray,
    cost_map: np.ndarray,
    start_rc: tuple[int, int],
    goal_positions: Iterable[tuple[int, int]],
    *,
    cost_weight: float = 1.0,
    resolution_m: float = 1.0,
    goal_terminal_costs: dict[tuple[int, int], float] | None = None,
    preferred_path_direction: DirectionField | None = None,
    direction_reward_weight: float = 0.0,
    reverse_penalty_weight: float = 0.0,
    turn_cost_weight: float = 0.0,
    max_turn_deg: float = 180.0,
) -> list[tuple[int, int]]:
    """Fast two-dimensional region-goal A* used to construct a refinement corridor."""
    if not 0.0 <= max_turn_deg <= 180.0:
        raise ValueError("max_turn_deg must be in [0, 180]")
    height, width = traversable_mask.shape
    goals = set(goal_positions)
    if not goals:
        return []
    start_position = (int(start_rc[0]), int(start_rc[1]))
    if traversable_mask[start_position] == 0:
        raise ValueError(f"Start position is not traversable: {start_position}")
    goal_mask = np.zeros((height, width), dtype=np.uint8)
    for row, col in goals:
        goal_mask[row, col] = 1
    distance = cv2.distanceTransform((goal_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    finite_costs = cost_map[np.isfinite(cost_map) & (traversable_mask > 0)]
    min_map_cost = float(finite_costs.min()) if len(finite_costs) else 0.0
    lower_step_cost = float(resolution_m) * max(
        1.0 + cost_weight * max(min_map_cost, 0.0) - direction_reward_weight,
        1.0e-6,
    )
    offsets = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    ]
    motion_lookup = {(dr, dc): index for index, (dr, dc, _) in enumerate(offsets)}
    turn_angles = _motion_turn_angles(offsets)
    start = start_position
    heap: list[tuple[float, int, tuple[int, int]]] = [
        (float(distance[start_position]) * lower_step_cost, 0, start)
    ]
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
        previous_motion = _incoming_motion_index_2d(
            current, came_from, motion_lookup
        )
        for motion_index, (dr, dc, step) in enumerate(offsets):
            turn_angle = (
                0.0 if previous_motion < 0
                else float(turn_angles[previous_motion, motion_index])
            )
            if turn_angle > math.radians(max_turn_deg) + 1.0e-9:
                continue
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width or traversable_mask[nr, nc] == 0:
                continue
            if dr and dc and (traversable_mask[row + dr, col] == 0 or traversable_mask[row, col + dc] == 0):
                continue
            traversal = max(float(cost_map[nr, nc]), 0.0)
            direction_cost = _preferred_path_direction_cost(
                preferred_path_direction,
                nr,
                nc,
                dr,
                dc,
                direction_reward_weight,
                reverse_penalty_weight,
            )
            step_factor = max(
                1.0
                + cost_weight * traversal
                + direction_cost,
                1.0e-6,
            )
            turn_cost = float(turn_cost_weight) * (1.0 - math.cos(turn_angle))
            tentative = (
                g_score[current]
                + step * float(resolution_m) * step_factor
                + turn_cost
            )
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


def _reconstruct(
    came_from: dict[State, State], state: State
) -> list[State]:
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


def _preferred_path_direction_cost(
    direction_field: DirectionField | None,
    row: int,
    col: int,
    dr: int,
    dc: int,
    reward_weight: float,
    reverse_penalty_weight: float,
) -> float:
    """Return reverse penalty minus forward reward on a directed path."""
    if direction_field is None or (
        reward_weight <= 0.0 and reverse_penalty_weight <= 0.0
    ):
        return 0.0
    step = math.hypot(float(dr), float(dc))
    if step <= 0.0:
        return 0.0
    move_x = float(dc) / step
    move_y = float(-dr) / step
    direction_x, direction_y = direction_field
    alignment = (
        move_x * float(direction_x[row, col])
        + move_y * float(direction_y[row, col])
    )
    forward_reward = float(reward_weight) * max(0.0, alignment)
    reverse_penalty = float(reverse_penalty_weight) * max(0.0, -alignment)
    return reverse_penalty - forward_reward


def _motion_turn_angles(
    motions: list[tuple[int, int, float]],
) -> np.ndarray:
    vectors = np.asarray([(dr, dc) for dr, dc, _ in motions], dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.clip(vectors @ vectors.T, -1.0, 1.0)
    return np.arccos(cosine)


def _incoming_motion_index(
    state: State,
    came_from: dict[State, State],
    motion_lookup: dict[tuple[int, int], int],
) -> int:
    """Find the last translation before this state, skipping in-place rotations."""
    current = state
    while current in came_from:
        previous = came_from[current]
        delta = (current[0] - previous[0], current[1] - previous[1])
        if delta != (0, 0):
            return motion_lookup.get(delta, -1)
        current = previous
    return -1


def _incoming_motion_index_2d(
    state: tuple[int, int],
    came_from: dict[tuple[int, int], tuple[int, int]],
    motion_lookup: dict[tuple[int, int], int],
) -> int:
    if state not in came_from:
        return -1
    previous = came_from[state]
    delta = (state[0] - previous[0], state[1] - previous[1])
    return motion_lookup.get(delta, -1)


def pose_region_astar_search(
    pose_free_masks: np.ndarray,
    cost_map: np.ndarray,
    start_state: State,
    goal_terminal_costs: dict[State, float],
    config: PoseAStarConfig,
    search_mask: np.ndarray | None = None,
    resolution_m: float = 1.0,
    preferred_path_direction: DirectionField | None = None,
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
    motion_lookup = {
        (dr, dc): index for index, (dr, dc, _) in enumerate(translations)
    }
    turn_angles = _motion_turn_angles(translations)

    g_score: dict[State, float] = {start_state: 0.0}
    came_from: dict[State, State] = {}
    closed: set[State] = set()
    heap: list[tuple[float, int, State]] = []
    counter = 0
    finite_costs = cost_map[np.isfinite(cost_map)]
    min_map_cost = float(finite_costs.min()) if len(finite_costs) else 0.0
    heuristic_scale = float(resolution_m) * max(
        1.0
        + config.cost_weight * max(min_map_cost, 0.0)
        - config.preferred_path_direction_reward,
        1.0e-6,
    )
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
        previous_motion = _incoming_motion_index(
            current, came_from, motion_lookup
        )

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
        for motion_index, (dr, dc, step) in enumerate(translations):
            turn_angle = (
                0.0 if previous_motion < 0
                else float(turn_angles[previous_motion, motion_index])
            )
            if turn_angle > math.radians(config.max_path_turn_deg) + 1.0e-9:
                continue
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
            direction_cost = _preferred_path_direction_cost(
                preferred_path_direction,
                nr,
                nc,
                dr,
                dc,
                config.preferred_path_direction_reward,
                config.preferred_path_reverse_penalty,
            )
            step_factor = max(
                1.0
                + config.cost_weight * traversal
                + lateral_penalty
                + direction_cost,
                1.0e-6,
            )
            turn_cost = config.path_turn_cost_weight * (
                1.0 - math.cos(turn_angle)
            )
            tentative = (
                current_g
                + step * float(resolution_m) * step_factor
                + turn_cost
            )
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


def hierarchical_pose_region_astar(
    pose_free_masks: np.ndarray,
    cost_map: np.ndarray,
    start_state: State,
    goal_terminal_costs: dict[State, float],
    config: PoseAStarConfig,
    *,
    resolution_m: float,
    corridor_radius_m: float,
    max_corridor_radius_m: float,
    preferred_path_direction: DirectionField | None = None,
) -> HierarchicalPoseAStarResult:
    """Plan a coarse region path, then refine position and heading in an expanding corridor."""
    if corridor_radius_m <= 0 or max_corridor_radius_m < corridor_radius_m:
        raise ValueError("Invalid hierarchical corridor radius range")
    position_free = np.any(np.asarray(pose_free_masks) > 0, axis=0).astype(np.uint8)
    position_terminal_costs: dict[tuple[int, int], float] = {}
    for state, terminal_cost in goal_terminal_costs.items():
        position = (state[0], state[1])
        position_terminal_costs[position] = min(
            position_terminal_costs.get(position, math.inf), float(terminal_cost)
        )
    coarse_path = region_astar_path(
        position_free,
        cost_map,
        (start_state[0], start_state[1]),
        position_terminal_costs,
        cost_weight=config.cost_weight,
        resolution_m=resolution_m,
        goal_terminal_costs=position_terminal_costs,
        preferred_path_direction=preferred_path_direction,
        direction_reward_weight=config.preferred_path_direction_reward,
        reverse_penalty_weight=config.preferred_path_reverse_penalty,
        turn_cost_weight=config.path_turn_cost_weight,
        max_turn_deg=config.max_path_turn_deg,
    )
    empty_corridor = np.zeros(position_free.shape, dtype=np.uint8)
    if not coarse_path:
        return HierarchicalPoseAStarResult(
            PoseAStarResult([], math.inf, math.inf, math.inf, 0),
            [], empty_corridor, float(corridor_radius_m),
        )

    radius = float(corridor_radius_m)
    last_result = PoseAStarResult([], math.inf, math.inf, math.inf, 0)
    last_corridor = empty_corridor
    while radius <= float(max_corridor_radius_m) + 1.0e-9:
        last_corridor = path_corridor_mask(
            position_free.shape,
            coarse_path,
            max(1, int(round(radius / resolution_m))),
        )
        corridor_goals = {
            state: cost
            for state, cost in goal_terminal_costs.items()
            if last_corridor[state[0], state[1]] > 0
        }
        last_result = pose_region_astar_search(
            pose_free_masks,
            cost_map,
            start_state,
            corridor_goals,
            config,
            search_mask=last_corridor,
            resolution_m=resolution_m,
            preferred_path_direction=preferred_path_direction,
        )
        if last_result.found:
            return HierarchicalPoseAStarResult(last_result, coarse_path, last_corridor, radius)
        radius *= 2.0
    return HierarchicalPoseAStarResult(
        last_result, coarse_path, last_corridor, min(radius / 2.0, float(max_corridor_radius_m))
    )
