"""Baseline grid A* search."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AStarConfig:
    connectivity: int = 8
    cost_weight: float = 1.0
    heuristic_weight: float = 1.0
    min_traversal_cost: float = 1.0e-6


@dataclass(frozen=True)
class AStarResult:
    path_rc: list[tuple[int, int]]
    total_cost: float
    expanded_nodes: int

    @property
    def found(self) -> bool:
        return len(self.path_rc) > 0


def neighbor_offsets(connectivity: int) -> list[tuple[int, int, float]]:
    if connectivity == 4:
        return [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if connectivity == 8:
        diag = math.sqrt(2.0)
        return [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, diag),
            (-1, 1, diag),
            (1, -1, diag),
            (1, 1, diag),
        ]
    raise ValueError(f"Unsupported A* connectivity: {connectivity}")


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    dr = float(a[0] - b[0])
    dc = float(a[1] - b[1])
    return math.hypot(dr, dc)


def reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_search(
    free_space_mask: np.ndarray,
    cost_map: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    config: AStarConfig,
) -> AStarResult:
    height, width = free_space_mask.shape
    for name, node in (("start", start_rc), ("goal", goal_rc)):
        row, col = node
        if row < 0 or row >= height or col < 0 or col >= width:
            raise ValueError(f"{name} grid cell is outside map: row={row}, col={col}")
        if free_space_mask[row, col] == 0 or not np.isfinite(cost_map[row, col]):
            raise ValueError(f"{name} grid cell is not traversable: row={row}, col={col}")

    offsets = neighbor_offsets(config.connectivity)
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_rc: 0.0}
    closed: set[tuple[int, int]] = set()
    heapq.heappush(open_heap, (config.heuristic_weight * heuristic(start_rc, goal_rc), counter, start_rc))
    expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_rc:
            return AStarResult(
                path_rc=reconstruct_path(came_from, current),
                total_cost=float(g_score[current]),
                expanded_nodes=expanded,
            )

        closed.add(current)
        expanded += 1
        row, col = current

        for dr, dc, step_length in offsets:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if free_space_mask[nr, nc] == 0 or not np.isfinite(cost_map[nr, nc]):
                continue

            traversal_cost = max(float(cost_map[nr, nc]), config.min_traversal_cost)
            tentative_g = g_score[current] + step_length * (1.0 + config.cost_weight * traversal_cost)
            neighbor = (nr, nc)
            if tentative_g >= g_score.get(neighbor, math.inf):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            counter += 1
            priority = tentative_g + config.heuristic_weight * heuristic(neighbor, goal_rc)
            heapq.heappush(open_heap, (priority, counter, neighbor))

    return AStarResult(path_rc=[], total_cost=math.inf, expanded_nodes=expanded)


def path_length_m(path_xy: Iterable[tuple[float, float]]) -> float:
    pts = np.asarray(list(path_xy), dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
