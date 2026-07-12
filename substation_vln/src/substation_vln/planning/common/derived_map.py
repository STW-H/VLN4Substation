"""Build derived planning layers and cost maps from base masks."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def distance_to_mask(mask: np.ndarray, resolution_m: float) -> np.ndarray:
    target = (mask > 0).astype(np.uint8)
    inverse = (target == 0).astype(np.uint8)
    return cv2.distanceTransform(inverse, cv2.DIST_L2, 5).astype(np.float32) * float(resolution_m)


def inflate_obstacles(obstacle_mask: np.ndarray, resolution_m: float, inflation_radius_m: float) -> np.ndarray:
    if inflation_radius_m <= 0:
        return (obstacle_mask > 0).astype(np.uint8)
    obstacle_distance = distance_to_mask(obstacle_mask, resolution_m)
    return (obstacle_distance <= float(inflation_radius_m)).astype(np.uint8)


def preferred_path_attraction(preferred_path_mask: np.ndarray, resolution_m: float, sigma_m: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(preferred_path_mask > 0):
        distance = np.full(preferred_path_mask.shape, np.inf, dtype=np.float32)
        attraction = np.zeros(preferred_path_mask.shape, dtype=np.float32)
        return distance, attraction
    distance = distance_to_mask(preferred_path_mask, resolution_m)
    if sigma_m <= 0:
        attraction = (preferred_path_mask > 0).astype(np.float32)
    else:
        attraction = np.exp(-np.square(distance) / (2.0 * sigma_m * sigma_m)).astype(np.float32)
    return distance, attraction


def obstacle_repulsion(distance_to_obstacle_m: np.ndarray, repulsion_radius_m: float, repulsion_weight: float) -> np.ndarray:
    if repulsion_radius_m <= 0 or repulsion_weight <= 0:
        return np.zeros_like(distance_to_obstacle_m, dtype=np.float32)
    repulsion = np.clip((repulsion_radius_m - distance_to_obstacle_m) / repulsion_radius_m, 0.0, 1.0)
    return (repulsion_weight * np.square(repulsion)).astype(np.float32)


def build_derived_layers(base_masks: dict[str, np.ndarray], resolution_m: float, params: dict[str, Any]) -> dict[str, np.ndarray]:
    boundary_mask = base_masks["boundary_mask"].astype(bool)
    obstacle_mask = base_masks["obstacle_mask"].astype(bool)
    preferred_road_mask = base_masks["preferred_road_mask"].astype(bool)
    preferred_path_mask = base_masks["preferred_path_mask"].astype(bool)

    inflated_obstacle_mask = inflate_obstacles(
        obstacle_mask.astype(np.uint8),
        resolution_m,
        float(params["obstacle_inflation_radius_m"]),
    ).astype(bool)
    free_space_mask = boundary_mask & (~inflated_obstacle_mask)
    preferred_road_mask = preferred_road_mask & free_space_mask
    preferred_path_mask = preferred_path_mask & free_space_mask

    distance_to_obstacle_m = distance_to_mask(obstacle_mask.astype(np.uint8), resolution_m)
    distance_to_preferred_path_m, preferred_path_attraction_field = preferred_path_attraction(
        preferred_path_mask.astype(np.uint8),
        resolution_m,
        float(params["preferred_path_sigma_m"]),
    )

    cost_map = np.full(boundary_mask.shape, np.inf, dtype=np.float32)
    cost_map[free_space_mask] = float(params["base_cost"])
    cost_map[preferred_road_mask] = float(params["preferred_road_cost"])
    cost_map[free_space_mask] -= float(params["preferred_path_alpha"]) * preferred_path_attraction_field[free_space_mask]
    cost_map[free_space_mask] += obstacle_repulsion(
        distance_to_obstacle_m,
        float(params["obstacle_repulsion_radius_m"]),
        float(params["obstacle_repulsion_weight"]),
    )[free_space_mask]
    cost_map[free_space_mask] = np.maximum(cost_map[free_space_mask], float(params["min_cost"]))

    return {
        "inflated_obstacle_mask": inflated_obstacle_mask.astype(np.uint8),
        "free_space_mask": free_space_mask.astype(np.uint8),
        "preferred_road_mask": preferred_road_mask.astype(np.uint8),
        "preferred_path_mask": preferred_path_mask.astype(np.uint8),
        "distance_to_obstacle_m": distance_to_obstacle_m.astype(np.float32),
        "distance_to_preferred_path_m": distance_to_preferred_path_m.astype(np.float32),
        "preferred_path_attraction": preferred_path_attraction_field.astype(np.float32),
        "cost_map": cost_map,
    }
