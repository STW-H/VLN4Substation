"""Build reusable planning fields and mode-specific traversal cost maps."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np


def distance_to_mask(mask: np.ndarray, resolution_m: float) -> np.ndarray:
    target = (mask > 0).astype(np.uint8)
    inverse = (target == 0).astype(np.uint8)
    return (
        cv2.distanceTransform(inverse, cv2.DIST_L2, 5).astype(np.float32)
        * float(resolution_m)
    )


def inflate_obstacles(
    obstacle_mask: np.ndarray,
    resolution_m: float,
    inflation_radius_m: float,
) -> np.ndarray:
    if inflation_radius_m <= 0:
        return (obstacle_mask > 0).astype(np.uint8)
    obstacle_distance = distance_to_mask(obstacle_mask, resolution_m)
    return (obstacle_distance <= float(inflation_radius_m)).astype(np.uint8)


def obstacle_repulsion(
    distance_to_obstacle_m: np.ndarray,
    repulsion_radius_m: float,
    repulsion_weight: float,
) -> np.ndarray:
    if repulsion_radius_m <= 0 or repulsion_weight <= 0:
        return np.zeros_like(distance_to_obstacle_m, dtype=np.float32)
    repulsion = np.clip(
        (repulsion_radius_m - distance_to_obstacle_m) / repulsion_radius_m,
        0.0,
        1.0,
    )
    return (repulsion_weight * np.square(repulsion)).astype(np.float32)


def build_derived_layers(
    base_masks: Mapping[str, np.ndarray],
    resolution_m: float,
    obstacle_inflation_radius_m: float,
) -> dict[str, np.ndarray]:
    """Build mode-independent collision masks and semantic distance fields."""
    boundary = base_masks["boundary_mask"].astype(bool)
    obstacle = base_masks["obstacle_mask"].astype(bool)
    equipment = base_masks.get(
        "equipment_mask", np.zeros_like(boundary, dtype=np.uint8)
    ).astype(bool)
    inflated = inflate_obstacles(
        obstacle.astype(np.uint8), resolution_m, obstacle_inflation_radius_m
    ).astype(bool)
    free_space = boundary & (~inflated)
    pose_center_space = boundary & (~obstacle) & (~equipment)

    preferred_path = base_masks["preferred_path_mask"]
    if np.any(preferred_path > 0):
        path_distance = distance_to_mask(preferred_path, resolution_m)
    else:
        path_distance = np.full(boundary.shape, np.inf, dtype=np.float32)

    return {
        "inflated_obstacle_mask": inflated.astype(np.uint8),
        "free_space_mask": free_space.astype(np.uint8),
        "pose_center_space_mask": pose_center_space.astype(np.uint8),
        "distance_to_obstacle_m": distance_to_mask(
            obstacle.astype(np.uint8), resolution_m
        ).astype(np.float32),
        "distance_to_preferred_path_m": path_distance.astype(np.float32),
    }


def build_traversal_cost_map(
    layers: Mapping[str, np.ndarray],
    params: Mapping[str, Any],
    *,
    pose_aware: bool,
) -> np.ndarray:
    """Compose one mode's cost map from stored semantic layers."""
    space_key = "pose_center_space_mask" if pose_aware else "free_space_mask"
    available = layers[space_key].astype(bool)
    preferred_road = layers["preferred_road_mask"].astype(bool) & available
    narrow_space = layers["narrow_space_mask"].astype(bool) & available

    sigma_m = float(params["preferred_path_sigma_m"])
    path_distance = np.asarray(layers["distance_to_preferred_path_m"])
    if sigma_m <= 0:
        path_attraction = (layers["preferred_path_mask"] > 0).astype(np.float32)
    else:
        path_attraction = np.exp(
            -np.square(path_distance) / (2.0 * sigma_m * sigma_m)
        ).astype(np.float32)

    cost_map = np.full(available.shape, np.inf, dtype=np.float32)
    cost_map[available] = float(params["base_cost"])
    cost_map[preferred_road] = float(params["preferred_road_cost"])
    cost_map[available] -= (
        float(params["preferred_path_alpha"]) * path_attraction[available]
    )
    cost_map[narrow_space] += float(params["narrow_space_penalty"])
    cost_map[available] += obstacle_repulsion(
        np.asarray(layers["distance_to_obstacle_m"]),
        float(params["obstacle_repulsion_radius_m"]),
        float(params["obstacle_repulsion_weight"]),
    )[available]
    cost_map[available] = np.maximum(
        cost_map[available], float(params["min_cost"])
    )
    return cost_map
