"""Generate feasible 2D inspection regions from 3D target visibility."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np

from substation_vln.inspection_regions.visibility_corridor import batch_visibility, dilate_occupancy
from substation_vln.inspection_regions.voxel_map import SparseVoxelGrid
from substation_vln.planning.common.grid import GridSpec


@dataclass(frozen=True)
class FeasibleInspectionRegionConfig:
    ground_z_m: float = 0.0
    camera_height_m: float = 1.0
    visibility_clearance_radius_m: float = 0.2
    camera_exclusion_radius_m: float = 0.3
    min_region_area_m2: float = 0.5
    morphology_open_radius_m: float = 0.0

    def validate(self) -> None:
        if self.camera_height_m <= 0:
            raise ValueError("camera_height_m must be positive.")
        if min(
            self.visibility_clearance_radius_m,
            self.camera_exclusion_radius_m,
            self.min_region_area_m2,
            self.morphology_open_radius_m,
        ) < 0:
            raise ValueError("Visibility and region-cleaning parameters must be non-negative.")


def _remove_small_components(mask: np.ndarray, resolution_m: float, min_area_m2: float) -> tuple[np.ndarray, int]:
    if min_area_m2 <= 0:
        count = max(0, cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)
        return mask.astype(bool), count
    minimum_cells = int(np.ceil(min_area_m2 / (resolution_m * resolution_m)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=np.bool_)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_cells
    cleaned = keep[labels]
    return cleaned, int(keep[1:].sum())


def _open_mask(mask: np.ndarray, resolution_m: float, radius_m: float) -> np.ndarray:
    if radius_m <= 0:
        return mask.astype(bool)
    radius_px = int(np.ceil(radius_m / resolution_m))
    size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)


def compute_feasible_inspection_region(
    *,
    target: dict[str, Any],
    voxel_grid: SparseVoxelGrid,
    grid: GridSpec,
    boundary_mask: np.ndarray,
    free_space_mask: np.ndarray,
    config: FeasibleInspectionRegionConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config.validate()
    target_xyz = np.asarray(target["target_xyz"], dtype=np.float64)
    min_distance = float(target["min_observation_distance_m"])
    max_distance = float(target["max_observation_distance_m"])
    target_exclusion = float(target["target_exclusion_radius_m"])
    if max_distance <= min_distance or min_distance < 0:
        raise ValueError(f"Invalid observation distance interval for {target['target_id']}")

    started = time.perf_counter()
    camera_z = config.ground_z_m + config.camera_height_m
    dz = float(target_xyz[2] - camera_z)
    horizontal_max = float(np.sqrt(max(0.0, max_distance**2 - dz**2)))
    horizontal_min = float(np.sqrt(max(0.0, min_distance**2 - dz**2))) if min_distance > abs(dz) else 0.0

    distance_valid = np.zeros((grid.height, grid.width), dtype=np.bool_)
    visibility = np.zeros_like(distance_valid)
    distance_to_target = np.full((grid.height, grid.width), np.inf, dtype=np.float32)
    if max_distance < abs(dz):
        layers = {
            "distance_candidate_mask": distance_valid.astype(np.uint8),
            "robust_visibility_mask": visibility.astype(np.uint8),
            "visible_inspection_region_mask": visibility.astype(np.uint8),
            "feasible_inspection_region_raw_mask": visibility.astype(np.uint8),
            "feasible_inspection_region_mask": visibility.astype(np.uint8),
            "distance_to_target_m": distance_to_target,
        }
        return layers, {"reason": "target_height_exceeds_maximum_3d_distance", "elapsed_seconds": time.perf_counter() - started}

    min_col = max(0, int(np.floor((target_xyz[0] - horizontal_max - grid.min_x) / grid.resolution_m)))
    max_col = min(grid.width - 1, int(np.floor((target_xyz[0] + horizontal_max - grid.min_x) / grid.resolution_m)))
    min_row = max(0, int(np.floor((grid.max_y - (target_xyz[1] + horizontal_max)) / grid.resolution_m)))
    max_row = min(grid.height - 1, int(np.floor((grid.max_y - (target_xyz[1] - horizontal_max)) / grid.resolution_m)))
    rows, cols = np.mgrid[min_row : max_row + 1, min_col : max_col + 1]
    xs, ys = grid.grid_to_xy(cols.ravel(), rows.ravel())
    horizontal_sq = np.square(xs - target_xyz[0]) + np.square(ys - target_xyz[1])
    distances = np.sqrt(horizontal_sq + dz * dz)
    valid = (distances >= min_distance) & (distances <= max_distance)
    flat_rows, flat_cols = rows.ravel(), cols.ravel()
    distance_to_target[flat_rows, flat_cols] = distances.astype(np.float32)
    distance_valid[flat_rows[valid], flat_cols[valid]] = True

    # Visibility outside the planning boundary cannot contribute to a safe stopping region.
    candidate = valid & (boundary_mask[flat_rows, flat_cols] > 0)
    candidate_rows = flat_rows[candidate]
    candidate_cols = flat_cols[candidate]
    candidate_x = xs[candidate]
    candidate_y = ys[candidate]
    camera_positions = np.column_stack(
        [candidate_x, candidate_y, np.full(len(candidate_x), camera_z, dtype=np.float64)]
    )

    margin = config.visibility_clearance_radius_m + voxel_grid.voxel_size_m * 2.0
    world_min = np.asarray(
        [
            target_xyz[0] - horizontal_max - margin,
            target_xyz[1] - horizontal_max - margin,
            min(camera_z, target_xyz[2]) - margin,
        ]
    )
    world_max = np.asarray(
        [
            target_xyz[0] + horizontal_max + margin,
            target_xyz[1] + horizontal_max + margin,
            max(camera_z, target_xyz[2]) + margin,
        ]
    )
    local_occupancy, local_origin = voxel_grid.local_dense_occupancy(world_min, world_max)
    occupied_before = int(local_occupancy.sum())
    inflated_occupancy = dilate_occupancy(
        local_occupancy,
        config.visibility_clearance_radius_m,
        voxel_grid.voxel_size_m,
    )
    occupied_after = int(inflated_occupancy.sum())

    # Inflating obstacles by r_los and testing the center ray approximates a
    # cylindrical visibility corridor of radius r_los. Extend endpoint
    # exclusions by r_los so target/camera endpoint occupancy does not leak
    # back into the truncated ray after dilation.
    effective_camera_exclusion = config.camera_exclusion_radius_m + config.visibility_clearance_radius_m
    effective_target_exclusion = target_exclusion + config.visibility_clearance_radius_m
    visible_values, backend = batch_visibility(
        inflated_occupancy,
        local_origin,
        voxel_grid.voxel_size_m,
        camera_positions,
        target_xyz,
        effective_camera_exclusion,
        effective_target_exclusion,
    )
    visibility[candidate_rows[visible_values], candidate_cols[visible_values]] = True
    observable = distance_valid & visibility
    safe_raw = observable & (free_space_mask > 0)
    safe_opened = _open_mask(safe_raw, grid.resolution_m, config.morphology_open_radius_m)
    safe_clean, component_count = _remove_small_components(
        safe_opened,
        grid.resolution_m,
        config.min_region_area_m2,
    )

    layers = {
        "distance_candidate_mask": distance_valid.astype(np.uint8),
        "robust_visibility_mask": visibility.astype(np.uint8),
        "visible_inspection_region_mask": observable.astype(np.uint8),
        "feasible_inspection_region_raw_mask": safe_raw.astype(np.uint8),
        "feasible_inspection_region_mask": safe_clean.astype(np.uint8),
        "distance_to_target_m": distance_to_target,
    }
    metadata = {
        "target_id": target["target_id"],
        "target_xyz": target_xyz.tolist(),
        "camera_z_m": camera_z,
        "horizontal_distance_range_m": [horizontal_min, horizontal_max],
        "candidate_bbox_rc": [min_row, min_col, max_row, max_col],
        "distance_candidate_cells": int(distance_valid.sum()),
        "raycast_candidate_cells": int(len(camera_positions)),
        "robust_visible_cells": int(visibility.sum()),
        "visible_inspection_region_cells": int(observable.sum()),
        "feasible_inspection_region_raw_cells": int(safe_raw.sum()),
        "feasible_inspection_region_cells": int(safe_clean.sum()),
        "feasible_inspection_region_components": component_count,
        "local_occupancy_shape": list(local_occupancy.shape),
        "local_occupied_voxels_before_dilation": occupied_before,
        "local_occupied_voxels_after_dilation": occupied_after,
        "effective_camera_exclusion_radius_m": effective_camera_exclusion,
        "effective_target_exclusion_radius_m": effective_target_exclusion,
        "raycast_backend": backend,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return layers, metadata
