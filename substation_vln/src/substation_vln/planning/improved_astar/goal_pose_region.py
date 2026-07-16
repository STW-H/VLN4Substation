"""Generate collision-free ROI-conical approach poses around inspection equipment."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from substation_vln.planning.common.grid import GridSpec
from substation_vln.planning.improved_astar.camera_model import (
    CameraConfig,
    normalized_preference_cost,
    wrap_angle,
)


def heading_yaws(heading_bins: int) -> np.ndarray:
    if heading_bins < 4:
        raise ValueError("heading_bins must be >= 4")
    return np.arange(heading_bins, dtype=np.float64) * (2.0 * math.pi / heading_bins)


def quantize_heading(yaw_rad: float, heading_bins: int) -> int:
    return int(round((yaw_rad % (2.0 * math.pi)) * heading_bins / (2.0 * math.pi))) % heading_bins


def robot_footprint_kernel(
    yaw_rad: float,
    resolution_m: float,
    length_m: float,
    width_m: float,
    safety_margin_m: float,
) -> np.ndarray:
    half_length = 0.5 * float(length_m) + float(safety_margin_m)
    half_width = 0.5 * float(width_m) + float(safety_margin_m)
    radius_px = int(math.ceil(math.hypot(half_length, half_width) / resolution_m)) + 2
    size = 2 * radius_px + 1
    center = np.array([radius_px, radius_px], dtype=np.float64)
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
    left = np.array([-forward[1], forward[0]], dtype=np.float64)
    corners_world = np.asarray(
        [
            half_length * forward + half_width * left,
            half_length * forward - half_width * left,
            -half_length * forward - half_width * left,
            -half_length * forward + half_width * left,
        ]
    )
    corners_px = np.column_stack(
        (center[0] + corners_world[:, 0] / resolution_m, center[1] - corners_world[:, 1] / resolution_m)
    )
    kernel = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(kernel, np.rint(corners_px).astype(np.int32), 1)
    return kernel


def build_pose_free_masks(
    boundary_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    equipment_mask: np.ndarray,
    grid: GridSpec,
    robot_config: dict[str, Any],
) -> np.ndarray:
    """Erode raw free space with a rotated rectangular robot footprint per heading."""
    bins = int(robot_config.get("heading_bins", 16))
    raw_free = (
        (np.asarray(boundary_mask) > 0)
        & (np.asarray(obstacle_mask) == 0)
        & (np.asarray(equipment_mask) == 0)
    ).astype(np.uint8)
    masks = np.empty((bins, grid.height, grid.width), dtype=np.uint8)
    for index, yaw in enumerate(heading_yaws(bins)):
        kernel = robot_footprint_kernel(
            float(yaw),
            grid.resolution_m,
            float(robot_config["length_m"]),
            float(robot_config["width_m"]),
            float(robot_config.get("safety_margin_m", 0.0)),
        )
        masks[index] = cv2.erode(raw_free, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return masks


def pack_pose_free_masks(masks: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(masks, dtype=np.uint8), axis=2, bitorder="little")


def unpack_pose_free_masks(packed: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=2, count=int(width), bitorder="little")


def generate_goal_pose_candidates(
    equipment: dict[str, Any],
    pose_free_masks: np.ndarray,
    grid: GridSpec,
    camera_config: CameraConfig,
    generation_config: dict[str, Any],
    observation_profiles: dict[str, Any] | None = None,
) -> dict[str, np.ndarray | float | int | str]:
    """Intersect an ROI-centered conical viewing shell with the camera-height plane."""
    model = str(generation_config.get("observation_model", "roi_conical_approach"))
    if model != "roi_conical_approach":
        raise ValueError(f"Unsupported observation model: {model}")

    bounds_min = np.asarray(equipment["robust_bounds_min_xyz"], dtype=np.float64)
    bounds_max = np.asarray(equipment["robust_bounds_max_xyz"], dtype=np.float64)
    center_xy = np.asarray(equipment["center_xyz"], dtype=np.float64)[:2]
    profiles = observation_profiles or {}
    profile = dict(profiles.get("default", {}))
    profile.update(profiles.get(str(equipment.get("equipment_type", "unknown")), {}))
    vertical_min_fraction = float(profile.get("vertical_min_fraction", 0.0))
    vertical_max_fraction = float(profile.get("vertical_max_fraction", 1.0))
    if not 0.0 <= vertical_min_fraction < vertical_max_fraction <= 1.0:
        raise ValueError(f"Invalid observation height fractions for {equipment.get('equipment_name')}")
    full_height = float(bounds_max[2] - bounds_min[2])
    observation_z_min = float(bounds_min[2] + vertical_min_fraction * full_height)
    observation_z_max = float(bounds_min[2] + vertical_max_fraction * full_height)

    tilt_min = float(profile.get("tilt_min_deg", camera_config.tilt_min_deg))
    tilt_max = float(profile.get("tilt_max_deg", camera_config.tilt_max_deg))
    preferred_tilt = float(profile.get("preferred_tilt_deg", camera_config.preferred_tilt_deg))
    if not camera_config.tilt_min_deg <= tilt_min < tilt_max <= camera_config.tilt_max_deg:
        raise ValueError(f"Observation tilt range [{tilt_min}, {tilt_max}] exceeds camera limits")
    if not tilt_min <= preferred_tilt <= tilt_max:
        raise ValueError("preferred_tilt_deg must lie inside the observation tilt range")

    if not 0.0 < tilt_min < tilt_max < 90.0:
        raise ValueError("ROI conical tilt limits must lie strictly inside (0, 90) degrees")
    observation_center_z = 0.5 * (observation_z_min + observation_z_max)
    height_delta = float(observation_center_z - camera_config.height_m)
    if height_delta <= 0.0:
        raise ValueError(f"ROI center is not above the camera for {equipment.get('equipment_name')}")

    def distance_at_tilt(tilt_deg: float) -> float:
        return height_delta / math.tan(math.radians(tilt_deg))

    min_distance = max(
        float(generation_config.get("min_candidate_distance_m", camera_config.near_clip_m)),
        distance_at_tilt(tilt_max),
    )
    max_distance = min(
        float(generation_config.get("max_search_radius_m", 30.0)),
        distance_at_tilt(tilt_min),
    )
    if max_distance <= min_distance:
        raise ValueError(f"Empty top-edge distance interval for {equipment.get('equipment_name')}")

    bbox = equipment.get("bbox_xy")
    if bbox:
        min_x, min_y = map(float, bbox["min"])
        max_x, max_y = map(float, bbox["max"])
    else:
        polygon_points = np.asarray([point for polygon in equipment.get("polygons_xy", []) for point in polygon])
        min_x, min_y = np.min(polygon_points, axis=0)
        max_x, max_y = np.max(polygon_points, axis=0)
    camera_offset = math.hypot(camera_config.forward_offset_m, camera_config.lateral_offset_m)
    crop_radius = max_distance + camera_offset
    crop_grid = grid.xy_to_grid(
        np.asarray(
            [[min_x - crop_radius, min_y - crop_radius], [max_x + crop_radius, max_y + crop_radius]],
            dtype=np.float64,
        )
    )
    col_min, col_max = sorted(map(int, crop_grid[:, 0]))
    row_min, row_max = sorted(map(int, crop_grid[:, 1]))
    col_min, col_max = max(0, col_min), min(grid.width - 1, col_max)
    row_min, row_max = max(0, row_min), min(grid.height - 1, row_max)

    bins = int(pose_free_masks.shape[0])
    stride = max(1, int(generation_config.get("candidate_stride_cells", 2)))
    yaw_values = heading_yaws(bins)
    candidates: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for row in range(row_min, row_max + 1, stride):
        cols = np.arange(col_min, col_max + 1, stride, dtype=np.int32)
        rows = np.full_like(cols, row)
        xs, ys = grid.grid_to_xy(cols, rows)
        for col, x, y in zip(cols, xs, ys, strict=True):
            radial_yaw = math.atan2(float(y - center_xy[1]), float(x - center_xy[0]))
            for tangent_yaw in (radial_yaw + 0.5 * math.pi, radial_yaw - 0.5 * math.pi):
                heading = quantize_heading(tangent_yaw, bins)
                if pose_free_masks[heading, row, col] == 0:
                    continue
                yaw = float(yaw_values[heading])
                forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
                left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
                camera_xy = (
                    np.asarray([x, y], dtype=np.float64)
                    + camera_config.forward_offset_m * forward
                    + camera_config.lateral_offset_m * left
                )
                radial_distance = float(np.linalg.norm(camera_xy - center_xy))
                if not min_distance <= radial_distance <= max_distance:
                    continue
                tilt = math.atan2(height_delta, radial_distance)
                tilt_deg = math.degrees(tilt)
                if not tilt_min <= tilt_deg <= tilt_max:
                    continue
                camera_world_yaw = math.atan2(
                    float(center_xy[1] - camera_xy[1]), float(center_xy[0] - camera_xy[0])
                )
                pan = wrap_angle(camera_world_yaw - yaw)
                if not camera_config.pan_min_deg <= math.degrees(pan) <= camera_config.pan_max_deg:
                    continue
                tilt_cost = normalized_preference_cost(tilt_deg, tilt_min, preferred_tilt, tilt_max)
                candidates[(int(row), int(col), int(heading))] = (float(tilt_cost), float(pan), float(tilt))

    if candidates:
        keys = np.asarray(list(candidates), dtype=np.int32)
        values = np.asarray(list(candidates.values()), dtype=np.float32)
    else:
        keys = np.empty((0, 3), dtype=np.int32)
        values = np.empty((0, 3), dtype=np.float32)
    roi_center = np.asarray([center_xy[0], center_xy[1], observation_center_z], dtype=np.float64)
    return {
        "rows": keys[:, 0],
        "cols": keys[:, 1],
        "heading_bins": keys[:, 2],
        "tilt_costs": values[:, 0],
        "camera_pan_rad": values[:, 1],
        "camera_tilt_rad": values[:, 2],
        "search_radius_m": float(max_distance),
        "minimum_distance_m": float(min_distance),
        "candidate_stride_cells": int(stride),
        "observation_model": model,
        "configured_tilt_min_deg": float(tilt_min),
        "configured_tilt_max_deg": float(tilt_max),
        "preferred_tilt_deg": float(preferred_tilt),
        "observation_vertical_min_fraction": float(vertical_min_fraction),
        "observation_vertical_max_fraction": float(vertical_max_fraction),
        "observation_z_min_m": float(observation_z_min),
        "observation_z_max_m": float(observation_z_max),
        "observation_center_xyz": roi_center,
    }
