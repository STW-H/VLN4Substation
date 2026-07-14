"""I/O and visualization helpers for feasible inspection regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from substation_vln.planning.common.grid import GridSpec


def load_targets(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") in ("inspection_targets_3d", "pointcloud_inspection_targets"):
        if not isinstance(payload.get("targets"), list):
            raise ValueError(f"Invalid inspection-target annotation: {path}")
        return payload

    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"Invalid merged annotation file: {path}")
    targets = [
        item
        for item in annotations
        if item.get("category") == "inspection_target" and item.get("geometry_type") == "point_3d"
    ]
    if not targets:
        raise ValueError(f"No 3D inspection targets found in merged annotations: {path}")
    ground_values = {float(item.get("ground_z_m", 0.0)) for item in targets}
    camera_values = {float(item.get("camera_height_m", 1.0)) for item in targets}
    if len(ground_values) != 1 or len(camera_values) != 1:
        raise ValueError("Merged 3D targets use inconsistent ground or camera heights.")
    return {
        "type": "merged_inspection_targets_3d",
        "source_annotation": str(path.expanduser().resolve()),
        "ground_plane": {"model": "constant_z", "z_m": ground_values.pop()},
        "camera": {"model": "omnidirectional", "height_above_ground_m": camera_values.pop()},
        "targets": targets,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def save_distance_image(path: Path, distance: np.ndarray, valid_mask: np.ndarray) -> None:
    image = np.zeros(distance.shape, dtype=np.uint8)
    valid = valid_mask.astype(bool) & np.isfinite(distance)
    if np.any(valid):
        values = distance[valid]
        lo, hi = float(values.min()), float(values.max())
        if hi > lo:
            image[valid] = np.clip(255.0 * (distance[valid] - lo) / (hi - lo), 0, 255).astype(np.uint8)
        else:
            image[valid] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS))


def save_feasible_inspection_region_overlay(
    path: Path,
    *,
    boundary_mask: np.ndarray,
    free_space_mask: np.ndarray,
    visible_inspection_region_mask: np.ndarray,
    feasible_inspection_region_mask: np.ndarray,
    target_xy: list[float],
    grid: GridSpec,
) -> None:
    overlay = np.full((*boundary_mask.shape, 3), 245, dtype=np.uint8)
    overlay[boundary_mask > 0] = (205, 205, 205)
    overlay[free_space_mask > 0] = (225, 205, 155)
    overlay[visible_inspection_region_mask > 0] = (40, 210, 255)
    overlay[feasible_inspection_region_mask > 0] = (60, 210, 60)
    col, row = grid.xy_to_grid(np.asarray([target_xy], dtype=np.float64))[0]
    cv2.drawMarker(overlay, (int(col), int(row)), (0, 0, 255), cv2.MARKER_CROSS, 25, 3, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)
