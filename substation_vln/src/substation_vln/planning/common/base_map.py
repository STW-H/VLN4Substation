"""Build base planning masks from merged annotation geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from substation_vln.planning.common.grid import GridSpec


def load_merged_annotations(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def planning_boundary_annotations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in payload["annotations"] if item.get("category") == "planning_boundary"]


def build_grid_spec(payload: dict[str, Any], resolution_m: float, padding_m: float) -> GridSpec:
    boundaries = planning_boundary_annotations(payload)
    if len(boundaries) != 1:
        raise ValueError(f"Expected exactly one planning_boundary annotation, got {len(boundaries)}")
    bbox = boundaries[0]["bbox_xy"]
    min_x = float(bbox["min"][0]) - padding_m
    min_y = float(bbox["min"][1]) - padding_m
    max_x = float(bbox["max"][0]) + padding_m
    max_y = float(bbox["max"][1]) + padding_m
    width = int(np.ceil((max_x - min_x) / resolution_m))
    height = int(np.ceil((max_y - min_y) / resolution_m))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid planning grid size: {width}x{height}")
    return GridSpec(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, resolution_m=resolution_m, width=width, height=height)


def empty_mask(grid: GridSpec) -> np.ndarray:
    return np.zeros((grid.height, grid.width), dtype=np.uint8)


def fill_polygon(mask: np.ndarray, grid: GridSpec, polygon_xy: list[list[float]], value: int = 1) -> None:
    if len(polygon_xy) < 3:
        return
    pts = grid.xy_to_grid(np.asarray(polygon_xy, dtype=np.float64)).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], int(value))


def draw_segment(mask: np.ndarray, grid: GridSpec, start_xy: list[float], end_xy: list[float], width_m: float, value: int = 1) -> None:
    pts = grid.xy_to_grid(np.asarray([start_xy, end_xy], dtype=np.float64))
    thickness = max(1, int(round(width_m / grid.resolution_m)))
    cv2.line(mask, tuple(pts[0]), tuple(pts[1]), int(value), thickness=thickness, lineType=cv2.LINE_AA)


def draw_polyline(mask: np.ndarray, grid: GridSpec, polyline_xy: list[list[float]], width_m: float, value: int = 1) -> None:
    if len(polyline_xy) < 2:
        return
    pts = grid.xy_to_grid(np.asarray(polyline_xy, dtype=np.float64)).reshape((-1, 1, 2))
    thickness = max(1, int(round(width_m / grid.resolution_m)))
    cv2.polylines(mask, [pts], isClosed=False, color=int(value), thickness=thickness, lineType=cv2.LINE_AA)


def add_directed_polyline_field(
    direction_sum_x: np.ndarray,
    direction_sum_y: np.ndarray,
    direction_count: np.ndarray,
    grid: GridSpec,
    polyline_xy: list[list[float]],
    width_m: float,
) -> None:
    """Accumulate each directed segment's local world-frame tangent."""
    for start, end in zip(polyline_xy[:-1], polyline_xy[1:], strict=True):
        delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        points = grid.xy_to_grid(np.asarray([start, end], dtype=np.float64))
        thickness = max(1, int(round(width_m / grid.resolution_m)))
        padding = thickness + 1
        col_min = max(0, int(points[:, 0].min()) - padding)
        col_max = min(grid.width - 1, int(points[:, 0].max()) + padding)
        row_min = max(0, int(points[:, 1].min()) - padding)
        row_max = min(grid.height - 1, int(points[:, 1].max()) + padding)
        if col_min > col_max or row_min > row_max:
            continue
        local_mask = np.zeros(
            (row_max - row_min + 1, col_max - col_min + 1), dtype=np.uint8
        )
        local_points = points - np.asarray([col_min, row_min], dtype=np.int32)
        cv2.line(
            local_mask,
            tuple(local_points[0]),
            tuple(local_points[1]),
            1,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )
        selected = local_mask > 0
        target = np.s_[row_min : row_max + 1, col_min : col_max + 1]
        tangent = delta / length
        direction_sum_x[target][selected] += float(tangent[0])
        direction_sum_y[target][selected] += float(tangent[1])
        direction_count[target][selected] += 1.0


def fill_circle(mask: np.ndarray, grid: GridSpec, center_xy: list[float], radius_m: float, value: int = 1) -> None:
    center = grid.xy_to_grid(np.asarray([center_xy], dtype=np.float64))[0]
    radius_px = max(1, int(round(radius_m / grid.resolution_m)))
    cv2.circle(mask, tuple(center), radius_px, int(value), thickness=-1, lineType=cv2.LINE_AA)


def equipment_primitives(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a grouped equipment annotation into independently planned primitives."""
    geometry_type = annotation.get("geometry_type")
    if geometry_type == "multi_polygon":
        primitives = []
        for polygon in annotation.get("polygons_xy", []):
            points = np.asarray(polygon, dtype=np.float64)
            area = 0.5 * abs(
                float(np.dot(points[:, 0], np.roll(points[:, 1], 1)))
                - float(np.dot(points[:, 1], np.roll(points[:, 0], 1)))
            )
            primitives.append(
                {
                    "geometry_type": "multi_polygon",
                    "polygons_xy": [polygon],
                    "circles": [],
                    "bbox_xy": {
                        "min": np.min(points, axis=0).tolist(),
                        "max": np.max(points, axis=0).tolist(),
                    },
                    "area_xy": area,
                }
            )
        return primitives
    if geometry_type == "multi_circle":
        return [
            {
                "geometry_type": "multi_circle",
                "polygons_xy": [],
                "circles": [circle],
                "bbox_xy": circle["bbox_xy"],
                "area_xy": float(circle["area_xy"]),
            }
            for circle in annotation.get("circles", [])
        ]
    raise ValueError(f"Unsupported equipment geometry: {geometry_type}")


def build_base_masks(payload: dict[str, Any], grid: GridSpec, preferred_path_width_m: float) -> dict[str, np.ndarray]:
    boundary_mask = empty_mask(grid)
    obstacle_mask = empty_mask(grid)
    preferred_road_mask = empty_mask(grid)
    preferred_path_mask = empty_mask(grid)
    direction_sum_x = np.zeros((grid.height, grid.width), dtype=np.float32)
    direction_sum_y = np.zeros((grid.height, grid.width), dtype=np.float32)
    direction_count = np.zeros((grid.height, grid.width), dtype=np.float32)
    narrow_space_mask = empty_mask(grid)
    equipment_mask = empty_mask(grid)
    equipment_index_mask = np.zeros((grid.height, grid.width), dtype=np.int32)
    equipment_index = 0

    for annotation in payload["annotations"]:
        category = annotation.get("category")
        geometry_type = annotation.get("geometry_type")

        if category == "planning_boundary" and geometry_type == "multi_polygon":
            for polygon in annotation.get("polygons_xy", []):
                fill_polygon(boundary_mask, grid, polygon)
        elif category == "obstacle":
            if geometry_type == "multi_polygon":
                for polygon in annotation.get("polygons_xy", []):
                    fill_polygon(obstacle_mask, grid, polygon)
            elif geometry_type == "multi_circle":
                for circle in annotation.get("circles", []):
                    fill_circle(obstacle_mask, grid, circle["center_xy"], float(circle["radius_xy"]))
        elif category == "preferred_road" and geometry_type == "multi_polygon":
            for polygon in annotation.get("polygons_xy", []):
                fill_polygon(preferred_road_mask, grid, polygon)
        elif category == "narrow_space":
            if geometry_type == "multi_polygon":
                for polygon in annotation.get("polygons_xy", []):
                    fill_polygon(narrow_space_mask, grid, polygon)
            elif geometry_type == "multi_circle":
                for circle in annotation.get("circles", []):
                    fill_circle(narrow_space_mask, grid, circle["center_xy"], float(circle["radius_xy"]))
        elif category == "preferred_path":
            if geometry_type == "multi_directed_segment":
                for segment in annotation.get("segments", []):
                    draw_segment(preferred_path_mask, grid, segment["start_xy"], segment["end_xy"], preferred_path_width_m)
                    add_directed_polyline_field(
                        direction_sum_x,
                        direction_sum_y,
                        direction_count,
                        grid,
                        [segment["start_xy"], segment["end_xy"]],
                        preferred_path_width_m,
                    )
            elif geometry_type in ("multi_directed_polyline", "multi_polyline"):
                for polyline in annotation.get("polylines", []):
                    draw_polyline(preferred_path_mask, grid, polyline["polyline_xy"], preferred_path_width_m)
                    if geometry_type == "multi_directed_polyline":
                        add_directed_polyline_field(
                            direction_sum_x,
                            direction_sum_y,
                            direction_count,
                            grid,
                            polyline["polyline_xy"],
                            preferred_path_width_m,
                        )
            elif geometry_type in ("directed_polyline", "polyline"):
                draw_polyline(preferred_path_mask, grid, annotation["polyline_xy"], preferred_path_width_m)
                if geometry_type == "directed_polyline":
                    add_directed_polyline_field(
                        direction_sum_x,
                        direction_sum_y,
                        direction_count,
                        grid,
                        annotation["polyline_xy"],
                        preferred_path_width_m,
                    )
        elif category == "equipment_region":
            for primitive in equipment_primitives(annotation):
                equipment_index += 1
                if primitive["geometry_type"] == "multi_polygon":
                    for polygon in primitive["polygons_xy"]:
                        fill_polygon(equipment_mask, grid, polygon)
                        fill_polygon(equipment_index_mask, grid, polygon, value=equipment_index)
                else:
                    for circle in primitive["circles"]:
                        fill_circle(equipment_mask, grid, circle["center_xy"], float(circle["radius_xy"]))
                        fill_circle(
                            equipment_index_mask,
                            grid,
                            circle["center_xy"],
                            float(circle["radius_xy"]),
                            value=equipment_index,
                        )

    boundary_mask = (boundary_mask > 0).astype(np.uint8)
    obstacle_mask = ((obstacle_mask > 0) & (boundary_mask > 0)).astype(np.uint8)
    preferred_road_mask = ((preferred_road_mask > 0) & (boundary_mask > 0) & (obstacle_mask == 0)).astype(np.uint8)
    preferred_path_mask = ((preferred_path_mask > 0) & (boundary_mask > 0) & (obstacle_mask == 0)).astype(np.uint8)
    narrow_space_mask = ((narrow_space_mask > 0) & (boundary_mask > 0) & (obstacle_mask == 0)).astype(np.uint8)
    equipment_mask = ((equipment_mask > 0) & (boundary_mask > 0)).astype(np.uint8)
    equipment_index_mask = np.where(boundary_mask > 0, equipment_index_mask, 0).astype(np.int32)
    direction_norm = np.hypot(direction_sum_x, direction_sum_y)
    directed_path_mask = (
        (direction_count > 0)
        & (direction_norm > 1.0e-6)
        & (preferred_path_mask > 0)
    )
    direction_x = np.zeros_like(direction_sum_x)
    direction_y = np.zeros_like(direction_sum_y)
    direction_x[directed_path_mask] = (
        direction_sum_x[directed_path_mask] / direction_norm[directed_path_mask]
    )
    direction_y[directed_path_mask] = (
        direction_sum_y[directed_path_mask] / direction_norm[directed_path_mask]
    )

    return {
        "boundary_mask": boundary_mask,
        "obstacle_mask": obstacle_mask,
        "preferred_road_mask": preferred_road_mask,
        "preferred_path_mask": preferred_path_mask,
        "directed_preferred_path_mask": directed_path_mask.astype(np.uint8),
        "preferred_path_direction_x": direction_x,
        "preferred_path_direction_y": direction_y,
        "narrow_space_mask": narrow_space_mask,
        "equipment_mask": equipment_mask,
        "equipment_index_mask": equipment_index_mask,
    }


def extract_equipment_regions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return stable equipment identity and footprint geometry in raster index order."""
    equipment: list[dict[str, Any]] = []
    for annotation in payload["annotations"]:
        if annotation.get("category") != "equipment_region":
            continue
        primitives = equipment_primitives(annotation)
        base_name = annotation.get("equipment_name") or annotation.get("label")
        for primitive_index, primitive in enumerate(primitives, start=1):
            equipment_name = base_name if len(primitives) == 1 else f"{base_name}_{primitive_index}"
            equipment.append(
                {
                    "equipment_index": len(equipment) + 1,
                    "annotation_id": annotation.get("id"),
                    "source_file": annotation.get("source_file"),
                    "source_id": annotation.get("source_id"),
                    "source_equipment_name": base_name,
                    "equipment_name": equipment_name,
                    "equipment_type": annotation.get("equipment_type") or "unknown",
                    "primitive_index": primitive_index,
                    "primitive_count": len(primitives),
                    **primitive,
                }
            )
    return equipment


def extract_patrol_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    patrol_points: list[dict[str, Any]] = []
    for annotation in payload["annotations"]:
        if annotation.get("category") != "patrol_point":
            continue
        for index, point in enumerate(annotation.get("directed_points", []), start=1):
            patrol_points.append(
                {
                    "annotation_id": annotation.get("id"),
                    "source_file": annotation.get("source_file"),
                    "source_id": annotation.get("source_id"),
                    "label": annotation.get("label"),
                    "point_index": index,
                    "stop_xy": point["stop_xy"],
                    "look_xy": point["look_xy"],
                    "direction_xy_unit": point["direction_xy_unit"],
                    "yaw_rad": point["yaw_rad"],
                    "yaw_deg": point["yaw_deg"],
                }
            )
    return patrol_points


def extract_robot_start_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand annotated point batches into uniquely selectable robot starts."""
    starts: list[dict[str, Any]] = []
    for annotation in payload["annotations"]:
        if annotation.get("category") != "robot_start_point":
            continue
        points_xy = annotation.get("points_xy", [])
        base_name = str(annotation.get("label") or "start").strip() or "start"
        point_names = annotation.get("point_names", [])
        for point_index, point_xy in enumerate(points_xy, start=1):
            name = (
                str(point_names[point_index - 1])
                if len(point_names) == len(points_xy)
                else (base_name if len(points_xy) == 1 else f"{base_name}_{point_index}")
            )
            starts.append(
                {
                    "start_point_index": len(starts) + 1,
                    "start_point_name": name,
                    "annotation_id": annotation.get("id"),
                    "source_file": annotation.get("source_file"),
                    "source_id": annotation.get("source_id"),
                    "point_index": point_index,
                    "xy": [float(point_xy[0]), float(point_xy[1])],
                }
            )
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in starts:
        groups.setdefault(str(item["start_point_name"]), []).append(item)
    for duplicated_name, items in groups.items():
        if len(items) <= 1:
            continue
        for occurrence, item in enumerate(items, start=1):
            item["start_point_name"] = f"{duplicated_name}_{occurrence}"
    return starts
