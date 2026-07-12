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


def fill_circle(mask: np.ndarray, grid: GridSpec, center_xy: list[float], radius_m: float, value: int = 1) -> None:
    center = grid.xy_to_grid(np.asarray([center_xy], dtype=np.float64))[0]
    radius_px = max(1, int(round(radius_m / grid.resolution_m)))
    cv2.circle(mask, tuple(center), radius_px, int(value), thickness=-1, lineType=cv2.LINE_AA)


def build_base_masks(payload: dict[str, Any], grid: GridSpec, preferred_path_width_m: float) -> dict[str, np.ndarray]:
    boundary_mask = empty_mask(grid)
    obstacle_mask = empty_mask(grid)
    preferred_road_mask = empty_mask(grid)
    preferred_path_mask = empty_mask(grid)

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
        elif category == "preferred_path":
            if geometry_type == "multi_directed_segment":
                for segment in annotation.get("segments", []):
                    draw_segment(preferred_path_mask, grid, segment["start_xy"], segment["end_xy"], preferred_path_width_m)
            elif geometry_type in ("multi_directed_polyline", "multi_polyline"):
                for polyline in annotation.get("polylines", []):
                    draw_polyline(preferred_path_mask, grid, polyline["polyline_xy"], preferred_path_width_m)
            elif geometry_type in ("directed_polyline", "polyline"):
                draw_polyline(preferred_path_mask, grid, annotation["polyline_xy"], preferred_path_width_m)

    boundary_mask = (boundary_mask > 0).astype(np.uint8)
    obstacle_mask = ((obstacle_mask > 0) & (boundary_mask > 0)).astype(np.uint8)
    preferred_road_mask = ((preferred_road_mask > 0) & (boundary_mask > 0) & (obstacle_mask == 0)).astype(np.uint8)
    preferred_path_mask = ((preferred_path_mask > 0) & (boundary_mask > 0) & (obstacle_mask == 0)).astype(np.uint8)
    return {
        "boundary_mask": boundary_mask,
        "obstacle_mask": obstacle_mask,
        "preferred_road_mask": preferred_road_mask,
        "preferred_path_mask": preferred_path_mask,
    }


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
