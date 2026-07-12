"""Build base planning masks from merged annotation geometry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class GridSpec:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    resolution_m: float
    width: int
    height: int

    def xy_to_grid(self, points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float64)
        cols = np.floor((pts[:, 0] - self.min_x) / self.resolution_m).astype(np.int32)
        rows = np.floor((self.max_y - pts[:, 1]) / self.resolution_m).astype(np.int32)
        cols = np.clip(cols, 0, self.width - 1)
        rows = np.clip(rows, 0, self.height - 1)
        return np.stack([cols, rows], axis=1)

    def grid_to_xy(self, cols: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.min_x + (np.asarray(cols, dtype=np.float64) + 0.5) * self.resolution_m
        y = self.max_y - (np.asarray(rows, dtype=np.float64) + 0.5) * self.resolution_m
        return x, y

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "resolution_m": self.resolution_m,
            "width": self.width,
            "height": self.height,
        }


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
        elif category == "preferred_path" and geometry_type == "multi_directed_segment":
            for segment in annotation.get("segments", []):
                draw_segment(preferred_path_mask, grid, segment["start_xy"], segment["end_xy"], preferred_path_width_m)

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
