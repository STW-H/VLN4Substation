"""Shared helpers for 2D orthographic annotation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CATEGORIES = {
    "1": {"key": "obstacle", "name": "障碍物", "default_label": "obstacle", "geometry": "polygon"},
    "2": {"key": "patrol_point", "name": "巡视点位", "default_label": "patrol_point", "geometry": "directed_point"},
    "3": {"key": "preferred_road", "name": "优先通过区", "default_label": "preferred_road", "geometry": "polygon"},
    "4": {"key": "planning_boundary", "name": "规划边界", "default_label": "planning_boundary", "geometry": "polygon", "single": True},
    "5": {"key": "preferred_path", "name": "优先路径", "default_label": "preferred_path", "geometry": "directed_polyline"},
    "6": {"key": "narrow_space", "name": "狭窄空间", "default_label": "narrow_space", "geometry": "polygon"},
}

LABEL_COLORS_BGR = [
    (30, 30, 255),
    (0, 145, 255),
    (255, 90, 20),
    (60, 210, 60),
    (255, 20, 220),
    (210, 210, 0),
]

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def metadata_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def apply_homogeneous(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack([points.astype(np.float64), ones])
    transformed = homogeneous @ matrix.T
    return transformed[:, :2] / transformed[:, 2:3]


def polygon_area(points: list[list[float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polyline_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def rectangle_polygon(start: list[float], end: list[float]) -> list[list[float]]:
    c0, r0 = start
    c1, r1 = end
    return [[c0, r0], [c1, r0], [c1, r1], [c0, r1]]


def has_planning_boundary(annotations: list[dict]) -> bool:
    return any(annotation.get("category") == "planning_boundary" for annotation in annotations)


def available_categories(annotations: list[dict]) -> dict[str, dict]:
    if has_planning_boundary(annotations):
        return {
            number: category
            for number, category in CATEGORIES.items()
            if category["key"] != "planning_boundary"
        }
    return CATEGORIES


def category_already_exists(annotations: list[dict], category: dict) -> bool:
    return any(annotation.get("category") == category["key"] for annotation in annotations)


def make_annotation(
    *,
    annotation_id: int,
    category: dict,
    label: str,
    pending: dict,
    pixel_to_world: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> dict:
    annotation = {
        "id": annotation_id,
        "category": category["key"],
        "category_name": category["name"],
        "label": label,
        "selection_type": pending["selection_type"],
        "geometry_type": pending["geometry_type"],
        "color_bgr": list(color_bgr),
    }
    if "shape" in category:
        annotation["shape"] = category["shape"]
        annotation["shape_name"] = category.get("shape_name", category["shape"])
    if "path_type" in category:
        annotation["path_type"] = category["path_type"]
        annotation["path_type_name"] = category.get("path_type_name", category["path_type"])

    if pending["geometry_type"] == "directed_point":
        add_directed_point_fields(annotation, pending, pixel_to_world)
    elif pending["geometry_type"] == "multi_directed_point":
        add_multi_directed_point_fields(annotation, pending, pixel_to_world)
    elif pending["geometry_type"] == "directed_polyline":
        add_directed_polyline_fields(annotation, pending, pixel_to_world)
    elif pending["geometry_type"] == "multi_directed_polyline":
        add_multi_directed_polyline_fields(annotation, pending, pixel_to_world)
    elif pending["geometry_type"] == "polyline":
        add_polyline_fields(annotation, pending, pixel_to_world)
    elif pending["geometry_type"] == "multi_polyline":
        add_multi_polyline_fields(annotation, pending, pixel_to_world)
    elif pending["geometry_type"] == "multi_polygon":
        add_multi_polygon_fields(annotation, pending, pixel_to_world)
    elif pending["geometry_type"] == "multi_circle":
        add_multi_circle_fields(annotation, pending, pixel_to_world)
    else:
        add_polygon_fields(annotation, pending, pixel_to_world)

    return annotation


def add_directed_point_fields(annotation: dict, pending: dict, pixel_to_world: np.ndarray) -> None:
    stop_pixel = pending["stop_pixel"]
    look_pixel = pending["look_pixel"]
    stop_xy, look_xy = apply_homogeneous(
        pixel_to_world,
        np.asarray([stop_pixel, look_pixel], dtype=np.float64),
    ).tolist()
    direction_pixel = (np.asarray(look_pixel, dtype=np.float64) - np.asarray(stop_pixel, dtype=np.float64)).tolist()
    direction_xy_arr = np.asarray(look_xy, dtype=np.float64) - np.asarray(stop_xy, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction_xy_arr))
    direction_xy_unit = (direction_xy_arr / direction_norm).tolist() if direction_norm > 0 else [0.0, 0.0]
    yaw_rad = float(np.arctan2(direction_xy_arr[1], direction_xy_arr[0])) if direction_norm > 0 else 0.0
    annotation.update(
        {
            "stop_pixel": stop_pixel,
            "look_pixel": look_pixel,
            "stop_xy": stop_xy,
            "look_xy": look_xy,
            "direction_pixel": direction_pixel,
            "direction_xy": direction_xy_arr.tolist(),
            "direction_xy_unit": direction_xy_unit,
            "yaw_rad": yaw_rad,
            "yaw_deg": float(np.degrees(yaw_rad)),
        }
    )


def directed_point_record(stop_pixel: list[float], look_pixel: list[float], pixel_to_world: np.ndarray) -> dict:
    stop_xy, look_xy = apply_homogeneous(
        pixel_to_world,
        np.asarray([stop_pixel, look_pixel], dtype=np.float64),
    ).tolist()
    direction_pixel = (np.asarray(look_pixel, dtype=np.float64) - np.asarray(stop_pixel, dtype=np.float64)).tolist()
    direction_xy_arr = np.asarray(look_xy, dtype=np.float64) - np.asarray(stop_xy, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction_xy_arr))
    direction_xy_unit = (direction_xy_arr / direction_norm).tolist() if direction_norm > 0 else [0.0, 0.0]
    yaw_rad = float(np.arctan2(direction_xy_arr[1], direction_xy_arr[0])) if direction_norm > 0 else 0.0
    return {
        "stop_pixel": stop_pixel,
        "look_pixel": look_pixel,
        "stop_xy": stop_xy,
        "look_xy": look_xy,
        "direction_pixel": direction_pixel,
        "direction_xy": direction_xy_arr.tolist(),
        "direction_xy_unit": direction_xy_unit,
        "yaw_rad": yaw_rad,
        "yaw_deg": float(np.degrees(yaw_rad)),
    }


def add_multi_directed_point_fields(annotation: dict, pending: dict, pixel_to_world: np.ndarray) -> None:
    directed_points_pixel = pending["directed_points_pixel"]
    directed_points = [
        directed_point_record(item["stop_pixel"], item["look_pixel"], pixel_to_world)
        for item in directed_points_pixel
    ]
    flat_pixel = [point for item in directed_points_pixel for point in (item["stop_pixel"], item["look_pixel"])]
    flat_xy = [point for item in directed_points for point in (item["stop_xy"], item["look_xy"])]
    annotation.update(
        {
            "directed_points_pixel": directed_points_pixel,
            "directed_points": directed_points,
            "bbox_pixel": bounds(flat_pixel),
            "bbox_xy": bounds(flat_xy),
            "count": len(directed_points),
        }
    )


def add_directed_polyline_fields(annotation: dict, pending: dict, pixel_to_world: np.ndarray) -> None:
    polyline_pixel = pending["polyline_pixel"]
    polyline_xy = apply_homogeneous(pixel_to_world, np.asarray(polyline_pixel, dtype=np.float64)).tolist()
    start_xy = np.asarray(polyline_xy[0], dtype=np.float64)
    end_xy = np.asarray(polyline_xy[-1], dtype=np.float64)
    direction_xy_arr = end_xy - start_xy
    direction_norm = float(np.linalg.norm(direction_xy_arr))
    direction_xy_unit = (direction_xy_arr / direction_norm).tolist() if direction_norm > 0 else [0.0, 0.0]
    yaw_rad = float(np.arctan2(direction_xy_arr[1], direction_xy_arr[0])) if direction_norm > 0 else 0.0
    annotation.update(
        {
            "polyline_pixel": polyline_pixel,
            "polyline_xy": polyline_xy,
            "direction_xy": direction_xy_arr.tolist(),
            "direction_xy_unit": direction_xy_unit,
            "yaw_rad": yaw_rad,
            "yaw_deg": float(np.degrees(yaw_rad)),
            "length_pixel": polyline_length(polyline_pixel),
            "length_xy": polyline_length(polyline_xy),
            "bbox_pixel": bounds(polyline_pixel),
            "bbox_xy": bounds(polyline_xy),
        }
    )


def directed_polyline_record(polyline_pixel: list[list[float]], pixel_to_world: np.ndarray) -> dict:
    polyline_xy = apply_homogeneous(pixel_to_world, np.asarray(polyline_pixel, dtype=np.float64)).tolist()
    start_xy = np.asarray(polyline_xy[0], dtype=np.float64)
    end_xy = np.asarray(polyline_xy[-1], dtype=np.float64)
    direction_xy_arr = end_xy - start_xy
    direction_norm = float(np.linalg.norm(direction_xy_arr))
    direction_xy_unit = (direction_xy_arr / direction_norm).tolist() if direction_norm > 0 else [0.0, 0.0]
    yaw_rad = float(np.arctan2(direction_xy_arr[1], direction_xy_arr[0])) if direction_norm > 0 else 0.0
    return {
        "polyline_pixel": polyline_pixel,
        "polyline_xy": polyline_xy,
        "direction_xy": direction_xy_arr.tolist(),
        "direction_xy_unit": direction_xy_unit,
        "yaw_rad": yaw_rad,
        "yaw_deg": float(np.degrees(yaw_rad)),
        "length_pixel": polyline_length(polyline_pixel),
        "length_xy": polyline_length(polyline_xy),
        "bbox_pixel": bounds(polyline_pixel),
        "bbox_xy": bounds(polyline_xy),
    }


def add_multi_directed_polyline_fields(annotation: dict, pending: dict, pixel_to_world: np.ndarray) -> None:
    polylines_pixel = pending["polylines_pixel"]
    polylines = [directed_polyline_record(polyline, pixel_to_world) for polyline in polylines_pixel]
    flat_pixel = [point for polyline in polylines_pixel for point in polyline]
    flat_xy = [point for polyline in polylines for point in polyline["polyline_xy"]]
    annotation.update(
        {
            "polylines_pixel": polylines_pixel,
            "polylines": polylines,
            "bbox_pixel": bounds(flat_pixel),
            "bbox_xy": bounds(flat_xy),
            "length_pixel": float(sum(polyline["length_pixel"] for polyline in polylines)),
            "length_xy": float(sum(polyline["length_xy"] for polyline in polylines)),
            "count": len(polylines),
        }
    )


def add_polyline_fields(annotation: dict, pending: dict, pixel_to_world: np.ndarray) -> None:
    polyline_pixel = pending["polyline_pixel"]
    polyline_xy = apply_homogeneous(pixel_to_world, np.asarray(polyline_pixel, dtype=np.float64)).tolist()
    annotation.update(
        {
            "polyline_pixel": polyline_pixel,
            "polyline_xy": polyline_xy,
            "length_pixel": polyline_length(polyline_pixel),
            "length_xy": polyline_length(polyline_xy),
            "bbox_pixel": bounds(polyline_pixel),
            "bbox_xy": bounds(polyline_xy),
        }
    )


def polyline_record(polyline_pixel: list[list[float]], pixel_to_world: np.ndarray) -> dict:
    polyline_xy = apply_homogeneous(pixel_to_world, np.asarray(polyline_pixel, dtype=np.float64)).tolist()
    return {
        "polyline_pixel": polyline_pixel,
        "polyline_xy": polyline_xy,
        "length_pixel": polyline_length(polyline_pixel),
        "length_xy": polyline_length(polyline_xy),
        "bbox_pixel": bounds(polyline_pixel),
        "bbox_xy": bounds(polyline_xy),
    }


def add_multi_polyline_fields(annotation: dict, pending: dict, pixel_to_world: np.ndarray) -> None:
    polylines_pixel = pending["polylines_pixel"]
    polylines = [polyline_record(polyline, pixel_to_world) for polyline in polylines_pixel]
    flat_pixel = [point for polyline in polylines_pixel for point in polyline]
    flat_xy = [point for polyline in polylines for point in polyline["polyline_xy"]]
    annotation.update(
        {
            "polylines_pixel": polylines_pixel,
            "polylines": polylines,
            "bbox_pixel": bounds(flat_pixel),
            "bbox_xy": bounds(flat_xy),
            "length_pixel": float(sum(polyline["length_pixel"] for polyline in polylines)),
            "length_xy": float(sum(polyline["length_xy"] for polyline in polylines)),
            "count": len(polylines),
        }
    )

def add_polygon_fields(annotation: dict, pending: dict, pixel_to_world: np.ndarray) -> None:
    polygon_pixel = pending["polygon_pixel"]
    polygon_xy = apply_homogeneous(pixel_to_world, np.asarray(polygon_pixel, dtype=np.float64)).tolist()
    annotation.update(
        {
            "polygon_pixel": polygon_pixel,
            "polygon_xy": polygon_xy,
            "bbox_pixel": bounds(polygon_pixel),
            "bbox_xy": bounds(polygon_xy),
            "area_pixel": polygon_area(polygon_pixel),
            "area_xy": polygon_area(polygon_xy),
        }
    )


def add_multi_polygon_fields(annotation: dict, pending: dict, pixel_to_world: np.ndarray) -> None:
    polygons_pixel = pending["polygons_pixel"]
    polygons_xy = [
        apply_homogeneous(pixel_to_world, np.asarray(polygon, dtype=np.float64)).tolist()
        for polygon in polygons_pixel
    ]
    flat_pixel = [point for polygon in polygons_pixel for point in polygon]
    flat_xy = [point for polygon in polygons_xy for point in polygon]
    annotation.update(
        {
            "polygons_pixel": polygons_pixel,
            "polygons_xy": polygons_xy,
            "bbox_pixel": bounds(flat_pixel),
            "bbox_xy": bounds(flat_xy),
            "area_pixel": float(sum(polygon_area(polygon) for polygon in polygons_pixel)),
            "area_xy": float(sum(polygon_area(polygon) for polygon in polygons_xy)),
        }
    )


def circle_record(circle_pixel: dict, pixel_to_world: np.ndarray) -> dict:
    center_pixel = circle_pixel["center_pixel"]
    radius_pixel = float(circle_pixel["radius_pixel"])
    center_xy = apply_homogeneous(pixel_to_world, np.asarray([center_pixel], dtype=np.float64))[0].tolist()
    sample_pixels = np.asarray(
        [
            [center_pixel[0] + radius_pixel, center_pixel[1]],
            [center_pixel[0], center_pixel[1] + radius_pixel],
        ],
        dtype=np.float64,
    )
    sample_xy = apply_homogeneous(pixel_to_world, sample_pixels)
    center_xy_arr = np.asarray(center_xy, dtype=np.float64)
    radius_x_xy = float(np.linalg.norm(sample_xy[0] - center_xy_arr))
    radius_y_xy = float(np.linalg.norm(sample_xy[1] - center_xy_arr))
    return {
        "center_pixel": center_pixel,
        "radius_pixel": radius_pixel,
        "center_xy": center_xy,
        "radius_x_xy": radius_x_xy,
        "radius_y_xy": radius_y_xy,
        "radius_xy": float((radius_x_xy + radius_y_xy) * 0.5),
        "area_pixel": float(np.pi * radius_pixel * radius_pixel),
        "area_xy": float(np.pi * radius_x_xy * radius_y_xy),
        "bbox_pixel": {
            "min": [center_pixel[0] - radius_pixel, center_pixel[1] - radius_pixel],
            "max": [center_pixel[0] + radius_pixel, center_pixel[1] + radius_pixel],
        },
        "bbox_xy": bounds(
            apply_homogeneous(
                pixel_to_world,
                np.asarray(
                    [
                        [center_pixel[0] - radius_pixel, center_pixel[1] - radius_pixel],
                        [center_pixel[0] + radius_pixel, center_pixel[1] + radius_pixel],
                    ],
                    dtype=np.float64,
                ),
            ).tolist()
        ),
    }


def add_multi_circle_fields(annotation: dict, pending: dict, pixel_to_world: np.ndarray) -> None:
    circles_pixel = pending["circles_pixel"]
    circles = [circle_record(circle, pixel_to_world) for circle in circles_pixel]
    flat_bbox_pixel = [
        point
        for circle in circles
        for point in (circle["bbox_pixel"]["min"], circle["bbox_pixel"]["max"])
    ]
    flat_bbox_xy = [
        point
        for circle in circles
        for point in (circle["bbox_xy"]["min"], circle["bbox_xy"]["max"])
    ]
    annotation.update(
        {
            "circles_pixel": circles_pixel,
            "circles": circles,
            "bbox_pixel": bounds(flat_bbox_pixel),
            "bbox_xy": bounds(flat_bbox_xy),
            "area_pixel": float(sum(circle["area_pixel"] for circle in circles)),
            "area_xy": float(sum(circle["area_xy"] for circle in circles)),
            "count": len(circles),
        }
    )


def bounds(points: list[list[float]]) -> dict:
    pts = np.asarray(points, dtype=np.float64)
    return {
        "min": np.min(pts, axis=0).tolist(),
        "max": np.max(pts, axis=0).tolist(),
    }
