#!/usr/bin/env python3
"""Annotate 2D objects on an orthographic image and map them to real XY coordinates."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional display enhancement
    Image = None
    ImageDraw = None
    ImageFont = None


from .schema import (
    CATEGORIES,
    FONT_CANDIDATES,
    LABEL_COLORS_BGR,
    apply_homogeneous,
    available_categories,
    category_already_exists,
    has_planning_boundary,
    make_annotation,
    metadata_path_for_image,
    polygon_area,
    polyline_length,
    rectangle_polygon,
)
from substation_vln.interactive import ask_yes_no, choose_numbered_option


AREA_SHAPE_OPTIONS = {
    "1": {"key": "polygon", "name": "多边形", "geometry": "polygon"},
    "2": {"key": "rectangle", "name": "矩形", "geometry": "rectangle"},
    "3": {"key": "circle", "name": "圆形", "geometry": "circle"},
}

# Kept in saved metadata for compatibility with existing annotation files.
OBSTACLE_SHAPE_OPTIONS = AREA_SHAPE_OPTIONS

PREFERRED_PATH_OPTIONS = {
    "1": {"key": "directed", "name": "有向路径", "geometry": "directed_polyline"},
    "2": {"key": "undirected", "name": "无向路径", "geometry": "polyline"},
}


def mouse_wheel_delta(flags: int) -> int:
    if hasattr(cv2, "getMouseWheelDelta"):
        return int(cv2.getMouseWheelDelta(flags))
    delta = (int(flags) >> 16) & 0xFFFF
    if delta >= 0x8000:
        delta -= 0x10000
    return delta


class OrthoImageAnnotator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.image_path = args.image.expanduser().resolve()
        self.metadata_path = args.metadata.expanduser().resolve() if args.metadata else metadata_path_for_image(self.image_path)
        self.output_path = args.output.expanduser().resolve()
        self.review_path = args.review_image.expanduser().resolve()

        if not self.image_path.exists():
            raise SystemExit(f"Image not found: {self.image_path}")
        if not self.metadata_path.exists():
            raise SystemExit(f"Metadata not found: {self.metadata_path}")

        self.image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise SystemExit(f"Failed to read image: {self.image_path}")
        self.height, self.width = self.image.shape[:2]
        print(f"Loaded annotation image: {self.image_path}")
        print(f"Image resolution: {self.width} x {self.height}")

        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.pixel_to_world = np.asarray(self.metadata["pixel_to_world_matrix"], dtype=np.float64)
        self.window_name = "Orthographic obstacle annotation"
        self.window_width = int(args.window_width)
        self.window_height = int(args.window_height) if args.window_height > 0 else self.default_window_height()
        self.pending_window_size: tuple[int, int] | None = None
        self.pending_window_time = 0.0
        self.fit_scale = self.compute_fit_scale()
        self.scale = 1.0 if args.initial_view == "native" else self.fit_scale
        self.min_scale = max(0.05, self.scale * 0.01)
        self.max_scale = max(self.scale * 20.0, 1.0)
        self.center = np.array([self.width * 0.5, self.height * 0.5], dtype=np.float64)

        self.annotations: list[dict] = []
        self.active_category: dict | None = None
        self.pending_annotation: dict | None = None
        self.road_vertices: list[list[float]] = []
        self.closed_polygons: list[list[list[float]]] = []
        self.closed_directed_points: list[dict] = []
        self.closed_polylines: list[list[list[float]]] = []
        self.closed_circles: list[dict] = []
        self.shape_points: list[list[float]] = []
        self.patrol_points: list[list[float]] = []
        self.text_font = self.load_text_font()
        print(f"Initial display scale: {self.scale:.3f} image pixels per screen pixel")
        print(f"Initial window size: {self.window_width} x {self.window_height}")
        print("1 image pixel per screen pixel means native 1:1 display.")
        print("Press 1 for 1:1 native pixels; press F to fit whole image.")

    def default_window_height(self) -> int:
        return max(200, int(round(self.window_width * self.height / self.width)))

    def load_text_font(self):
        if ImageFont is None:
            return None
        for font_path in FONT_CANDIDATES:
            if Path(font_path).exists():
                try:
                    return ImageFont.truetype(font_path, int(self.args.label_font_size))
                except OSError:
                    continue
        return ImageFont.load_default()

    def compute_fit_scale(self) -> float:
        fit_scale_x = self.width / self.window_width
        fit_scale_y = self.height / self.window_height
        return max(fit_scale_x, fit_scale_y, 1.0)

    def read_window_image_size(self) -> tuple[int, int] | None:
        if self.args.disable_dynamic_window_size:
            return None
        try:
            _, _, width, height = cv2.getWindowImageRect(self.window_name)
        except cv2.error:
            return None
        if width < self.args.min_window_width or height < self.args.min_window_height:
            return None
        if not self.args.allow_free_window_aspect:
            expected_ratio = self.width / self.height
            actual_ratio = width / height
            if actual_ratio < expected_ratio * 0.5 or actual_ratio > expected_ratio * 2.0:
                return None
        return int(width), int(height)

    def update_window_image_size(self, *, force: bool = False) -> bool:
        size = self.read_window_image_size()
        if size is None:
            return False
        width, height = size
        if width == self.window_width and height == self.window_height:
            self.pending_window_size = None
            return False

        now = time.monotonic()
        if self.pending_window_size != size:
            self.pending_window_size = size
            self.pending_window_time = now
            return False

        stable_seconds = max(0.0, self.args.window_resize_debounce_ms / 1000.0)
        if force or now - self.pending_window_time >= stable_seconds:
            old_center = self.center.copy()
            old_width = self.window_width
            old_height = self.window_height
            self.window_width = width
            self.window_height = height
            self.center = old_center
            self.clamp_center()
            self.pending_window_size = None
            print(f"Display viewport resized: {old_width}x{old_height} -> {self.window_width}x{self.window_height}")
            return True
        return False

    def set_fit_view(self) -> None:
        self.scale = self.compute_fit_scale()
        self.center = np.array([self.width * 0.5, self.height * 0.5], dtype=np.float64)
        print(f"Fit view scale: {self.scale:.3f} image pixels per screen pixel")

    def set_one_to_one_view(self) -> None:
        self.scale = 1.0
        self.clamp_center()
        print("1:1 view enabled.")

    def zoom_at_window_center(self, factor: float) -> None:
        self.scale = float(np.clip(self.scale * factor, self.min_scale, self.max_scale))
        self.clamp_center()

    def pan_by_fraction(self, dx_fraction: float, dy_fraction: float) -> None:
        self.center[0] += self.window_width * self.scale * dx_fraction
        self.center[1] += self.window_height * self.scale * dy_fraction
        self.clamp_center()

    def screen_to_image(self, x: int, y: int) -> tuple[float, float]:
        col = self.center[0] + (x - self.window_width * 0.5) * self.scale
        row = self.center[1] + (y - self.window_height * 0.5) * self.scale
        return float(col), float(row)

    def image_to_screen(self, col: float, row: float) -> tuple[int, int]:
        x = int(round((col - self.center[0]) / self.scale + self.window_width * 0.5))
        y = int(round((row - self.center[1]) / self.scale + self.window_height * 0.5))
        return x, y

    def clamp_center(self) -> None:
        margin_x = self.window_width * self.scale * 0.5
        margin_y = self.window_height * self.scale * 0.5
        self.center[0] = float(np.clip(self.center[0], -margin_x, self.width + margin_x))
        self.center[1] = float(np.clip(self.center[1], -margin_y, self.height + margin_y))

    def world_radius_to_pixel_radius(self, radius_m: float, center_pixel: list[float] | None = None) -> float:
        center = np.asarray(center_pixel if center_pixel is not None else [self.width * 0.5, self.height * 0.5], dtype=np.float64)
        sample_pixels = np.asarray(
            [
                center,
                center + np.asarray([1.0, 0.0], dtype=np.float64),
                center + np.asarray([0.0, 1.0], dtype=np.float64),
            ],
            dtype=np.float64,
        )
        sample_xy = apply_homogeneous(self.pixel_to_world, sample_pixels)
        meters_per_pixel_x = float(np.linalg.norm(sample_xy[1] - sample_xy[0]))
        meters_per_pixel_y = float(np.linalg.norm(sample_xy[2] - sample_xy[0]))
        meters_per_pixel = float((meters_per_pixel_x + meters_per_pixel_y) * 0.5)
        if meters_per_pixel <= 0:
            raise ValueError("Invalid pixel-to-world scale; cannot convert circle radius.")
        return float(radius_m / meters_per_pixel)

    def horizontal_meters_per_image_pixel(self) -> float:
        center = np.asarray([self.width * 0.5, self.height * 0.5], dtype=np.float64)
        sample_xy = apply_homogeneous(
            self.pixel_to_world,
            np.asarray([center, center + np.asarray([1.0, 0.0])], dtype=np.float64),
        )
        meters_per_pixel = float(np.linalg.norm(sample_xy[1] - sample_xy[0]))
        if meters_per_pixel <= 0:
            raise ValueError("Invalid pixel-to-world scale; cannot draw scale bars.")
        return meters_per_pixel

    def draw_scale_bars(self, view: np.ndarray) -> None:
        viewport_height, viewport_width = view.shape[:2]
        meters_per_screen_pixel = self.horizontal_meters_per_image_pixel() * self.scale
        candidate_distances_m = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
        target_length_px = 130.0
        min_length_px = 65.0
        max_length_px = 180.0
        fitting = [
            (distance, distance / meters_per_screen_pixel)
            for distance in candidate_distances_m
            if min_length_px <= distance / meters_per_screen_pixel <= max_length_px
        ]
        if fitting:
            distance_m, length_px = min(fitting, key=lambda item: abs(item[1] - target_length_px))
        else:
            distance_m, length_px = min(
                ((distance, distance / meters_per_screen_pixel) for distance in candidate_distances_m),
                key=lambda item: abs(item[1] - target_length_px),
            )

        panel_width = min(250, max(170, viewport_width - 24))
        panel_height = 66
        panel_right = viewport_width - 12
        panel_left = max(12, panel_right - panel_width)
        panel_bottom = viewport_height - 12
        panel_top = max(12, panel_bottom - panel_height)
        overlay = view.copy()
        cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), (20, 20, 20), -1)
        view[:] = cv2.addWeighted(overlay, 0.58, view, 0.42, 0.0)

        left = panel_left + 18
        y = panel_bottom - 20
        right = min(panel_right - 18, left + max(1, int(round(length_px))))
        label = f"{distance_m:g} m"
        cv2.line(view, (left, y), (right, y), (0, 0, 0), 6, cv2.LINE_AA)
        cv2.line(view, (left, y), (right, y), (255, 255, 255), 3, cv2.LINE_AA)
        for x in (left, right):
            cv2.line(view, (x, y - 7), (x, y + 7), (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(view, (x, y - 7), (x, y + 7), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, label, (left, panel_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, label, (left, panel_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

    def mouse_callback(self, event, x, y, flags, param) -> None:
        category = self.active_category
        geometry = category["geometry"] if category else None

        if event == cv2.EVENT_MOUSEWHEEL:
            self.update_window_image_size(force=True)
            wheel_delta = mouse_wheel_delta(flags)
            if wheel_delta == 0:
                return
            before = np.asarray(self.screen_to_image(x, y), dtype=np.float64)
            factor = 0.85 if wheel_delta > 0 else 1.18
            self.scale = float(np.clip(self.scale * factor, self.min_scale, self.max_scale))
            after = np.asarray(self.screen_to_image(x, y), dtype=np.float64)
            self.center += before - after
            self.clamp_center()
            return

        if event == cv2.EVENT_RBUTTONDOWN and geometry == "polygon":
            self.close_current_polygon()
            return

        if event == cv2.EVENT_RBUTTONDOWN and geometry in ("directed_polyline", "polyline"):
            self.close_current_polyline()
            return

        if event == cv2.EVENT_LBUTTONDOWN and geometry == "directed_point":
            point = list(self.screen_to_image(x, y))
            self.patrol_points.append(point)
            if len(self.patrol_points) == 1:
                print("Added patrol stop point. Left-click a second point to set viewing direction.")
            elif len(self.patrol_points) >= 2:
                stop_pixel, look_pixel = self.patrol_points[:2]
                if np.linalg.norm(np.asarray(look_pixel) - np.asarray(stop_pixel)) < self.args.min_direction_pixel_length:
                    print("Ignored patrol direction: second point is too close to stop point.")
                    self.patrol_points = [stop_pixel]
                else:
                    self.closed_directed_points.append({"stop_pixel": stop_pixel, "look_pixel": look_pixel})
                    self.patrol_points = []
                    print(
                        f"Completed patrol point #{len(self.closed_directed_points)}. "
                        "Press Enter to finish this annotation, or keep adding patrol points."
                    )
            return

        if event == cv2.EVENT_LBUTTONDOWN and geometry == "polygon":
            if category and category.get("single") and self.closed_polygons:
                print(f"{category['name']}只能包含一个闭合多边形。按 Enter 完成本次标注。")
                return
            point = list(self.screen_to_image(x, y))
            self.road_vertices.append(point)
            category_name = category["name"] if category else "polygon"
            print(f"Added {category_name} vertex #{len(self.road_vertices)}. Right-click to close polygon.")
            return

        if event == cv2.EVENT_LBUTTONDOWN and geometry == "rectangle":
            point = list(self.screen_to_image(x, y))
            self.shape_points.append(point)
            if len(self.shape_points) == 1:
                print("Added rectangle corner. Left-click the opposite corner.")
            elif len(self.shape_points) >= 2:
                polygon = rectangle_polygon(self.shape_points[0], self.shape_points[1])
                if polygon_area(polygon) < self.args.min_pixel_area:
                    print("Ignored tiny rectangle.")
                else:
                    self.closed_polygons.append(polygon)
                    print(f"Completed rectangle #{len(self.closed_polygons)}. Press Enter to finish, or keep adding rectangles.")
                self.shape_points = []
            return

        if event == cv2.EVENT_LBUTTONDOWN and geometry == "circle":
            point = list(self.screen_to_image(x, y))
            radius = self.world_radius_to_pixel_radius(float(self.args.default_circle_radius_m), point)
            self.closed_circles.append(
                {
                    "center_pixel": point,
                    "radius_pixel": radius,
                    "default_radius_m": float(self.args.default_circle_radius_m),
                }
            )
            print(
                f"Completed circle #{len(self.closed_circles)} "
                f"with radius {self.args.default_circle_radius_m:.2f} m. "
                "Press Enter to finish, or keep adding circles."
            )
            return

        if event == cv2.EVENT_LBUTTONDOWN and geometry in ("directed_polyline", "polyline"):
            point = list(self.screen_to_image(x, y))
            self.road_vertices.append(point)
            category_name = category["name"] if category else "polyline"
            print(f"Added {category_name} waypoint #{len(self.road_vertices)}. Right-click to finish current path.")
            return

        return

    def close_current_polygon(self) -> None:
        if len(self.road_vertices) < 3:
            print("Polygon needs at least 3 vertices.")
            return
        polygon = [list(point) for point in self.road_vertices]
        if polygon_area(polygon) < self.args.min_pixel_area:
            print("Ignored tiny polygon.")
            self.road_vertices = []
            return
        self.closed_polygons.append(polygon)
        self.road_vertices = []
        print(f"Closed polygon #{len(self.closed_polygons)}. Press Enter to finish this annotation, or keep drawing another polygon.")

    def finish_current_polygons(self) -> None:
        if self.road_vertices:
            print("Current polygon is not closed. Right-click to close it before pressing Enter.")
            return
        if self.shape_points:
            print("Current rectangle is incomplete. Left-click the opposite corner before pressing Enter.")
            return
        if not self.closed_polygons:
            if self.active_category and self.active_category["geometry"] == "rectangle":
                print("No completed rectangle in current annotation.")
            else:
                print("No closed polygon in current annotation.")
            return
        selection_type = "image_multi_rectangle" if self.active_category and self.active_category["geometry"] == "rectangle" else "image_multi_polygon"
        self.pending_annotation = {
            "selection_type": selection_type,
            "geometry_type": "multi_polygon",
            "polygons_pixel": [list(polygon) for polygon in self.closed_polygons],
        }
        self.closed_polygons = []

    def finish_current_circles(self) -> None:
        if not self.closed_circles:
            print("No completed circle in current annotation.")
            return
        self.pending_annotation = {
            "selection_type": "image_multi_circle",
            "geometry_type": "multi_circle",
            "circles_pixel": [dict(circle) for circle in self.closed_circles],
        }
        self.closed_circles = []

    def close_current_polyline(self) -> None:
        if len(self.road_vertices) < 2:
            print("Path needs at least 2 waypoints.")
            return
        polyline = [list(point) for point in self.road_vertices]
        if polyline_length(polyline) < self.args.min_direction_pixel_length:
            print("Ignored tiny path.")
            self.road_vertices = []
            return
        self.closed_polylines.append(polyline)
        self.road_vertices = []
        print(f"Closed path #{len(self.closed_polylines)}. Press Enter to finish this annotation, or keep drawing another path.")

    def finish_current_polylines(self) -> None:
        if self.road_vertices:
            print("Current path is not finished. Right-click to finish it before pressing Enter.")
            return
        if not self.closed_polylines:
            print("No closed path in current annotation.")
            return
        is_directed = self.active_category and self.active_category["geometry"] == "directed_polyline"
        self.pending_annotation = {
            "selection_type": "image_multi_directed_polyline" if is_directed else "image_multi_polyline",
            "geometry_type": "multi_directed_polyline" if is_directed else "multi_polyline",
            "polylines_pixel": [list(polyline) for polyline in self.closed_polylines],
        }
        self.closed_polylines = []

    def finish_current_directed_points(self) -> None:
        if self.patrol_points:
            print("Current patrol point is incomplete. Left-click the direction point before pressing Enter.")
            return
        if not self.closed_directed_points:
            print("No completed patrol point in current annotation.")
            return
        self.pending_annotation = {
            "selection_type": "image_multi_directed_point",
            "geometry_type": "multi_directed_point",
            "directed_points_pixel": [dict(item) for item in self.closed_directed_points],
        }
        self.closed_directed_points = []

    def render_view(self) -> np.ndarray:
        if self.args.display_interpolation == "auto":
            interpolation = self.auto_interpolation()
        else:
            interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "area": cv2.INTER_AREA,
                "cubic": cv2.INTER_CUBIC,
                "lanczos": cv2.INTER_LANCZOS4,
            }[self.args.display_interpolation]

        view = self.render_image_crop(interpolation)

        for annotation in self.annotations:
            if annotation.get("category") == "planning_boundary":
                continue
            color = tuple(annotation["color_bgr"])
            if annotation["geometry_type"] == "directed_point":
                stop = self.image_to_screen(*annotation["stop_pixel"])
                look = self.image_to_screen(*annotation["look_pixel"])
                cv2.arrowedLine(view, stop, look, color, thickness=max(2, int(self.args.line_width)), tipLength=0.25)
                cv2.circle(view, stop, max(5, int(self.args.line_width * 3)), color, thickness=-1)
                cv2.circle(view, stop, max(8, int(self.args.line_width * 5)), (255, 255, 255), thickness=max(1, int(self.args.line_width)))
            elif annotation["geometry_type"] == "multi_directed_point":
                for item in annotation["directed_points"]:
                    stop = self.image_to_screen(*item["stop_pixel"])
                    look = self.image_to_screen(*item["look_pixel"])
                    cv2.arrowedLine(view, stop, look, color, thickness=max(2, int(self.args.line_width)), tipLength=0.25)
                    cv2.circle(view, stop, max(5, int(self.args.line_width * 3)), color, thickness=-1)
                    cv2.circle(view, stop, max(8, int(self.args.line_width * 5)), (255, 255, 255), thickness=max(1, int(self.args.line_width)))
            elif annotation["geometry_type"] == "directed_polyline":
                pts = np.asarray([self.image_to_screen(c, r) for c, r in annotation["polyline_pixel"]], dtype=np.int32)
                self.draw_directed_polyline(view, pts, color)
            elif annotation["geometry_type"] == "multi_directed_polyline":
                for polyline in annotation["polylines"]:
                    pts = np.asarray([self.image_to_screen(c, r) for c, r in polyline["polyline_pixel"]], dtype=np.int32)
                    self.draw_directed_polyline(view, pts, color)
            elif annotation["geometry_type"] == "polyline":
                pts = np.asarray([self.image_to_screen(c, r) for c, r in annotation["polyline_pixel"]], dtype=np.int32)
                self.draw_polyline(view, pts, color)
            elif annotation["geometry_type"] == "multi_polyline":
                for polyline in annotation["polylines"]:
                    pts = np.asarray([self.image_to_screen(c, r) for c, r in polyline["polyline_pixel"]], dtype=np.int32)
                    self.draw_polyline(view, pts, color)
            elif annotation["geometry_type"] == "multi_polygon":
                for polygon in annotation["polygons_pixel"]:
                    pts = np.asarray([self.image_to_screen(c, r) for c, r in polygon], dtype=np.int32)
                    overlay = view.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    view = cv2.addWeighted(overlay, 0.18, view, 0.82, 0.0)
                    cv2.polylines(view, [pts], isClosed=True, color=color, thickness=max(2, int(self.args.line_width)))
            elif annotation["geometry_type"] == "multi_circle":
                for circle in annotation["circles"]:
                    self.draw_circle(view, circle["center_pixel"], circle["radius_pixel"], color, fill_alpha=0.18)
            else:
                pts = np.asarray([self.image_to_screen(c, r) for c, r in annotation["polygon_pixel"]], dtype=np.int32)
                overlay = view.copy()
                cv2.fillPoly(overlay, [pts], color)
                view = cv2.addWeighted(overlay, 0.18, view, 0.82, 0.0)
                cv2.polylines(view, [pts], isClosed=True, color=color, thickness=max(2, int(self.args.line_width)))

        for polygon in self.closed_polygons:
            pts = np.asarray([self.image_to_screen(c, r) for c, r in polygon], dtype=np.int32)
            overlay = view.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            view = cv2.addWeighted(overlay, 0.10, view, 0.90, 0.0)
            cv2.polylines(view, [pts], isClosed=True, color=(0, 255, 255), thickness=max(2, int(self.args.line_width)))

        for circle in self.closed_circles:
            self.draw_circle(view, circle["center_pixel"], circle["radius_pixel"], (0, 255, 255), fill_alpha=0.10)

        for item in self.closed_directed_points:
            stop = self.image_to_screen(*item["stop_pixel"])
            look = self.image_to_screen(*item["look_pixel"])
            cv2.arrowedLine(view, stop, look, (0, 255, 255), thickness=max(2, int(self.args.line_width)), tipLength=0.25)
            cv2.circle(view, stop, max(5, int(self.args.line_width * 3)), (0, 255, 255), thickness=-1)

        for polyline in self.closed_polylines:
            pts = np.asarray([self.image_to_screen(c, r) for c, r in polyline], dtype=np.int32)
            if self.active_category and self.active_category["geometry"] == "polyline":
                self.draw_polyline(view, pts, (0, 255, 255))
            else:
                self.draw_directed_polyline(view, pts, (0, 255, 255))

        if self.road_vertices:
            pts = np.asarray([self.image_to_screen(c, r) for c, r in self.road_vertices], dtype=np.int32)
            if self.active_category and self.active_category["geometry"] in ("directed_polyline", "polyline"):
                if self.active_category["geometry"] == "polyline":
                    self.draw_polyline(view, pts, (0, 255, 255))
                else:
                    self.draw_directed_polyline(view, pts, (0, 255, 255))
            else:
                for point in pts:
                    cv2.circle(view, tuple(point), max(4, int(self.args.line_width * 2)), (0, 255, 255), thickness=-1)
                if len(pts) >= 2:
                    cv2.polylines(view, [pts], isClosed=False, color=(0, 255, 255), thickness=max(2, int(self.args.line_width)))

        if self.shape_points:
            pts = [self.image_to_screen(c, r) for c, r in self.shape_points]
            for point in pts:
                cv2.circle(view, point, max(4, int(self.args.line_width * 2)), (0, 255, 255), thickness=-1)
            if len(self.shape_points) == 2 and self.active_category:
                if self.active_category["geometry"] == "rectangle":
                    polygon = rectangle_polygon(self.shape_points[0], self.shape_points[1])
                    rect_pts = np.asarray([self.image_to_screen(c, r) for c, r in polygon], dtype=np.int32)
                    cv2.polylines(view, [rect_pts], isClosed=True, color=(0, 255, 255), thickness=max(2, int(self.args.line_width)))
                elif self.active_category["geometry"] == "circle":
                    radius = float(np.linalg.norm(np.asarray(self.shape_points[1]) - np.asarray(self.shape_points[0])))
                    self.draw_circle(view, self.shape_points[0], radius, (0, 255, 255), fill_alpha=0.0)

        if self.patrol_points:
            stop = self.image_to_screen(*self.patrol_points[0])
            cv2.circle(view, stop, max(5, int(self.args.line_width * 3)), (0, 255, 255), thickness=-1)

        if self.pending_annotation is not None:
            self.draw_pending_annotation(view)

        self.draw_annotation_labels(view)
        self.draw_scale_bars(view)

        category_name = self.active_category["key"] if self.active_category else "none"
        if self.active_category and self.active_category["geometry"] == "directed_point":
            tool_hint = "Left: stop+direction | Enter: finish | W/A/S/D: pan"
        elif self.active_category and self.active_category["geometry"] == "directed_polyline":
            tool_hint = "Left: waypoint | Right-click: close path | Enter: finish | W/A/S/D: pan"
        elif self.active_category and self.active_category["geometry"] == "polyline":
            tool_hint = "Left: waypoint | Right-click: close path | Enter: finish | W/A/S/D: pan"
        elif self.active_category and self.active_category["geometry"] == "polygon":
            tool_hint = "Left: vertex | Right-click: close | Enter: finish | W/A/S/D: pan"
        elif self.active_category and self.active_category["geometry"] == "rectangle":
            tool_hint = "Left: corner/opposite | Enter: finish | W/A/S/D: pan"
        elif self.active_category and self.active_category["geometry"] == "circle":
            tool_hint = "Left: circle center | Enter: finish | W/A/S/D: pan"
        else:
            tool_hint = "Left: annotate | W/A/S/D: pan"
        cv2.putText(
            view,
            f"Category: {category_name} | {tool_hint} | Q: save",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            f"Category: {category_name} | {tool_hint} | Q: save",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return view

    def auto_interpolation(self) -> int:
        if self.scale > 1.0:
            return cv2.INTER_AREA
        if self.scale < 1.0:
            return cv2.INTER_LANCZOS4
        return cv2.INTER_NEAREST

    def draw_pending_annotation(self, view: np.ndarray) -> None:
        pending = self.pending_annotation
        if pending is None:
            return
        color = (0, 255, 255)
        if pending["geometry_type"] == "directed_point":
            stop = self.image_to_screen(*pending["stop_pixel"])
            look = self.image_to_screen(*pending["look_pixel"])
            cv2.arrowedLine(view, stop, look, color, thickness=max(2, int(self.args.line_width)), tipLength=0.25)
            cv2.circle(view, stop, max(5, int(self.args.line_width * 3)), color, thickness=-1)
        elif pending["geometry_type"] == "multi_directed_point":
            for item in pending["directed_points_pixel"]:
                stop = self.image_to_screen(*item["stop_pixel"])
                look = self.image_to_screen(*item["look_pixel"])
                cv2.arrowedLine(view, stop, look, color, thickness=max(2, int(self.args.line_width)), tipLength=0.25)
                cv2.circle(view, stop, max(5, int(self.args.line_width * 3)), color, thickness=-1)
        elif pending["geometry_type"] == "directed_polyline":
            pts = np.asarray([self.image_to_screen(c, r) for c, r in pending["polyline_pixel"]], dtype=np.int32)
            self.draw_directed_polyline(view, pts, color)
        elif pending["geometry_type"] == "multi_directed_polyline":
            for polyline in pending["polylines_pixel"]:
                pts = np.asarray([self.image_to_screen(c, r) for c, r in polyline], dtype=np.int32)
                self.draw_directed_polyline(view, pts, color)
        elif pending["geometry_type"] == "polyline":
            pts = np.asarray([self.image_to_screen(c, r) for c, r in pending["polyline_pixel"]], dtype=np.int32)
            self.draw_polyline(view, pts, color)
        elif pending["geometry_type"] == "multi_polyline":
            for polyline in pending["polylines_pixel"]:
                pts = np.asarray([self.image_to_screen(c, r) for c, r in polyline], dtype=np.int32)
                self.draw_polyline(view, pts, color)
        elif pending["geometry_type"] == "multi_polygon":
            for polygon in pending["polygons_pixel"]:
                pts = np.asarray([self.image_to_screen(c, r) for c, r in polygon], dtype=np.int32)
                overlay = view.copy()
                cv2.fillPoly(overlay, [pts], color)
                blended = cv2.addWeighted(overlay, 0.12, view, 0.88, 0.0)
                view[:] = blended
                cv2.polylines(view, [pts], isClosed=True, color=color, thickness=max(2, int(self.args.line_width)))
        elif pending["geometry_type"] == "multi_circle":
            for circle in pending["circles_pixel"]:
                self.draw_circle(view, circle["center_pixel"], circle["radius_pixel"], color, fill_alpha=0.12)
        else:
            pts = np.asarray([self.image_to_screen(c, r) for c, r in pending["polygon_pixel"]], dtype=np.int32)
            overlay = view.copy()
            cv2.fillPoly(overlay, [pts], color)
            blended = cv2.addWeighted(overlay, 0.12, view, 0.88, 0.0)
            view[:] = blended
            cv2.polylines(view, [pts], isClosed=True, color=color, thickness=max(2, int(self.args.line_width)))

    def draw_directed_polyline(self, view: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
        if len(pts) == 0:
            return
        for point in pts:
            cv2.circle(view, tuple(point), max(3, int(self.args.line_width * 2)), color, thickness=-1)
        if len(pts) < 2:
            return
        cv2.polylines(view, [pts], isClosed=False, color=color, thickness=max(2, int(self.args.line_width)))
        start = tuple(pts[-2])
        end = tuple(pts[-1])
        cv2.arrowedLine(view, start, end, color, thickness=max(2, int(self.args.line_width)), tipLength=0.25)

    def draw_polyline(self, view: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
        if len(pts) == 0:
            return
        for point in pts:
            cv2.circle(view, tuple(point), max(3, int(self.args.line_width * 2)), color, thickness=-1)
        if len(pts) >= 2:
            cv2.polylines(view, [pts], isClosed=False, color=color, thickness=max(2, int(self.args.line_width)))

    def draw_circle(
        self,
        view: np.ndarray,
        center_pixel: list[float],
        radius_pixel: float,
        color: tuple[int, int, int],
        *,
        fill_alpha: float,
    ) -> None:
        center = self.image_to_screen(*center_pixel)
        radius = max(1, int(round(float(radius_pixel) / self.scale)))
        if fill_alpha > 0:
            overlay = view.copy()
            cv2.circle(overlay, center, radius, color, thickness=-1)
            blended = cv2.addWeighted(overlay, fill_alpha, view, 1.0 - fill_alpha, 0.0)
            view[:] = blended
        cv2.circle(view, center, radius, color, thickness=max(2, int(self.args.line_width)))
        cv2.circle(view, center, max(3, int(self.args.line_width * 2)), color, thickness=-1)

    def draw_annotation_labels(self, view: np.ndarray) -> None:
        if not self.annotations:
            return

        labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
        for annotation in self.annotations:
            if annotation.get("category") == "planning_boundary":
                continue
            label = str(annotation.get("label", ""))
            if not label:
                continue
            if annotation["geometry_type"] == "directed_point":
                x, y = self.image_to_screen(*annotation["stop_pixel"])
                position = (x + 10, y - 10)
            elif annotation["geometry_type"] == "multi_directed_point":
                first = annotation["directed_points"][0]
                x, y = self.image_to_screen(*first["stop_pixel"])
                position = (x + 10, y - 10)
            elif annotation["geometry_type"] == "directed_polyline":
                pts = np.asarray(annotation["polyline_pixel"], dtype=np.float64)
                mid = pts[len(pts) // 2]
                position = self.image_to_screen(float(mid[0]), float(mid[1]))
            elif annotation["geometry_type"] == "multi_directed_polyline":
                first = annotation["polylines"][0]
                pts = np.asarray(first["polyline_pixel"], dtype=np.float64)
                mid = pts[len(pts) // 2]
                position = self.image_to_screen(float(mid[0]), float(mid[1]))
            elif annotation["geometry_type"] == "polyline":
                pts = np.asarray(annotation["polyline_pixel"], dtype=np.float64)
                mid = pts[len(pts) // 2]
                position = self.image_to_screen(float(mid[0]), float(mid[1]))
            elif annotation["geometry_type"] == "multi_polyline":
                first = annotation["polylines"][0]
                pts = np.asarray(first["polyline_pixel"], dtype=np.float64)
                mid = pts[len(pts) // 2]
                position = self.image_to_screen(float(mid[0]), float(mid[1]))
            elif annotation["geometry_type"] == "multi_polygon":
                pts = np.asarray([point for polygon in annotation["polygons_pixel"] for point in polygon], dtype=np.float64)
                cx, cy = np.mean(pts, axis=0)
                position = self.image_to_screen(float(cx), float(cy))
            elif annotation["geometry_type"] == "multi_circle":
                first = annotation["circles"][0]
                x, y = self.image_to_screen(*first["center_pixel"])
                position = (x + 10, y - 10)
            else:
                pts = np.asarray(annotation["polygon_pixel"], dtype=np.float64)
                cx, cy = np.mean(pts, axis=0)
                position = self.image_to_screen(float(cx), float(cy))
            labels.append((label, position, tuple(annotation["color_bgr"])))

        if not labels:
            return

        if Image is None or ImageDraw is None or self.text_font is None:
            for label, position, color in labels:
                cv2.putText(
                    view,
                    label,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    view,
                    label,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            return

        image_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        for label, position, color_bgr in labels:
            x, y = position
            color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
            bbox = draw.textbbox((x, y), label, font=self.text_font)
            pad = 3
            draw.rectangle(
                (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
                fill=(255, 255, 255),
                outline=(0, 0, 0),
            )
            draw.text((x, y), label, fill=color_rgb, font=self.text_font)
        view[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    def render_image_crop(self, interpolation: int) -> np.ndarray:
        """Render the current viewport by cropping source pixels before resizing.

        This behaves closer to a normal image viewer than warping the full image:
        every zoom/pan frame samples the corresponding region from the original
        8K image, then scales only that region to the display window.
        """
        visible_width = self.window_width * self.scale
        visible_height = self.window_height * self.scale
        left = self.center[0] - visible_width * 0.5
        top = self.center[1] - visible_height * 0.5
        right = self.center[0] + visible_width * 0.5
        bottom = self.center[1] + visible_height * 0.5

        src_left = int(np.floor(left))
        src_top = int(np.floor(top))
        src_right = int(np.ceil(right))
        src_bottom = int(np.ceil(bottom))
        src_width = max(1, src_right - src_left)
        src_height = max(1, src_bottom - src_top)

        crop = np.full((src_height, src_width, 3), 255, dtype=np.uint8)

        image_left = max(0, src_left)
        image_top = max(0, src_top)
        image_right = min(self.width, src_right)
        image_bottom = min(self.height, src_bottom)

        if image_right > image_left and image_bottom > image_top:
            dst_left = image_left - src_left
            dst_top = image_top - src_top
            crop[
                dst_top : dst_top + (image_bottom - image_top),
                dst_left : dst_left + (image_right - image_left),
            ] = self.image[image_top:image_bottom, image_left:image_right]

        if crop.shape[1] == self.window_width and crop.shape[0] == self.window_height:
            return crop.copy()

        return cv2.resize(crop, (self.window_width, self.window_height), interpolation=interpolation)

    def add_pending_annotation(self) -> bool:
        if self.pending_annotation is None:
            return False
        if self.active_category is None:
            self.pending_annotation = None
            return False

        pending = self.pending_annotation
        category = self.active_category
        if category.get("single") and category_already_exists(self.annotations, category):
            self.pending_annotation = None
            print(f"{category['name']}只能标注一次，已忽略本次标注。")
            return False

        cv2.imshow(self.window_name, self.render_view())
        cv2.waitKey(1)
        equipment_fields: dict = {}
        if category["key"] == "equipment_region":
            equipment_name = input("设备名称（应与操作票中的名称一致）: ").strip()
            while not equipment_name:
                print("设备名称不能为空。")
                equipment_name = input("设备名称: ").strip()
            equipment_type = input("设备类型 [unknown_device]: ").strip() or "unknown_device"
            label = equipment_name
            equipment_fields = {
                "equipment_name": equipment_name,
                "equipment_type": equipment_type,
            }
        else:
            default_label = category["default_label"]
            label = input(f"{category['name']} label [{default_label}]: ").strip() or default_label
        annotation_id = len(self.annotations) + 1
        color = LABEL_COLORS_BGR[(annotation_id - 1) % len(LABEL_COLORS_BGR)]
        annotation = make_annotation(
            annotation_id=annotation_id,
            category=category,
            label=label,
            pending=pending,
            pixel_to_world=self.pixel_to_world,
            color_bgr=color,
        )
        annotation.update(equipment_fields)

        if annotation["geometry_type"] == "directed_point":
            print(
                f"Pending annotation #{annotation_id}: "
                f"category={category['key']}, label={label}, yaw_deg={annotation['yaw_deg']:.2f}"
            )
        elif annotation["geometry_type"] == "multi_directed_point":
            print(
                f"Pending annotation #{annotation_id}: "
                f"category={category['key']}, label={label}, count={annotation['count']}"
            )
        elif annotation["geometry_type"] == "directed_polyline":
            print(
                f"Pending annotation #{annotation_id}: "
                f"category={category['key']}, label={label}, length_xy={annotation['length_xy']:.3f}"
            )
        elif annotation["geometry_type"] == "multi_directed_polyline":
            print(
                f"Pending annotation #{annotation_id}: "
                f"category={category['key']}, label={label}, count={annotation['count']}, length_xy={annotation['length_xy']:.3f}"
            )
        elif annotation["geometry_type"] == "polyline":
            print(
                f"Pending annotation #{annotation_id}: "
                f"category={category['key']}, label={label}, length_xy={annotation['length_xy']:.3f}"
            )
        elif annotation["geometry_type"] == "multi_polyline":
            print(
                f"Pending annotation #{annotation_id}: "
                f"category={category['key']}, label={label}, count={annotation['count']}, length_xy={annotation['length_xy']:.3f}"
            )
        elif annotation["geometry_type"] == "multi_circle":
            print(
                f"Pending annotation #{annotation_id}: "
                f"category={category['key']}, label={label}, count={annotation['count']}, area_xy={annotation['area_xy']:.3f}"
            )
        else:
            print(
                f"Pending annotation #{annotation_id}: "
                f"category={category['key']}, label={label}, area_xy={annotation['area_xy']:.3f}"
            )

        if not self.confirm_annotation():
            self.pending_annotation = None
            print("Discarded current annotation.")
            return False

        self.annotations.append(annotation)
        self.pending_annotation = None
        print(f"Confirmed annotation #{annotation_id}: category={category['key']}, label={label}")
        return True

    def confirm_annotation(self) -> bool:
        return ask_yes_no("确认保存该标注结果？", default=True)

    def undo(self) -> None:
        if self.patrol_points:
            removed = self.patrol_points.pop()
            print(f"Removed current patrol point: {removed}")
            return
        if self.road_vertices:
            removed = self.road_vertices.pop()
            print(f"Removed current polygon vertex: {removed}")
            return
        if self.shape_points:
            removed = self.shape_points.pop()
            print(f"Removed current shape point: {removed}")
            return
        if self.closed_polylines:
            removed = self.closed_polylines.pop()
            print(f"Removed path with {len(removed)} waypoints from current annotation.")
            return
        if self.closed_directed_points:
            removed = self.closed_directed_points.pop()
            print(f"Removed completed patrol point: {removed}")
            return
        if self.closed_polygons:
            removed = self.closed_polygons.pop()
            print(f"Removed closed polygon with {len(removed)} vertices from current annotation.")
            return
        if self.closed_circles:
            removed = self.closed_circles.pop()
            print(f"Removed circle: {removed}")
            return
        if not self.annotations:
            print("No annotation to undo.")
            return
        annotation = self.annotations.pop()
        print(f"Undid annotation #{annotation['id']}: label={annotation['label']}")

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "annotation_kind": "ortho_2d",
            "image": str(self.image_path),
            "metadata": str(self.metadata_path),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "coordinate_system": "axis_corrected_pointcloud",
            "pixel_to_world_matrix": self.pixel_to_world.tolist(),
            "world_to_pixel_matrix": self.metadata.get("world_to_pixel_matrix"),
            "image_size": {"width": int(self.width), "height": int(self.height)},
            "categories": CATEGORIES,
            "obstacle_shape_options": OBSTACLE_SHAPE_OPTIONS,
            "preferred_path_options": PREFERRED_PATH_OPTIONS,
            "annotations": self.annotations,
        }
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved annotations: {self.output_path}")
        self.save_review_image()

    def save_review_image(self) -> None:
        review = self.image.copy()
        for annotation in self.annotations:
            if annotation.get("category") == "planning_boundary":
                continue
            color = tuple(annotation["color_bgr"])
            if annotation["geometry_type"] == "directed_point":
                stop = tuple(np.asarray(annotation["stop_pixel"], dtype=np.int32))
                look = tuple(np.asarray(annotation["look_pixel"], dtype=np.int32))
                cv2.arrowedLine(review, stop, look, color, thickness=max(4, int(self.args.line_width * 2)), tipLength=0.25)
                cv2.circle(review, stop, max(12, int(self.args.line_width * 6)), color, thickness=-1)
                cv2.circle(review, stop, max(18, int(self.args.line_width * 9)), (255, 255, 255), thickness=max(3, int(self.args.line_width * 2)))
            elif annotation["geometry_type"] == "multi_directed_point":
                for item in annotation["directed_points"]:
                    stop = tuple(np.asarray(item["stop_pixel"], dtype=np.int32))
                    look = tuple(np.asarray(item["look_pixel"], dtype=np.int32))
                    cv2.arrowedLine(review, stop, look, color, thickness=max(4, int(self.args.line_width * 2)), tipLength=0.25)
                    cv2.circle(review, stop, max(12, int(self.args.line_width * 6)), color, thickness=-1)
                    cv2.circle(review, stop, max(18, int(self.args.line_width * 9)), (255, 255, 255), thickness=max(3, int(self.args.line_width * 2)))
            elif annotation["geometry_type"] == "directed_polyline":
                pts = np.asarray(annotation["polyline_pixel"], dtype=np.int32)
                self.draw_directed_polyline(review, pts, color)
            elif annotation["geometry_type"] == "multi_directed_polyline":
                for polyline in annotation["polylines"]:
                    pts = np.asarray(polyline["polyline_pixel"], dtype=np.int32)
                    self.draw_directed_polyline(review, pts, color)
            elif annotation["geometry_type"] == "polyline":
                pts = np.asarray(annotation["polyline_pixel"], dtype=np.int32)
                self.draw_polyline(review, pts, color)
            elif annotation["geometry_type"] == "multi_polyline":
                for polyline in annotation["polylines"]:
                    pts = np.asarray(polyline["polyline_pixel"], dtype=np.int32)
                    self.draw_polyline(review, pts, color)
            elif annotation["geometry_type"] == "multi_polygon":
                for polygon in annotation["polygons_pixel"]:
                    pts = np.asarray(polygon, dtype=np.int32)
                    overlay = review.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    review = cv2.addWeighted(overlay, 0.18, review, 0.82, 0.0)
                    cv2.polylines(review, [pts], isClosed=True, color=color, thickness=max(3, int(self.args.line_width * 2)))
            elif annotation["geometry_type"] == "multi_circle":
                for circle in annotation["circles"]:
                    self.draw_review_circle(review, circle["center_pixel"], circle["radius_pixel"], color)
            else:
                pts = np.asarray(annotation["polygon_pixel"], dtype=np.int32)
                overlay = review.copy()
                cv2.fillPoly(overlay, [pts], color)
                review = cv2.addWeighted(overlay, 0.18, review, 0.82, 0.0)
                cv2.polylines(review, [pts], isClosed=True, color=color, thickness=max(3, int(self.args.line_width * 2)))
        max_resolution = int(self.args.review_max_resolution)
        if max_resolution <= 0:
            raise ValueError("review_max_resolution must be positive")
        height, width = review.shape[:2]
        scale = min(1.0, max_resolution / max(width, height))
        if scale < 1.0:
            review = cv2.resize(
                review,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.review_path), review)
        print(f"Saved review thumbnail: {self.review_path} ({review.shape[1]} x {review.shape[0]})")

    def draw_review_circle(self, image: np.ndarray, center_pixel: list[float], radius_pixel: float, color: tuple[int, int, int]) -> None:
        center = tuple(np.asarray(center_pixel, dtype=np.int32))
        radius = max(1, int(round(float(radius_pixel))))
        overlay = image.copy()
        cv2.circle(overlay, center, radius, color, thickness=-1)
        image[:] = cv2.addWeighted(overlay, 0.18, image, 0.82, 0.0)
        cv2.circle(image, center, radius, color, thickness=max(3, int(self.args.line_width * 2)))
        cv2.circle(image, center, max(10, int(self.args.line_width * 5)), color, thickness=-1)

    def has_planning_boundary(self) -> bool:
        return has_planning_boundary(self.annotations)

    def available_categories(self) -> dict[str, dict]:
        return available_categories(self.annotations)

    def choose_category(self) -> dict | None:
        if self.has_planning_boundary():
            print("\n已完成规划边界标注，后续轨迹规划将在该边界内进行。")
        category = choose_numbered_option(
            prompt="请选择下一次标注类型",
            options=self.available_categories(),
            quit_label="完成标注并退出",
        )
        if category is None:
            return category
        if category["key"] == "preferred_path":
            path_type = choose_numbered_option(
                prompt="请选择优先路径标注方式",
                options=PREFERRED_PATH_OPTIONS,
                quit_label="返回标注类型选择",
            )
            if path_type is None:
                return self.choose_category()
            path_category = dict(category)
            path_category["geometry"] = path_type["geometry"]
            path_category["path_type"] = path_type["key"]
            path_category["path_type_name"] = path_type["name"]
            return path_category
        if category["key"] not in (
            "obstacle",
            "narrow_space",
            "equipment_region",
        ):
            return category

        shape = choose_numbered_option(
            prompt=f"请选择{category['name']}标注图形",
            options=AREA_SHAPE_OPTIONS,
            quit_label="返回标注类型选择",
        )
        if shape is None:
            return self.choose_category()

        shaped_category = dict(category)
        shaped_category["geometry"] = shape["geometry"]
        shaped_category["shape"] = shape["key"]
        shaped_category["shape_name"] = shape["name"]
        return shaped_category

    def print_controls(self) -> None:
        category = self.active_category
        print("\nControls")
        print("  W/A/S/D: pan")
        print("  Mouse wheel: zoom")
        print("  + / -: zoom with keyboard")
        print("  1: show image at 1:1 pixels")
        print("  F: fit whole image to window")
        print("  U or Backspace: undo last annotation")
        print("  Esc: cancel current unfinished annotation")
        print("  Q: save and quit")
        if category is None:
            return
        if category["geometry"] == "directed_point":
            print("  Left-click: first point is patrol stop")
            print("  Left-click again: second point sets viewing direction")
            print("  Enter: finish this annotation after at least one patrol point is completed")
        elif category["geometry"] in ("directed_polyline", "polyline"):
            print(f"  Left-click: add {category['name']} waypoint")
            print(f"  Right-click: finish current {category['name']} path")
            print("  Enter: finish this annotation after at least one path is closed")
        elif category["geometry"] == "polygon":
            print(f"  Left-click: add {category['name']} polygon vertex")
            print(f"  Right-click: close current {category['name']} polygon")
            print("  Enter: finish this annotation after at least one polygon is closed")
        elif category["geometry"] == "rectangle":
            print("  Left-click: first rectangle corner")
            print("  Left-click again: opposite rectangle corner")
            print("  Enter: finish this annotation after at least one rectangle is completed")
        elif category["geometry"] == "circle":
            print(f"  Left-click: circle center, radius defaults to {self.args.default_circle_radius_m:.2f} m")
            print("  Enter: finish this annotation after at least one circle is completed")

    def set_active_category(self, category: dict) -> None:
        self.active_category = category
        self.pending_annotation = None
        self.road_vertices = []
        self.closed_polygons = []
        self.closed_directed_points = []
        self.closed_polylines = []
        self.closed_circles = []
        self.shape_points = []
        self.patrol_points = []
        shape_text = f", shape={category['shape_name']}" if category.get("shape_name") else ""
        path_text = f", path_type={category['path_type_name']}" if category.get("path_type_name") else ""
        print(f"\nCurrent category: {category['name']} ({category['key']}{shape_text}{path_text})")
        self.print_controls()

    def run_annotation_window(self) -> str:
        window_flags = cv2.WINDOW_AUTOSIZE if self.args.disable_dynamic_window_size else cv2.WINDOW_NORMAL
        cv2.namedWindow(self.window_name, window_flags)
        if not self.args.disable_dynamic_window_size:
            cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            if self.add_pending_annotation():
                self.save()
                self.update_window_image_size()
                cv2.imshow(self.window_name, self.render_view())
                cv2.waitKey(1)
                category = self.choose_category()
                if category is None:
                    return "quit"
                self.set_active_category(category)
                continue
            self.update_window_image_size()
            cv2.imshow(self.window_name, self.render_view())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("u"), 8, 127):
                self.undo()
            elif key == ord("1"):
                self.set_one_to_one_view()
            elif key in (ord("f"), ord("F")):
                self.set_fit_view()
            elif key in (ord("+"), ord("=")):
                self.update_window_image_size(force=True)
                self.zoom_at_window_center(0.85)
            elif key in (ord("-"), ord("_")):
                self.update_window_image_size(force=True)
                self.zoom_at_window_center(1.18)
            elif key in (ord("a"), ord("A")):
                self.pan_by_fraction(-0.15, 0.0)
            elif key in (ord("d"), ord("D")):
                self.pan_by_fraction(0.15, 0.0)
            elif key in (ord("w"), ord("W")):
                self.pan_by_fraction(0.0, -0.15)
            elif key in (ord("s"), ord("S")):
                self.pan_by_fraction(0.0, 0.15)
            elif key in (10, 13):
                if self.active_category:
                    if self.active_category["geometry"] == "polygon":
                        self.finish_current_polygons()
                    elif self.active_category["geometry"] == "rectangle":
                        self.finish_current_polygons()
                    elif self.active_category["geometry"] == "circle":
                        self.finish_current_circles()
                    elif self.active_category["geometry"] == "directed_point":
                        self.finish_current_directed_points()
                    elif self.active_category["geometry"] in ("directed_polyline", "polyline"):
                        self.finish_current_polylines()
            elif key == 27:
                self.pending_annotation = None
                self.road_vertices = []
                self.closed_polygons = []
                self.closed_directed_points = []
                self.closed_polylines = []
                self.closed_circles = []
                self.shape_points = []
                self.patrol_points = []
            elif key in (ord("q"), ord("Q")):
                return "quit"
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Annotation window closed.")
                    return "closed"
            except cv2.error:
                print("Annotation window closed.")
                return "closed"

    def run(self) -> None:
        category = self.choose_category()
        if category is None:
            self.save()
            return

        self.window_name = "Orthographic annotation"
        self.set_active_category(category)
        self.run_annotation_window()
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        self.save()
