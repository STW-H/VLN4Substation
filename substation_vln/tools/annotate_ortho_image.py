#!/usr/bin/env python3
"""Annotate obstacle rectangles on an orthographic image and map them to real XY coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.paths import ANNOTATION_OUTPUTS_ERFEISHAN_DIR  # noqa: E402


DEFAULT_IMAGE = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / "axis_corrected_pointcloud_ortho_8k.png"
DEFAULT_OUTPUT = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / "obstacles_2d_from_ortho_image.json"
DEFAULT_REVIEW = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / "obstacles_2d_from_ortho_image_review.png"

LABEL_COLORS_BGR = [
    (30, 30, 255),
    (0, 145, 255),
    (255, 90, 20),
    (60, 210, 60),
    (255, 20, 220),
    (210, 210, 0),
]


def metadata_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def apply_homogeneous(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack([points.astype(np.float64), ones])
    transformed = homogeneous @ matrix.T
    return transformed[:, :2] / transformed[:, 2:3]


def rectangle_polygon(start: tuple[float, float], end: tuple[float, float]) -> list[list[float]]:
    c0, r0 = start
    c1, r1 = end
    return [[c0, r0], [c1, r0], [c1, r1], [c0, r1]]


def polygon_area(points: list[list[float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


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

        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.pixel_to_world = np.asarray(self.metadata["pixel_to_world_matrix"], dtype=np.float64)

        self.window_name = "Orthographic obstacle annotation"
        self.window_width = int(args.window_width)
        self.window_height = int(args.window_height)
        fit_scale_x = self.width / self.window_width
        fit_scale_y = self.height / self.window_height
        self.scale = max(fit_scale_x, fit_scale_y, 1.0)
        self.min_scale = max(0.05, self.scale * 0.01)
        self.max_scale = max(self.scale * 20.0, 1.0)
        self.center = np.array([self.width * 0.5, self.height * 0.5], dtype=np.float64)

        self.annotations: list[dict] = []
        self.dragging_box = False
        self.panning = False
        self.drag_start: tuple[float, float] | None = None
        self.drag_end: tuple[float, float] | None = None
        self.pan_last: tuple[int, int] | None = None
        self.pending_polygon: list[list[float]] | None = None

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

    def mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            wheel_delta = cv2.getMouseWheelDelta(flags)
            before = np.asarray(self.screen_to_image(x, y), dtype=np.float64)
            factor = 0.85 if wheel_delta > 0 else 1.18
            self.scale = float(np.clip(self.scale * factor, self.min_scale, self.max_scale))
            after = np.asarray(self.screen_to_image(x, y), dtype=np.float64)
            self.center += before - after
            self.clamp_center()
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self.panning = True
            self.pan_last = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.panning and self.pan_last is not None:
            dx = x - self.pan_last[0]
            dy = y - self.pan_last[1]
            self.center[0] -= dx * self.scale
            self.center[1] -= dy * self.scale
            self.pan_last = (x, y)
            self.clamp_center()
            return

        if event == cv2.EVENT_RBUTTONUP:
            self.panning = False
            self.pan_last = None
            return

        shift_left = bool(flags & cv2.EVENT_FLAG_SHIFTKEY) and bool(flags & cv2.EVENT_FLAG_LBUTTON)
        if event == cv2.EVENT_LBUTTONDOWN and bool(flags & cv2.EVENT_FLAG_SHIFTKEY):
            self.drag_start = self.screen_to_image(x, y)
            self.drag_end = self.drag_start
            self.dragging_box = True
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging_box and shift_left:
            self.drag_end = self.screen_to_image(x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and self.dragging_box:
            self.drag_end = self.screen_to_image(x, y)
            self.dragging_box = False
            if self.drag_start is None or self.drag_end is None:
                return
            polygon = rectangle_polygon(self.drag_start, self.drag_end)
            if polygon_area(polygon) < self.args.min_pixel_area:
                print("Ignored tiny rectangle.")
                self.drag_start = None
                self.drag_end = None
                return
            self.pending_polygon = polygon
            self.drag_start = None
            self.drag_end = None

    def render_view(self) -> np.ndarray:
        transform = np.array(
            [
                [1.0 / self.scale, 0.0, -self.center[0] / self.scale + self.window_width * 0.5],
                [0.0, 1.0 / self.scale, -self.center[1] / self.scale + self.window_height * 0.5],
            ],
            dtype=np.float64,
        )
        view = cv2.warpAffine(
            self.image,
            transform,
            (self.window_width, self.window_height),
            flags=cv2.INTER_AREA if self.scale > 1.0 else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        for annotation in self.annotations:
            color = tuple(annotation["color_bgr"])
            pts = np.asarray([self.image_to_screen(c, r) for c, r in annotation["polygon_pixel"]], dtype=np.int32)
            overlay = view.copy()
            cv2.fillPoly(overlay, [pts], color)
            view = cv2.addWeighted(overlay, 0.18, view, 0.82, 0.0)
            cv2.polylines(view, [pts], isClosed=True, color=color, thickness=max(2, int(self.args.line_width)))

        if self.dragging_box and self.drag_start is not None and self.drag_end is not None:
            pts = np.asarray([self.image_to_screen(c, r) for c, r in rectangle_polygon(self.drag_start, self.drag_end)], dtype=np.int32)
            cv2.polylines(view, [pts], isClosed=True, color=(0, 255, 255), thickness=max(2, int(self.args.line_width)))

        cv2.putText(
            view,
            "Shift+Left drag: box | Right drag: pan | Wheel: zoom | U/Backspace: undo | Q/S: save | Esc: cancel",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            "Shift+Left drag: box | Right drag: pan | Wheel: zoom | U/Backspace: undo | Q/S: save | Esc: cancel",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return view

    def add_pending_annotation(self) -> None:
        if self.pending_polygon is None:
            return
        polygon_pixel = self.pending_polygon
        polygon_xy = apply_homogeneous(self.pixel_to_world, np.asarray(polygon_pixel, dtype=np.float64)).tolist()
        label = input("Obstacle label for selected rectangle [obstacle]: ").strip() or "obstacle"
        annotation_id = len(self.annotations) + 1
        color = LABEL_COLORS_BGR[(annotation_id - 1) % len(LABEL_COLORS_BGR)]
        annotation = {
            "id": annotation_id,
            "label": label,
            "selection_type": "image_rectangle",
            "polygon_pixel": polygon_pixel,
            "polygon_xy": polygon_xy,
            "bbox_pixel": {
                "min": np.min(np.asarray(polygon_pixel), axis=0).tolist(),
                "max": np.max(np.asarray(polygon_pixel), axis=0).tolist(),
            },
            "bbox_xy": {
                "min": np.min(np.asarray(polygon_xy), axis=0).tolist(),
                "max": np.max(np.asarray(polygon_xy), axis=0).tolist(),
            },
            "area_pixel": polygon_area(polygon_pixel),
            "area_xy": polygon_area(polygon_xy),
            "color_bgr": list(color),
        }
        self.annotations.append(annotation)
        self.pending_polygon = None
        print(f"Added annotation #{annotation_id}: label={label}, area_xy={annotation['area_xy']:.3f}")

    def undo(self) -> None:
        if not self.annotations:
            print("No annotation to undo.")
            return
        annotation = self.annotations.pop()
        print(f"Undid annotation #{annotation['id']}: label={annotation['label']}")

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": str(self.image_path),
            "metadata": str(self.metadata_path),
            "coordinate_system": "axis_corrected_pointcloud",
            "pixel_to_world_matrix": self.pixel_to_world.tolist(),
            "world_to_pixel_matrix": self.metadata.get("world_to_pixel_matrix"),
            "image_size": {"width": int(self.width), "height": int(self.height)},
            "annotations": self.annotations,
        }
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved annotations: {self.output_path}")
        self.save_review_image()

    def save_review_image(self) -> None:
        review = self.image.copy()
        for annotation in self.annotations:
            color = tuple(annotation["color_bgr"])
            pts = np.asarray(annotation["polygon_pixel"], dtype=np.int32)
            overlay = review.copy()
            cv2.fillPoly(overlay, [pts], color)
            review = cv2.addWeighted(overlay, 0.18, review, 0.82, 0.0)
            cv2.polylines(review, [pts], isClosed=True, color=color, thickness=max(3, int(self.args.line_width * 2)))
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.review_path), review)
        print(f"Saved review image: {self.review_path}")

    def run(self) -> None:
        print("\nControls")
        print("  Shift + left-drag: draw obstacle rectangle")
        print("  Right-drag: pan")
        print("  Mouse wheel: zoom")
        print("  U or Backspace: undo last annotation")
        print("  Esc: cancel current rectangle")
        print("  S or Q: save and quit")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            self.add_pending_annotation()
            cv2.imshow(self.window_name, self.render_view())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("u"), 8, 127):
                self.undo()
            elif key == 27:
                self.pending_polygon = None
                self.dragging_box = False
                self.drag_start = None
                self.drag_end = None
            elif key in (ord("s"), ord("q")):
                self.save()
                break
        cv2.destroyWindow(self.window_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate an orthographic image and convert labels to real XY.")
    parser.add_argument("image", type=Path, nargs="?", default=DEFAULT_IMAGE)
    parser.add_argument("--metadata", type=Path, help="Mapping metadata JSON. Default is image path with .json suffix.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-image", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=1000)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--min-pixel-area", type=float, default=16.0)
    args = parser.parse_args()

    annotator = OrthoImageAnnotator(args)
    annotator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
