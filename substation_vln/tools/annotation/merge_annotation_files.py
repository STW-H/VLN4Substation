#!/usr/bin/env python3
"""Merge split orthographic annotation JSON files into one planning annotation file."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.config import config_path, load_yaml_config  # noqa: E402
from substation_vln.paths import ANNOTATION_OUTPUTS_ERFEISHAN_DIR, CONFIGS_DIR  # noqa: E402


DEFAULT_IMAGE = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / "axis_corrected_pointcloud_ortho_8k.png"
DEFAULT_OUTPUT = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / "annotations_merged.json"
DEFAULT_REVIEW = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / "annotations_merged_review.png"
DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "annotation" / "merge_annotation_files_erfeishan.yaml"

LABEL_TRANSLATIONS = {
    "daolu": "preferred_road",
    "zuidabianjie": "planning_boundary",
    "youxianlujing": "preferred_path",
    "zhubian": "main_transformer",
    "二次室": "secondary_control_room",
    "bangongqu": "office_area",
    "yingjifang": "emergency_room",
    "shang_zhicheng": "upper_support",
    "xia_zhicheng": "lower_support",
    "longmen": "gantry",
    "ercizhichengjia": "secondary_support_frame",
    "zhalan": "fence",
    "bianyaqi": "transformer",
    "dainliuhuganqi": "current_transformer",
    "dianyahuganqi": "voltage_transformer",
    "galikaiguan": "disconnect_switch",
}

TRANSLATION_REVIEW_REQUIRED = {
    "galikaiguan": "Assumed to mean geli/隔离开关, translated as disconnect_switch.",
    "ercizhichengjia": "Assumed to mean 二次支撑架, translated as secondary_support_frame.",
    "yingjifang": "Assumed to mean 应急房, translated as emergency_room.",
}

CATEGORY_COLORS_BGR = {
    "planning_boundary": (30, 220, 30),
    "obstacle": (35, 35, 235),
    "preferred_road": (255, 140, 0),
    "preferred_path": (255, 30, 220),
    "patrol_point": (40, 190, 255),
}


def load_annotation_files(annotation_dir: Path) -> list[Path]:
    files = sorted(annotation_dir.glob("annotation_*.json"))
    return [path for path in files if not path.name.endswith("_merged.json")]


def translate_label(label: str) -> str:
    return LABEL_TRANSLATIONS.get(label, label)


def merge_annotations(files: list[Path]) -> dict:
    if not files:
        raise SystemExit("No annotation_*.json files found.")

    merged_annotations: list[dict] = []
    source_files: list[dict] = []
    base_payload: dict | None = None
    review_required: dict[str, str] = {}

    for source_index, path in enumerate(files, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if base_payload is None:
            base_payload = payload
        source_files.append(
            {
                "index": source_index,
                "path": str(path),
                "file": path.name,
                "saved_at": payload.get("saved_at"),
                "annotation_count": len(payload.get("annotations", [])),
            }
        )

        for annotation in payload.get("annotations", []):
            item = dict(annotation)
            original_label = str(item.get("label", ""))
            translated_label = translate_label(original_label)
            if original_label in TRANSLATION_REVIEW_REQUIRED:
                review_required[original_label] = TRANSLATION_REVIEW_REQUIRED[original_label]
            item["source_file"] = path.name
            item["source_id"] = item.get("id")
            item["original_label"] = original_label
            item["label"] = translated_label
            item["id"] = len(merged_annotations) + 1
            merged_annotations.append(item)

    assert base_payload is not None
    return {
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": source_files,
        "image": base_payload.get("image"),
        "metadata": base_payload.get("metadata"),
        "coordinate_system": base_payload.get("coordinate_system", "axis_corrected_pointcloud"),
        "pixel_to_world_matrix": base_payload.get("pixel_to_world_matrix"),
        "world_to_pixel_matrix": base_payload.get("world_to_pixel_matrix"),
        "image_size": base_payload.get("image_size"),
        "categories": base_payload.get("categories"),
        "obstacle_shape_options": base_payload.get("obstacle_shape_options"),
        "label_translation_map": LABEL_TRANSLATIONS,
        "translation_review_required": review_required,
        "annotations": merged_annotations,
    }


def blend_polygon(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    image[:] = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)


def draw_circle(image: np.ndarray, center_pixel: list[float], radius_pixel: float, color: tuple[int, int, int], alpha: float) -> None:
    center = tuple(np.asarray(center_pixel, dtype=np.int32))
    radius = max(1, int(round(float(radius_pixel))))
    overlay = image.copy()
    cv2.circle(overlay, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    image[:] = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)
    cv2.circle(image, center, radius, color, thickness=3, lineType=cv2.LINE_AA)


def draw_small_arrow(image: np.ndarray, start_pixel: list[float], end_pixel: list[float], color: tuple[int, int, int]) -> None:
    start = np.asarray(start_pixel, dtype=np.float64)
    end = np.asarray(end_pixel, dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return
    max_draw_length = 70.0
    if length > max_draw_length:
        end = start + direction / length * max_draw_length
    cv2.arrowedLine(
        image,
        tuple(start.astype(np.int32)),
        tuple(end.astype(np.int32)),
        color,
        thickness=2,
        line_type=cv2.LINE_AA,
        tipLength=0.12,
    )


def draw_label(image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = position
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def annotation_label_position(annotation: dict) -> tuple[int, int] | None:
    geometry_type = annotation.get("geometry_type")
    if geometry_type == "multi_polygon":
        points = [point for polygon in annotation.get("polygons_pixel", []) for point in polygon]
        if points:
            center = np.mean(np.asarray(points, dtype=np.float64), axis=0)
            return tuple(center.astype(np.int32))
    if geometry_type == "multi_circle" and annotation.get("circles"):
        center = annotation["circles"][0]["center_pixel"]
        return int(center[0] + 10), int(center[1] - 10)
    if geometry_type == "multi_directed_segment" and annotation.get("segments"):
        segment = annotation["segments"][0]
        start = np.asarray(segment["start_pixel"], dtype=np.float64)
        end = np.asarray(segment["end_pixel"], dtype=np.float64)
        return tuple(((start + end) * 0.5).astype(np.int32))
    if geometry_type == "multi_directed_point" and annotation.get("directed_points"):
        stop = annotation["directed_points"][0]["stop_pixel"]
        return int(stop[0] + 10), int(stop[1] - 10)
    return None


def draw_review(image_path: Path, merged_payload: dict, review_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    for annotation in merged_payload["annotations"]:
        category = annotation.get("category")
        color = CATEGORY_COLORS_BGR.get(category, tuple(annotation.get("color_bgr", [255, 255, 255])))
        geometry_type = annotation.get("geometry_type")

        if geometry_type == "multi_polygon":
            for polygon in annotation.get("polygons_pixel", []):
                points = np.asarray(polygon, dtype=np.int32)
                if category == "planning_boundary":
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=5, lineType=cv2.LINE_AA)
                else:
                    alpha = 0.16 if category == "preferred_road" else 0.20
                    blend_polygon(image, points, color, alpha)
        elif geometry_type == "multi_circle":
            for circle in annotation.get("circles", []):
                draw_circle(image, circle["center_pixel"], circle["radius_pixel"], color, alpha=0.20)
        elif geometry_type == "multi_directed_segment":
            for segment in annotation.get("segments", []):
                draw_small_arrow(image, segment["start_pixel"], segment["end_pixel"], color)
        elif geometry_type == "multi_directed_point":
            for point in annotation.get("directed_points", []):
                stop = tuple(np.asarray(point["stop_pixel"], dtype=np.int32))
                look = np.asarray(point["look_pixel"], dtype=np.float64)
                stop_arr = np.asarray(point["stop_pixel"], dtype=np.float64)
                direction = look - stop_arr
                length = float(np.linalg.norm(direction))
                if length > 1e-6:
                    end = stop_arr + direction / length * min(length, 55.0)
                    cv2.arrowedLine(image, stop, tuple(end.astype(np.int32)), color, thickness=3, line_type=cv2.LINE_AA, tipLength=0.18)
                cv2.circle(image, stop, 8, color, thickness=-1, lineType=cv2.LINE_AA)

    for annotation in merged_payload["annotations"]:
        if annotation.get("category") == "planning_boundary":
            continue
        position = annotation_label_position(annotation)
        if position is not None:
            color = CATEGORY_COLORS_BGR.get(annotation.get("category"), tuple(annotation.get("color_bgr", [255, 255, 255])))
            draw_label(image, str(annotation.get("label", "")), position, color)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(review_path), image)


def summarize(merged_payload: dict) -> None:
    by_category: dict[str, int] = {}
    for annotation in merged_payload["annotations"]:
        by_category[annotation["category"]] = by_category.get(annotation["category"], 0) + 1
    print("Merged annotation records:", len(merged_payload["annotations"]))
    print("By category:")
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count}")
    if merged_payload["translation_review_required"]:
        print("Translation mappings that should be reviewed:")
        for label, note in merged_payload["translation_review_required"].items():
            print(f"  {label}: {LABEL_TRANSLATIONS[label]} ({note})")


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML file with default tool arguments")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Merge split 2D annotation JSON files.", parents=[pre_parser])
    parser.add_argument("--annotation-dir", type=Path, default=config_path(config, "annotation_dir", ANNOTATION_OUTPUTS_ERFEISHAN_DIR))
    parser.add_argument("--image", type=Path, default=config_path(config, "image", DEFAULT_IMAGE))
    parser.add_argument("--output", type=Path, default=config_path(config, "output", DEFAULT_OUTPUT))
    parser.add_argument("--review-image", type=Path, default=config_path(config, "review_image", DEFAULT_REVIEW))
    args = parser.parse_args()

    files = load_annotation_files(args.annotation_dir)
    merged_payload = merge_annotations(files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_review(args.image, merged_payload, args.review_image)
    summarize(merged_payload)
    print(f"Saved merged annotations: {args.output}")
    print(f"Saved merged review image: {args.review_image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
