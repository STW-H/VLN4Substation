#!/usr/bin/env python3
"""Unified command-line entry for 2D map and 3D inspection-target annotation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.annotation.annotator import OrthoImageAnnotator  # noqa: E402
from substation_vln.annotation.inspection_targets_3d import InspectionTarget3DAnnotator, InspectionTargetDefaults  # noqa: E402
from substation_vln.config import config_path, config_value, load_yaml_config  # noqa: E402
from substation_vln.interactive import choose_numbered_option  # noqa: E402
from substation_vln.paths import ANNOTATION_OUTPUTS_ERFEISHAN_DIR, CONFIGS_DIR, DEFAULT_AXIS_CORRECTED_POINTCLOUD  # noqa: E402


DEFAULT_IMAGE = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / "axis_corrected_pointcloud_ortho_8k.png"
DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "annotation" / "annotate.yaml"


def default_annotation_output_paths() -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"annotation_{timestamp}"
    output = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / f"{stem}.json"
    review = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / f"{stem}_review.png"
    index = 2
    while output.exists() or review.exists():
        numbered_stem = f"{stem}_{index:02d}"
        output = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / f"{numbered_stem}.json"
        review = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / f"{numbered_stem}_review.png"
        index += 1
    return output, review


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML file with default tool arguments")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Annotate either 2D map semantics or 3D inspection targets.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--mode",
        choices=("choose", "map", "target3d"),
        default=config_value(config, "mode", "choose"),
        help="Annotation mode; choose opens an interactive mode menu.",
    )
    parser.add_argument("image", type=Path, nargs="?", default=config_path(config, "image", DEFAULT_IMAGE))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=config_path(config, "metadata"),
        help="Mapping metadata JSON. Default is image path with .json suffix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config_path(config, "output"),
        help="Annotation JSON output. Default: annotation_<current time>.json",
    )
    parser.add_argument(
        "--review-image",
        type=Path,
        default=config_path(config, "review_image"),
        help="Review image output. Default: same timestamp as annotation JSON.",
    )
    parser.add_argument("--window-width", type=int, default=config_value(config, "window_width", 1200))
    parser.add_argument(
        "--window-height",
        type=int,
        default=config_value(config, "window_height", 0),
        help="0 means auto height from image aspect ratio",
    )
    parser.add_argument(
        "--disable-dynamic-window-size",
        action="store_true",
        default=config_value(config, "disable_dynamic_window_size", False),
        help="Use a fixed OpenCV autosize window. By default, resizing is supported with delayed redraw.",
    )
    parser.add_argument(
        "--window-resize-debounce-ms",
        type=float,
        default=config_value(config, "window_resize_debounce_ms", 300.0),
        help="Only resample the source image after the window size has been stable for this many milliseconds.",
    )
    parser.add_argument("--min-window-width", type=int, default=config_value(config, "min_window_width", 320))
    parser.add_argument("--min-window-height", type=int, default=config_value(config, "min_window_height", 240))
    parser.add_argument(
        "--allow-free-window-aspect",
        action="store_true",
        default=config_value(config, "allow_free_window_aspect", False),
        help="Accept very wide/tall resized windows. Default rejects suspicious viewport sizes reported by some OpenCV Qt builds.",
    )
    parser.add_argument(
        "--initial-view",
        choices=("native", "fit"),
        default=config_value(config, "initial_view", "native"),
        help="native starts at 1:1 image pixels for crisp display; fit shows the whole image.",
    )
    parser.add_argument("--line-width", type=int, default=config_value(config, "line_width", 2))
    parser.add_argument("--label-font-size", type=int, default=config_value(config, "label_font_size", 22))
    parser.add_argument("--min-pixel-area", type=float, default=config_value(config, "min_pixel_area", 16.0))
    parser.add_argument(
        "--min-direction-pixel-length",
        type=float,
        default=config_value(config, "min_direction_pixel_length", 8.0),
    )
    parser.add_argument(
        "--default-circle-radius-m",
        type=float,
        default=config_value(config, "default_circle_radius_m", 0.5),
        help="Default obstacle circle radius in meters. Used when selecting circle obstacles.",
    )
    parser.add_argument(
        "--display-interpolation",
        choices=("auto", "nearest", "linear", "area", "cubic", "lanczos"),
        default=config_value(config, "display_interpolation", "auto"),
        help="Interpolation used for the OpenCV display view. auto uses AREA when downsampling and LANCZOS when zooming in.",
    )
    parser.add_argument("--pointcloud", type=Path, default=config_path(config, "pointcloud", DEFAULT_AXIS_CORRECTED_POINTCLOUD))
    parser.add_argument("--max-display-points", type=int, default=config_value(config, "max_display_points", 20_000_000))
    parser.add_argument("--point-size", type=float, default=config_value(config, "point_size", 3.0))
    parser.add_argument(
        "--display-color-contrast",
        type=float,
        default=config_value(config, "display_color_contrast", 1.0),
        help="Display-only RGB contrast around 0.5; coordinates and saved annotations are unchanged.",
    )
    parser.add_argument(
        "--display-color-brightness",
        type=float,
        default=config_value(config, "display_color_brightness", 0.0),
        help="Display-only RGB brightness offset.",
    )
    parser.add_argument(
        "--display-background-color",
        type=float,
        nargs=4,
        default=config_value(config, "display_background_color", [0.02, 0.02, 0.02, 1.0]),
        metavar=("R", "G", "B", "A"),
        help="SceneWidget background RGBA in [0, 1].",
    )
    parser.add_argument(
        "--selection-marker-point-size",
        type=float,
        default=config_value(config, "selection_marker_point_size", 8.0),
        help="Yellow recorded-point size in the SceneWidget picker.",
    )
    parser.add_argument("--ground-z-m", type=float, default=config_value(config, "ground_z_m", 0.0))
    parser.add_argument("--camera-height-m", type=float, default=config_value(config, "camera_height_m", 1.0))
    parser.add_argument("--default-equipment-type", default=config_value(config, "default_equipment_type", "unknown_device"))
    parser.add_argument("--default-task-type", default=config_value(config, "default_task_type", "visual_inspection"))
    parser.add_argument("--default-min-distance-m", type=float, default=config_value(config, "default_min_distance_m", 2.0))
    parser.add_argument("--default-max-distance-m", type=float, default=config_value(config, "default_max_distance_m", 6.0))
    parser.add_argument("--default-exclusion-radius-m", type=float, default=config_value(config, "default_exclusion_radius_m", 0.2))
    parser.add_argument("--prompt-target-id", action="store_true", default=config_value(config, "prompt_target_id", False))
    parser.add_argument("--prompt-task-type", action="store_true", default=config_value(config, "prompt_task_type", False))
    parser.add_argument("--prompt-observation-parameters", action="store_true", default=config_value(config, "prompt_observation_parameters", False))
    parser.add_argument("--no-resume", action="store_true", default=config_value(config, "no_resume", False))
    parser.add_argument("--no-review", action="store_true", default=config_value(config, "no_review", False))
    parser.add_argument(
        "--review-sphere-radius-m",
        type=float,
        default=config_value(config, "review_sphere_radius_m", 0.3),
        help="Radius of the target marker sphere in the final review window.",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "choose":
        selected = choose_numbered_option(
            prompt="请选择标注模式",
            options={
                "1": {"key": "map", "name": "二维地图语义标注"},
                "2": {"key": "target3d", "name": "三维巡视目标标注"},
            },
            quit_label="退出",
            default_quit=False,
        )
        if selected is None:
            return 0
        mode = selected["key"]

    if args.output is None:
        args.output, args.review_image = default_annotation_output_paths()
    elif args.review_image is None:
        args.review_image = args.output.with_name(f"{args.output.stem}_review.png")

    if mode == "map":
        annotator = OrthoImageAnnotator(args)
        annotator.run()
        return 0

    defaults = InspectionTargetDefaults(
        category=args.default_equipment_type,
        task_type=args.default_task_type,
        min_observation_distance_m=args.default_min_distance_m,
        max_observation_distance_m=args.default_max_distance_m,
        target_exclusion_radius_m=args.default_exclusion_radius_m,
    )
    annotator = InspectionTarget3DAnnotator(
        pointcloud_path=args.pointcloud,
        output_path=args.output,
        max_display_points=args.max_display_points,
        point_size=args.point_size,
        ground_z_m=args.ground_z_m,
        camera_height_m=args.camera_height_m,
        defaults=defaults,
        resume=not args.no_resume,
        show_review=not args.no_review,
        review_sphere_radius_m=args.review_sphere_radius_m,
        selection_marker_point_size=args.selection_marker_point_size,
        display_color_contrast=args.display_color_contrast,
        display_color_brightness=args.display_color_brightness,
        display_background_color=tuple(args.display_background_color),
        prompt_target_id=args.prompt_target_id,
        prompt_task_type=args.prompt_task_type,
        prompt_observation_parameters=args.prompt_observation_parameters,
    )
    annotator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
