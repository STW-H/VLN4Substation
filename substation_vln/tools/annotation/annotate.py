#!/usr/bin/env python3
"""Command-line entry for 2D semantic and equipment-region annotation."""

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
from substation_vln.config import config_path, config_value, load_yaml_config  # noqa: E402
from substation_vln.paths import (  # noqa: E402
    ANNOTATION_SESSIONS_ERFEISHAN_DIR,
    CONFIGS_DIR,
    ORTHOPHOTO_ERFEISHAN_DIR,
)


DEFAULT_IMAGE = ORTHOPHOTO_ERFEISHAN_DIR / "axis_corrected_pointcloud_ortho_8k.png"
DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "annotation" / "annotate.yaml"


def default_annotation_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"annotation_{timestamp}"
    output = ANNOTATION_SESSIONS_ERFEISHAN_DIR / f"{stem}.json"
    index = 2
    while output.exists() or output.with_suffix(".png").exists():
        numbered_stem = f"{stem}_{index:02d}"
        output = ANNOTATION_SESSIONS_ERFEISHAN_DIR / f"{numbered_stem}.json"
        index += 1
    return output


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML file with default tool arguments")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Annotate 2D map semantics and inspection-equipment footprints.",
        parents=[pre_parser],
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
    parser.add_argument(
        "--review-max-resolution",
        type=int,
        default=config_value(config, "review_max_resolution", 2048),
        help="Maximum width or height of the saved review thumbnail.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = default_annotation_output_path()
    args.review_image = args.output.with_suffix(".png")

    annotator = OrthoImageAnnotator(args)
    annotator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
