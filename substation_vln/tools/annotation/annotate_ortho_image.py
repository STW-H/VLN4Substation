#!/usr/bin/env python3
"""Command line entry for orthographic image annotation."""

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
from substation_vln.paths import ANNOTATION_OUTPUTS_ERFEISHAN_DIR  # noqa: E402


DEFAULT_IMAGE = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / "axis_corrected_pointcloud_ortho_8k.png"


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
    parser = argparse.ArgumentParser(description="Annotate an orthographic image and convert labels to real XY.")
    parser.add_argument("image", type=Path, nargs="?", default=DEFAULT_IMAGE)
    parser.add_argument("--metadata", type=Path, help="Mapping metadata JSON. Default is image path with .json suffix.")
    parser.add_argument("--output", type=Path, help="Annotation JSON output. Default: annotation_<current time>.json")
    parser.add_argument("--review-image", type=Path, help="Review image output. Default: same timestamp as annotation JSON.")
    parser.add_argument("--window-width", type=int, default=1200)
    parser.add_argument("--window-height", type=int, default=0, help="0 means auto height from image aspect ratio")
    parser.add_argument(
        "--disable-dynamic-window-size",
        action="store_true",
        help="Use a fixed OpenCV autosize window. By default, resizing is supported with delayed redraw.",
    )
    parser.add_argument(
        "--window-resize-debounce-ms",
        type=float,
        default=300.0,
        help="Only resample the source image after the window size has been stable for this many milliseconds.",
    )
    parser.add_argument("--min-window-width", type=int, default=320)
    parser.add_argument("--min-window-height", type=int, default=240)
    parser.add_argument(
        "--allow-free-window-aspect",
        action="store_true",
        help="Accept very wide/tall resized windows. Default rejects suspicious viewport sizes reported by some OpenCV Qt builds.",
    )
    parser.add_argument(
        "--initial-view",
        choices=("native", "fit"),
        default="native",
        help="native starts at 1:1 image pixels for crisp display; fit shows the whole image.",
    )
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--label-font-size", type=int, default=22)
    parser.add_argument("--min-pixel-area", type=float, default=16.0)
    parser.add_argument("--min-direction-pixel-length", type=float, default=8.0)
    parser.add_argument(
        "--default-circle-radius-m",
        type=float,
        default=0.5,
        help="Default obstacle circle radius in meters. Used when selecting circle obstacles.",
    )
    parser.add_argument(
        "--display-interpolation",
        choices=("auto", "nearest", "linear", "area", "cubic", "lanczos"),
        default="auto",
        help="Interpolation used for the OpenCV display view. auto uses AREA when downsampling and LANCZOS when zooming in.",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output, args.review_image = default_annotation_output_paths()
    elif args.review_image is None:
        args.review_image = args.output.with_name(f"{args.output.stem}_review.png")

    annotator = OrthoImageAnnotator(args)
    annotator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
