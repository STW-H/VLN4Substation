#!/usr/bin/env python3
"""View or render 3D Gaussian Splatting PLY files with Habitat-GS."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.preprocessing.coordinate_transforms import (  # noqa: E402
    ensure_gaussian_z_up_to_y_up_cache,
    transform_z_up_points_to_habitat_y_up,
)
from substation_vln.visualization.habitat_gs import render_gaussian_snapshot  # noqa: E402
from substation_vln.config import config_path, config_value, load_yaml_config  # noqa: E402
from substation_vln.paths import (  # noqa: E402
    CONFIGS_DIR,
    DEFAULT_ZUP_GAUSSIAN,
    HABITAT_GS_ROOT,
    HABITAT_GS_VIEWER,
    OUTPUTS_ERFEISHAN_DIR,
)


def default_y_up_cache_path(scene: Path) -> Path:
    stem = scene.name.removesuffix(".gs.ply").removesuffix(".ply")
    return OUTPUTS_ERFEISHAN_DIR / "gaussian_yup_cache" / f"{stem}_habitat_yup.gs.ply"


def habitat_position(position: list[float], already_y_up: bool) -> list[float]:
    if already_y_up:
        return position
    return transform_z_up_points_to_habitat_y_up([position])[0].tolist()


def main() -> int:
    default_config = CONFIGS_DIR / "tools" / "visualization" / "view_gaussian_erfeishan.yaml"
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=default_config, help="YAML file with default tool arguments")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="View a 3DGS PLY using Habitat-GS.", parents=[pre_parser])
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=config_path(config, "input", DEFAULT_ZUP_GAUSSIAN),
        help="Z-up 3DGS PLY file; default is the raw Erfeishan Gaussian",
    )
    parser.add_argument("--width", type=int, default=config_value(config, "width", 1920))
    parser.add_argument("--height", type=int, default=config_value(config, "height", 1080))
    parser.add_argument("--snapshot", action="store_true", default=config_value(config, "snapshot", False), help="Render one offscreen RGB image instead of opening the viewer")
    parser.add_argument("--output", type=Path, default=config_path(config, "output", OUTPUTS_ERFEISHAN_DIR / "gaussian" / "snapshot.png"))
    parser.add_argument("--viewer", type=Path, default=config_path(config, "viewer", HABITAT_GS_VIEWER), help="Path to Habitat-GS gaussian_viewer.py")
    parser.add_argument(
        "--y-up-cache",
        type=Path,
        default=config_path(config, "y_up_cache"),
        help="Y-up cache path under outputs; default is outputs/220kv_erfeishan/gaussian_yup_cache/<input>_habitat_yup.gs.ply",
    )
    parser.add_argument(
        "--already-y-up",
        action="store_true",
        default=config_value(config, "already_y_up", False),
        help="Skip Z-up to Y-up conversion and pass the input directly to Habitat-GS",
    )
    parser.add_argument("--position", type=float, nargs=3, default=config_value(config, "position", [0.0, 1.5, 3.0]), metavar=("X", "Y", "Z"))
    parser.add_argument("--yaw-deg", type=float, default=config_value(config, "yaw_deg", 0.0), help="Camera yaw for --snapshot")
    parser.add_argument("--pitch-deg", type=float, default=config_value(config, "pitch_deg", 0.0), help="Camera pitch for --snapshot")
    args = parser.parse_args()

    scene = args.input.expanduser()
    if not scene.is_absolute():
        scene = (Path.cwd() / scene).absolute()
    if not scene.exists():
        raise SystemExit(f"File not found: {scene}")
    if scene.suffix.lower() != ".ply":
        raise SystemExit("Habitat-GS expects a PLY Gaussian file.")
    if args.already_y_up and not (scene.name.endswith(".gs.ply") or scene.name.endswith(".3dgs.ply")):
        print(
            "warning: Habitat-GS recognizes Gaussian stages by suffix. "
            "Use the processed *.gs.ply links when possible.",
            file=sys.stderr,
        )

    viewer = args.viewer.expanduser().resolve()
    if not viewer.exists():
        raise SystemExit(f"Habitat-GS viewer not found: {viewer}")

    habitat_scene = scene
    if not args.already_y_up:
        cache_path = args.y_up_cache.expanduser().resolve() if args.y_up_cache else default_y_up_cache_path(scene)
        if cache_path.exists() and cache_path.stat().st_mtime >= scene.stat().st_mtime:
            print(f"Using cached Habitat Y-up Gaussian: {cache_path}")
        else:
            print(f"Creating/updating Habitat Y-up Gaussian cache: {cache_path}")
        habitat_scene = ensure_gaussian_z_up_to_y_up_cache(scene, cache_path, HABITAT_GS_ROOT)

    start_position = habitat_position(args.position, args.already_y_up)

    if args.snapshot:
        render_gaussian_snapshot(
            habitat_scene,
            args.output.expanduser().resolve(),
            args.width,
            args.height,
            start_position,
            args.yaw_deg,
            args.pitch_deg,
        )
        print(f"saved: {args.output}")
        return 0

    cmd = [
        sys.executable,
        str(viewer),
        "--input",
        str(habitat_scene),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--start-position",
        *(str(v) for v in start_position),
        "--start-yaw-deg",
        str(args.yaw_deg),
    ]
    return subprocess.call(cmd, cwd=str(viewer.parent.parent))


if __name__ == "__main__":
    sys.exit(main())
