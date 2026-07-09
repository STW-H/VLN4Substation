#!/usr/bin/env python3
"""View or render 3D Gaussian Splatting PLY files with Habitat-GS."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.habitat_gs import render_gaussian_snapshot  # noqa: E402
from substation_vln.paths import HABITAT_GS_ROOT, HABITAT_GS_VIEWER, OUTPUTS_ERFEISHAN_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="View a 3DGS PLY using Habitat-GS.")
    parser.add_argument("input", type=Path, help="3DGS PLY file, preferably named *.gs.ply")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--snapshot", action="store_true", help="Render one offscreen RGB image instead of opening the viewer")
    parser.add_argument("--output", type=Path, default=OUTPUTS_ERFEISHAN_DIR / "gaussian" / "snapshot.png")
    parser.add_argument("--viewer", type=Path, default=HABITAT_GS_VIEWER, help="Path to Habitat-GS gaussian_viewer.py")
    parser.add_argument("--position", type=float, nargs=3, default=[0.0, 1.5, 3.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Camera yaw for --snapshot")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="Camera pitch for --snapshot")
    args = parser.parse_args()

    scene = args.input.expanduser()
    if not scene.is_absolute():
        scene = (Path.cwd() / scene).absolute()
    if not scene.exists():
        raise SystemExit(f"File not found: {scene}")
    if scene.suffix.lower() != ".ply":
        raise SystemExit("Habitat-GS expects a PLY Gaussian file.")
    if not (scene.name.endswith(".gs.ply") or scene.name.endswith(".3dgs.ply")):
        print(
            "warning: Habitat-GS recognizes Gaussian stages by suffix. "
            "Use the processed *.gs.ply links when possible.",
            file=sys.stderr,
        )

    if args.snapshot:
        render_gaussian_snapshot(
            scene,
            args.output.expanduser().resolve(),
            args.width,
            args.height,
            args.position,
            args.yaw_deg,
            args.pitch_deg,
        )
        print(f"saved: {args.output}")
        return 0

    viewer = args.viewer.expanduser().resolve()
    if not viewer.exists():
        raise SystemExit(f"Habitat-GS viewer not found: {viewer}")

    cmd = [
        sys.executable,
        str(viewer),
        "--input",
        str(scene),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--start-position",
        *(str(v) for v in args.position),
        "--start-yaw-deg",
        str(args.yaw_deg),
    ]
    return subprocess.call(cmd, cwd=str(viewer.parent.parent))


if __name__ == "__main__":
    sys.exit(main())
