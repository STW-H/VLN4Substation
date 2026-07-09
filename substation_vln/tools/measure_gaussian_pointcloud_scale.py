#!/usr/bin/env python3
"""Measure scale difference between Gaussian centers and complete point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.geometry import segment_length  # noqa: E402
from substation_vln.paths import DEFAULT_GAUSSIAN, DEFAULT_POINTCLOUD  # noqa: E402
from substation_vln.picking import pick_points  # noqa: E402
from substation_vln.pointcloud_io import import_open3d, make_pcd, sample_ply_points  # noqa: E402


def pick_segment(o3d, pcd, title: str) -> np.ndarray:
    print("\n" + "-" * 72)
    print(f"Next window: {title}")
    print("Pick exactly 2 endpoints of the same physical segment, then press Q.")
    print("Controls: Shift + left click = pick, Shift + right click = undo.")
    input("Press Enter here to open this picking window...")
    return pick_points(o3d, pcd, title, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare one segment length in Gaussian and complete point cloud.")
    parser.add_argument("--gaussian", type=Path, default=DEFAULT_GAUSSIAN, help="Processed Y-up Gaussian PLY")
    parser.add_argument("--pointcloud", type=Path, default=DEFAULT_POINTCLOUD, help="Processed complete point cloud PLY")
    parser.add_argument("--gaussian-sample-points", type=int, default=1_000_000)
    parser.add_argument(
        "--pointcloud-sample-points",
        type=int,
        default=0,
        help="Points shown from complete point cloud; default 0 uses all points",
    )
    parser.add_argument(
        "--pick-order",
        choices=["gaussian-first", "pointcloud-first"],
        default="gaussian-first",
        help="Manual picking order. Endpoints must correspond between the two windows.",
    )
    args = parser.parse_args()

    gaussian_path = args.gaussian.expanduser().resolve()
    pointcloud_path = args.pointcloud.expanduser().resolve()
    if not gaussian_path.exists():
        raise SystemExit(f"Gaussian not found: {gaussian_path}")
    if not pointcloud_path.exists():
        raise SystemExit(f"Point cloud not found: {pointcloud_path}")

    o3d = import_open3d()
    print(f"Loading Gaussian centers: {gaussian_path}")
    gaussian_points, gaussian_colors = sample_ply_points(gaussian_path, args.gaussian_sample_points)
    print(f"Loading complete point cloud: {pointcloud_path}")
    pointcloud_points, pointcloud_colors = sample_ply_points(pointcloud_path, args.pointcloud_sample_points)

    gaussian_pcd = make_pcd(o3d, gaussian_points, color=(1.0, 0.15, 0.05), colors=gaussian_colors)
    pointcloud_pcd = make_pcd(o3d, pointcloud_points, color=(0.1, 0.45, 1.0), colors=pointcloud_colors)

    if args.pick_order == "gaussian-first":
        gaussian_segment = pick_segment(o3d, gaussian_pcd, "Step 1/2: pick 2 endpoints in Gaussian centers")
        pointcloud_segment = pick_segment(
            o3d,
            pointcloud_pcd,
            "Step 2/2: pick corresponding 2 endpoints in complete point cloud",
        )
    else:
        pointcloud_segment = pick_segment(o3d, pointcloud_pcd, "Step 1/2: pick 2 endpoints in complete point cloud")
        gaussian_segment = pick_segment(o3d, gaussian_pcd, "Step 2/2: pick corresponding 2 endpoints in Gaussian centers")

    gaussian_length = segment_length(gaussian_segment)
    pointcloud_length = segment_length(pointcloud_segment)
    if gaussian_length <= 0:
        raise SystemExit("Gaussian segment length is zero; please pick two different points.")

    ratio = pointcloud_length / gaussian_length
    print("\nScale check result")
    print("Gaussian endpoints:")
    print(gaussian_segment)
    print("Point-cloud endpoints:")
    print(pointcloud_segment)
    print(f"Gaussian segment length:    {gaussian_length:.6f} m")
    print(f"Point-cloud segment length: {pointcloud_length:.6f} m")
    print(f"Length difference:          {pointcloud_length - gaussian_length:.6f} m")
    print(f"Scale ratio pc/gaussian:    {ratio:.9f}")
    print(f"Scale difference:           {(ratio - 1.0) * 100.0:.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

