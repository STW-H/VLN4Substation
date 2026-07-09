#!/usr/bin/env python3
"""Overlay raw Gaussian centers and LAS point cloud in a Z-up local frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.geometry import bounds_text  # noqa: E402
from substation_vln.las import sample_las_local  # noqa: E402
from substation_vln.paths import DEFAULT_RAW_GAUSSIAN_DIR, DEFAULT_RAW_LAS  # noqa: E402
from substation_vln.pointcloud_io import import_open3d, make_pcd, sample_ply_points  # noqa: E402


def bbox_center_extent(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    return (bounds_min + bounds_max) * 0.5, bounds_max - bounds_min


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay LAS and raw Gaussian point clouds.")
    parser.add_argument("--las", type=Path, default=DEFAULT_RAW_LAS)
    parser.add_argument(
        "--gaussian",
        type=Path,
        default=DEFAULT_RAW_GAUSSIAN_DIR / "layer_2_point_cloud.ply",
        help="Raw Gaussian PLY file, not Y-up rotated",
    )
    parser.add_argument("--max-las-points", type=int, default=1_000_000)
    parser.add_argument("--max-gaussian-points", type=int, default=1_000_000)
    parser.add_argument("--no-view", action="store_true", help="Only print statistics, do not open Open3D window")
    parser.add_argument(
        "--align-bbox-centers",
        action="store_true",
        help="Translate LAS local points so its bbox center matches Gaussian bbox center for visual scale comparison",
    )
    parser.add_argument("--no-frame", action="store_true")
    args = parser.parse_args()

    las_path = args.las.expanduser().resolve()
    gaussian_path = args.gaussian.expanduser().resolve()
    if not las_path.exists():
        raise SystemExit(f"LAS file not found: {las_path}")
    if not gaussian_path.exists():
        raise SystemExit(f"Gaussian file not found: {gaussian_path}")

    o3d = import_open3d()
    las_points, las_origin = sample_las_local(las_path, args.max_las_points)
    gaussian_points, _ = sample_ply_points(gaussian_path, args.max_gaussian_points)

    print(f"LAS origin subtracted: {las_origin}")
    print("\n" + bounds_text("LAS local Z-up", las_points))
    print("\n" + bounds_text("Raw Gaussian Z-up", gaussian_points))
    las_center, las_extent = bbox_center_extent(las_points)
    gaussian_center, gaussian_extent = bbox_center_extent(gaussian_points)
    print("\nExtent ratio Gaussian / LAS local:")
    print(gaussian_extent / np.maximum(las_extent, 1e-9))

    if args.align_bbox_centers:
        delta = gaussian_center - las_center
        las_points = las_points + delta
        print(f"\nApplied visual-only bbox center translation to LAS: {delta}")
        print(bounds_text("LAS local after visual center alignment", las_points))

    if args.no_view:
        return 0

    geometries = [
        make_pcd(o3d, las_points, [0.1, 0.45, 1.0]),
        make_pcd(o3d, gaussian_points, [1.0, 0.15, 0.05]),
    ]
    if not args.no_frame:
        all_points = np.vstack([las_points, gaussian_points])
        frame_size = max(float(np.ptp(all_points, axis=0).max()) * 0.06, 1.0)
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))

    print("\nColor legend: LAS=blue, raw Gaussian=red")
    o3d.visualization.draw_geometries(geometries, window_name="LAS local Z-up + raw Gaussian Z-up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
