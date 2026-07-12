#!/usr/bin/env python3
"""Visualize point clouds for the substation VLN project."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.paths import DEFAULT_PROCESSED_POINTCLOUD_DIR  # noqa: E402
from substation_vln.preprocessing.pointcloud_io import (  # noqa: E402
    SUPPORTED_LAS,
    SUPPORTED_OPEN3D,
    describe_pcd,
    import_open3d,
    load_las_as_pcd,
)
from substation_vln.visualization.pointcloud import centered_display_pcd, coordinate_frame_for_points  # noqa: E402


def load_point_cloud(path: Path, max_points: int):
    suffix = path.suffix.lower()
    if suffix == ".lidata":
        raise SystemExit(
            f"{path} is a .LiData file. Please export/convert it to PLY, PCD, "
            "LAS, LAZ, XYZ, or PTS before visualization."
        )

    o3d = import_open3d()
    if suffix in SUPPORTED_OPEN3D:
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise SystemExit(f"Open3D could not read any points from: {path}")
        if max_points > 0 and len(pcd.points) > max_points:
            pcd = pcd.random_down_sample(max_points / len(pcd.points))
        return pcd

    if suffix in SUPPORTED_LAS:
        return load_las_as_pcd(path, max_points)

    raise SystemExit(
        f"Unsupported file extension: {suffix}\n"
        f"Supported: {', '.join(sorted(SUPPORTED_OPEN3D | SUPPORTED_LAS))}"
    )


def default_converted_output(path: Path) -> Path:
    return DEFAULT_PROCESSED_POINTCLOUD_DIR / f"{path.stem}_real_coords.ply"


def save_converted_las(path: Path, output: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_LAS:
        raise SystemExit("--save-converted currently supports only LAS/LAZ inputs")

    converter = PROJECT_ROOT / "substation_vln" / "tools" / "preprocessing" / "convert_las_to_real_ply.py"
    metadata = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(converter),
        str(path),
        "--output",
        str(output),
        "--metadata",
        str(metadata),
    ]
    print(f"Saving converted real-coordinate point cloud to: {output}")
    print("Existing file will be overwritten if present.")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="View a substation point cloud.")
    parser.add_argument("input", type=Path, help="Point cloud file")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Voxel downsample size")
    parser.add_argument("--max-points", type=int, default=0, help="Randomly downsample if above this count; default 0 loads all points")
    parser.add_argument("--no-center", action="store_true", help="Do not center the visualization")
    parser.add_argument("--no-frame", action="store_true", help="Do not draw the coordinate frame")
    parser.add_argument("--info", action="store_true", help="Only print point-cloud metadata")
    parser.add_argument(
        "--save-converted",
        action="store_true",
        help="For LAS/LAZ input, save real-coordinate PLY to the processed folder before viewing",
    )
    parser.add_argument(
        "--converted-output",
        type=Path,
        help="Output path for --save-converted; default is processed/220kv_erfeishan/pointcloud/<stem>_real_coords.ply",
    )
    args = parser.parse_args()

    path = args.input.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    if args.save_converted:
        converted_output = (
            args.converted_output.expanduser().resolve()
            if args.converted_output
            else default_converted_output(path)
        )
        save_converted_las(path, converted_output)

    o3d = import_open3d()
    pcd = load_point_cloud(path, args.max_points)

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)

    print("Original/downsampled point cloud:")
    print(describe_pcd(pcd))

    view_pcd = pcd
    if not args.no_center:
        view_pcd, display_center = centered_display_pcd(o3d, pcd)
        print(f"display center subtracted only for visualization: {display_center}")
        print("\nVisualization coordinates after centering:")
        print(describe_pcd(view_pcd))

    if not args.info:
        geometries = [view_pcd]
        if not args.no_frame:
            geometries.append(coordinate_frame_for_points(o3d, np.asarray(view_pcd.points)))
        o3d.visualization.draw_geometries(geometries, window_name=path.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
