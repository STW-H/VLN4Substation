#!/usr/bin/env python3
"""Convert LAS/LAZ integer coordinates to real-world-coordinate binary PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.preprocessing.las import write_las_real_ply  # noqa: E402
from substation_vln.picking import pick_with_pause  # noqa: E402
from substation_vln.paths import DEFAULT_PROCESSED_POINTCLOUD_DIR  # noqa: E402
from substation_vln.preprocessing.pointcloud_io import (  # noqa: E402
    describe_pcd,
    import_open3d,
    load_las_as_pcd,
    transform_binary_ply_xyz,
)
from substation_vln.serialization import json_ready  # noqa: E402
from substation_vln.visualization.pointcloud import (  # noqa: E402
    centered_display_pcd,
    crop_point_cloud,
    draw_point_cloud,
)


def normalize(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} is too small to normalize")
    return np.asarray(vector, dtype=np.float64) / norm


def fit_plane_normal(points: np.ndarray, up_hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vt[-1], "plane normal")

    if up_hint is not None and float(np.dot(normal, up_hint)) < 0:
        normal = -normal

    residuals = centered @ normal
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return normal, centroid, rmse


def build_axis_correction(
    ground_points: np.ndarray,
    x_axis_points: np.ndarray,
) -> dict:
    z_axis, plane_centroid, plane_rmse = fit_plane_normal(ground_points, up_hint=np.array([0.0, 0.0, 1.0]))

    raw_x = x_axis_points[1] - x_axis_points[0]
    projected_x = raw_x - float(np.dot(raw_x, z_axis)) * z_axis
    x_axis = normalize(projected_x, "projected X axis")
    y_axis = normalize(np.cross(z_axis, x_axis), "Y axis")
    x_axis = normalize(np.cross(y_axis, z_axis), "orthogonalized X axis")

    rotation = np.vstack([x_axis, y_axis, z_axis])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation

    return {
        "matrix": matrix,
        "rotation": rotation,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "plane_centroid": plane_centroid,
        "plane_rmse": plane_rmse,
        "ground_region_point_count": int(len(ground_points)),
        "x_axis_points": x_axis_points,
    }


def default_output(path: Path) -> Path:
    return DEFAULT_PROCESSED_POINTCLOUD_DIR / f"{path.stem}_real_coords.ply"


def default_axis_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}_axis_corrected{path.suffix}")


def crop_ground_region(o3d, display_pcd, display_center: np.ndarray, title: str) -> np.ndarray:
    print("\n" + "=" * 72)
    print(title)
    print("Crop one ground region in this window.")
    print("Suggested Open3D workflow:")
    print("  1. Rotate/zoom until the target ground area is clear.")
    print("  2. Press K to enter selection/cropping mode.")
    print("  3. Drag/select a ground area.")
    print("  4. Press C to crop the selected area.")
    print("  5. Press Q to finish this region.")
    print("Choose flat road/ground points and avoid equipment, poles, fences, and vegetation.")
    print("=" * 72)
    input("Press Enter here to open this cropping window...")

    cropped = crop_point_cloud(o3d, display_pcd, title, point_size=3.0)
    if cropped is None or cropped.is_empty():
        raise SystemExit("No points were cropped. Please run again and crop a visible ground region.")
    points = np.asarray(cropped.points) + display_center
    print(f"Cropped ground region points: {len(points):,}")
    return points


def visualize_las_sample(o3d, display_pcd, display_center: np.ndarray, title: str, point_size: float) -> None:
    print("\nLAS sample point cloud:")
    print(describe_pcd(display_pcd))
    print(f"display center subtracted only for visualization/selection: {display_center}")
    print("Color information is included if the LAS file contains RGB fields.")
    draw_point_cloud(o3d, display_pcd, title, point_size=point_size, show_frame=True)


def run_axis_correction(args: argparse.Namespace, o3d, display_pcd, display_center: np.ndarray, real_ply: Path) -> None:
    axis_output = args.axis_output.expanduser().resolve() if args.axis_output else default_axis_output(real_ply)
    axis_metadata = args.axis_metadata.expanduser().resolve() if args.axis_metadata else axis_output.with_suffix(".json")

    print("\nAxis correction selection guide")
    print("  Step 1-3: crop three separated flat ground regions for plane fitting.")
    print("  Step 4: pick two points along the desired X+ direction.")
    print("  Display coordinates are centered for Open3D only; saved coordinates remain real LAS coordinates.")

    ground_regions = [
        crop_ground_region(o3d, display_pcd, display_center, f"Axis correction 1/4: crop ground region {idx}/3")
        for idx in range(1, 4)
    ]
    ground_points = np.vstack(ground_regions)
    x_axis_points = (
        pick_with_pause(
            o3d,
            display_pcd,
            "Axis correction 4/4: pick 2 points defining X+ direction",
            2,
        )
        + display_center
    )

    correction = build_axis_correction(
        ground_points=ground_points,
        x_axis_points=x_axis_points,
    )

    print("\nAxis correction result")
    print(f"  ground region points used: {correction['ground_region_point_count']:,}")
    print(f"  fitted plane RMSE: {correction['plane_rmse']:.6f}")
    print(f"  X+ axis in original coordinates: {correction['x_axis']}")
    print(f"  Y+ axis in original coordinates: {correction['y_axis']}")
    print(f"  Z+ axis in original coordinates: {correction['z_axis']}")
    print("  origin: unchanged")
    print("  matrix old -> axis-corrected:\n", correction["matrix"])

    stats = transform_binary_ply_xyz(real_ply, axis_output, correction["matrix"], args.chunk_size)
    payload = {
        "input": str(real_ply),
        "output": str(axis_output),
        "transform_direction": "p_axis_corrected = R_old_to_axis_corrected @ p_real_coords; origin unchanged",
        "origin_policy": "unchanged",
        "las_selection_sample_points": int(len(display_pcd.points)),
        "display_center_subtracted": display_center,
        "ground_region_counts": [int(len(points)) for points in ground_regions],
        "axis_correction": correction,
        "output_stats": stats,
    }
    axis_metadata.parent.mkdir(parents=True, exist_ok=True)
    axis_metadata.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"axis metadata: {axis_metadata}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert LAS/LAZ to real-coordinate PLY.")
    parser.add_argument("input", type=Path, help="Input LAS/LAZ file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output binary PLY file; default is processed/220kv_erfeishan/pointcloud/<input_stem>_real_coords.ply",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Points per streaming chunk")
    parser.add_argument("--metadata", type=Path, help="Optional JSON metadata output")
    parser.add_argument(
        "--axis-correct",
        action="store_true",
        help="After real-coordinate conversion, interactively fit ground/X axes and write an axis-corrected PLY",
    )
    parser.add_argument(
        "--axis-output",
        type=Path,
        help="Output PLY for --axis-correct; default is <output_stem>_axis_corrected.ply",
    )
    parser.add_argument("--axis-metadata", type=Path, help="Output JSON for axis-correction metadata")
    parser.add_argument(
        "--axis-sample-points",
        type=int,
        default=20_000_000,
        help="LAS points shown/used for visualization and interactive axis fitting; default 20,000,000",
    )
    parser.add_argument(
        "--no-las-view",
        action="store_true",
        help="Skip the initial colored LAS visualization window",
    )
    parser.add_argument("--view-point-size", type=float, default=2.0, help="Point size for LAS visualization")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else default_output(input_path).resolve()

    o3d = import_open3d()
    print(f"Loading colored LAS sample for visualization/selection: {input_path}")
    las_pcd = load_las_as_pcd(input_path, args.axis_sample_points)
    display_pcd, display_center = centered_display_pcd(o3d, las_pcd)
    if not args.no_las_view and not args.axis_correct:
        visualize_las_sample(o3d, display_pcd, display_center, input_path.name, args.view_point_size)

    write_las_real_ply(input_path, output_path, args.chunk_size, args.metadata)
    if args.axis_correct:
        run_axis_correction(args, o3d, display_pcd, display_center, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
