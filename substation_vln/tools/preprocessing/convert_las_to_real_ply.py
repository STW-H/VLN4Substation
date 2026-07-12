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
from substation_vln.config import config_path, config_value, load_yaml_config  # noqa: E402
from substation_vln.picking import pick_with_pause  # noqa: E402
from substation_vln.paths import CONFIGS_DIR, DEFAULT_PROCESSED_POINTCLOUD_DIR  # noqa: E402
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
    # Preserve the rotated X/Y coordinates while translating only along the
    # corrected Z axis, so the fitted reference ground plane is centered at
    # z=0 in the axis-corrected coordinate frame.
    ground_z_after_rotation = float((rotation @ plane_centroid)[2])
    matrix[2, 3] = -ground_z_after_rotation

    return {
        "matrix": matrix,
        "rotation": rotation,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "plane_centroid": plane_centroid,
        "plane_rmse": plane_rmse,
        "ground_z_after_rotation": ground_z_after_rotation,
        "z_translation": float(matrix[2, 3]),
        "ground_region_point_count": int(len(ground_points)),
        "x_axis_points": x_axis_points,
    }


def default_output(path: Path) -> Path:
    return DEFAULT_PROCESSED_POINTCLOUD_DIR / f"{path.stem}_real_coords.ply"


def default_axis_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}_axis_corrected{path.suffix}")


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "preprocessing" / "convert_las_to_real_ply_erfeishan.yaml"


def crop_ground_region(o3d, display_pcd, display_center: np.ndarray, title: str) -> np.ndarray:
    print("\n" + "=" * 72)
    print(title)
    print("本步骤需要选择一块平坦地面点云，用于拟合基准地平面和确定 Z+ 方向。")
    print("请尽量选择道路或硬化地面，并与其他步骤选择的区域保持较大间距。")
    print("Suggested Open3D workflow:")
    print("  1. Rotate/zoom until the target ground area is clear.")
    print("  2. Press K to enter selection/cropping mode.")
    print("  3. Drag/select a ground area.")
    print("  4. Press C to crop the selected area.")
    print("  5. Press Q to finish this region.")
    print("请避开设备、底座、立杆、围栏、沟槽、植被及其他非地面点。")
    print("=" * 72)
    input("确认已了解本步骤后，按 Enter 打开地面区域裁剪窗口...")

    cropped = crop_point_cloud(o3d, display_pcd, title, point_size=3.0)
    if cropped is None or cropped.is_empty():
        raise SystemExit("No points were cropped. Please run again and crop a visible ground region.")
    points = np.asarray(cropped.points) + display_center
    print(f"本步骤完成：已选择地面点 {len(points):,} 个。")
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

    print("\n" + "=" * 72)
    print("开始点云坐标轴矫正")
    print("=" * 72)
    print("  轴矫正 1/5～3/5：选择三块相互分散的平坦地面区域。")
    print("  轴矫正 4/5：依次选择两个点，定义从第一个点指向第二个点的 X+ 方向。")
    print("  轴矫正 5/5：拟合地平面、构建变换、将基准地面设为 z=0 并保存结果。")
    print("  Open3D 中仅为显示而临时中心化；保存结果使用完整工程坐标变换。")

    ground_regions = []
    for idx in range(1, 4):
        ground_regions.append(
            crop_ground_region(
                o3d,
                display_pcd,
                display_center,
                f"轴矫正 {idx}/5：选择地面区域 {idx}/3",
            )
        )
    ground_points = np.vstack(ground_regions)
    print(f"\n地面选择完成：3 个区域，共 {len(ground_points):,} 个点。")
    print("下一步将选择 X+ 方向。请在一条清晰、较长且近似水平的道路边缘或结构线上选点。")
    print("选择顺序决定方向：第一个点 → 第二个点 = X+。")
    x_axis_points = (
        pick_with_pause(
            o3d,
            display_pcd,
            "轴矫正 4/5：选择两个点定义 X+ 方向",
            2,
        )
        + display_center
    )
    print("本步骤完成：已获取 X+ 方向的两个参考点。")

    print("\n轴矫正 5/5：正在拟合地平面并计算坐标变换...")
    correction = build_axis_correction(
        ground_points=ground_points,
        x_axis_points=x_axis_points,
    )

    print("\n轴矫正计算完成，结果如下：")
    print(f"  ground region points used: {correction['ground_region_point_count']:,}")
    print(f"  fitted plane RMSE: {correction['plane_rmse']:.6f}")
    print(f"  X+ axis in original coordinates: {correction['x_axis']}")
    print(f"  Y+ axis in original coordinates: {correction['y_axis']}")
    print(f"  Z+ axis in original coordinates: {correction['z_axis']}")
    print(f"  fitted ground Z after rotation: {correction['ground_z_after_rotation']:.6f}")
    print(f"  Z translation: {correction['z_translation']:.6f}")
    print("  fitted reference ground: z=0; rotated X/Y coordinates unchanged")
    print("  matrix old -> axis-corrected:\n", correction["matrix"])

    print(f"\n正在变换完整点云并写入：{axis_output}")
    stats = transform_binary_ply_xyz(real_ply, axis_output, correction["matrix"], args.chunk_size)
    payload = {
        "input": str(real_ply),
        "output": str(axis_output),
        "transform_direction": "p_axis_corrected = R_old_to_axis_corrected @ p_real_coords + [0, 0, t_z]; fitted reference ground is z=0",
        "origin_policy": "preserve rotated X/Y coordinates; translate corrected Z so fitted ground centroid has z=0",
        "las_selection_sample_points": int(len(display_pcd.points)),
        "display_center_subtracted": display_center,
        "ground_region_counts": [int(len(points)) for points in ground_regions],
        "axis_correction": correction,
        "output_stats": stats,
    }
    axis_metadata.parent.mkdir(parents=True, exist_ok=True)
    axis_metadata.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n点云轴矫正全部完成。")
    print(f"  矫正点云：{axis_output}")
    print(f"  变换元数据：{axis_metadata}")
    print("  请检查拟合平面 RMSE，并确认所选基准地面在新坐标系中位于 z=0 附近。")


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML file with default tool arguments")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Convert LAS/LAZ to real-coordinate PLY.", parents=[pre_parser])
    parser.add_argument("input", type=Path, nargs="?", default=config_path(config, "input"), help="Input LAS/LAZ file")
    parser.add_argument(
        "--output",
        type=Path,
        default=config_path(config, "output"),
        help="Output binary PLY file; default is processed/220kv_erfeishan/pointcloud/<input_stem>_real_coords.ply",
    )
    parser.add_argument("--chunk-size", type=int, default=config_value(config, "chunk_size", 1_000_000), help="Points per streaming chunk")
    parser.add_argument("--metadata", type=Path, default=config_path(config, "metadata"), help="Optional JSON metadata output")
    parser.add_argument(
        "--axis-correct",
        action="store_true",
        default=config_value(config, "axis_correct", False),
        help="After real-coordinate conversion, interactively fit ground/X axes and write an axis-corrected PLY",
    )
    parser.add_argument(
        "--axis-output",
        type=Path,
        default=config_path(config, "axis_output"),
        help="Output PLY for --axis-correct; default is <output_stem>_axis_corrected.ply",
    )
    parser.add_argument("--axis-metadata", type=Path, default=config_path(config, "axis_metadata"), help="Output JSON for axis-correction metadata")
    parser.add_argument(
        "--axis-sample-points",
        type=int,
        default=config_value(config, "axis_sample_points", 20_000_000),
        help="LAS points shown/used for visualization and interactive axis fitting; default 20,000,000",
    )
    parser.add_argument(
        "--no-las-view",
        action="store_true",
        default=config_value(config, "no_las_view", False),
        help="Skip the initial colored LAS visualization window",
    )
    parser.add_argument("--view-point-size", type=float, default=config_value(config, "view_point_size", 2.0), help="Point size for LAS visualization")
    args = parser.parse_args()
    if args.input is None:
        parser.error("input is required either as a positional argument or in the YAML config")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else default_output(input_path).resolve()

    print("\n[点云预处理 1/3] 读取 LAS/LAZ 点云样本")
    print(f"  输入文件：{input_path}")
    print(f"  最大采样点数：{args.axis_sample_points:,}")
    o3d = import_open3d()
    las_pcd = load_las_as_pcd(input_path, args.axis_sample_points)
    display_pcd, display_center = centered_display_pcd(o3d, las_pcd)
    print(f"  读取完成：当前样本包含 {len(las_pcd.points):,} 个点。")
    if not args.no_las_view and not args.axis_correct:
        visualize_las_sample(o3d, display_pcd, display_center, input_path.name, args.view_point_size)

    print("\n[点云预处理 2/3] 恢复 LAS 真实坐标并写入 PLY")
    print("  坐标规则：real = integer × LAS header scale + LAS header offset")
    print(f"  输出文件：{output_path}")
    write_las_real_ply(input_path, output_path, args.chunk_size, args.metadata)
    if args.axis_correct:
        print("\n[点云预处理 3/3] 交互式坐标轴矫正与地面归零")
        run_axis_correction(args, o3d, display_pcd, display_center, output_path)
    else:
        print("\n[点云预处理 3/3] 已跳过坐标轴矫正（axis_correct=false）。")
    print("\n点云预处理命令执行完毕。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
