#!/usr/bin/env python3
"""Register Gaussian centers to the processed full point cloud.

The reusable PLY loading, picking, geometry, and ICP logic lives under
substation_vln/src/substation_vln. This file is intentionally a command-line
workflow wrapper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.config import config_path, config_value, load_yaml_config  # noqa: E402
from substation_vln.geometry import bounds_text, transform_points, umeyama_similarity  # noqa: E402
from substation_vln.paths import (  # noqa: E402
    CONFIGS_DIR,
    DEFAULT_ALIGNED_GAUSSIAN,
    DEFAULT_AXIS_CORRECTED_POINTCLOUD,
    DEFAULT_GAUSSIAN,
    DEFAULT_REGISTRATION,
)
from substation_vln.picking import pick_with_pause  # noqa: E402
from substation_vln.preprocessing.pointcloud_io import import_open3d, make_pcd, sample_ply_points  # noqa: E402
from substation_vln.preprocessing.registration import (  # noqa: E402
    filter_gaussian_for_icp,
    filter_target_for_icp,
    load_correspondences,
    run_icp,
    save_aligned_gaussian,
    save_transform,
)
from substation_vln.visualization.pointcloud import configure_default_camera, configure_visualizer, coordinate_frame_for_points  # noqa: E402


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "preprocessing" / "register_gaussian_to_pointcloud_erfeishan.yaml"


def visualize_registration(o3d: Any, pointcloud_pcd, gaussian_pcd, final_matrix: np.ndarray) -> None:
    pointcloud_points = np.asarray(pointcloud_pcd.points)
    transformed_gaussian_points = transform_points(np.asarray(gaussian_pcd.points), final_matrix)

    pc_min = pointcloud_points.min(axis=0)
    pc_max = pointcloud_points.max(axis=0)
    gs_min = transformed_gaussian_points.min(axis=0)
    gs_max = transformed_gaussian_points.max(axis=0)
    scene_min = np.minimum(pc_min, gs_min)
    scene_max = np.maximum(pc_max, gs_max)
    scene_center = (scene_min + scene_max) * 0.5

    print("\nRegistration overlay bounds before display centering:")
    print(bounds_text("complete point cloud", pointcloud_points))
    print(bounds_text("transformed Gaussian", transformed_gaussian_points))
    print(f"display center subtracted only for visualization: {scene_center}")

    display_pointcloud = make_pcd(
        o3d,
        pointcloud_points - scene_center,
        color=(0.55, 0.55, 0.55),
        colors=np.asarray(pointcloud_pcd.colors) if pointcloud_pcd.has_colors() else None,
    )
    display_gaussian = make_pcd(o3d, transformed_gaussian_points - scene_center, color=(1.0, 0.05, 0.02))
    display_gaussian.paint_uniform_color([1.0, 0.05, 0.02])

    frame = coordinate_frame_for_points(
        o3d,
        np.vstack([pointcloud_points - scene_center, transformed_gaussian_points - scene_center]),
        ratio=0.06,
    )

    print("\nColor legend: complete point cloud=original colors, transformed Gaussian=red")
    print("Note: coordinates are temporarily centered for display only; saved transform remains in real coordinates.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Registration check: original point cloud + transformed Gaussian", width=1280, height=800)
    for geometry in (display_pointcloud, display_gaussian, frame):
        vis.add_geometry(geometry)

    configure_visualizer(vis, point_size=2.0)
    configure_default_camera(vis)

    vis.run()
    vis.destroy_window()


def load_filter_preview_matrix(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "initial_similarity" in data and "matrix" in data["initial_similarity"]:
        matrix = data["initial_similarity"]["matrix"]
        print(f"Using initial_similarity.matrix from: {path}")
    elif "final_matrix" in data:
        matrix = data["final_matrix"]
        print(f"Using final_matrix from: {path}")
    else:
        raise SystemExit(f"No transform matrix found in: {path}")
    return np.asarray(matrix, dtype=np.float64)


def resolve_pointcloud_input(path: Path) -> Path:
    """Accept either a PLY point cloud or an axis-correction JSON metadata file."""
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".json":
        return resolved

    data = json.loads(resolved.read_text(encoding="utf-8"))
    output = data.get("output")
    if not output:
        raise SystemExit(f"Point-cloud JSON has no output field: {resolved}")
    pointcloud_path = Path(output).expanduser().resolve()
    print(f"Using axis-corrected point cloud from JSON output field: {pointcloud_path}")
    return pointcloud_path


def visualize_icp_filter(o3d: Any, gaussian_pcd, keep_mask: np.ndarray, matrix: np.ndarray) -> None:
    gaussian_points = np.asarray(gaussian_pcd.points)
    gaussian_colors = np.asarray(gaussian_pcd.colors) if gaussian_pcd.has_colors() else None
    kept_points = gaussian_points[keep_mask]
    removed_points = gaussian_points[~keep_mask]
    kept_colors = gaussian_colors[keep_mask] if gaussian_colors is not None else None

    transformed_gaussian = transform_points(gaussian_points, matrix)
    transformed_kept = transform_points(kept_points, matrix) if len(kept_points) else np.empty((0, 3))
    transformed_removed = transform_points(removed_points, matrix) if len(removed_points) else np.empty((0, 3))

    scene_center = (transformed_gaussian.min(axis=0) + transformed_gaussian.max(axis=0)) * 0.5

    print("\nICP filter preview bounds before display centering:")
    print(bounds_text("Gaussian all points", transformed_gaussian))
    if len(transformed_kept):
        print(bounds_text("Gaussian kept for ICP", transformed_kept))
    if len(transformed_removed):
        print(bounds_text("Gaussian removed by filter", transformed_removed))
    print(f"display center subtracted only for visualization: {scene_center}")

    geometries = []
    if len(transformed_kept):
        display_kept = make_pcd(
            o3d,
            transformed_kept - scene_center,
            color=(0.65, 0.65, 0.65),
            colors=kept_colors,
        )
        geometries.append(display_kept)
    if len(transformed_removed):
        display_removed = make_pcd(o3d, transformed_removed - scene_center, color=(1.0, 0.0, 0.0))
        display_removed.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(display_removed)

    geometries.append(coordinate_frame_for_points(o3d, transformed_gaussian - scene_center, ratio=0.06))

    print("\nColor legend: Gaussian kept for ICP=original RGB, Gaussian removed by filter=red")
    print("Tip: red points are filtered out and will not participate in ICP.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ICP filter preview: Gaussian filtering", width=1280, height=800)
    for geometry in geometries:
        vis.add_geometry(geometry)

    configure_visualizer(vis, point_size=2.0)
    configure_default_camera(vis)

    vis.run()
    vis.destroy_window()


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML file with default tool arguments")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Register Gaussian centers to the processed full point cloud.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--pointcloud",
        type=Path,
        default=config_path(config, "pointcloud", DEFAULT_AXIS_CORRECTED_POINTCLOUD),
        help="Processed complete point cloud PLY, or axis-correction JSON whose output field points to the PLY",
    )
    parser.add_argument("--gaussian", type=Path, default=config_path(config, "gaussian", DEFAULT_GAUSSIAN), help="Gaussian PLY; default is raw Z-up Gaussian")
    parser.add_argument("--output", type=Path, default=config_path(config, "output", DEFAULT_REGISTRATION), help="Output transform JSON")
    parser.add_argument(
        "--aligned-gaussian-output",
        type=Path,
        default=config_path(config, "aligned_gaussian_output", DEFAULT_ALIGNED_GAUSSIAN),
        help="Processed aligned Gaussian point-cloud PLY output",
    )
    parser.add_argument("--aligned-gaussian-metadata", type=Path, default=config_path(config, "aligned_gaussian_metadata"), help="Processed aligned Gaussian metadata JSON")
    parser.add_argument("--no-save-aligned-gaussian", action="store_true", default=config_value(config, "no_save_aligned_gaussian", False), help="Do not save processed aligned Gaussian")
    parser.add_argument("--num-points", type=int, default=config_value(config, "num_points", 6), help="Number of manual correspondence points")
    parser.add_argument(
        "--pointcloud-sample-points",
        type=int,
        default=config_value(config, "pointcloud_sample_points", 0),
        help="Points shown from complete point cloud; default 0 uses all points",
    )
    parser.add_argument("--gaussian-sample-points", type=int, default=config_value(config, "gaussian_sample_points", 1_000_000))
    parser.add_argument("--correspondences", type=Path, default=config_path(config, "correspondences"), help="Optional JSON correspondences to skip interactive picking")
    parser.add_argument("--preview-icp-filter", type=Path, default=config_path(config, "preview_icp_filter"), help="Only visualize Gaussian ICP filtering using a transform JSON")
    parser.add_argument("--icp-method", choices=["point_to_point", "point_to_plane"], default=config_value(config, "icp_method", "point_to_point"))
    parser.add_argument("--icp-voxel-size", type=float, default=config_value(config, "icp_voxel_size", 0.5))
    parser.add_argument("--max-correspondence-distance", type=float, default=config_value(config, "max_correspondence_distance", 2.0))
    parser.add_argument("--icp-iterations", type=int, default=config_value(config, "icp_iterations", 80))
    parser.add_argument("--icp-multiscale", action="store_true", default=config_value(config, "icp_multiscale", False), help="Run coarse-to-fine multi-scale ICP")
    parser.add_argument(
        "--icp-multiscale-spec",
        default=config_value(config, "icp_multiscale_spec", "2.0:5.0,1.0:3.0,0.5:1.5,0.2:0.8"),
        help="Comma-separated voxel:distance stages for --icp-multiscale",
    )
    parser.add_argument("--no-icp-filter-target-bounds", action="store_true", default=config_value(config, "no_icp_filter_target_bounds", False))
    parser.add_argument("--icp-filter-target-bounds-margin", type=float, default=config_value(config, "icp_filter_target_bounds_margin", 5.0))
    parser.add_argument("--no-icp-filter-statistical", action="store_true", default=config_value(config, "no_icp_filter_statistical", False))
    parser.add_argument("--no-icp-filter-height", action="store_true", default=config_value(config, "no_icp_filter_height", False))
    parser.add_argument("--icp-min-height-above-target-min", type=float, default=config_value(config, "icp_min_height_above_target_min", 5.0))
    parser.add_argument("--icp-filter-nb-neighbors", type=int, default=config_value(config, "icp_filter_nb_neighbors", 30))
    parser.add_argument("--icp-filter-std-ratio", type=float, default=config_value(config, "icp_filter_std_ratio", 2.0))
    parser.add_argument("--no-icp", action="store_true", default=config_value(config, "no_icp", False), help="Only use manual similarity transform")
    parser.add_argument("--no-view", action="store_true", default=config_value(config, "no_view", False), help="Do not show final overlay window")
    parser.add_argument(
        "--pick-order",
        choices=["pointcloud-first", "gaussian-first"],
        default=config_value(config, "pick_order", "pointcloud-first"),
        help="Manual picking order. Correspondence order must still match between the two windows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pointcloud_path = resolve_pointcloud_input(args.pointcloud)
    gaussian_path = args.gaussian.expanduser().resolve()
    if not pointcloud_path.exists():
        raise SystemExit(f"Point cloud not found: {pointcloud_path}")
    if not gaussian_path.exists():
        raise SystemExit(f"Gaussian not found: {gaussian_path}")

    print("\n" + "=" * 72)
    print("Gaussian 到完整点云配准")
    print("=" * 72)
    print("  源点云（待变换）：Gaussian 中心点")
    print(f"    {gaussian_path}")
    print("  目标点云（参考坐标系）：轴矫正后的完整点云")
    print(f"    {pointcloud_path}")
    print("  变换方向：Gaussian → 完整点云")
    if args.preview_icp_filter:
        print("  当前模式：仅预览 ICP 过滤结果，不执行配准和保存。")
    else:
        print("  处理流程：读取数据 → 获取对应点 → 相似变换粗配准 → ICP 精配准 → 保存 → 复核")
        print(f"  手工对应点数量：{args.num_points}")
        print(f"  选点顺序：{args.pick_order}")
        print("  注意：两个窗口中的对应点必须按完全相同的编号顺序选择。")

    print("\n[配准 1/7] 读取源点云和目标点云")
    o3d = import_open3d()

    if args.pointcloud_sample_points == 0:
        print("  将读取完整目标点云，可能耗时较长并占用较多内存。")
    print(f"  正在读取目标/参考点云：{pointcloud_path}")
    pointcloud_pts, pointcloud_colors = sample_ply_points(pointcloud_path, args.pointcloud_sample_points)
    print(f"  目标点云读取完成：{len(pointcloud_pts):,} 个点。")
    print(f"  正在读取源/待移动 Gaussian 中心点：{gaussian_path}")
    gaussian_pts, gaussian_colors = sample_ply_points(gaussian_path, args.gaussian_sample_points)
    print(f"  Gaussian 采样读取完成：{len(gaussian_pts):,} 个点。")

    pointcloud_pcd = make_pcd(o3d, pointcloud_pts, color=(0.1, 0.45, 1.0), colors=pointcloud_colors)
    gaussian_pcd = make_pcd(o3d, gaussian_pts, color=(1.0, 0.15, 0.05), colors=gaussian_colors)

    if args.preview_icp_filter:
        print("\n[ICP过滤预览 1/2] 加载已有粗配准矩阵并计算过滤结果")
        preview_matrix = load_filter_preview_matrix(args.preview_icp_filter.expanduser().resolve())
        _, _, keep_mask = filter_gaussian_for_icp(o3d, gaussian_pcd, pointcloud_pcd, preview_matrix, args)
        print(f"  保留点数：{int(keep_mask.sum()):,}")
        print(f"  移除点数：{int((~keep_mask).sum()):,}")
        print("\n[ICP过滤预览 2/2] 打开复核窗口")
        print("  原始颜色表示保留并参与 ICP 的点，红色表示被过滤的点；按 Q 关闭窗口。")
        visualize_icp_filter(o3d, gaussian_pcd, keep_mask, preview_matrix)
        print("\nICP过滤预览完成；本次未修改或保存配准结果。")
        return 0

    print("\n[配准 2/7] 获取人工对应点")
    if args.correspondences:
        print(f"  使用已有对应点文件，跳过交互选点：{args.correspondences.expanduser().resolve()}")
        manual_gaussian, manual_pointcloud = load_correspondences(args.correspondences.expanduser().resolve())
    elif args.pick_order == "gaussian-first":
        print("  先在 Gaussian 中选点，再在完整点云中按相同顺序选择对应结构点。")
        manual_gaussian = pick_with_pause(
            o3d,
            gaussian_pcd,
            "Step 1/2: pick source/moving points on Gaussian centers",
            args.num_points,
        )
        manual_pointcloud = pick_with_pause(
            o3d,
            pointcloud_pcd,
            "Step 2/2: pick target/reference points on complete processed point cloud",
            args.num_points,
        )
    else:
        print("  先在完整点云中选点，再在 Gaussian 中按相同顺序选择对应结构点。")
        manual_pointcloud = pick_with_pause(
            o3d,
            pointcloud_pcd,
            "Step 1/2: pick target/reference points on complete processed point cloud",
            args.num_points,
        )
        manual_gaussian = pick_with_pause(
            o3d,
            gaussian_pcd,
            "Step 2/2: pick source/moving points on Gaussian centers",
            args.num_points,
        )

    print(f"  对应点获取完成：{len(manual_gaussian)} 对。")
    print("\n[配准 3/7] 使用 Umeyama 相似变换进行粗配准")
    scale, rotation, translation, initial_matrix, initial_rmse = umeyama_similarity(manual_gaussian, manual_pointcloud)
    print("\nInitial Gaussian -> pointcloud similarity transform")
    print("  scale:", scale)
    print("  rotation:\n", rotation)
    print("  translation:", translation)
    print("  manual-pair RMSE:", initial_rmse)
    print("  matrix:\n", initial_matrix)

    if args.no_icp:
        print("\n[配准 4/7] 已按配置跳过 ICP，粗配准矩阵直接作为最终矩阵。")
        final_matrix = initial_matrix
        icp_payload = None
    else:
        print("\n[配准 4/7] 过滤点云并执行 ICP 精配准")
        print("  Gaussian 过滤用于移除超出目标范围、低处点和统计离群点。")
        print("  完整点云过滤用于缩小 ICP 目标范围；过滤仅影响 ICP，不修改原始文件。")
        icp_gaussian_pcd, icp_filter_payload, _ = filter_gaussian_for_icp(
            o3d, gaussian_pcd, pointcloud_pcd, initial_matrix, args
        )
        icp_target_pcd, target_filter_payload = filter_target_for_icp(o3d, pointcloud_pcd, args)
        print("\n  正在执行 ICP：变换后的 Gaussian → 完整点云")
        final_matrix, icp_result, icp_stage_payload = run_icp(o3d, icp_gaussian_pcd, icp_target_pcd, initial_matrix, args)
        print("ICP result")
        print("  fitness:", icp_result.fitness)
        print("  inlier_rmse:", icp_result.inlier_rmse)
        print("  matrix:\n", final_matrix)
        icp_payload = {
            "method": args.icp_method,
            "multiscale": bool(args.icp_multiscale),
            "multiscale_spec": args.icp_multiscale_spec if args.icp_multiscale else None,
            "voxel_size": args.icp_voxel_size,
            "max_correspondence_distance": args.max_correspondence_distance,
            "iterations": args.icp_iterations,
            "gaussian_filter": icp_filter_payload,
            "target_filter": target_filter_payload,
            "stages": icp_stage_payload,
            "fitness": float(icp_result.fitness),
            "inlier_rmse": float(icp_result.inlier_rmse),
            "matrix": np.asarray(final_matrix).tolist(),
        }

    transformed_manual = transform_points(manual_gaussian, final_matrix)
    final_manual_rmse = float(np.sqrt(np.mean(np.sum((transformed_manual - manual_pointcloud) ** 2, axis=1))))

    payload = {
        "source": str(gaussian_path),
        "target": str(pointcloud_path),
        "transform_direction": "p_pointcloud = T_gaussian_to_pointcloud @ p_gaussian_homogeneous",
        "manual_correspondence_count": int(len(manual_gaussian)),
        "manual_correspondences": {
            "source_gaussian": manual_gaussian.tolist(),
            "target_pointcloud": manual_pointcloud.tolist(),
            "pairs": [
                {"gaussian": gaussian.tolist(), "pointcloud": pointcloud.tolist()}
                for gaussian, pointcloud in zip(manual_gaussian, manual_pointcloud)
            ],
        },
        "initial_similarity": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "manual_pair_rmse": initial_rmse,
            "matrix": initial_matrix.tolist(),
        },
        "icp_refined": icp_payload,
        "final_manual_pair_rmse": final_manual_rmse,
        "final_matrix": np.asarray(final_matrix).tolist(),
    }
    registration_output = args.output.expanduser().resolve()
    print("\n[配准 5/7] 保存配准矩阵及过程元数据")
    print(f"  最终人工对应点 RMSE：{final_manual_rmse:.6f}")
    print(f"  输出文件：{registration_output}")
    save_transform(registration_output, payload)

    if not args.no_save_aligned_gaussian:
        print("\n[配准 6/7] 将最终矩阵应用到完整 Gaussian 文件并保存")
        aligned_output = args.aligned_gaussian_output.expanduser().resolve()
        aligned_metadata = (
            args.aligned_gaussian_metadata.expanduser().resolve()
            if args.aligned_gaussian_metadata
            else aligned_output.with_suffix(".json")
        )
        save_aligned_gaussian(
            gaussian_path,
            aligned_output,
            aligned_metadata,
            final_matrix,
            pointcloud_path,
            registration_output,
        )
        print(f"  对齐后 Gaussian：{aligned_output}")
        print(f"  对齐结果元数据：{aligned_metadata}")
    else:
        print("\n[配准 6/7] 已按配置跳过对齐 Gaussian 文件保存。")

    if not args.no_view:
        print("\n[配准 7/7] 打开最终叠加复核窗口")
        print("  完整点云显示原始颜色，对齐后的 Gaussian 显示红色；请检查主要结构是否重合。")
        print("  按 Q 关闭窗口并结束流程。")
        visualize_registration(o3d, pointcloud_pcd, gaussian_pcd, final_matrix)
    else:
        print("\n[配准 7/7] 已按配置跳过最终叠加复核窗口。")

    print("\nGaussian 到完整点云配准流程完成。")
    print(f"  最终变换：{registration_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
