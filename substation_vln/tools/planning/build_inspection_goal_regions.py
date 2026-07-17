#!/usr/bin/env python3
"""Precompute camera-feasible terminal robot poses for every annotated equipment."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.config import load_yaml_config  # noqa: E402
from substation_vln.paths import CONFIGS_DIR  # noqa: E402
from substation_vln.planning.common.grid import GridSpec  # noqa: E402
from substation_vln.planning.common.io import read_json, resolve_project_path, write_json  # noqa: E402
from substation_vln.planning.improved_astar.camera_model import CameraConfig  # noqa: E402
from substation_vln.planning.improved_astar.equipment_geometry import extract_equipment_geometries  # noqa: E402
from substation_vln.planning.improved_astar.goal_pose_region import (  # noqa: E402
    build_pose_free_masks,
    generate_goal_pose_candidates,
    pack_pose_free_masks,
)
from substation_vln.planning.improved_astar.visibility import (  # noqa: E402
    build_or_load_voxel_visibility_map,
    candidate_visibility,
)


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "planning" / "build_inspection_goal_regions.yaml"


def equipment_geometry_signature(items: list[dict]) -> list[dict]:
    """Return the annotation fields that determine point-cloud equipment cropping."""
    return [
        {
            "equipment_name": item.get("equipment_name"),
            "equipment_type": item.get("equipment_type"),
            "geometry_type": item.get("geometry_type"),
            "polygons_xy": item.get("polygons_xy", []),
            "circles": item.get("circles", []),
        }
        for item in items
    ]


def draw_goal_overlay(
    output_path: Path,
    candidates: list[dict[str, np.ndarray | float | int]],
    equipment: list[dict],
    equipment_index_mask: np.ndarray,
    grid: GridSpec,
    visualization_config: dict,
) -> None:
    """Draw every equipment target and terminal-cost-valued conical approach region."""
    image = np.full((grid.height, grid.width, 3), 255, dtype=np.uint8)
    tilt_weight = float(visualization_config.get("tilt_cost_weight", 15.0))
    maximum_cost = max(tilt_weight, 1.0e-6)
    marker_radius = max(0, int(visualization_config.get("goal_marker_radius_px", 1)))

    labels: list[tuple[int, int, int, str, int, bool]] = []
    for item, geometry in zip(candidates, equipment, strict=True):
        index = int(geometry["equipment_index"])
        rows = np.asarray(item["rows"], dtype=np.int32)
        cols = np.asarray(item["cols"], dtype=np.int32)
        terminal_costs = tilt_weight * np.asarray(item["tilt_costs"], dtype=np.float32)
        if len(rows):
            cost_raster = np.full((grid.height, grid.width), np.inf, dtype=np.float32)
            np.minimum.at(cost_raster, (rows, cols), terminal_costs)
            unique_rows, unique_cols = np.nonzero(np.isfinite(cost_raster))
            normalized = np.clip(cost_raster[unique_rows, unique_cols] / maximum_cost, 0.0, 1.0)
            saturation = np.rint(255.0 - 205.0 * normalized).astype(np.uint8)
            hsv = np.column_stack(
                (
                    np.full(len(unique_rows), 60, dtype=np.uint8),
                    saturation,
                    np.full(len(unique_rows), 220, dtype=np.uint8),
                )
            ).reshape((-1, 1, 3))
            colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape((-1, 3))
            for row, col, color in zip(unique_rows, unique_cols, colors, strict=True):
                bgr = tuple(int(channel) for channel in color)
                if marker_radius:
                    cv2.circle(image, (int(col), int(row)), marker_radius, bgr, -1, cv2.LINE_8)
                else:
                    image[row, col] = color

        target_mask = np.asarray(equipment_index_mask) == index
        image[target_mask] = (35, 90, 210) if len(rows) else (40, 40, 190)
        center = np.asarray(geometry["center_xyz"], dtype=np.float64)
        col, row = grid.xy_to_grid(center[None, :2])[0]
        labels.append((int(col), int(row), index, str(geometry["equipment_name"]), len(rows), not len(rows)))

    for col, row, index, _, _, empty in labels:
        color = (0, 0, 180) if empty else (25, 25, 25)
        cv2.putText(
            image,
            f"{index:02d}",
            (int(col) - 8, int(row) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # Absolute terminal-cost saturation legend, matching region-goal A* weights.
    legend_x, legend_y = 30, 45
    legend_width, legend_height = 360, 22
    for offset in range(legend_width):
        normalized = offset / max(legend_width - 1, 1)
        saturation = int(round(255.0 - 205.0 * normalized))
        color = cv2.cvtColor(
            np.asarray([[[60, saturation, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        image[legend_y : legend_y + legend_height, legend_x + offset] = color
    cv2.rectangle(
        image,
        (legend_x, legend_y),
        (legend_x + legend_width, legend_y + legend_height),
        (30, 30, 30),
        1,
    )
    cv2.putText(
        image,
        f"terminal cost: 0 (recommended) -> {maximum_cost:g} (limit)",
        (legend_x, legend_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    list_y = legend_y + legend_height + 35
    for _, _, index, name, count, empty in labels:
        cv2.putText(
            image,
            f"{index:02d}  {name}  poses={count}",
            (legend_x, list_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 180) if empty else (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        list_y += 24
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build camera-feasible inspection goal-pose regions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rebuild-geometry", action="store_true", help="Ignore cached equipment geometry and rescan the point cloud")
    parser.add_argument("--rebuild-visibility", action="store_true", help="Ignore cached scene visibility voxels")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    paths = config["paths"]
    planning_map_path = resolve_project_path(paths["planning_map"])
    planning_metadata_path = resolve_project_path(paths["planning_metadata"])
    equipment_regions_path = resolve_project_path(paths["equipment_regions"])
    pointcloud_path = resolve_project_path(paths["pointcloud"])
    output_dir = resolve_project_path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = config["outputs"]
    geometry_path = output_dir / outputs["equipment_geometry"]

    print("\n[巡视目标区域 1/5] 读取通用规划地图与设备标注")
    metadata = read_json(planning_metadata_path)
    grid = GridSpec.from_dict(metadata["grid"])
    map_data = np.load(planning_map_path)
    equipment_regions = read_json(equipment_regions_path)
    print(f"  巡视设备：{len(equipment_regions)} 台")
    print(f"  规划栅格：{grid.width} × {grid.height}，分辨率 {grid.resolution_m:g} m")
    equipment_index_mask = map_data["equipment_index_mask"]
    missing_indices = [
        (int(item["equipment_index"]), str(item["equipment_name"]))
        for item in equipment_regions
        if not np.any(equipment_index_mask == int(item["equipment_index"]))
    ]
    if missing_indices:
        missing_text = ", ".join(f"{index}:{name}" for index, name in missing_indices)
        raise SystemExit(
            "规划地图中的设备索引为空："
            f"{missing_text}。请先重新运行 build_planning_map.py，再生成巡视区域。"
        )

    print("\n[巡视目标区域 2/5] 提取设备三维点云几何")
    if geometry_path.exists() and not args.rebuild_geometry and bool(config.get("geometry", {}).get("reuse_cache", True)):
        equipment_geometry = read_json(geometry_path)
        cached_signature = equipment_geometry_signature(equipment_geometry)
        current_signature = equipment_geometry_signature(equipment_regions)
        if cached_signature != current_signature:
            print("  缓存设备列表或轮廓与当前标注不一致，将重新扫描点云。")
            equipment_geometry = []
        else:
            print(f"  使用缓存：{geometry_path}")
    else:
        equipment_geometry = []
    if not equipment_geometry:
        geometry_config = config.get("geometry", {})
        equipment_geometry = extract_equipment_geometries(
            pointcloud_path,
            equipment_regions,
            map_data["equipment_index_mask"],
            grid,
            ground_z_m=float(geometry_config.get("ground_z_m", 0.0)),
            ground_clearance_m=float(geometry_config.get("ground_clearance_m", 0.15)),
            scan_stride=int(geometry_config.get("scan_stride", 1)),
            chunk_size=int(geometry_config.get("chunk_size", 2_000_000)),
            max_points_per_equipment=int(geometry_config.get("max_points_per_equipment", 500_000)),
            lower_percentile=float(geometry_config.get("lower_percentile", 1.0)),
            upper_percentile=float(geometry_config.get("upper_percentile", 99.5)),
        )
        write_json(geometry_path, equipment_geometry)
        print(f"  已保存设备几何：{geometry_path}")

    print("\n[巡视目标区域 3/5] 构建局部视线遮挡体素索引")
    visibility_config = config.get("visibility", {})
    visibility_map = None
    visibility_equipment_labels = None
    visibility_cache_path = output_dir / outputs["visibility_voxels"]
    if bool(visibility_config.get("enabled", True)):
        visibility_map = build_or_load_voxel_visibility_map(
            pointcloud_path,
            visibility_cache_path,
            grid,
            visibility_config,
            force_rebuild=args.rebuild_visibility,
        )
        voxel_xy = np.asarray(visibility_map.centers_xyz[:, :2], dtype=np.float64)
        voxel_cols = np.floor((voxel_xy[:, 0] - grid.min_x) / grid.resolution_m).astype(np.int32)
        voxel_rows = np.floor((grid.max_y - voxel_xy[:, 1]) / grid.resolution_m).astype(np.int32)
        voxel_inside = (
            (voxel_rows >= 0)
            & (voxel_rows < grid.height)
            & (voxel_cols >= 0)
            & (voxel_cols < grid.width)
        )
        visibility_equipment_labels = np.zeros(len(voxel_xy), dtype=np.int32)
        visibility_equipment_labels[voxel_inside] = map_data["equipment_index_mask"][
            voxel_rows[voxel_inside], voxel_cols[voxel_inside]
        ]
        print(f"  场景占用体素：{len(visibility_map.centers_xyz):,}")
        print(f"  体素尺寸：{visibility_map.voxel_size_m:g} m")
        print(f"  缓存：{visibility_cache_path}")
    else:
        print("  已在配置中关闭点云遮挡检测。")

    print("\n[巡视目标区域 4/5] 计算机器狗矩形足迹可行状态")
    pose_free_masks = build_pose_free_masks(
        map_data["boundary_mask"],
        map_data["obstacle_mask"],
        map_data["equipment_mask"],
        grid,
        config["robot"],
    )
    print(f"  航向离散数：{pose_free_masks.shape[0]}")

    print("\n[巡视目标区域 5/5] 应用ROI圆锥、碰撞与点云遮挡约束")
    camera = CameraConfig.from_dict(config["camera"])
    generated: list[dict[str, np.ndarray | float | int]] = []
    all_equipment_index: list[np.ndarray] = []
    all_arrays: dict[str, list[np.ndarray]] = {
        "goal_rows": [],
        "goal_cols": [],
        "goal_heading_bins": [],
        "goal_tilt_costs": [],
        "goal_camera_pan_rad": [],
        "goal_camera_tilt_rad": [],
    }
    summaries = []
    mapping = {
        "goal_rows": "rows",
        "goal_cols": "cols",
        "goal_heading_bins": "heading_bins",
        "goal_tilt_costs": "tilt_costs",
        "goal_camera_pan_rad": "camera_pan_rad",
        "goal_camera_tilt_rad": "camera_tilt_rad",
    }
    for equipment in equipment_geometry:
        result = generate_goal_pose_candidates(
            equipment,
            pose_free_masks,
            grid,
            camera,
            config.get("generation", {}),
            config.get("observation_profiles", {}),
        )
        pre_visibility_count = len(np.asarray(result["rows"]))
        if visibility_map is not None and pre_visibility_count:
            rows = np.asarray(result["rows"], dtype=np.int32)
            cols = np.asarray(result["cols"], dtype=np.int32)
            headings = np.asarray(result["heading_bins"], dtype=np.int32)
            xs, ys = grid.grid_to_xy(cols, rows)
            yaw = headings.astype(np.float64) * (2.0 * np.pi / pose_free_masks.shape[0])
            camera_config_values = config["camera"]
            forward_offset = float(camera_config_values.get("forward_offset_m", 0.0))
            lateral_offset = float(camera_config_values.get("lateral_offset_m", 0.0))
            camera_x = xs + forward_offset * np.cos(yaw) - lateral_offset * np.sin(yaw)
            camera_y = ys + forward_offset * np.sin(yaw) + lateral_offset * np.cos(yaw)
            camera_origins = np.column_stack(
                (
                    camera_x,
                    camera_y,
                    np.full(len(rows), float(camera_config_values["height_m"]), dtype=np.float64),
                )
            )
            unique_origins, inverse = np.unique(camera_origins, axis=0, return_inverse=True)
            target_visibility_map = type(visibility_map)(
                centers_xyz=visibility_map.centers_xyz[
                    visibility_equipment_labels != int(equipment["equipment_index"])
                ],
                voxel_size_m=visibility_map.voxel_size_m,
            )
            target_visibility_tree = target_visibility_map.build_tree()
            unique_feasible = candidate_visibility(
                unique_origins,
                np.asarray(result["observation_center_xyz"], dtype=np.float64),
                target_visibility_map,
                visibility_config,
                tree=target_visibility_tree,
            )
            visibility_mask = unique_feasible[inverse]
            for field in (
                "rows",
                "cols",
                "heading_bins",
                "tilt_costs",
                "camera_pan_rad",
                "camera_tilt_rad",
            ):
                result[field] = np.asarray(result[field])[visibility_mask]
            visibility_unique_origin_count = len(unique_origins)
        else:
            visibility_unique_origin_count = pre_visibility_count
        generated.append(result)
        count = len(np.asarray(result["rows"]))
        index = int(equipment["equipment_index"])
        all_equipment_index.append(np.full(count, index, dtype=np.int16))
        for output_name, result_name in mapping.items():
            all_arrays[output_name].append(np.asarray(result[result_name]))
        summaries.append(
            {
                "equipment_index": index,
                "equipment_name": equipment["equipment_name"],
                "equipment_type": equipment["equipment_type"],
                "candidate_pose_count": int(count),
                "candidate_pose_count_before_visibility": int(pre_visibility_count),
                "visibility_rejected_pose_count": int(pre_visibility_count - count),
                "visibility_unique_origin_count": int(visibility_unique_origin_count),
                "search_radius_m": float(result["search_radius_m"]),
                "minimum_distance_m": float(result.get("minimum_distance_m", 0.0)),
                "candidate_stride_cells": int(result["candidate_stride_cells"]),
                "observation_model": str(result["observation_model"]),
                "configured_tilt_min_deg": float(result.get("configured_tilt_min_deg", camera.tilt_min_deg)),
                "configured_tilt_max_deg": float(result.get("configured_tilt_max_deg", camera.tilt_max_deg)),
                "preferred_tilt_deg": float(result.get("preferred_tilt_deg", camera.preferred_tilt_deg)),
                "observation_vertical_min_fraction": float(result["observation_vertical_min_fraction"]),
                "observation_vertical_max_fraction": float(result["observation_vertical_max_fraction"]),
                "observation_z_min_m": float(result["observation_z_min_m"]),
                "observation_z_max_m": float(result["observation_z_max_m"]),
                "observation_center_xyz": np.asarray(result["observation_center_xyz"]).tolist(),
            }
        )
        print(
            f"  {index:02d}. {equipment['equipment_name']}: "
            f"{count:,}/{pre_visibility_count:,} 个无遮挡/ROI圆锥可行位姿"
        )

    def concatenate(parts: list[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype=dtype)

    regions_npz = output_dir / outputs["goal_regions_npz"]
    np.savez_compressed(
        regions_npz,
        pose_free_packed=pack_pose_free_masks(pose_free_masks),
        goal_equipment_index=concatenate(all_equipment_index, np.int16),
        goal_rows=concatenate(all_arrays["goal_rows"], np.int32),
        goal_cols=concatenate(all_arrays["goal_cols"], np.int32),
        goal_heading_bins=concatenate(all_arrays["goal_heading_bins"], np.int16),
        goal_tilt_costs=concatenate(all_arrays["goal_tilt_costs"], np.float32),
        goal_camera_pan_rad=concatenate(all_arrays["goal_camera_pan_rad"], np.float32),
        goal_camera_tilt_rad=concatenate(all_arrays["goal_camera_tilt_rad"], np.float32),
    )
    result_metadata = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config.expanduser().resolve()),
        "planning_map": str(planning_map_path),
        "planning_metadata": str(planning_metadata_path),
        "pointcloud": str(pointcloud_path),
        "grid": grid.to_dict(),
        "robot": config["robot"],
        "camera": config["camera"],
        "geometry": config.get("geometry", {}),
        "generation": config.get("generation", {}),
        "observation_profiles": config.get("observation_profiles", {}),
        "visibility": config.get("visibility", {}),
        "visibility_voxel_count": int(len(visibility_map.centers_xyz)) if visibility_map is not None else 0,
        "equipment": summaries,
        "total_candidate_pose_count": int(sum(item["candidate_pose_count"] for item in summaries)),
        "outputs": {key: str(output_dir / filename) for key, filename in outputs.items()},
    }
    metadata_output = output_dir / outputs["metadata"]
    write_json(metadata_output, result_metadata)
    overlay_output = output_dir / outputs["overlay_png"]
    draw_goal_overlay(
        overlay_output,
        generated,
        equipment_geometry,
        map_data["equipment_index_mask"],
        grid,
        config.get("visualization", {}),
    )
    print(f"\n已保存区域目标：{regions_npz}")
    print(f"已保存元数据：{metadata_output}")
    print(f"已保存复核图：{overlay_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
