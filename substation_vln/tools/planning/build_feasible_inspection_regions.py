#!/usr/bin/env python3
"""Build feasible 2D inspection regions from 3D target visibility."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.config import config_path, config_value, load_yaml_config  # noqa: E402
from substation_vln.inspection_regions.io import (  # noqa: E402
    load_targets,
    save_distance_image,
    save_mask,
    save_feasible_inspection_region_overlay,
    write_json,
)
from substation_vln.inspection_regions.feasible_region import FeasibleInspectionRegionConfig, compute_feasible_inspection_region  # noqa: E402
from substation_vln.inspection_regions.voxel_map import build_or_load_voxel_grid  # noqa: E402
from substation_vln.paths import CONFIGS_DIR  # noqa: E402
from substation_vln.planning.common.grid import GridSpec  # noqa: E402


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "planning" / "build_feasible_inspection_regions.yaml"


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Compute feasible 2D inspection regions from 3D targets and a voxelized point cloud.", parents=[pre_parser])
    parser.add_argument("--rebuild-voxel-cache", action="store_true", default=config_value(config.get("voxel", {}), "rebuild_cache", False))
    parser.add_argument("--target-id", action="append", default=None, help="Process only this target ID; repeat for multiple targets")
    args = parser.parse_args()

    paths = config["paths"]
    pointcloud_path = config_path(paths, "pointcloud")
    targets_path = config_path(paths, "targets")
    planning_map_path = config_path(paths, "planning_map")
    planning_metadata_path = config_path(paths, "planning_metadata")
    output_dir = config_path(paths, "output_dir")
    cache_path = config_path(paths, "voxel_cache")
    for name, path in (
        ("pointcloud", pointcloud_path),
        ("targets", targets_path),
        ("planning_map", planning_map_path),
        ("planning_metadata", planning_metadata_path),
    ):
        if path is None or not path.exists():
            raise SystemExit(f"Missing {name}: {path}")
    if output_dir is None or cache_path is None:
        raise SystemExit("output_dir and voxel_cache must be configured.")

    try:
        target_payload = load_targets(targets_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    target_ids = set(args.target_id or config.get("target_ids", []))
    targets = [item for item in target_payload["targets"] if not target_ids or item["target_id"] in target_ids]
    missing_ids = target_ids - {item["target_id"] for item in targets}
    if missing_ids:
        raise SystemExit(f"Unknown target IDs: {sorted(missing_ids)}")
    if not targets:
        raise SystemExit("No inspection targets selected.")

    metadata = json.loads(planning_metadata_path.read_text(encoding="utf-8"))
    grid = GridSpec.from_dict(metadata["grid"])
    with np.load(planning_map_path) as planning_data:
        boundary_mask = planning_data["boundary_mask"].astype(np.uint8)
        free_space_mask = planning_data["free_space_mask"].astype(np.uint8)

    voxel_config = config["voxel"]
    print("\n[可行巡视区域 1/3] 构建或加载三维稀疏体素地图")
    voxel_grid, cache_reused = build_or_load_voxel_grid(
        pointcloud_path,
        cache_path,
        voxel_size_m=float(voxel_config["voxel_size_m"]),
        chunk_size=int(voxel_config.get("chunk_size", 2_000_000)),
        rebuild=bool(args.rebuild_voxel_cache),
    )

    visibility_config = config["visibility"]
    region_config = FeasibleInspectionRegionConfig(
        ground_z_m=float(visibility_config.get("ground_z_m", 0.0)),
        camera_height_m=float(visibility_config.get("camera_height_m", 1.0)),
        visibility_clearance_radius_m=float(visibility_config.get("visibility_clearance_radius_m", 0.2)),
        camera_exclusion_radius_m=float(visibility_config.get("camera_exclusion_radius_m", 0.3)),
        min_region_area_m2=float(visibility_config.get("min_region_area_m2", 0.5)),
        morphology_open_radius_m=float(visibility_config.get("morphology_open_radius_m", 0.0)),
    )
    annotated_ground = float(target_payload["ground_plane"]["z_m"])
    annotated_camera = float(target_payload["camera"]["height_above_ground_m"])
    if not np.isclose(region_config.ground_z_m, annotated_ground):
        raise SystemExit(f"ground_z mismatch: targets={annotated_ground}, config={region_config.ground_z_m}")
    if not np.isclose(region_config.camera_height_m, annotated_camera):
        raise SystemExit(f"camera_height mismatch: targets={annotated_camera}, config={region_config.camera_height_m}")

    print("\n[可行巡视区域 2/3] 逐目标计算距离、鲁棒视线和二维安全区域")
    summaries = []
    for index, target in enumerate(targets, start=1):
        target_category = target.get("device_category", target.get("category", ""))
        print(f"\n  目标 {index}/{len(targets)}：{target['target_id']} ({target_category})")
        layers, result_metadata = compute_feasible_inspection_region(
            target=target,
            voxel_grid=voxel_grid,
            grid=grid,
            boundary_mask=boundary_mask,
            free_space_mask=free_space_mask,
            config=region_config,
        )
        target_dir = output_dir / target["target_id"]
        target_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target_dir / "feasible_inspection_region.npz", **layers)
        for key in (
            "distance_candidate_mask",
            "robust_visibility_mask",
            "visible_inspection_region_mask",
            "feasible_inspection_region_raw_mask",
            "feasible_inspection_region_mask",
        ):
            if key in layers:
                save_mask(target_dir / f"{key}.png", layers[key])
        save_distance_image(target_dir / "distance_to_target.png", layers["distance_to_target_m"], layers["distance_candidate_mask"])
        save_feasible_inspection_region_overlay(
            target_dir / "feasible_inspection_region_overlay.png",
            boundary_mask=boundary_mask,
            free_space_mask=free_space_mask,
            visible_inspection_region_mask=layers["visible_inspection_region_mask"],
            feasible_inspection_region_mask=layers["feasible_inspection_region_mask"],
            target_xy=target["target_xyz"][:2],
            grid=grid,
        )
        target_metadata = {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "source_pointcloud": str(pointcloud_path),
            "targets_file": str(targets_path),
            "planning_map": str(planning_map_path),
            "target": target,
            "voxel": {
                "voxel_size_m": voxel_grid.voxel_size_m,
                "occupied_voxels": voxel_grid.occupied_count,
                "cache": str(cache_path),
                "cache_reused": cache_reused,
            },
            "visibility_config": visibility_config,
            "result": result_metadata,
        }
        write_json(target_dir / "metadata.json", target_metadata)
        summaries.append(target_metadata)
        print(
            f"    distance={result_metadata.get('distance_candidate_cells', 0):,}, "
            f"visible={result_metadata.get('robust_visible_cells', 0):,}, "
            f"feasible={result_metadata.get('feasible_inspection_region_cells', 0):,}, "
            f"time={result_metadata.get('elapsed_seconds', 0.0):.2f}s"
        )

    print("\n[可行巡视区域 3/3] 保存任务汇总")
    write_json(
        output_dir / "build_summary.json",
        {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "config": str(args.config.expanduser().resolve()),
            "target_count": len(summaries),
            "targets": summaries,
        },
    )
    print(f"完成。输出目录：{output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
