"""Registration utilities for Gaussian-to-point-cloud alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import transform_points
from .pointcloud_io import make_pcd, transform_binary_ply_xyz


def load_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "manual_correspondences" in data:
        manual = data["manual_correspondences"]
        source = np.asarray(manual["source_gaussian"], dtype=np.float64)
        target = np.asarray(manual["target_pointcloud"], dtype=np.float64)
    elif "pairs" in data:
        source = np.asarray([p["source"] if "source" in p else p["gaussian"] for p in data["pairs"]], dtype=np.float64)
        target = np.asarray([p["target"] if "target" in p else p["pointcloud"] for p in data["pairs"]], dtype=np.float64)
    elif "source" in data and "target" in data:
        source = np.asarray(data["source"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
    else:
        raise SystemExit(
            f"No reusable manual correspondences found in: {path}\n"
            "Expected one of these JSON formats:\n"
            "  1) a previous transform JSON containing manual_correspondences\n"
            "  2) {'source': [[...]], 'target': [[...]]}\n"
            "  3) {'pairs': [{'gaussian': [...], 'pointcloud': [...]}]}"
        )
    return source, target


def save_transform(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved transform: {path}")


def save_aligned_gaussian(
    source_gaussian: Path,
    output_path: Path,
    metadata_path: Path,
    matrix: np.ndarray,
    target_pointcloud: Path,
    registration_json: Path,
) -> None:
    """Save Gaussian centers transformed into the processed point-cloud frame."""
    output_path = output_path.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    print(f"\nSaving aligned Gaussian point cloud: {output_path}")
    stats = transform_binary_ply_xyz(source_gaussian, output_path, matrix)
    payload = {
        "source_gaussian": str(source_gaussian),
        "target_pointcloud": str(target_pointcloud),
        "registration_json": str(registration_json),
        "output": str(output_path),
        "transform_direction": "p_processed_gaussian = T_gaussian_to_pointcloud @ p_raw_gaussian_homogeneous",
        "matrix": np.asarray(matrix).tolist(),
        "note": (
            "This file is the Gaussian center point cloud transformed into the processed point-cloud coordinate "
            "system. It is intended for downstream point-cloud/map workflows."
        ),
        "output_stats": stats,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"aligned Gaussian metadata: {metadata_path}")


def preprocess_for_icp(o3d: Any, pcd, voxel_size: float, estimate_normals: bool):
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    if estimate_normals:
        radius = max(voxel_size * 3.0, 1.0)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
    return pcd


def filter_gaussian_for_icp(o3d: Any, gaussian_pcd, target_pcd, initial_matrix: np.ndarray, args):
    filtered = gaussian_pcd
    before = len(filtered.points)
    keep_mask = np.ones(before, dtype=bool)
    steps = []

    if not args.no_icp_filter_target_bounds:
        source_points = np.asarray(gaussian_pcd.points)
        source_colors = np.asarray(gaussian_pcd.colors) if gaussian_pcd.has_colors() else None
        transformed = transform_points(source_points, initial_matrix)
        target_points = np.asarray(target_pcd.points)
        bounds_min = target_points.min(axis=0) - args.icp_filter_target_bounds_margin
        bounds_max = target_points.max(axis=0) + args.icp_filter_target_bounds_margin
        mask = np.all((transformed >= bounds_min) & (transformed <= bounds_max), axis=1)
        if int(mask.sum()) >= 100:
            keep_mask &= mask
            filtered = make_pcd(
                o3d,
                source_points[keep_mask],
                color=(1.0, 0.15, 0.05),
                colors=source_colors[keep_mask] if source_colors is not None and len(source_colors) == len(source_points) else None,
            )
            steps.append(
                {
                    "method": "target_bounds_after_initial_transform",
                    "margin": args.icp_filter_target_bounds_margin,
                    "before": int(len(source_points)),
                    "after": int(mask.sum()),
                }
            )
        else:
            steps.append(
                {
                    "method": "target_bounds_after_initial_transform",
                    "margin": args.icp_filter_target_bounds_margin,
                    "before": int(len(source_points)),
                    "after": int(mask.sum()),
                    "skipped": "too_few_points_remaining",
                }
            )

    if not args.no_icp_filter_height:
        source_points = np.asarray(gaussian_pcd.points)
        source_colors = np.asarray(gaussian_pcd.colors) if gaussian_pcd.has_colors() else None
        transformed = transform_points(source_points, initial_matrix)
        target_points = np.asarray(target_pcd.points)
        z_threshold = float(target_points[:, 2].min() + args.icp_min_height_above_target_min)
        mask = transformed[:, 2] >= z_threshold
        if int((keep_mask & mask).sum()) >= 100:
            keep_mask &= mask
            filtered = make_pcd(
                o3d,
                source_points[keep_mask],
                color=(1.0, 0.15, 0.05),
                colors=source_colors[keep_mask] if source_colors is not None and len(source_colors) == len(source_points) else None,
            )
            steps.append(
                {
                    "method": "source_height_after_initial_transform",
                    "target_min_z": float(target_points[:, 2].min()),
                    "min_height_above_target_min": args.icp_min_height_above_target_min,
                    "z_threshold": z_threshold,
                    "before": int(len(source_points)),
                    "after": int(keep_mask.sum()),
                }
            )
        else:
            steps.append(
                {
                    "method": "source_height_after_initial_transform",
                    "target_min_z": float(target_points[:, 2].min()),
                    "min_height_above_target_min": args.icp_min_height_above_target_min,
                    "z_threshold": z_threshold,
                    "after": int((keep_mask & mask).sum()),
                    "skipped": "too_few_points_remaining",
                }
            )

    if not args.no_icp_filter_statistical:
        current_before = len(filtered.points)
        filtered, inlier_indices = filtered.remove_statistical_outlier(
            nb_neighbors=args.icp_filter_nb_neighbors,
            std_ratio=args.icp_filter_std_ratio,
        )
        current_global_indices = np.flatnonzero(keep_mask)
        statistical_keep_mask = np.zeros_like(keep_mask)
        statistical_keep_mask[current_global_indices[np.asarray(inlier_indices, dtype=np.int64)]] = True
        keep_mask &= statistical_keep_mask
        steps.append(
            {
                "method": "statistical_outlier_removal",
                "nb_neighbors": args.icp_filter_nb_neighbors,
                "std_ratio": args.icp_filter_std_ratio,
                "before": int(current_before),
                "after": int(len(inlier_indices)),
            }
        )

    after = len(filtered.points)
    print("\nGaussian filtering for ICP only")
    print(f"  before: {before:,}")
    print(f"  after:  {after:,}")
    for step in steps:
        print(f"  {step}")

    return filtered, {"before": int(before), "after": int(after), "steps": steps}, keep_mask


def filter_target_for_icp(o3d: Any, target_pcd, args):
    if args.no_icp_filter_height:
        return target_pcd, {"enabled": False}

    target_points = np.asarray(target_pcd.points)
    target_colors = np.asarray(target_pcd.colors) if target_pcd.has_colors() else None
    z_min = float(target_points[:, 2].min())
    z_threshold = z_min + args.icp_min_height_above_target_min
    mask = target_points[:, 2] >= z_threshold
    if int(mask.sum()) < 100:
        print("\nTarget height filtering skipped: too few points would remain.")
        return target_pcd, {
            "enabled": True,
            "skipped": "too_few_points_remaining",
            "before": int(len(target_points)),
            "after": int(mask.sum()),
            "target_min_z": z_min,
            "z_threshold": z_threshold,
        }

    filtered = make_pcd(
        o3d,
        target_points[mask],
        color=(0.1, 0.45, 1.0),
        colors=target_colors[mask] if target_colors is not None and len(target_colors) == len(target_points) else None,
    )
    payload = {
        "enabled": True,
        "method": "target_height",
        "before": int(len(target_points)),
        "after": int(mask.sum()),
        "target_min_z": z_min,
        "min_height_above_target_min": args.icp_min_height_above_target_min,
        "z_threshold": z_threshold,
    }
    print("\nTarget filtering for ICP only")
    print(f"  before: {len(target_points):,}")
    print(f"  after:  {int(mask.sum()):,}")
    print(f"  z_threshold: {z_threshold:.6f}")
    return filtered, payload


def run_single_icp(o3d: Any, source, target, init: np.ndarray, method: str, voxel_size: float, max_distance: float, iterations: int):
    src = preprocess_for_icp(o3d, source, voxel_size, method == "point_to_plane")
    tgt = preprocess_for_icp(o3d, target, voxel_size, method == "point_to_plane")

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
    if method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    return o3d.pipelines.registration.registration_icp(src, tgt, max_distance, init, estimation, criteria)


def parse_multiscale_spec(spec: str) -> list[tuple[float, float]]:
    stages = []
    for raw_stage in spec.split(","):
        stage = raw_stage.strip()
        if not stage:
            continue
        parts = stage.split(":")
        if len(parts) != 2:
            raise SystemExit(f"Invalid --icp-multiscale-spec stage: {stage}. Expected voxel:distance")
        stages.append((float(parts[0]), float(parts[1])))
    if not stages:
        raise SystemExit("--icp-multiscale-spec produced no stages")
    return stages


def run_icp(o3d: Any, source, target, init: np.ndarray, args) -> tuple[np.ndarray, Any, list[dict[str, Any]]]:
    stages = parse_multiscale_spec(args.icp_multiscale_spec) if args.icp_multiscale else [
        (args.icp_voxel_size, args.max_correspondence_distance)
    ]

    current = init
    results = []
    result = None
    for index, (voxel_size, max_distance) in enumerate(stages, start=1):
        print(
            f"\nICP stage {index}/{len(stages)}: "
            f"method={args.icp_method}, voxel={voxel_size}, distance={max_distance}, iterations={args.icp_iterations}"
        )
        result = run_single_icp(
            o3d,
            source,
            target,
            current,
            args.icp_method,
            voxel_size,
            max_distance,
            args.icp_iterations,
        )
        current = result.transformation
        print("  fitness:", result.fitness)
        print("  inlier_rmse:", result.inlier_rmse)
        results.append(
            {
                "stage": index,
                "method": args.icp_method,
                "voxel_size": voxel_size,
                "max_correspondence_distance": max_distance,
                "iterations": args.icp_iterations,
                "fitness": float(result.fitness),
                "inlier_rmse": float(result.inlier_rmse),
                "matrix": np.asarray(result.transformation).tolist(),
            }
        )

    return current, result, results
