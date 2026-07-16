"""Extract robust three-dimensional equipment geometry from an aligned point cloud."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from substation_vln.planning.common.grid import GridSpec
from substation_vln.preprocessing.pointcloud_io import binary_ply_dtype, parse_binary_ply_vertex


def _limit_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def extract_equipment_geometries(
    pointcloud_path: Path,
    equipment_regions: list[dict[str, Any]],
    equipment_index_mask: np.ndarray,
    grid: GridSpec,
    *,
    ground_z_m: float = 0.0,
    ground_clearance_m: float = 0.15,
    scan_stride: int = 1,
    chunk_size: int = 2_000_000,
    max_points_per_equipment: int = 500_000,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
) -> list[dict[str, Any]]:
    """Scan the PLY once and compute robust bounds for every annotated equipment footprint.

    Point membership is evaluated using the planning grid's equipment-index raster. This
    keeps extraction fast for very large point clouds and makes it consistent with the
    two-dimensional planning representation.
    """
    pointcloud_path = pointcloud_path.expanduser().resolve()
    vertex_count, props, data_offset = parse_binary_ply_vertex(pointcloud_path)
    dtype = binary_ply_dtype(props)
    names = set(dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"Point cloud has no x/y/z fields: {pointcloud_path}")
    if scan_stride < 1:
        raise ValueError("scan_stride must be >= 1")

    records = np.memmap(pointcloud_path, dtype=dtype, mode="r", offset=data_offset, shape=(vertex_count,))
    samples: dict[int, list[np.ndarray]] = {int(item["equipment_index"]): [] for item in equipment_regions}
    sampled_counts = {index: 0 for index in samples}
    z_threshold = float(ground_z_m) + float(ground_clearance_m)

    for start in range(0, vertex_count, chunk_size * scan_stride):
        stop = min(vertex_count, start + chunk_size * scan_stride)
        chunk = records[start:stop:scan_stride]
        x = np.asarray(chunk["x"], dtype=np.float64)
        y = np.asarray(chunk["y"], dtype=np.float64)
        z = np.asarray(chunk["z"], dtype=np.float64)
        inside = (
            (x >= grid.min_x)
            & (x < grid.max_x)
            & (y >= grid.min_y)
            & (y < grid.max_y)
            & np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(z)
            & (z > z_threshold)
        )
        if np.any(inside):
            valid_indices = np.flatnonzero(inside)
            cols = np.floor((x[inside] - grid.min_x) / grid.resolution_m).astype(np.int32)
            rows = np.floor((grid.max_y - y[inside]) / grid.resolution_m).astype(np.int32)
            labels = equipment_index_mask[rows, cols]
            for index in np.unique(labels[labels > 0]):
                selected = valid_indices[labels == index]
                xyz = np.column_stack((x[selected], y[selected], z[selected]))
                samples[int(index)].append(xyz)
                sampled_counts[int(index)] += len(xyz)
                current = samples[int(index)]
                if sum(len(part) for part in current) > max(2 * max_points_per_equipment, 1):
                    reduced = _limit_sample(np.concatenate(current, axis=0), max_points_per_equipment)
                    samples[int(index)] = [reduced]
        print(f"\r  扫描点云：{stop:,}/{vertex_count:,}", end="", flush=True)
    print()

    results: list[dict[str, Any]] = []
    for equipment in equipment_regions:
        index = int(equipment["equipment_index"])
        if not samples[index]:
            raise ValueError(
                f"No above-ground points found for equipment {equipment['equipment_name']!r}; "
                "check its footprint or ground filtering parameters."
            )
        points = _limit_sample(np.concatenate(samples[index], axis=0), max_points_per_equipment)
        lower = np.percentile(points, lower_percentile, axis=0)
        upper = np.percentile(points, upper_percentile, axis=0)
        robust = points[np.all((points >= lower) & (points <= upper), axis=1)]
        if len(robust) < 8:
            robust = points
        robust_min = np.min(robust, axis=0)
        robust_max = np.max(robust, axis=0)
        median = np.median(robust, axis=0)
        aim_center = 0.5 * (robust_min + robust_max)
        result = dict(equipment)
        result.update(
            {
                "pointcloud_sample_count": int(len(points)),
                "pointcloud_points_before_cap": int(sampled_counts[index]),
                "estimated_full_point_count": int(sampled_counts[index] * scan_stride),
                "ground_filter_z_min_m": z_threshold,
                "robust_percentiles": [float(lower_percentile), float(upper_percentile)],
                "robust_bounds_min_xyz": robust_min.tolist(),
                "robust_bounds_max_xyz": robust_max.tolist(),
                "point_median_xyz": median.tolist(),
                "center_xyz": aim_center.tolist(),
                "center_method": "center of percentile-filtered equipment point-cloud bounds",
            }
        )
        results.append(result)
    return results
