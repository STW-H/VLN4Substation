"""Voxel-cached point-cloud line-of-sight checks for inspection goal poses."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from substation_vln.planning.common.grid import GridSpec
from substation_vln.preprocessing.pointcloud_io import binary_ply_dtype, parse_binary_ply_vertex


@dataclass(frozen=True)
class VoxelVisibilityMap:
    centers_xyz: np.ndarray
    voxel_size_m: float

    def build_tree(self) -> cKDTree:
        return cKDTree(np.asarray(self.centers_xyz, dtype=np.float64), compact_nodes=True, balanced_tree=True)


def _voxel_keys(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return ix.astype(np.int64) + int(nx) * (iy.astype(np.int64) + int(ny) * iz.astype(np.int64))


def build_or_load_voxel_visibility_map(
    pointcloud_path: Path,
    cache_path: Path,
    grid: GridSpec,
    config: dict[str, Any],
    *,
    force_rebuild: bool = False,
) -> VoxelVisibilityMap:
    pointcloud_path = pointcloud_path.expanduser().resolve()
    cache_path = cache_path.expanduser().resolve()
    voxel_size = float(config.get("voxel_size_m", 0.2))
    z_min = float(config.get("ground_z_m", 0.0)) + float(config.get("ground_clearance_m", 0.15))
    z_max = float(config.get("max_z_m", 35.0))
    signature = {
        "pointcloud": str(pointcloud_path),
        "pointcloud_size": int(pointcloud_path.stat().st_size),
        "pointcloud_mtime_ns": int(pointcloud_path.stat().st_mtime_ns),
        "voxel_size_m": voxel_size,
        "bounds": [grid.min_x, grid.max_x, grid.min_y, grid.max_y, z_min, z_max],
    }
    if cache_path.exists() and not force_rebuild and bool(config.get("reuse_cache", True)):
        cached = np.load(cache_path, allow_pickle=False)
        cached_signature = json.loads(str(cached["signature_json"].item()))
        if cached_signature == signature:
            return VoxelVisibilityMap(
                centers_xyz=np.asarray(cached["centers_xyz"], dtype=np.float32),
                voxel_size_m=voxel_size,
            )

    vertex_count, props, data_offset = parse_binary_ply_vertex(pointcloud_path)
    dtype = binary_ply_dtype(props)
    records = np.memmap(pointcloud_path, dtype=dtype, mode="r", offset=data_offset, shape=(vertex_count,))
    nx = int(math.ceil((grid.max_x - grid.min_x) / voxel_size))
    ny = int(math.ceil((grid.max_y - grid.min_y) / voxel_size))
    chunk_size = int(config.get("chunk_size", 2_000_000))
    key_parts: list[np.ndarray] = []
    for start in range(0, vertex_count, chunk_size):
        stop = min(vertex_count, start + chunk_size)
        chunk = records[start:stop]
        x = np.asarray(chunk["x"], dtype=np.float64)
        y = np.asarray(chunk["y"], dtype=np.float64)
        z = np.asarray(chunk["z"], dtype=np.float64)
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(z)
            & (x >= grid.min_x)
            & (x < grid.max_x)
            & (y >= grid.min_y)
            & (y < grid.max_y)
            & (z >= z_min)
            & (z < z_max)
        )
        if np.any(valid):
            ix = np.floor((x[valid] - grid.min_x) / voxel_size).astype(np.int32)
            iy = np.floor((y[valid] - grid.min_y) / voxel_size).astype(np.int32)
            iz = np.floor((z[valid] - z_min) / voxel_size).astype(np.int32)
            key_parts.append(np.unique(_voxel_keys(ix, iy, iz, nx, ny)))
        print(f"\r  构建遮挡体素：{stop:,}/{vertex_count:,}", end="", flush=True)
    print()
    keys = np.unique(np.concatenate(key_parts)) if key_parts else np.empty(0, dtype=np.int64)
    iz = keys // (nx * ny)
    remainder = keys - iz * nx * ny
    iy = remainder // nx
    ix = remainder - iy * nx
    centers = np.column_stack(
        (
            grid.min_x + (ix + 0.5) * voxel_size,
            grid.min_y + (iy + 0.5) * voxel_size,
            z_min + (iz + 0.5) * voxel_size,
        )
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        centers_xyz=centers,
        signature_json=np.asarray(json.dumps(signature, sort_keys=True)),
    )
    return VoxelVisibilityMap(centers_xyz=centers, voxel_size_m=voxel_size)


def _ray_clear_batch(
    origins_xyz: np.ndarray,
    target_xyz: np.ndarray,
    tree: cKDTree,
    config: dict[str, Any],
    voxel_size_m: float,
) -> np.ndarray:
    origins = np.asarray(origins_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    vectors = target[None, :] - origins
    distances = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(distances[:, None], 1.0e-9)
    camera_exclusion = float(config.get("camera_exclusion_m", 0.4))
    target_exclusion = float(config.get("target_exclusion_m", 0.4))
    step = float(config.get("ray_step_m", max(0.05, 0.5 * voxel_size_m)))
    usable = np.maximum(0.0, distances - camera_exclusion - target_exclusion)
    max_steps = int(math.ceil(float(usable.max(initial=0.0)) / step))
    if max_steps <= 0:
        return np.ones(len(origins), dtype=bool)
    sample_distance = camera_exclusion + np.arange(max_steps, dtype=np.float64) * step
    valid = sample_distance[None, :] < (distances - target_exclusion)[:, None]
    points = origins[:, None, :] + directions[:, None, :] * sample_distance[None, :, None]
    flat_valid = valid.ravel()
    flat_points = points.reshape((-1, 3))[flat_valid]
    candidate_ids = np.repeat(np.arange(len(origins), dtype=np.int32), max_steps)[flat_valid]

    clearance = float(config.get("clearance_radius_m", 0.2))
    if bool(config.get("include_voxel_uncertainty", True)):
        clearance += 0.5 * math.sqrt(3.0) * voxel_size_m
    nearest, _ = tree.query(flat_points, k=1, distance_upper_bound=clearance, workers=1)
    blocked = np.zeros(len(origins), dtype=bool)
    if len(candidate_ids):
        np.logical_or.at(blocked, candidate_ids, np.isfinite(nearest))
    return ~blocked


def candidate_visibility(
    origins_xyz: np.ndarray,
    target_xyz: np.ndarray,
    visibility_map: VoxelVisibilityMap,
    config: dict[str, Any],
    tree: cKDTree | None = None,
) -> np.ndarray:
    """Return whether the ROI-center sightline is clear for each camera origin."""
    origins = np.asarray(origins_xyz, dtype=np.float64)
    tree = tree if tree is not None else visibility_map.build_tree()
    batch_size = max(1, int(config.get("batch_size", 512)))
    feasible = np.empty(len(origins), dtype=bool)
    for start in range(0, len(origins), batch_size):
        stop = min(len(origins), start + batch_size)
        feasible[start:stop] = _ray_clear_batch(
            origins[start:stop],
            target_xyz,
            tree,
            config,
            visibility_map.voxel_size_m,
        )
    return feasible
