"""Sparse point-cloud voxelization for feasible inspection regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from substation_vln.preprocessing.pointcloud_io import binary_ply_dtype, parse_binary_ply_vertex


@dataclass(frozen=True)
class SparseVoxelGrid:
    origin: np.ndarray
    shape: tuple[int, int, int]
    voxel_size_m: float
    occupied_indices: np.ndarray
    source_path: str
    source_mtime_ns: int

    @property
    def occupied_count(self) -> int:
        return int(len(self.occupied_indices))

    def local_dense_occupancy(
        self,
        world_min: np.ndarray,
        world_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a dense occupancy crop and its voxel-aligned world origin."""
        lo = np.floor((np.asarray(world_min, dtype=np.float64) - self.origin) / self.voxel_size_m).astype(np.int64)
        hi = np.ceil((np.asarray(world_max, dtype=np.float64) - self.origin) / self.voxel_size_m).astype(np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(self.shape, dtype=np.int64))
        if np.any(hi <= lo):
            raise ValueError("Requested local voxel crop does not overlap the global voxel grid.")

        indices = self.occupied_indices
        keep = np.all((indices >= lo) & (indices < hi), axis=1)
        local_indices = indices[keep].astype(np.int64, copy=False) - lo
        local_shape = tuple(int(v) for v in (hi - lo))
        occupancy = np.zeros(local_shape, dtype=np.bool_)
        if len(local_indices):
            occupancy[local_indices[:, 0], local_indices[:, 1], local_indices[:, 2]] = True
        local_origin = self.origin + lo.astype(np.float64) * self.voxel_size_m
        return occupancy, local_origin


def _scan_bounds(records: np.memmap, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        xyz = np.column_stack([chunk["x"], chunk["y"], chunk["z"]]).astype(np.float64, copy=False)
        bounds_min = np.minimum(bounds_min, xyz.min(axis=0))
        bounds_max = np.maximum(bounds_max, xyz.max(axis=0))
    return bounds_min, bounds_max


def _encode_indices(indices: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    ny, nz = int(shape[1]), int(shape[2])
    values = indices.astype(np.uint64, copy=False)
    return (values[:, 0] * np.uint64(ny) + values[:, 1]) * np.uint64(nz) + values[:, 2]


def _decode_keys(keys: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    ny, nz = int(shape[1]), int(shape[2])
    keys = keys.astype(np.uint64, copy=False)
    ix = keys // np.uint64(ny * nz)
    remainder = keys % np.uint64(ny * nz)
    iy = remainder // np.uint64(nz)
    iz = remainder % np.uint64(nz)
    return np.column_stack([ix, iy, iz]).astype(np.int32)


def build_sparse_voxel_grid(
    pointcloud_path: Path,
    voxel_size_m: float,
    chunk_size: int = 2_000_000,
) -> SparseVoxelGrid:
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive.")
    pointcloud_path = pointcloud_path.expanduser().resolve()
    vertex_count, props, data_offset = parse_binary_ply_vertex(pointcloud_path)
    dtype = binary_ply_dtype(props)
    records = np.memmap(pointcloud_path, dtype=dtype, mode="r", offset=data_offset, shape=(vertex_count,))

    print("  扫描点云边界...")
    bounds_min, bounds_max = _scan_bounds(records, chunk_size)
    origin = np.floor(bounds_min / voxel_size_m) * voxel_size_m
    shape_arr = np.floor((bounds_max - origin) / voxel_size_m).astype(np.int64) + 1
    shape = tuple(int(v) for v in shape_arr)
    if int(np.prod(shape_arr, dtype=np.int64)) >= np.iinfo(np.uint64).max:
        raise ValueError("Voxel grid is too large for uint64 indexing.")

    print(f"  点云点数：{vertex_count:,}")
    print(f"  体素尺寸：{voxel_size_m:g} m")
    print(f"  全局体素范围：origin={origin}, shape={shape}")
    key_chunks: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, vertex_count, chunk_size):
        end = min(start + chunk_size, vertex_count)
        chunk = records[start:end]
        xyz = np.column_stack([chunk["x"], chunk["y"], chunk["z"]]).astype(np.float64, copy=False)
        indices = np.floor((xyz - origin) / voxel_size_m).astype(np.int64)
        keys = np.unique(_encode_indices(indices, shape))
        key_chunks.append(keys)
        print(f"\r  体素化 {end:,}/{vertex_count:,}", end="", flush=True)
    print()
    occupied_keys = np.unique(np.concatenate(key_chunks))
    occupied_indices = _decode_keys(occupied_keys, shape)
    print(f"  占据体素：{len(occupied_indices):,}，耗时 {time.perf_counter() - started:.2f} s")
    return SparseVoxelGrid(
        origin=origin.astype(np.float64),
        shape=shape,
        voxel_size_m=float(voxel_size_m),
        occupied_indices=occupied_indices,
        source_path=str(pointcloud_path),
        source_mtime_ns=pointcloud_path.stat().st_mtime_ns,
    )


def save_voxel_grid(path: Path, grid: SparseVoxelGrid) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        origin=grid.origin,
        shape=np.asarray(grid.shape, dtype=np.int32),
        voxel_size_m=np.asarray(grid.voxel_size_m, dtype=np.float64),
        occupied_indices=grid.occupied_indices,
        source_path=np.asarray(grid.source_path),
        source_mtime_ns=np.asarray(grid.source_mtime_ns, dtype=np.int64),
    )


def load_voxel_grid(path: Path) -> SparseVoxelGrid:
    with np.load(path, allow_pickle=False) as data:
        return SparseVoxelGrid(
            origin=data["origin"].astype(np.float64),
            shape=tuple(int(v) for v in data["shape"]),
            voxel_size_m=float(data["voxel_size_m"]),
            occupied_indices=data["occupied_indices"].astype(np.int32),
            source_path=str(data["source_path"]),
            source_mtime_ns=int(data["source_mtime_ns"]),
        )


def build_or_load_voxel_grid(
    pointcloud_path: Path,
    cache_path: Path,
    voxel_size_m: float,
    chunk_size: int = 2_000_000,
    rebuild: bool = False,
) -> tuple[SparseVoxelGrid, bool]:
    pointcloud_path = pointcloud_path.expanduser().resolve()
    cache_path = cache_path.expanduser().resolve()
    if cache_path.exists() and not rebuild:
        cached = load_voxel_grid(cache_path)
        if (
            Path(cached.source_path).resolve() == pointcloud_path
            and cached.source_mtime_ns == pointcloud_path.stat().st_mtime_ns
            and np.isclose(cached.voxel_size_m, voxel_size_m)
        ):
            print(f"  使用体素缓存：{cache_path}")
            return cached, True
        print("  体素缓存与当前点云或参数不一致，将重新构建。")
    grid = build_sparse_voxel_grid(pointcloud_path, voxel_size_m, chunk_size)
    save_voxel_grid(cache_path, grid)
    print(f"  已保存体素缓存：{cache_path}")
    return grid, False
