"""Accelerated robust visibility-corridor testing."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range


def spherical_structure(radius_m: float, voxel_size_m: float) -> np.ndarray:
    if radius_m <= 0:
        return np.ones((1, 1, 1), dtype=np.bool_)
    radius_voxels = int(np.ceil(radius_m / voxel_size_m))
    axis = np.arange(-radius_voxels, radius_voxels + 1, dtype=np.float64)
    dx, dy, dz = np.meshgrid(axis, axis, axis, indexing="ij")
    return ((dx * dx + dy * dy + dz * dz) * voxel_size_m**2 <= radius_m**2 + 1.0e-12)


def dilate_occupancy(occupancy: np.ndarray, radius_m: float, voxel_size_m: float) -> np.ndarray:
    if radius_m <= 0:
        return occupancy.copy()
    from scipy.ndimage import binary_dilation

    return binary_dilation(occupancy, structure=spherical_structure(radius_m, voxel_size_m))


if njit is not None:

    @njit(cache=True)
    def _one_ray_visible(occupancy, local_origin, voxel_size, start, end):
        p0 = (start - local_origin) / voxel_size
        p1 = (end - local_origin) / voxel_size
        ix, iy, iz = int(np.floor(p0[0])), int(np.floor(p0[1])), int(np.floor(p0[2]))
        ex, ey, ez = int(np.floor(p1[0])), int(np.floor(p1[1])), int(np.floor(p1[2]))
        nx, ny, nz = occupancy.shape
        dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]

        sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
        inf = 1.0e30
        tdx = voxel_size / abs(dx) if sx != 0 else inf
        tdy = voxel_size / abs(dy) if sy != 0 else inf
        tdz = voxel_size / abs(dz) if sz != 0 else inf
        bx = local_origin[0] + (ix + (1 if sx > 0 else 0)) * voxel_size
        by = local_origin[1] + (iy + (1 if sy > 0 else 0)) * voxel_size
        bz = local_origin[2] + (iz + (1 if sz > 0 else 0)) * voxel_size
        tmx = (bx - start[0]) / dx if sx != 0 else inf
        tmy = (by - start[1]) / dy if sy != 0 else inf
        tmz = (bz - start[2]) / dz if sz != 0 else inf

        max_steps = nx + ny + nz + 16
        for _ in range(max_steps):
            if ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= ny or iz >= nz:
                return False
            if occupancy[ix, iy, iz]:
                return False
            if ix == ex and iy == ey and iz == ez:
                return True
            minimum = min(tmx, tmy, tmz)
            tolerance = 1.0e-12
            if tmx <= minimum + tolerance:
                ix += sx
                tmx += tdx
            if tmy <= minimum + tolerance:
                iy += sy
                tmy += tdy
            if tmz <= minimum + tolerance:
                iz += sz
                tmz += tdz
        return False


    @njit(parallel=True, cache=True)
    def _batch_visible_numba(occupancy, local_origin, voxel_size, cameras, target, camera_exclusion, target_exclusion):
        result = np.zeros(len(cameras), dtype=np.bool_)
        for index in prange(len(cameras)):
            camera = cameras[index]
            vector = target - camera
            distance = np.sqrt(np.dot(vector, vector))
            if distance <= camera_exclusion + target_exclusion:
                continue
            direction = vector / distance
            start = camera + camera_exclusion * direction
            end = target - target_exclusion * direction
            result[index] = _one_ray_visible(occupancy, local_origin, voxel_size, start, end)
        return result


def batch_visibility(
    occupancy: np.ndarray,
    local_origin: np.ndarray,
    voxel_size_m: float,
    camera_positions: np.ndarray,
    target_xyz: np.ndarray,
    camera_exclusion_radius_m: float,
    target_exclusion_radius_m: float,
) -> tuple[np.ndarray, str]:
    cameras = np.ascontiguousarray(camera_positions, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if njit is None:  # pragma: no cover
        raise RuntimeError("Numba is required for batch visibility in the current implementation.")
    result = _batch_visible_numba(
        np.ascontiguousarray(occupancy),
        np.asarray(local_origin, dtype=np.float64),
        float(voxel_size_m),
        cameras,
        target,
        float(camera_exclusion_radius_m),
        float(target_exclusion_radius_m),
    )
    return result, "numba_parallel_3d_dda"
