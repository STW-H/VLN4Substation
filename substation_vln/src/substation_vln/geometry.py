"""Small geometry helpers for point-cloud registration."""

from __future__ import annotations

from typing import Any

import numpy as np


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def segment_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[1] - points[0]))


def bounds_text(name: str, points: np.ndarray) -> str:
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    return "\n".join(
        [
            f"{name}:",
            f"  points: {len(points):,}",
            f"  min:    {bounds_min}",
            f"  max:    {bounds_max}",
            f"  extent: {bounds_max - bounds_min}",
        ]
    )


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("Need at least three source/target point pairs with matching shape")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    rotation = u @ np.diag(d) @ vt
    source_variance = np.sum(source_centered**2) / source.shape[0]
    scale = float(np.sum(singular_values * d) / source_variance)
    translation = target_mean - scale * rotation @ source_mean

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    transformed = transform_points(source, matrix)
    rmse = float(np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1))))
    return scale, rotation, translation, matrix, rmse


def make_coordinate_frame(o3d: Any, points: np.ndarray, ratio: float = 0.05, min_size: float = 1.0):
    frame_size = max(float(np.ptp(points, axis=0).max()) * ratio, min_size)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
