"""Visualization helpers shared by planning test tools."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from substation_vln.planning.common.grid import GridSpec


def load_aligned_ortho_thumbnail(
    source_image_path: Path,
    source_metadata_path: Path,
    thumbnail_path: Path,
    grid: GridSpec,
    *,
    max_resolution: int = 2048,
) -> np.ndarray:
    """Load a clean 2K ortho thumbnail and warp it onto the planning grid."""
    source_image_path = Path(source_image_path)
    source_metadata_path = Path(source_metadata_path)
    thumbnail_path = Path(thumbnail_path)
    if max_resolution <= 0:
        raise ValueError("max_resolution must be positive")

    rebuild = not thumbnail_path.exists() or thumbnail_path.stat().st_mtime < source_image_path.stat().st_mtime
    if rebuild:
        source = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(f"Failed to read orthographic image: {source_image_path}")
        source_height, source_width = source.shape[:2]
        scale = min(1.0, float(max_resolution) / max(source_width, source_height))
        thumbnail = cv2.resize(
            source,
            (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(thumbnail_path), thumbnail):
            raise OSError(f"Failed to save orthographic thumbnail: {thumbnail_path}")
    else:
        thumbnail = cv2.imread(str(thumbnail_path), cv2.IMREAD_COLOR)
        if thumbnail is None:
            raise FileNotFoundError(f"Failed to read orthographic thumbnail: {thumbnail_path}")

    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_width = int(metadata["image_size"]["width"])
    source_height = int(metadata["image_size"]["height"])
    thumb_height, thumb_width = thumbnail.shape[:2]
    scale_x = thumb_width / source_width
    scale_y = thumb_height / source_height
    pixel_to_world = np.asarray(metadata["pixel_to_world_matrix"], dtype=np.float64)

    # OpenCV resize maps pixel centers, so derive the thumbnail pixel-to-world
    # affine transform with the corresponding half-pixel correction.
    world_x_per_col = float(pixel_to_world[0, 0]) / scale_x
    world_y_per_row = float(pixel_to_world[1, 1]) / scale_y
    world_x_at_zero = float(pixel_to_world[0, 2]) + float(pixel_to_world[0, 0]) * (0.5 / scale_x - 0.5)
    world_y_at_zero = float(pixel_to_world[1, 2]) + float(pixel_to_world[1, 1]) * (0.5 / scale_y - 0.5)

    thumbnail_to_grid = np.asarray(
        [
            [world_x_per_col / grid.resolution_m, 0.0, (world_x_at_zero - grid.min_x) / grid.resolution_m - 0.5],
            [0.0, -world_y_per_row / grid.resolution_m, (grid.max_y - world_y_at_zero) / grid.resolution_m - 0.5],
        ],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        thumbnail,
        thumbnail_to_grid,
        (grid.width, grid.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
