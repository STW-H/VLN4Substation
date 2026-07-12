#!/usr/bin/env python3
"""Render an orthographic annotation image from the processed point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.paths import ANNOTATION_OUTPUTS_ERFEISHAN_DIR, DEFAULT_AXIS_CORRECTED_POINTCLOUD  # noqa: E402
from substation_vln.pointcloud_io import binary_ply_dtype, parse_binary_ply_vertex  # noqa: E402


DEFAULT_OUTPUT = ANNOTATION_OUTPUTS_ERFEISHAN_DIR / "axis_corrected_pointcloud_ortho_8k.png"


def parse_background(value: str) -> tuple[int, int, int]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("background must be R,G,B")
    if any(part < 0.0 or part > 1.0 for part in parts):
        raise argparse.ArgumentTypeError("background values must be in [0, 1]")
    return tuple(int(round(part * 255.0)) for part in parts)  # type: ignore[return-value]


def compute_resolution(extent_xy: np.ndarray, min_resolution: int) -> tuple[int, int]:
    width_extent = float(extent_xy[0])
    height_extent = float(extent_xy[1])
    if width_extent <= 0 or height_extent <= 0:
        raise SystemExit(f"Invalid XY extent: {extent_xy}")
    if width_extent >= height_extent:
        height = int(min_resolution)
        width = int(np.ceil(min_resolution * width_extent / height_extent))
    else:
        width = int(min_resolution)
        height = int(np.ceil(min_resolution * height_extent / width_extent))
    return max(width, 1), max(height, 1)


def ply_bounds(path: Path, chunk_size: int) -> tuple[int, list[tuple[str, str]], int, np.ndarray, np.ndarray]:
    vertex_count, props, data_offset = parse_binary_ply_vertex(path)
    dtype = binary_ply_dtype(props)
    records = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(vertex_count,))
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for start in range(0, vertex_count, chunk_size):
        end = min(start + chunk_size, vertex_count)
        chunk = records[start:end]
        xyz_min = np.array([chunk["x"].min(), chunk["y"].min(), chunk["z"].min()], dtype=np.float64)
        xyz_max = np.array([chunk["x"].max(), chunk["y"].max(), chunk["z"].max()], dtype=np.float64)
        bounds_min = np.minimum(bounds_min, xyz_min)
        bounds_max = np.maximum(bounds_max, xyz_max)
    return vertex_count, props, data_offset, bounds_min, bounds_max


def transform_matrices(x_min: float, x_max: float, y_min: float, y_max: float, width: int, height: int):
    sx = (x_max - x_min) / max(width - 1, 1)
    sy = (y_max - y_min) / max(height - 1, 1)
    pixel_to_world = np.array(
        [
            [sx, 0.0, x_min],
            [0.0, -sy, y_max],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    world_to_pixel = np.linalg.inv(pixel_to_world)
    return pixel_to_world, world_to_pixel


def render_streaming(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    min_resolution: int,
    margin: float,
    camera_side: str,
    background: tuple[int, int, int],
    chunk_size: int,
    point_radius: int,
) -> None:
    vertex_count, props, data_offset, bounds_min, bounds_max = ply_bounds(input_path, chunk_size)
    names = [name for name, _ in props]
    if not {"x", "y", "z"}.issubset(names):
        raise SystemExit(f"PLY has no x/y/z properties: {input_path}")
    has_rgb = {"red", "green", "blue"}.issubset(names)

    extent = bounds_max - bounds_min
    x_min = float(bounds_min[0] - margin)
    x_max = float(bounds_max[0] + margin)
    y_min = float(bounds_min[1] - margin)
    y_max = float(bounds_max[1] + margin)
    render_extent_xy = np.array([x_max - x_min, y_max - y_min], dtype=np.float64)
    width, height = compute_resolution(render_extent_xy, min_resolution)
    pixel_to_world, world_to_pixel = transform_matrices(x_min, x_max, y_min, y_max, width, height)

    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = np.asarray(background, dtype=np.uint8)
    z_buffer = np.full(width * height, -np.inf if camera_side == "zplus" else np.inf, dtype=np.float32)

    dtype = binary_ply_dtype(props)
    records = np.memmap(input_path, dtype=dtype, mode="r", offset=data_offset, shape=(vertex_count,))
    x_scale = (width - 1) / (x_max - x_min)
    y_scale = (height - 1) / (y_max - y_min)

    for start in range(0, vertex_count, chunk_size):
        end = min(start + chunk_size, vertex_count)
        chunk = records[start:end]
        xs = np.asarray(chunk["x"], dtype=np.float64)
        ys = np.asarray(chunk["y"], dtype=np.float64)
        zs = np.asarray(chunk["z"], dtype=np.float32)

        cols = np.floor((xs - x_min) * x_scale).astype(np.int64)
        rows = np.floor((y_max - ys) * y_scale).astype(np.int64)
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        if not np.any(valid):
            continue

        cols = cols[valid]
        rows = rows[valid]
        zs = zs[valid]
        flat = rows * width + cols

        if has_rgb:
            rgb = np.column_stack([chunk["red"][valid], chunk["green"][valid], chunk["blue"][valid]])
            max_val = 65535.0 if rgb.max(initial=0) > 255 else 255.0
            rgb = np.clip(rgb.astype(np.float32) / max_val * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = np.full((len(flat), 3), 191, dtype=np.uint8)

        if camera_side == "zplus":
            order = np.lexsort((-zs, flat))
            sorted_flat = flat[order]
            first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
            chosen = order[first]
            update = zs[chosen] > z_buffer[flat[chosen]]
        else:
            order = np.lexsort((zs, flat))
            sorted_flat = flat[order]
            first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
            chosen = order[first]
            update = zs[chosen] < z_buffer[flat[chosen]]

        chosen = chosen[update]
        if len(chosen) > 0:
            z_buffer[flat[chosen]] = zs[chosen]
            image.reshape(-1, 3)[flat[chosen]] = rgb[chosen]

        print(f"\rrendered {end:,}/{vertex_count:,}", end="", flush=True)
    print()

    if point_radius > 0:
        print(f"expanding projected points with radius={point_radius} px")
        image = expand_non_background_pixels(image, background, point_radius)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)

    metadata = {
        "input_pointcloud": str(input_path),
        "output_image": str(output_path),
        "image_size": {"width": int(width), "height": int(height)},
        "camera": {
            "projection": "orthographic",
            "camera_side": camera_side,
            "view_direction": "Z-" if camera_side == "zplus" else "Z+",
            "screen_x_axis": "pointcloud_X",
            "screen_y_axis": "pointcloud_Y",
        },
        "pointcloud_bounds": {
            "min": bounds_min.tolist(),
            "max": bounds_max.tolist(),
            "extent": extent.tolist(),
        },
        "orthographic_bounds": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "margin": float(margin),
        },
        "pixel_to_world_matrix": pixel_to_world.tolist(),
        "world_to_pixel_matrix": world_to_pixel.tolist(),
        "pixel_coordinate": "Use homogeneous [col, row, 1]. pixel_to_world gives [x, y, 1].",
        "render_style": {
            "point_radius_pixels": int(point_radius),
            "note": "Point expansion only changes image appearance; coordinate mapping remains pixel-center based.",
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved image: {output_path}")
    print(f"saved metadata: {metadata_path}")
    print(f"resolution: {width} x {height}")
    print(f"ground sample distance: x={pixel_to_world[0, 0]:.6f} m/px, y={abs(pixel_to_world[1, 1]):.6f} m/px")


def expand_non_background_pixels(image: np.ndarray, background: tuple[int, int, int], radius: int) -> np.ndarray:
    """Expand colored point pixels into nearby background pixels for easier annotation."""
    if radius <= 0:
        return image
    bg = np.asarray(background, dtype=np.uint8)
    base = image.copy()
    result = image.copy()
    source_mask = np.any(base != bg, axis=2)
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dx, dy))

    for dx, dy in offsets:
        src_y0 = max(0, -dy)
        src_y1 = min(base.shape[0], base.shape[0] - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(base.shape[0], base.shape[0] + dy)
        src_x0 = max(0, -dx)
        src_x1 = min(base.shape[1], base.shape[1] - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(base.shape[1], base.shape[1] + dx)

        src_mask = source_mask[src_y0:src_y1, src_x0:src_x1]
        dst = result[dst_y0:dst_y1, dst_x0:dst_x1]
        dst_bg = np.all(dst == bg, axis=2)
        fill = src_mask & dst_bg
        if np.any(fill):
            dst[fill] = base[src_y0:src_y1, src_x0:src_x1][fill]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an orthographic image for 2D annotation.")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_AXIS_CORRECTED_POINTCLOUD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, help="Metadata JSON. Default is output path with .json suffix.")
    parser.add_argument("--min-resolution", type=int, default=8192, help="Shorter image side in pixels")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--camera-side", choices=("zplus", "zminus"), default="zplus")
    parser.add_argument("--background", type=parse_background, default=(255, 255, 255))
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--point-radius",
        type=int,
        default=0,
        help="Expand each projected point by this pixel radius for a denser annotation image; 0 disables expansion.",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Point cloud not found: {input_path}")
    output_path = args.output.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve() if args.metadata else output_path.with_suffix(".json")

    render_streaming(
        input_path=input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        min_resolution=args.min_resolution,
        margin=args.margin,
        camera_side=args.camera_side,
        background=args.background,
        chunk_size=args.chunk_size,
        point_radius=args.point_radius,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
