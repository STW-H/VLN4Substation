"""Point-cloud loading and Open3D conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_OPEN3D = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}
SUPPORTED_LAS = {".las", ".laz"}

PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("Please install Open3D in the active environment.") from exc
    return o3d


def parse_binary_ply_vertex(path: Path) -> tuple[int, list[tuple[str, str]], int]:
    with open(path, "rb") as f:
        header_bytes = b""
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {path}")
            header_bytes += line
            text = line.decode("ascii", errors="ignore").strip()
            header.append(text)
            if line.strip() == b"end_header":
                break

    if not any(line == "format binary_little_endian 1.0" for line in header):
        raise SystemExit(f"Only binary_little_endian PLY is supported: {path}")

    vertex_count = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element ") and not line.startswith("element vertex"):
            in_vertex = False
        elif in_vertex and line.startswith("property"):
            parts = line.split()
            if len(parts) != 3:
                raise SystemExit(f"Unsupported PLY property line: {line}")
            prop_type, prop_name = parts[1], parts[2]
            if prop_type not in PLY_TYPES:
                raise SystemExit(f"Unsupported PLY property type {prop_type}: {path}")
            props.append((prop_name, prop_type))

    if vertex_count <= 0:
        raise SystemExit(f"No vertex element found: {path}")
    return vertex_count, props, len(header_bytes)


def sample_ply_points(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    vertex_count, props, data_offset = parse_binary_ply_vertex(path)
    names = [name for name, _ in props]
    dtype = np.dtype([(name, PLY_TYPES[prop_type]) for name, prop_type in props])
    if not {"x", "y", "z"}.issubset(names):
        raise SystemExit(f"PLY has no x/y/z properties: {path}")

    requested_points = vertex_count if max_points <= 0 else min(max_points, vertex_count)
    step = max(1, int(np.ceil(vertex_count / requested_points)))
    indices = np.arange(0, vertex_count, step, dtype=np.int64)[:requested_points]

    records = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(vertex_count,))
    selected = records if len(indices) == vertex_count and step == 1 else records[indices]

    points = np.column_stack([selected["x"], selected["y"], selected["z"]]).astype(np.float64, copy=False)
    colors = None
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([selected["red"], selected["green"], selected["blue"]]).astype(
            np.float64, copy=False
        )
        max_val = 65535.0 if colors.max(initial=0) > 255 else 255.0
        colors = np.clip(colors / max_val, 0.0, 1.0)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh_c0 = 0.28209479177387814
        colors = np.column_stack([selected["f_dc_0"], selected["f_dc_1"], selected["f_dc_2"]]).astype(
            np.float64, copy=False
        )
        colors = np.clip(colors * sh_c0 + 0.5, 0.0, 1.0)
    return points, colors


def make_pcd(o3d: Any, points: np.ndarray, color=(0.7, 0.7, 0.7), colors: np.ndarray | None = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(points), 1)))
    return pcd


def describe_pcd(pcd) -> str:
    pts = np.asarray(pcd.points)
    bounds_min = pts.min(axis=0)
    bounds_max = pts.max(axis=0)
    extent = bounds_max - bounds_min
    return "\n".join(
        [
            f"points: {len(pts):,}",
            f"min:    {bounds_min}",
            f"max:    {bounds_max}",
            f"extent: {extent}",
            f"has colors:  {pcd.has_colors()}",
            f"has normals: {pcd.has_normals()}",
        ]
    )


def load_las_as_pcd(path: Path, max_points: int):
    try:
        import laspy
    except ImportError as exc:
        raise SystemExit("LAS/LAZ support requires laspy. Install it with: pip install laspy lazrs") from exc

    o3d = import_open3d()
    with laspy.open(path) as reader:
        point_count = int(reader.header.point_count)
        if max_points > 0 and point_count > max_points:
            step = max(1, int(np.ceil(point_count / max_points)))
            xs, ys, zs = [], [], []
            rs, gs, bs = [], [], []
            has_rgb = all(name in reader.header.point_format.dimension_names for name in ("red", "green", "blue"))
            seen = 0
            kept = 0
            for points in reader.chunk_iterator(1_000_000):
                local = np.arange(len(points))
                mask = ((seen + local) % step) == 0
                if np.any(mask):
                    xs.append(points.x[mask])
                    ys.append(points.y[mask])
                    zs.append(points.z[mask])
                    if has_rgb:
                        rs.append(points.red[mask])
                        gs.append(points.green[mask])
                        bs.append(points.blue[mask])
                    kept += int(mask.sum())
                seen += len(points)
                if kept >= max_points:
                    break
            xyz = np.vstack((np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))).T[:max_points]
            rgb = np.vstack((np.concatenate(rs), np.concatenate(gs), np.concatenate(bs))).T[: len(xyz)] if has_rgb and rs else None
        else:
            las = reader.read()
            xyz = np.vstack((las.x, las.y, las.z)).T
            rgb = np.vstack((las.red, las.green, las.blue)).T if all(hasattr(las, name) for name in ("red", "green", "blue")) else None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        rgb = rgb.astype(np.float64)
        max_val = 65535.0 if rgb.max(initial=0) > 255 else 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb / max_val, 0, 1))
    return pcd
