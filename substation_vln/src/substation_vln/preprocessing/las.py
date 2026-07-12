"""LAS/LAZ conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_las_real_ply(input_path: Path, output_path: Path, chunk_size: int, metadata_path: Path | None = None) -> None:
    try:
        import laspy
    except ImportError as exc:
        raise SystemExit("Please install LAS support first: pip install laspy lazrs") from exc

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(input_path) as reader:
        header = reader.header
        point_count = int(header.point_count)
        dims = set(header.point_format.dimension_names)
        has_rgb = {"red", "green", "blue"}.issubset(dims)

        ply_header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {point_count}",
            "property double x",
            "property double y",
            "property double z",
        ]
        dtype_fields = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
        if has_rgb:
            ply_header.extend(["property uchar red", "property uchar green", "property uchar blue"])
            dtype_fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
        ply_header.append("end_header")

        written = 0
        with open(output_path, "wb") as f:
            f.write(("\n".join(ply_header) + "\n").encode("ascii"))
            for points in reader.chunk_iterator(chunk_size):
                arr = np.empty(len(points), dtype=np.dtype(dtype_fields))
                arr["x"] = points.x
                arr["y"] = points.y
                arr["z"] = points.z
                if has_rgb:
                    rgb = np.vstack((points.red, points.green, points.blue)).T.astype(np.float64, copy=False)
                    max_val = 65535.0 if rgb.max(initial=0) > 255 else 255.0
                    rgb8 = np.clip(rgb / max_val * 255.0, 0, 255).astype(np.uint8)
                    arr["red"] = rgb8[:, 0]
                    arr["green"] = rgb8[:, 1]
                    arr["blue"] = rgb8[:, 2]
                arr.tofile(f)
                written += len(points)
                print(f"\rwritten {written:,}/{point_count:,}", end="", flush=True)
        print(f"\nsaved: {output_path}")

        metadata = {
            "input": str(input_path),
            "output": str(output_path),
            "point_count": point_count,
            "las_scales": np.asarray(header.scales).tolist(),
            "las_offsets": np.asarray(header.offsets).tolist(),
            "las_mins": np.asarray(header.mins).tolist(),
            "las_maxs": np.asarray(header.maxs).tolist(),
            "has_rgb": has_rgb,
            "coordinate_rule": "real = integer * LAS header scale + LAS header offset",
        }
        metadata_output = metadata_path.expanduser().resolve() if metadata_path else output_path.with_suffix(".json")
        with open(metadata_output, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"metadata: {metadata_output}")
