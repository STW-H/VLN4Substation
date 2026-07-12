#!/usr/bin/env python3
"""Build base and derived planning maps from merged annotations."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.paths import CONFIGS_DIR  # noqa: E402
from substation_vln.planning.common.base_map import (  # noqa: E402
    build_base_masks,
    build_grid_spec,
    extract_patrol_points,
    load_merged_annotations,
)
from substation_vln.planning.common.derived_map import build_derived_layers  # noqa: E402
from substation_vln.planning.common.io import (  # noqa: E402
    load_yaml_config,
    resolve_project_path,
    save_cost_png,
    save_mask_png,
    save_overlay_png,
    write_json,
)


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "planning" / "build_planning_map_erfeishan.yaml"


def output_path(output_dir: Path, outputs: dict, key: str) -> Path:
    return output_dir / outputs[key]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build planning masks, distance fields, and cost map.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    annotation_path = resolve_project_path(config["paths"]["annotation"])
    output_dir = resolve_project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = config["outputs"]

    payload = load_merged_annotations(annotation_path)
    base_params = config["base_map"]
    derived_params = config["derived_map"]

    grid = build_grid_spec(
        payload,
        resolution_m=float(base_params["resolution_m"]),
        padding_m=float(base_params["bounds_padding_m"]),
    )
    base_masks = build_base_masks(
        payload,
        grid,
        preferred_path_width_m=float(base_params["preferred_path_width_m"]),
    )
    derived_layers = build_derived_layers(
        base_masks,
        resolution_m=grid.resolution_m,
        params=derived_params,
    )
    layers = {**base_masks, **derived_layers}
    patrol_points = extract_patrol_points(payload)

    np.savez_compressed(
        output_path(output_dir, outputs, "npz"),
        boundary_mask=layers["boundary_mask"],
        obstacle_mask=layers["obstacle_mask"],
        inflated_obstacle_mask=layers["inflated_obstacle_mask"],
        free_space_mask=layers["free_space_mask"],
        preferred_road_mask=layers["preferred_road_mask"],
        preferred_path_mask=layers["preferred_path_mask"],
        distance_to_obstacle_m=layers["distance_to_obstacle_m"],
        distance_to_preferred_path_m=layers["distance_to_preferred_path_m"],
        preferred_path_attraction=layers["preferred_path_attraction"],
        cost_map=layers["cost_map"],
    )

    metadata = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config.expanduser().resolve()),
        "annotation": str(annotation_path),
        "grid": grid.to_dict(),
        "base_map": base_params,
        "derived_map": derived_params,
        "outputs": {key: str(output_path(output_dir, outputs, key)) for key in outputs},
        "annotation_source_files": payload.get("source_files", []),
        "counts": {
            "patrol_points": len(patrol_points),
            "boundary_cells": int(layers["boundary_mask"].sum()),
            "obstacle_cells": int(layers["obstacle_mask"].sum()),
            "inflated_obstacle_cells": int(layers["inflated_obstacle_mask"].sum()),
            "free_space_cells": int(layers["free_space_mask"].sum()),
            "preferred_road_cells": int(layers["preferred_road_mask"].sum()),
            "preferred_path_cells": int(layers["preferred_path_mask"].sum()),
        },
    }
    write_json(output_path(output_dir, outputs, "metadata"), metadata)
    write_json(output_path(output_dir, outputs, "patrol_points"), patrol_points)

    save_mask_png(output_path(output_dir, outputs, "boundary_mask_png"), layers["boundary_mask"])
    save_mask_png(output_path(output_dir, outputs, "obstacle_mask_png"), layers["obstacle_mask"])
    save_mask_png(output_path(output_dir, outputs, "inflated_obstacle_mask_png"), layers["inflated_obstacle_mask"])
    save_mask_png(output_path(output_dir, outputs, "free_space_mask_png"), layers["free_space_mask"])
    save_mask_png(output_path(output_dir, outputs, "preferred_road_mask_png"), layers["preferred_road_mask"])
    save_mask_png(output_path(output_dir, outputs, "preferred_path_mask_png"), layers["preferred_path_mask"])
    save_cost_png(output_path(output_dir, outputs, "cost_map_png"), layers["cost_map"])
    save_overlay_png(output_path(output_dir, outputs, "planning_overlay_png"), layers, layers["cost_map"])

    print(f"Built planning map: {output_path(output_dir, outputs, 'npz')}")
    print(f"Metadata: {output_path(output_dir, outputs, 'metadata')}")
    print(f"Patrol points: {len(patrol_points)}")
    print("Grid:", grid.to_dict())
    print("Counts:", metadata["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
