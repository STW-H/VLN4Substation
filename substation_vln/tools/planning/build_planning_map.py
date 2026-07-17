#!/usr/bin/env python3
"""Build mode-independent planning masks and distance fields."""

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
    extract_equipment_regions,
    extract_patrol_points,
    extract_robot_start_points,
    load_merged_annotations,
)
from substation_vln.planning.common.derived_map import (  # noqa: E402
    build_derived_layers,
    build_traversal_cost_map,
)
from substation_vln.planning.common.io import (  # noqa: E402
    load_yaml_config,
    resolve_project_path,
    save_cost_png,
    save_mask_png,
    save_overlay_png,
    write_json,
)


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "planning" / "build_planning_map.yaml"


def output_path(output_dir: Path, outputs: dict, key: str) -> Path:
    return output_dir / outputs[key]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build planning masks and distance fields.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    annotation_path = resolve_project_path(config["paths"]["annotation"])
    output_dir = resolve_project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = config["outputs"]

    payload = load_merged_annotations(annotation_path)
    base_params = config["base_map"]
    collision_params = config["collision"]

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
        obstacle_inflation_radius_m=float(
            collision_params["obstacle_inflation_radius_m"]
        ),
    )
    layers = {**base_masks, **derived_layers}
    mode_dir = resolve_project_path(config["paths"]["movement_mode_dir"])
    mode_cost_maps: dict[str, np.ndarray] = {}
    mode_metadata: dict[str, dict] = {}
    for mode in config["movement_modes"]:
        mode_name = str(mode)
        mode_config_path = mode_dir / f"{mode_name}.yaml"
        mode_config = load_yaml_config(mode_config_path)
        if str(mode_config.get("movement_mode")) != mode_name:
            raise ValueError(
                f"Mode name mismatch in {mode_config_path}: "
                f"{mode_config.get('movement_mode')!r}"
            )
        mode_cost_maps[f"cost_map_{mode_name}"] = build_traversal_cost_map(
            layers, mode_config["cost_map"], pose_aware=False
        )
        mode_cost_maps[f"pose_cost_map_{mode_name}"] = build_traversal_cost_map(
            layers, mode_config["cost_map"], pose_aware=True
        )
        mode_metadata[mode_name] = {
            "config": str(mode_config_path),
            "cost_map": mode_config["cost_map"],
        }
    patrol_points = extract_patrol_points(payload)
    robot_start_points = extract_robot_start_points(payload)
    equipment_regions = extract_equipment_regions(payload)

    map_arrays = {
        "boundary_mask": layers["boundary_mask"],
        "obstacle_mask": layers["obstacle_mask"],
        "inflated_obstacle_mask": layers["inflated_obstacle_mask"],
        "free_space_mask": layers["free_space_mask"],
        "preferred_road_mask": layers["preferred_road_mask"],
        "preferred_path_mask": layers["preferred_path_mask"],
        "directed_preferred_path_mask": layers["directed_preferred_path_mask"],
        "preferred_path_direction_x": layers["preferred_path_direction_x"],
        "preferred_path_direction_y": layers["preferred_path_direction_y"],
        "narrow_space_mask": layers["narrow_space_mask"],
        "equipment_mask": layers["equipment_mask"],
        "equipment_index_mask": layers["equipment_index_mask"],
        "distance_to_obstacle_m": layers["distance_to_obstacle_m"],
        "distance_to_preferred_path_m": layers["distance_to_preferred_path_m"],
        "pose_center_space_mask": layers["pose_center_space_mask"],
        **mode_cost_maps,
    }
    np.savez_compressed(output_path(output_dir, outputs, "npz"), **map_arrays)

    metadata = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config.expanduser().resolve()),
        "annotation": str(annotation_path),
        "grid": grid.to_dict(),
        "base_map": base_params,
        "collision": collision_params,
        "movement_modes": mode_metadata,
        "outputs": {key: str(output_path(output_dir, outputs, key)) for key in outputs},
        "annotation_source_files": payload.get("source_files", []),
        "counts": {
            "patrol_points": len(patrol_points),
            "robot_start_points": len(robot_start_points),
            "equipment_regions": len(equipment_regions),
            "boundary_cells": int(layers["boundary_mask"].sum()),
            "obstacle_cells": int(layers["obstacle_mask"].sum()),
            "inflated_obstacle_cells": int(layers["inflated_obstacle_mask"].sum()),
            "free_space_cells": int(layers["free_space_mask"].sum()),
            "preferred_road_cells": int(layers["preferred_road_mask"].sum()),
            "preferred_path_cells": int(layers["preferred_path_mask"].sum()),
            "directed_preferred_path_cells": int(
                layers["directed_preferred_path_mask"].sum()
            ),
            "narrow_space_cells": int(layers["narrow_space_mask"].sum()),
            "equipment_cells": int(layers["equipment_mask"].sum()),
            "pose_center_space_cells": int(layers["pose_center_space_mask"].sum()),
        },
    }
    write_json(output_path(output_dir, outputs, "metadata"), metadata)
    write_json(output_path(output_dir, outputs, "patrol_points"), patrol_points)
    write_json(output_path(output_dir, outputs, "robot_start_points"), robot_start_points)
    if "equipment_regions" in outputs:
        write_json(output_path(output_dir, outputs, "equipment_regions"), equipment_regions)

    save_mask_png(output_path(output_dir, outputs, "boundary_mask_png"), layers["boundary_mask"])
    save_mask_png(output_path(output_dir, outputs, "obstacle_mask_png"), layers["obstacle_mask"])
    save_mask_png(output_path(output_dir, outputs, "inflated_obstacle_mask_png"), layers["inflated_obstacle_mask"])
    save_mask_png(output_path(output_dir, outputs, "free_space_mask_png"), layers["free_space_mask"])
    save_mask_png(output_path(output_dir, outputs, "preferred_road_mask_png"), layers["preferred_road_mask"])
    save_mask_png(output_path(output_dir, outputs, "preferred_path_mask_png"), layers["preferred_path_mask"])
    save_mask_png(
        output_path(output_dir, outputs, "directed_preferred_path_mask_png"),
        layers["directed_preferred_path_mask"],
    )
    save_mask_png(output_path(output_dir, outputs, "narrow_space_mask_png"), layers["narrow_space_mask"])
    if "equipment_mask_png" in outputs:
        save_mask_png(output_path(output_dir, outputs, "equipment_mask_png"), layers["equipment_mask"])
    for mode in mode_metadata:
        cost_map = mode_cost_maps[f"cost_map_{mode}"]
        pose_cost_map = mode_cost_maps[f"pose_cost_map_{mode}"]
        save_cost_png(output_dir / f"cost_map_{mode}.png", cost_map)
        save_cost_png(output_dir / f"pose_cost_map_{mode}.png", pose_cost_map)
        save_overlay_png(
            output_dir / f"planning_overlay_{mode}.png",
            layers,
            cost_map,
        )
    print(f"Built planning map: {output_path(output_dir, outputs, 'npz')}")
    print(f"Metadata: {output_path(output_dir, outputs, 'metadata')}")
    print(f"Patrol points: {len(patrol_points)}")
    print(f"Robot start points: {len(robot_start_points)}")
    print(f"Equipment regions: {len(equipment_regions)}")
    print("Grid:", grid.to_dict())
    print("Counts:", metadata["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
