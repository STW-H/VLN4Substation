#!/usr/bin/env python3
"""Parse a natural-language inspection request and generate a complete displayed route."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.config import load_yaml_config  # noqa: E402
from substation_vln.paths import CONFIGS_DIR  # noqa: E402
from substation_vln.planning.astar import path_length_m  # noqa: E402
from substation_vln.planning.common.grid import GridSpec  # noqa: E402
from substation_vln.planning.common.io import read_json, resolve_project_path, write_json  # noqa: E402
from substation_vln.planning.common.visualization import (  # noqa: E402
    draw_route_segments,
    load_aligned_ortho_thumbnail,
)
from substation_vln.planning.improved_astar import (  # noqa: E402
    PoseAStarConfig,
    hierarchical_pose_region_astar,
    unpack_pose_free_masks,
)
from substation_vln.planning.improved_astar.route_planning import (  # noqa: E402
    equipment_goal_states,
    pose_state_payload,
    resolve_route_start,
    waypoint_goal_states,
)
from substation_vln.tasks import (  # noqa: E402
    DeepSeekRouteParser,
    RoutePlan,
    build_semantic_catalog,
    validate_catalog_references,
)


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "planning" / "run_natural_language_route.yaml"


def load_saved_plan(path: Path) -> RoutePlan:
    payload = read_json(path)
    plan_payload = payload.get("instruction_plan", payload)
    parser = plan_payload.get("parser", {})
    return RoutePlan.from_model_response(
        plan_payload,
        raw_instruction=str(plan_payload.get("raw_instruction", "loaded plan")),
        provider=str(parser.get("provider", "deepseek")),
        model=str(parser.get("model", "deepseek-v4-pro")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse natural language and plan a displayed inspection route.")
    parser.add_argument("--plan-json", type=Path, help="Use an already parsed plan without calling DeepSeek")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    configured_instruction = str(config.get("instruction", "")).strip()
    if args.plan_json is None and not configured_instruction:
        raise SystemExit("config.instruction 不能为空。")
    paths = config["paths"]
    planning_metadata = read_json(resolve_project_path(paths["planning_metadata"]))
    regions_metadata = read_json(resolve_project_path(paths["goal_regions_metadata"]))
    starts = read_json(resolve_project_path(paths["robot_start_points"]))
    equipment_items = regions_metadata["equipment"]
    semantic_catalog = build_semantic_catalog(starts, equipment_items)
    start_catalog = semantic_catalog["robot_start_points"]
    equipment_catalog = semantic_catalog["inspection_equipment"]
    write_json(resolve_project_path(paths["task_semantic_catalog"]), semantic_catalog)

    if args.plan_json:
        plan = load_saved_plan(args.plan_json.expanduser().resolve())
    else:
        deepseek = config["deepseek"]
        task_parser = DeepSeekRouteParser(
            api_key_env=str(deepseek.get("api_key_env", "DEEPSEEK_API_KEY")),
            api_key_file=(
                resolve_project_path(deepseek["api_key_file"])
                if deepseek.get("api_key_file")
                else None
            ),
            base_url=str(deepseek.get("base_url", "https://api.deepseek.com")),
            model=str(deepseek.get("model", "deepseek-v4-pro")),
            timeout_s=float(deepseek.get("timeout_s", 60.0)),
        )
        try:
            plan = task_parser.parse(
                configured_instruction,
                available_start_points=start_catalog,
                available_equipment=equipment_catalog,
            )
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"DeepSeek任务解析失败：{exc}") from exc

    start_by_name = {str(item["start_point_name"]): item for item in starts}
    equipment_by_name = {str(item["equipment_name"]): item for item in equipment_items}
    try:
        validate_catalog_references(plan, semantic_catalog)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    mode_path = resolve_project_path(paths["movement_mode_dir"]) / f"{plan.movement_mode}.yaml"
    if not mode_path.exists():
        raise SystemExit(f"Movement-mode config does not exist: {mode_path}")
    mode_config = load_yaml_config(mode_path)
    planner_config = PoseAStarConfig(**mode_config["astar"])
    hierarchical = mode_config["hierarchical"]
    waypoint_config = mode_config["waypoint"]

    grid = GridSpec.from_dict(planning_metadata["grid"])
    map_data = np.load(resolve_project_path(paths["planning_map"]))
    region_data = np.load(resolve_project_path(paths["goal_regions"]))
    pose_free = unpack_pose_free_masks(region_data["pose_free_packed"], grid.width)
    seed = args.random_seed
    if seed is None:
        seed = config.get("start", {}).get("random_seed")
    try:
        start_state, selected_start = resolve_route_start(
            starts,
            plan,
            pose_free,
            grid,
            float(config.get("start", {}).get("yaw_deg", 0.0)),
            None if seed is None else int(seed),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cost_key = f"pose_cost_map_{plan.movement_mode}"
    if cost_key not in map_data.files:
        raise SystemExit(
            f"规划地图缺少 {cost_key}，请重新运行 build_planning_map.py。"
        )
    cost_map = map_data[cost_key]
    direction_keys = ("preferred_path_direction_x", "preferred_path_direction_y")
    if any(key not in map_data.files for key in direction_keys):
        raise SystemExit("规划地图缺少推荐路径方向场，请重新运行 build_planning_map.py。")
    preferred_path_direction = (
        map_data[direction_keys[0]],
        map_data[direction_keys[1]],
    )

    current_state = start_state
    segment_states: list[list[tuple[int, int, int]]] = []
    segment_records: list[dict] = []
    checkpoints = [(start_state[0], start_state[1])]
    total_cost = 0.0
    total_expanded = 0

    def plan_segment(label: str, goals: dict[tuple[int, int, int], float]):
        nonlocal current_state, total_cost, total_expanded
        result = hierarchical_pose_region_astar(
            pose_free,
            cost_map,
            current_state,
            goals,
            planner_config,
            resolution_m=grid.resolution_m,
            corridor_radius_m=float(hierarchical["corridor_radius_m"]),
            max_corridor_radius_m=float(hierarchical["max_corridor_radius_m"]),
            preferred_path_direction=preferred_path_direction,
        )
        if not result.pose_result.found:
            raise SystemExit(f"No route found for segment: {label}")
        states = result.pose_result.path_states
        segment_states.append(states)
        current_state = states[-1]
        checkpoints.append((current_state[0], current_state[1]))
        total_cost += result.pose_result.total_cost
        total_expanded += result.pose_result.expanded_nodes
        record = {
            "label": label,
            "start": pose_state_payload(grid, states[0], pose_free.shape[0]),
            "goal": pose_state_payload(grid, states[-1], pose_free.shape[0]),
            "node_count": len(states),
            "path_cost": result.pose_result.path_cost,
            "terminal_cost": result.pose_result.terminal_goal_cost,
            "total_cost": result.pose_result.total_cost,
            "expanded_nodes": result.pose_result.expanded_nodes,
            "used_corridor_radius_m": result.used_corridor_radius_m,
        }
        segment_records.append(record)
        return record

    for waypoint_name in plan.intermediate_points:
        waypoint = start_by_name[waypoint_name]
        goals = waypoint_goal_states(
            waypoint,
            pose_free,
            grid,
            float(waypoint_config["tolerance_m"]),
            float(waypoint_config["terminal_distance_weight"]),
        )
        plan_segment(f"waypoint:{waypoint_name}", goals)

    equipment = equipment_by_name[plan.target_point]
    goals, lookup = equipment_goal_states(
        int(equipment["equipment_index"]), region_data, planner_config.tilt_cost_weight
    )
    if not goals:
        raise SystemExit(f"Target equipment has no feasible goal pose: {plan.target_point}")
    target_record = plan_segment(f"equipment:{plan.target_point}", goals)
    global_index = lookup[current_state]
    target_record["camera"] = {
        "pan_rad": float(region_data["goal_camera_pan_rad"][global_index]),
        "pan_deg": float(math.degrees(region_data["goal_camera_pan_rad"][global_index])),
        "tilt_rad": float(region_data["goal_camera_tilt_rad"][global_index]),
        "tilt_deg": float(math.degrees(region_data["goal_camera_tilt_rad"][global_index])),
        "tilt_cost": float(region_data["goal_tilt_costs"][global_index]),
    }

    full_states: list[tuple[int, int, int]] = []
    for states in segment_states:
        full_states.extend(states if not full_states else states[1:])
    full_xy = [
        pose_state_payload(grid, state, pose_free.shape[0])["xy"]
        for state in full_states
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_project_path(paths["output_dir"])
    output_json = output_dir / f"natural_language_route_{timestamp}.json"
    output_png = output_dir / f"natural_language_route_{timestamp}.png"
    payload = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "instruction_plan": plan.to_dict(),
        "movement_mode_config": str(mode_path),
        "selected_start": selected_start,
        "start_state": pose_state_payload(grid, start_state, pose_free.shape[0]),
        "segments": segment_records,
        "target": {"equipment": equipment, "route_segment": target_record},
        "route": {
            "states": [
                pose_state_payload(grid, state, pose_free.shape[0])
                for state in full_states
            ],
            "node_count": len(full_states),
            "length_m": path_length_m(tuple(tuple(xy) for xy in full_xy)),
            "total_cost": total_cost,
            "expanded_nodes": total_expanded,
        },
        "outputs": {"json": str(output_json), "image": str(output_png)},
    }
    write_json(output_json, payload)
    background = load_aligned_ortho_thumbnail(
        resolve_project_path(paths["ortho_image"]),
        resolve_project_path(paths["ortho_metadata"]),
        resolve_project_path(paths["ortho_thumbnail_2k"]),
        grid,
        max_resolution=int(config.get("display", {}).get("ortho_max_resolution", 2048)),
    )
    rendered = draw_route_segments(
        background,
        segment_states,
        checkpoints,
        plan.movement_mode,
        output_png,
        int(config.get("display", {}).get("path_line_width", 3)),
    )
    print(f"解析模式：{plan.movement_mode}")
    if plan.movement_mode_reason:
        print(f"模式理由：{plan.movement_mode_reason}")
    print(f"选择起点：{selected_start['start_point_name']}")
    print(f"规划分段：{len(segment_records)}，总长度：{payload['route']['length_m']:.3f} m")
    print(f"结果：{output_json}")
    print(f"展示：{output_png}")
    if not args.no_display:
        display = config.get("display", {})
        scale = min(
            int(display.get("max_window_width", 1400)) / grid.width,
            int(display.get("max_window_height", 1000)) / grid.height,
            1.0,
        )
        shown = cv2.resize(
            rendered,
            (max(1, round(grid.width * scale)), max(1, round(grid.height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow(str(display.get("window_name", "Natural-language inspection route")), shown)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
