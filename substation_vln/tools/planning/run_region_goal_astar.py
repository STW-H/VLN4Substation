#!/usr/bin/env python3
"""Run pose-aware A* from a selected start pose to an equipment goal-pose region."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

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
from substation_vln.planning.common.io import resolve_project_path, write_json  # noqa: E402
from substation_vln.planning.common.visualization import load_aligned_ortho_thumbnail  # noqa: E402
from substation_vln.planning.improved_astar import (  # noqa: E402
    PoseAStarConfig,
    path_corridor_mask,
    pose_region_astar_search,
    quantize_heading,
    region_astar_path,
    unpack_pose_free_masks,
)


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "planning" / "run_region_goal_astar.yaml"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def choose_equipment(items: list[dict], requested: str | None) -> dict:
    if requested:
        matches = [item for item in items if str(item["equipment_index"]) == requested or item["equipment_name"] == requested]
        if len(matches) != 1:
            raise SystemExit(f"Equipment {requested!r} was not found uniquely.")
        return matches[0]
    print("\n请选择巡视设备")
    for item in items:
        print(
            f"  {int(item['equipment_index']):02d}: {item['equipment_name']} "
            f"({item['equipment_type']}), 可行位姿 {int(item['candidate_pose_count']):,}"
        )
    while True:
        raw = input("设备编号: ").strip()
        matches = [item for item in items if str(item["equipment_index"]) == raw]
        if matches:
            return matches[0]
        print("请输入列表中的设备编号。")


def choose_start_cell(image: np.ndarray, valid_mask: np.ndarray, display: dict) -> tuple[int, int]:
    height, width = valid_mask.shape
    max_w = int(display.get("max_window_width", 1400))
    max_h = int(display.get("max_window_height", 1000))
    scale = min(max_w / width, max_h / height, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    shown = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    selected: list[tuple[int, int]] = []
    name = str(display.get("window_name", "Region-goal A* start pose"))

    def mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        row = int(np.clip(round(y / scale), 0, height - 1))
        col = int(np.clip(round(x / scale), 0, width - 1))
        if valid_mask[row, col] == 0:
            print("该位置无法容纳当前方向下的机器狗矩形足迹，请重新选择。")
            return
        selected[:] = [(row, col)]
        print(f"  已选择起点栅格：row={row}, col={col}")

    print("\n请左键选择机器狗起点，按 Enter 确认，Q/Esc 取消。")
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, *size)
    cv2.setMouseCallback(name, mouse)
    while True:
        canvas = shown.copy()
        if selected:
            row, col = selected[0]
            cv2.circle(canvas, (round(col * scale), round(row * scale)), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.imshow(name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13) and selected:
            break
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(name)
            raise SystemExit("Start selection canceled.")
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            raise SystemExit("Start selection window closed.")
    cv2.destroyWindow(name)
    return selected[0]


def path_to_payload(grid: GridSpec, states: list[tuple[int, int, int]], heading_bins: int) -> tuple[list[dict], list[list[float]]]:
    rows = np.asarray([state[0] for state in states], dtype=np.int32)
    cols = np.asarray([state[1] for state in states], dtype=np.int32)
    xs, ys = grid.grid_to_xy(cols, rows)
    payload = []
    xy = []
    for state, x, y in zip(states, xs, ys, strict=True):
        yaw = state[2] * 2.0 * math.pi / heading_bins
        payload.append(
            {
                "grid_rc_heading": [int(state[0]), int(state[1]), int(state[2])],
                "xy": [float(x), float(y)],
                "yaw_rad": float(yaw),
                "yaw_deg": float(math.degrees(yaw)),
            }
        )
        xy.append([float(x), float(y)])
    return payload, xy


def make_target_cost_image(
    background_image: np.ndarray,
    equipment_index_mask: np.ndarray,
    target_equipment_index: int,
    goal_rows: np.ndarray,
    goal_cols: np.ndarray,
    terminal_costs: np.ndarray,
    maximum_terminal_cost: float,
    display: dict,
) -> np.ndarray:
    """Render only the target footprint and its cost-valued feasible region."""
    image = np.asarray(background_image, dtype=np.uint8).copy()
    height, width = image.shape[:2]
    target_mask = np.asarray(equipment_index_mask) == int(target_equipment_index)
    target_color = np.asarray((35, 90, 210), dtype=np.float32)
    target_alpha = float(np.clip(display.get("target_overlay_alpha", 0.65), 0.0, 1.0))
    image[target_mask] = np.rint(
        (1.0 - target_alpha) * image[target_mask].astype(np.float32) + target_alpha * target_color
    ).astype(np.uint8)

    cost_raster = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(cost_raster, (goal_rows, goal_cols), np.asarray(terminal_costs, dtype=np.float32))
    valid_rows, valid_cols = np.nonzero(np.isfinite(cost_raster))
    if len(valid_rows):
        normalized = np.clip(
            cost_raster[valid_rows, valid_cols] / max(float(maximum_terminal_cost), 1.0e-6),
            0.0,
            1.0,
        )
        # Fixed green hue/value: low cost is vivid, high cost is pale.
        saturation = np.rint(255.0 - 205.0 * normalized).astype(np.uint8)
        hsv = np.column_stack(
            (
                np.full(len(valid_rows), 60, dtype=np.uint8),
                saturation,
                np.full(len(valid_rows), 220, dtype=np.uint8),
            )
        ).reshape((-1, 1, 3))
        colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape((-1, 3))
        radius = max(0, int(display.get("goal_marker_radius_px", 1)))
        goal_alpha = float(np.clip(display.get("goal_overlay_alpha", 0.80), 0.0, 1.0))
        for row, col, color in zip(valid_rows, valid_cols, colors, strict=True):
            bgr = tuple(int(channel) for channel in color)
            if radius:
                r0, r1 = max(0, row - radius), min(height, row + radius + 1)
                c0, c1 = max(0, col - radius), min(width, col + radius + 1)
                patch = image[r0:r1, c0:c1]
                yy, xx = np.ogrid[r0:r1, c0:c1]
                circle_mask = (yy - row) ** 2 + (xx - col) ** 2 <= radius**2
                patch[circle_mask] = np.rint(
                    goal_alpha * np.asarray(bgr, dtype=np.float32)
                    + (1.0 - goal_alpha) * patch[circle_mask].astype(np.float32)
                ).astype(np.uint8)
            else:
                image[row, col] = np.rint(
                    goal_alpha * color.astype(np.float32)
                    + (1.0 - goal_alpha) * image[row, col].astype(np.float32)
                ).astype(np.uint8)
    return image


def draw_result(
    image: np.ndarray,
    states: list[tuple[int, int, int]],
    output: Path,
    display: dict,
) -> np.ndarray:
    overlay = image.copy()
    if len(states) >= 2:
        points = np.asarray([(col, row) for row, col, _ in states], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], False, (0, 0, 255), int(display.get("path_line_width", 3)), cv2.LINE_AA)
    start = states[0]
    goal = states[-1]
    cv2.circle(overlay, (start[1], start[0]), 6, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(overlay, (goal[1], goal[0]), 7, (255, 0, 0), -1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overlay)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Run region-goal, pose-aware A* for inspection navigation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--equipment", help="Equipment index or exact equipment name")
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--start-yaw-deg", type=float)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    paths = config["paths"]
    planning_map_path = resolve_project_path(paths["planning_map"])
    planning_metadata_path = resolve_project_path(paths["planning_metadata"])
    regions_path = resolve_project_path(paths["goal_regions"])
    regions_metadata_path = resolve_project_path(paths["goal_regions_metadata"])
    output_dir = resolve_project_path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    planning_metadata = load_json(planning_metadata_path)
    regions_metadata = load_json(regions_metadata_path)
    grid = GridSpec.from_dict(planning_metadata["grid"])
    target = choose_equipment(regions_metadata["equipment"], args.equipment)
    equipment_index = int(target["equipment_index"])
    region_data = np.load(regions_path)
    pose_free = unpack_pose_free_masks(region_data["pose_free_packed"], grid.width)
    heading_bins = int(pose_free.shape[0])
    yaw_deg = float(args.start_yaw_deg if args.start_yaw_deg is not None else config.get("start", {}).get("yaw_deg", 0.0))
    start_heading = quantize_heading(math.radians(yaw_deg), heading_bins)

    selected = region_data["goal_equipment_index"] == equipment_index
    rows = region_data["goal_rows"][selected].astype(np.int32)
    cols = region_data["goal_cols"][selected].astype(np.int32)
    headings = region_data["goal_heading_bins"][selected].astype(np.int32)
    tilt_costs = region_data["goal_tilt_costs"][selected]
    if len(rows) == 0:
        raise SystemExit(f"Equipment {target['equipment_name']} has no camera-feasible goal poses.")
    planner_config = PoseAStarConfig(**config.get("astar", {}))
    terminal_costs = (planner_config.tilt_cost_weight * tilt_costs).astype(np.float32)
    maximum_terminal_cost = planner_config.tilt_cost_weight
    map_data = np.load(planning_map_path)
    ortho_base = load_aligned_ortho_thumbnail(
        resolve_project_path(paths["ortho_image"]),
        resolve_project_path(paths["ortho_metadata"]),
        resolve_project_path(paths["ortho_thumbnail_2k"]),
        grid,
        max_resolution=int(config.get("display", {}).get("ortho_max_resolution", 2048)),
    )
    base_image = make_target_cost_image(
        ortho_base,
        map_data["equipment_index_mask"],
        equipment_index,
        rows,
        cols,
        terminal_costs,
        maximum_terminal_cost,
        config.get("display", {}),
    )
    if (args.start_x is None) != (args.start_y is None):
        raise SystemExit("--start-x and --start-y must be provided together.")
    if args.start_x is not None:
        col, row = grid.xy_to_grid(np.asarray([[args.start_x, args.start_y]], dtype=np.float64))[0]
        start_rc = (int(row), int(col))
    else:
        start_rc = choose_start_cell(base_image, pose_free[start_heading], config.get("display", {}))
    start_state = (start_rc[0], start_rc[1], start_heading)

    goal_cost_map: dict[tuple[int, int, int], float] = {}
    candidate_lookup: dict[tuple[int, int, int], int] = {}
    for index, (row, col, heading, terminal_cost) in enumerate(
        zip(rows, cols, headings, terminal_costs, strict=True)
    ):
        state = (int(row), int(col), int(heading))
        if state not in goal_cost_map or float(terminal_cost) < goal_cost_map[state]:
            goal_cost_map[state] = float(terminal_cost)
            candidate_lookup[state] = index

    planning_cost_map = map_data["pose_cost_map"] if "pose_cost_map" in map_data.files else map_data["cost_map"]
    print("\n[区域目标位姿 A*]")
    print(f"  目标设备：{target['equipment_name']}")
    print(f"  候选终点位姿：{len(goal_cost_map):,}")
    print(f"  起点状态：{start_state}，航向 {start_heading * 360.0 / heading_bins:.1f}°")
    print("  阶段1：二维区域目标 A* 生成全局路径走廊")
    position_free = np.any(pose_free > 0, axis=0).astype(np.uint8)
    position_terminal_costs: dict[tuple[int, int], float] = {}
    for state, terminal_cost in goal_cost_map.items():
        position = (state[0], state[1])
        position_terminal_costs[position] = min(
            position_terminal_costs.get(position, math.inf), float(terminal_cost)
        )
    coarse_path = region_astar_path(
        position_free,
        planning_cost_map,
        start_rc,
        {(row, col) for row, col, _ in goal_cost_map},
        cost_weight=planner_config.cost_weight,
        resolution_m=grid.resolution_m,
        goal_terminal_costs=position_terminal_costs,
    )
    if not coarse_path:
        raise SystemExit("二维区域目标 A* 未找到通往目标区域的路径。")
    corridor_radius_m = float(config.get("hierarchical", {}).get("corridor_radius_m", 2.0))
    max_corridor_radius_m = float(config.get("hierarchical", {}).get("max_corridor_radius_m", 8.0))
    result = None
    corridor = None
    corridor_goals = {}
    radius = corridor_radius_m
    while radius <= max_corridor_radius_m + 1.0e-9:
        corridor = path_corridor_mask(
            (grid.height, grid.width),
            coarse_path,
            max(1, int(round(radius / grid.resolution_m))),
        )
        corridor_goals = {state: cost for state, cost in goal_cost_map.items() if corridor[state[0], state[1]] > 0}
        print(f"  阶段2：在 {radius:g} m 走廊内细化机身航向（目标位姿 {len(corridor_goals):,}）")
        result = pose_region_astar_search(
            pose_free,
            planning_cost_map,
            start_state,
            corridor_goals,
            planner_config,
            search_mask=corridor,
            resolution_m=grid.resolution_m,
        )
        if result.found:
            corridor_radius_m = radius
            break
        radius *= 2.0
    assert result is not None and corridor is not None
    if not result.found:
        raise SystemExit("区域目标位姿 A* 未找到路径。")

    path, path_xy = path_to_payload(grid, result.path_states, heading_bins)
    goal_state = result.path_states[-1]
    candidate_index = candidate_lookup[goal_state]
    final_camera = {
        "observation_model": str(target.get("observation_model", "roi_conical_approach")),
        "roi_center_xyz": target.get("observation_center_xyz"),
        "pan_rad": float(region_data["goal_camera_pan_rad"][selected][candidate_index]),
        "pan_deg": float(math.degrees(region_data["goal_camera_pan_rad"][selected][candidate_index])),
        "tilt_rad": float(region_data["goal_camera_tilt_rad"][selected][candidate_index]),
        "tilt_deg": float(math.degrees(region_data["goal_camera_tilt_rad"][selected][candidate_index])),
        "tilt_cost": float(tilt_costs[candidate_index]),
        "terminal_goal_cost": float(terminal_costs[candidate_index]),
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_json = output_dir / f"region_goal_astar_{timestamp}.json"
    result_png = output_dir / f"region_goal_astar_{timestamp}_overlay.png"
    payload = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config.expanduser().resolve()),
        "target_equipment": target,
        "start": path[0],
        "goal": {**path[-1], "camera": final_camera},
        "robot": regions_metadata["robot"],
        "camera": regions_metadata["camera"],
        "astar": config.get("astar", {}),
        "hierarchical": {
            **config.get("hierarchical", {}),
            "used_corridor_radius_m": corridor_radius_m,
            "coarse_path_node_count": len(coarse_path),
            "corridor_cell_count": int(corridor.sum()),
        },
        "path": {
            "states": path,
            "node_count": len(path),
            "length_m": path_length_m(tuple(tuple(point) for point in path_xy)),
            "path_cost": result.path_cost,
            "terminal_goal_cost": result.terminal_goal_cost,
            "total_cost": result.total_cost,
            "expanded_nodes": result.expanded_nodes,
        },
        "outputs": {"result_json": str(result_json), "overlay_png": str(result_png)},
    }
    write_json(result_json, payload)
    rendered = draw_result(base_image, result.path_states, result_png, config.get("display", {}))
    print(f"  路径长度：{payload['path']['length_m']:.3f} m")
    print(f"  终点机身航向：{path[-1]['yaw_deg']:.1f}°")
    print(f"  云台：pan={final_camera['pan_deg']:.1f}°，tilt={final_camera['tilt_deg']:.1f}°")
    print(f"  俯仰软代价：{final_camera['tilt_cost']:.3f}")
    print(f"  终点观测代价：{final_camera['terminal_goal_cost']:.3f}")
    print("  ROI中心视线：无遮挡")
    print(f"  已保存：{result_json}")
    print(f"  已保存：{result_png}")
    if not args.no_display:
        max_w = int(config.get("display", {}).get("max_window_width", 1400))
        max_h = int(config.get("display", {}).get("max_window_height", 1000))
        scale = min(max_w / grid.width, max_h / grid.height, 1.0)
        shown = cv2.resize(rendered, (round(grid.width * scale), round(grid.height * scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow("Region-goal pose A* result", shown)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
