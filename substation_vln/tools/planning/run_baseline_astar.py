#!/usr/bin/env python3
"""Interactively run baseline A* from a clicked start point to a patrol point."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
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
from substation_vln.planning.astar import AStarConfig, astar_search, path_length_m  # noqa: E402
from substation_vln.planning.common.grid import GridSpec  # noqa: E402
from substation_vln.planning.common.io import resolve_project_path, write_json  # noqa: E402


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "planning" / "run_baseline_astar.yaml"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_to_uint8(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    image = np.zeros(values.shape, dtype=np.uint8)
    valid = valid_mask & np.isfinite(values)
    if np.any(valid):
        data = values[valid]
        lo = float(data.min())
        hi = float(data.max())
        if hi > lo:
            image[valid] = np.clip(255.0 * (data - lo) / (hi - lo), 0, 255).astype(np.uint8)
        else:
            image[valid] = 128
    return image


def make_display_image(layers: dict[str, np.ndarray], cost_map: np.ndarray) -> np.ndarray:
    boundary = layers["boundary_mask"] > 0
    free = layers["free_space_mask"] > 0
    cost_vis = normalize_to_uint8(cost_map, free)
    heat = cv2.applyColorMap(cost_vis, cv2.COLORMAP_VIRIDIS)

    image = np.full((*boundary.shape, 3), 245, dtype=np.uint8)
    image[boundary] = (230, 230, 230)
    image[free] = cv2.addWeighted(image[free], 0.35, heat[free], 0.65, 0.0)
    image[layers["preferred_road_mask"] > 0] = (255, 190, 80)
    image[layers["preferred_path_mask"] > 0] = (255, 60, 220)
    image[layers["narrow_space_mask"] > 0] = (180, 60, 180)
    image[layers["inflated_obstacle_mask"] > 0] = (95, 95, 235)
    image[layers["obstacle_mask"] > 0] = (30, 30, 210)
    return image


def choose_start_cell(image: np.ndarray, free_space_mask: np.ndarray, display_config: dict) -> tuple[int, int]:
    height, width = image.shape[:2]
    max_w = int(display_config.get("max_window_width", 1400))
    max_h = int(display_config.get("max_window_height", 1000))
    scale = min(max_w / width, max_h / height, 1.0)
    display_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    display = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST)
    selected: list[tuple[int, int]] = []
    window_name = str(display_config.get("window_name", "Baseline A* start selection"))

    def on_mouse(event, x, y, flags, userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        col = int(np.clip(round(x / scale), 0, width - 1))
        row = int(np.clip(round(y / scale), 0, height - 1))
        if free_space_mask[row, col] == 0:
            print(f"Selected cell is not free space: row={row}, col={col}. Please click another start point.")
            return
        selected[:] = [(row, col)]
        print(f"Selected start cell: row={row}, col={col}")

    print("\nStart selection")
    print("  Left-click a free-space cell as the A* start point.")
    print("  Press Enter after selecting; press Q or Esc to cancel.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_size[0], display_size[1])
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        canvas = display.copy()
        if selected:
            row, col = selected[0]
            cv2.circle(canvas, (int(round(col * scale)), int(round(row * scale))), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and selected:
            break
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window_name)
            raise SystemExit("Start selection canceled.")
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            raise SystemExit("Start selection window closed.")
    cv2.destroyWindow(window_name)
    return selected[0]


def print_patrol_points(patrol_points: list[dict]) -> None:
    print("\nTarget patrol points")
    for idx, point in enumerate(patrol_points, start=1):
        label = point.get("label") or "unlabeled"
        stop = point["stop_xy"]
        yaw = float(point.get("yaw_deg", 0.0))
        print(f"  {idx:02d}. {label} #{point.get('point_index')} stop=({stop[0]:.3f}, {stop[1]:.3f}) yaw={yaw:.1f} deg")


def choose_goal_patrol_point(patrol_points: list[dict]) -> tuple[int, dict]:
    print_patrol_points(patrol_points)
    while True:
        raw = input("\nSelect target patrol point index: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter an integer index.")
            continue
        if 1 <= idx <= len(patrol_points):
            point = patrol_points[idx - 1]
            label = point.get("label") or "unlabeled"
            confirm = input(f"Use target {idx:02d} ({label} #{point.get('point_index')})? [Y/N]: ").strip().lower()
            if confirm == "y":
                return idx, point
            if confirm == "n":
                continue
            print("Please answer Y or N.")
            continue
        print(f"Index must be in [1, {len(patrol_points)}].")


def path_rc_to_xy(grid: GridSpec, path_rc: list[tuple[int, int]]) -> list[list[float]]:
    rows = np.asarray([p[0] for p in path_rc], dtype=np.int32)
    cols = np.asarray([p[1] for p in path_rc], dtype=np.int32)
    xs, ys = grid.grid_to_xy(cols, rows)
    return [[float(x), float(y)] for x, y in zip(xs, ys, strict=True)]


def draw_path_overlay(
    base_image: np.ndarray,
    path_rc: list[tuple[int, int]],
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    output_path: Path,
    display_config: dict,
) -> np.ndarray:
    overlay = base_image.copy()
    if len(path_rc) >= 2:
        pts = np.asarray([(col, row) for row, col in path_rc], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            overlay,
            [pts],
            isClosed=False,
            color=(0, 0, 255),
            thickness=int(display_config.get("path_line_width", 3)),
            lineType=cv2.LINE_AA,
        )
    radius = int(display_config.get("point_radius", 5))
    cv2.circle(overlay, (start_rc[1], start_rc[0]), radius, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(overlay, (goal_rc[1], goal_rc[0]), radius, (255, 0, 0), -1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return overlay


def show_result(image: np.ndarray, display_config: dict) -> None:
    height, width = image.shape[:2]
    max_w = int(display_config.get("max_window_width", 1400))
    max_h = int(display_config.get("max_window_height", 1000))
    scale = min(max_w / width, max_h / height, 1.0)
    display_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    display = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST)
    window_name = "Baseline A* result"
    print("\nShowing result. Press any key in the image window to close.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_size[0], display_size[1])
    cv2.imshow(window_name, display)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run interactive baseline A* from clicked start to selected patrol point.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML file with default tool arguments")
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    paths = config["paths"]
    planning_map_path = resolve_project_path(paths["planning_map"])
    metadata_path = resolve_project_path(paths["metadata"])
    patrol_points_path = resolve_project_path(paths["patrol_points"])
    output_dir = resolve_project_path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(metadata_path)
    grid = GridSpec.from_dict(metadata["grid"])
    patrol_points = load_json(patrol_points_path)
    data = np.load(planning_map_path)
    layers = {name: data[name] for name in data.files if name.endswith("_mask")}
    cost_map = data["cost_map"]
    free_space_mask = data["free_space_mask"]
    base_image = make_display_image(layers, cost_map)

    start_rc = choose_start_cell(base_image, free_space_mask, config.get("display", {}))
    goal_index, goal_point = choose_goal_patrol_point(patrol_points)
    goal_col, goal_row = grid.xy_to_grid(np.asarray([goal_point["stop_xy"]], dtype=np.float64))[0]
    goal_rc = (int(goal_row), int(goal_col))

    astar_config = AStarConfig(**config.get("astar", {}))
    result = astar_search(free_space_mask, cost_map, start_rc, goal_rc, astar_config)
    if not result.found:
        raise SystemExit("A* did not find a path.")

    path_xy = path_rc_to_xy(grid, result.path_rc)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_json = output_dir / f"baseline_astar_{timestamp}.json"
    overlay_png = output_dir / f"baseline_astar_{timestamp}_overlay.png"

    payload = {
        "config": str(args.config.expanduser().resolve()),
        "planning_map": str(planning_map_path),
        "metadata": str(metadata_path),
        "patrol_points": str(patrol_points_path),
        "start": {
            "grid_rc": [int(start_rc[0]), int(start_rc[1])],
            "xy": path_xy[0],
        },
        "goal": {
            "patrol_point_index": int(goal_index),
            "grid_rc": [int(goal_rc[0]), int(goal_rc[1])],
            "patrol_point": goal_point,
        },
        "astar": config.get("astar", {}),
        "path": {
            "grid_rc": [[int(r), int(c)] for r, c in result.path_rc],
            "xy": path_xy,
            "node_count": len(result.path_rc),
            "length_m": path_length_m((tuple(p) for p in path_xy)),
            "total_cost": result.total_cost,
            "expanded_nodes": result.expanded_nodes,
        },
        "outputs": {
            "result_json": str(result_json),
            "overlay_png": str(overlay_png),
        },
    }
    write_json(result_json, payload)
    overlay = draw_path_overlay(base_image, result.path_rc, start_rc, goal_rc, overlay_png, config.get("display", {}))

    print(f"\nSaved result: {result_json}")
    print(f"Saved overlay: {overlay_png}")
    print(f"Path nodes: {len(result.path_rc)}")
    print(f"Path length: {payload['path']['length_m']:.3f} m")
    print(f"Path cost: {result.total_cost:.3f}")
    print(f"Expanded nodes: {result.expanded_nodes}")
    show_result(overlay, config.get("display", {}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
