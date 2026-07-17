"""Planning-map configuration, serialization, and visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from substation_vln.paths import PROJECT_ROOT


def load_yaml_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def save_cost_png(path: Path, cost_map: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(cost_map)
    image = np.zeros(cost_map.shape, dtype=np.uint8)
    if np.any(finite):
        values = cost_map[finite]
        lo = float(values.min())
        hi = float(values.max())
        if hi > lo:
            normalized = (cost_map[finite] - lo) / (hi - lo)
        else:
            normalized = np.zeros_like(values, dtype=np.float32)
        image[finite] = np.clip(255.0 * (1.0 - normalized), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def save_overlay_png(path: Path, layers: dict[str, np.ndarray], cost_map: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = layers["boundary_mask"].shape
    overlay = np.full((height, width, 3), 245, dtype=np.uint8)
    overlay[layers["boundary_mask"] > 0] = (235, 235, 235)
    overlay[layers["preferred_road_mask"] > 0] = (255, 190, 80)
    overlay[layers["preferred_path_mask"] > 0] = (255, 60, 220)
    overlay[layers["narrow_space_mask"] > 0] = (180, 60, 180)
    if "equipment_mask" in layers:
        overlay[layers["equipment_mask"] > 0] = (0, 165, 255)
    overlay[layers["free_space_mask"] == 0] = (215, 215, 215)
    overlay[layers["inflated_obstacle_mask"] > 0] = (95, 95, 235)
    overlay[layers["obstacle_mask"] > 0] = (30, 30, 210)

    finite = np.isfinite(cost_map)
    if np.any(finite):
        cost_vis = np.zeros(cost_map.shape, dtype=np.uint8)
        values = cost_map[finite]
        lo = float(values.min())
        hi = float(values.max())
        if hi > lo:
            cost_vis[finite] = np.clip(255.0 * (cost_map[finite] - lo) / (hi - lo), 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(cost_vis, cv2.COLORMAP_VIRIDIS)
            overlay[finite] = cv2.addWeighted(overlay[finite], 0.65, heat[finite], 0.35, 0.0)
    cv2.imwrite(str(path), overlay)
