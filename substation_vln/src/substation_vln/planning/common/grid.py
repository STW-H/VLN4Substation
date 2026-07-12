"""Grid metadata and coordinate conversion helpers for planning maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    resolution_m: float
    width: int
    height: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GridSpec":
        return cls(
            min_x=float(payload["min_x"]),
            max_x=float(payload["max_x"]),
            min_y=float(payload["min_y"]),
            max_y=float(payload["max_y"]),
            resolution_m=float(payload["resolution_m"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )

    def xy_to_grid(self, points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float64)
        cols = np.floor((pts[:, 0] - self.min_x) / self.resolution_m).astype(np.int32)
        rows = np.floor((self.max_y - pts[:, 1]) / self.resolution_m).astype(np.int32)
        cols = np.clip(cols, 0, self.width - 1)
        rows = np.clip(rows, 0, self.height - 1)
        return np.stack([cols, rows], axis=1)

    def grid_to_xy(self, cols: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.min_x + (np.asarray(cols, dtype=np.float64) + 0.5) * self.resolution_m
        y = self.max_y - (np.asarray(rows, dtype=np.float64) + 0.5) * self.resolution_m
        return x, y

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "resolution_m": self.resolution_m,
            "width": self.width,
            "height": self.height,
        }
