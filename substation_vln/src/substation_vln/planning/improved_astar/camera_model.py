"""Camera parameters used by ROI-conical inspection approach planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def normalized_preference_cost(value: float, lower: float, preferred: float, upper: float) -> float:
    """Piecewise-linear cost: zero at preferred and one at either configured limit."""
    if not lower <= preferred <= upper:
        raise ValueError("preferred value must lie inside [lower, upper]")
    if value <= preferred:
        return float((preferred - value) / max(preferred - lower, 1.0e-6))
    return float((value - preferred) / max(upper - preferred, 1.0e-6))


@dataclass(frozen=True)
class CameraConfig:
    height_m: float = 1.0
    forward_offset_m: float = 0.0
    lateral_offset_m: float = 0.0
    pan_min_deg: float = -180.0
    pan_max_deg: float = 180.0
    tilt_min_deg: float = 20.0
    tilt_max_deg: float = 70.0
    preferred_tilt_deg: float = 45.0
    near_clip_m: float = 0.2

    def __post_init__(self) -> None:
        if not self.pan_min_deg < self.pan_max_deg:
            raise ValueError("pan_min_deg must be less than pan_max_deg")
        if not self.tilt_min_deg <= self.preferred_tilt_deg <= self.tilt_max_deg:
            raise ValueError("preferred_tilt_deg must lie inside the hard tilt range")
        if self.near_clip_m <= 0.0:
            raise ValueError("near_clip_m must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CameraConfig":
        return cls(**values)
