"""Project paths used by command-line tools."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "substation_vln"

RAW_ERFEISHAN_DIR = PACKAGE_ROOT / "data" / "raw" / "220kv_erfeishan"
PROCESSED_ERFEISHAN_DIR = PACKAGE_ROOT / "data" / "processed" / "220kv_erfeishan"
OUTPUTS_ERFEISHAN_DIR = PACKAGE_ROOT / "outputs" / "220kv_erfeishan"

DEFAULT_POINTCLOUD = PROCESSED_ERFEISHAN_DIR / "pointcloud" / "erfeishan_0.02_resampled_real_coords.ply"
DEFAULT_GAUSSIAN = PROCESSED_ERFEISHAN_DIR / "gaussian_yup" / "layer_2_yup.gs.ply"
DEFAULT_REGISTRATION = PROCESSED_ERFEISHAN_DIR / "registration" / "gaussian_to_pointcloud_transform.json"

DEFAULT_RAW_LAS = RAW_ERFEISHAN_DIR / "pointcloud" / "erfeishan_0.02_resampled.las"
DEFAULT_RAW_GAUSSIAN_DIR = RAW_ERFEISHAN_DIR / "gaussian"
DEFAULT_PROCESSED_POINTCLOUD_DIR = PROCESSED_ERFEISHAN_DIR / "pointcloud"

HABITAT_GS_ROOT = PROJECT_ROOT / "external" / "habitat-gs"
HABITAT_GS_VIEWER = HABITAT_GS_ROOT / "examples" / "gaussian_viewer.py"
