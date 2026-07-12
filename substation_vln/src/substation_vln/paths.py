"""Project paths used by command-line tools."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "substation_vln"
CONFIGS_DIR = PACKAGE_ROOT / "configs"

RAW_ERFEISHAN_DIR = PACKAGE_ROOT / "data" / "raw" / "220kv_erfeishan"
PROCESSED_ERFEISHAN_DIR = PACKAGE_ROOT / "data" / "processed" / "220kv_erfeishan"
OUTPUTS_ERFEISHAN_DIR = PACKAGE_ROOT / "outputs" / "220kv_erfeishan"
ANNOTATION_OUTPUTS_ERFEISHAN_DIR = OUTPUTS_ERFEISHAN_DIR / "annotation"
PLANNING_OUTPUTS_ERFEISHAN_DIR = OUTPUTS_ERFEISHAN_DIR / "planning"

DEFAULT_AXIS_CORRECTED_POINTCLOUD = (
    PROCESSED_ERFEISHAN_DIR / "pointcloud" / "erfeishan_0.02_resampled_real_coords_axis_corrected.ply"
)
DEFAULT_ZUP_GAUSSIAN = RAW_ERFEISHAN_DIR / "gaussian" / "layer_2_point_cloud.ply"
DEFAULT_GAUSSIAN = DEFAULT_ZUP_GAUSSIAN
DEFAULT_ALIGNED_GAUSSIAN = PROCESSED_ERFEISHAN_DIR / "gaussian" / "layer_2_aligned_to_axis_corrected_pointcloud.ply"
DEFAULT_REGISTRATION = PROCESSED_ERFEISHAN_DIR / "registration" / "gaussian_to_pointcloud_transform.json"

DEFAULT_PROCESSED_POINTCLOUD_DIR = PROCESSED_ERFEISHAN_DIR / "pointcloud"

HABITAT_GS_ROOT = PROJECT_ROOT / "external" / "habitat-gs"
HABITAT_GS_VIEWER = HABITAT_GS_ROOT / "examples" / "gaussian_viewer.py"
