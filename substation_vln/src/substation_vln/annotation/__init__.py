"""Annotation data structures and interactive annotation utilities."""

from .annotator import OrthoImageAnnotator
from .inspection_targets_3d import InspectionTarget3DAnnotator, InspectionTargetDefaults, make_target_record
from .schema import (
    CATEGORIES,
    FONT_CANDIDATES,
    LABEL_COLORS_BGR,
    apply_homogeneous,
    available_categories,
    category_already_exists,
    has_planning_boundary,
    make_annotation,
    metadata_path_for_image,
    polygon_area,
    polyline_length,
    rectangle_polygon,
)

__all__ = [
    "CATEGORIES",
    "FONT_CANDIDATES",
    "LABEL_COLORS_BGR",
    "OrthoImageAnnotator",
    "InspectionTarget3DAnnotator",
    "InspectionTargetDefaults",
    "apply_homogeneous",
    "available_categories",
    "category_already_exists",
    "has_planning_boundary",
    "make_annotation",
    "make_target_record",
    "metadata_path_for_image",
    "polygon_area",
    "polyline_length",
    "rectangle_polygon",
]
