"""Build and validate the annotation names exposed to the route parser."""

from __future__ import annotations

from typing import Any

from .schema import RoutePlan


def build_semantic_catalog(
    start_points: list[dict[str, Any]],
    equipment: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "robot_start_points": [
            {
                "category": "robot_start_point",
                "start_point_index": int(item["start_point_index"]),
                "start_point_name": str(item["start_point_name"]),
            }
            for item in start_points
        ],
        "inspection_equipment": [
            {
                "category": "equipment_region",
                "equipment_index": int(item["equipment_index"]),
                "equipment_name": str(item["equipment_name"]),
                "equipment_type": str(item["equipment_type"]),
            }
            for item in equipment
        ],
    }


def validate_catalog_references(
    plan: RoutePlan,
    catalog: dict[str, list[dict[str, Any]]],
) -> None:
    start_names = {
        item["start_point_name"] for item in catalog["robot_start_points"]
    }
    equipment_names = {
        item["equipment_name"] for item in catalog["inspection_equipment"]
    }
    if plan.start_point is not None and plan.start_point not in start_names:
        raise ValueError(f"起点不在标注目录中：{plan.start_point}")
    if plan.target_point not in equipment_names:
        raise ValueError(f"终点设备不在标注目录中：{plan.target_point}")
    unknown = [name for name in plan.intermediate_points if name not in start_names]
    if unknown:
        raise ValueError(f"经过点不在标注目录中：{unknown}")
