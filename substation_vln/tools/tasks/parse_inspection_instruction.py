#!/usr/bin/env python3
"""Parse natural-language route semantics using the DeepSeek API."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.config import load_yaml_config  # noqa: E402
from substation_vln.paths import CONFIGS_DIR, OUTPUTS_ERFEISHAN_DIR  # noqa: E402
from substation_vln.planning.common.io import (  # noqa: E402
    read_json,
    resolve_project_path,
    write_json,
)
from substation_vln.tasks import (  # noqa: E402
    DeepSeekRouteParser,
    build_semantic_catalog,
    validate_catalog_references,
)


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "tasks" / "parse_inspection_instruction.yaml"


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUTS_ERFEISHAN_DIR / "tasks" / f"inspection_plan_{timestamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse navigation and capture instructions with DeepSeek.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    instruction = str(config.get("instruction", "")).strip()
    if not instruction:
        raise SystemExit("config.instruction 不能为空。")
    deepseek = config.get("deepseek", {})
    catalog_config = config.get("catalog", {})
    starts_path = resolve_project_path(catalog_config["robot_start_points"])
    equipment_path = resolve_project_path(catalog_config["goal_regions_metadata"])
    if not starts_path.exists() or not equipment_path.exists():
        raise SystemExit(
            "任务目录不存在，请先运行 build_planning_map.py 和 build_inspection_goal_regions.py。"
        )
    starts_payload = read_json(starts_path)
    equipment_payload = read_json(equipment_path)["equipment"]
    semantic_catalog = build_semantic_catalog(starts_payload, equipment_payload)
    available_starts = semantic_catalog["robot_start_points"]
    available_equipment = semantic_catalog["inspection_equipment"]
    semantic_catalog_path = resolve_project_path(catalog_config["semantic_catalog"])
    write_json(semantic_catalog_path, semantic_catalog)
    task_parser = DeepSeekRouteParser(
        api_key_env=str(deepseek.get("api_key_env", "DEEPSEEK_API_KEY")),
        api_key_file=resolve_project_path(deepseek.get("api_key_file")) if deepseek.get("api_key_file") else None,
        base_url=str(deepseek.get("base_url", "https://api.deepseek.com")),
        model=str(deepseek.get("model", "deepseek-chat")),
        timeout_s=float(deepseek.get("timeout_s", 60.0)),
    )
    try:
        plan = task_parser.parse(
            instruction,
            available_start_points=available_starts,
            available_equipment=available_equipment,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"任务解析失败：{exc}") from exc
    try:
        validate_catalog_references(plan, semantic_catalog)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output = args.output or default_output_path()
    output = output.expanduser().resolve()
    write_json(output, plan.to_dict())
    print("解析完成")
    print(f"  起点：{plan.start_point or '随机'}")
    print(f"  经过点：{plan.intermediate_points or '无'}")
    print(f"  终点：{plan.target_point}")
    print(f"  运动模式：{plan.movement_mode}")
    if plan.movement_mode_reason:
        print(f"  模式理由：{plan.movement_mode_reason}")
    print(f"输出：{output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
