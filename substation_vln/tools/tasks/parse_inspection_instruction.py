#!/usr/bin/env python3
"""Parse an operation ticket or natural-language instruction using the DeepSeek API."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.config import load_yaml_config  # noqa: E402
from substation_vln.paths import CONFIGS_DIR, OUTPUTS_ERFEISHAN_DIR  # noqa: E402
from substation_vln.tasks import DeepSeekInspectionTaskParser  # noqa: E402


DEFAULT_CONFIG = CONFIGS_DIR / "tools" / "tasks" / "parse_inspection_instruction.yaml"


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUTS_ERFEISHAN_DIR / "tasks" / f"inspection_plan_{timestamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a capture-only inspection instruction with DeepSeek.")
    parser.add_argument("instruction", nargs="?", help="Natural-language instruction; omit when --instruction-file is used.")
    parser.add_argument("--instruction-file", type=Path, help="UTF-8 operation-ticket or instruction text file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if bool(args.instruction) == bool(args.instruction_file):
        raise SystemExit("Provide exactly one of positional instruction or --instruction-file.")
    instruction = (
        args.instruction_file.expanduser().read_text(encoding="utf-8")
        if args.instruction_file
        else str(args.instruction)
    )
    deepseek = config.get("deepseek", {})
    task_parser = DeepSeekInspectionTaskParser(
        api_key_env=str(deepseek.get("api_key_env", "DEEPSEEK_API_KEY")),
        base_url=str(deepseek.get("base_url", "https://api.deepseek.com")),
        model=str(deepseek.get("model", "deepseek-chat")),
        timeout_s=float(deepseek.get("timeout_s", 60.0)),
    )
    try:
        plan = task_parser.parse(instruction)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"任务解析失败：{exc}") from exc
    output = args.output or default_output_path()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"解析完成：{len(plan.tasks)} 个拍摄任务")
    for task in plan.tasks:
        print(f"  {task.sequence}. {task.equipment_name} / {task.inspection_part} / 拍摄{task.image_count}张")
    print(f"输出：{output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
