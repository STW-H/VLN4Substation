"""Remote DeepSeek parser for operation tickets and natural-language inspection instructions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
import unicodedata
from typing import Any

from .schema import RoutePlan


SYSTEM_PROMPT = """你是变电站机器人路线任务解析器。请结合用户指令和标注语义目录，提取机器人起点、按顺序经过的点、终点设备和运动模式。

输出一个JSON对象：
{"start_point":"起点名称或random","intermediate_points":["经过点1","经过点2"],"target_point":"终点设备名称","movement_mode":"normal或fast或safe","movement_mode_reason":"简短理由"}

解析说明：
1. 起点、经过点和终点使用标注语义目录中的规范名称；
2. 用户未指定起点时，start_point填写random；未指定经过点时，intermediate_points填写空列表；
3. 保持经过点的原始先后顺序；
4. movement_mode结合整句语义判断，并在movement_mode_reason中简要说明依据。

运动模式参考示例：
- 日常巡视、环境和时间要求一般时，通常可选择normal；
- 用户强调尽快、紧急或效率时，可以考虑fast；
- 天气、道路、现场条件存在风险，或用户强调谨慎与安全时，可以考虑safe；
- 当指令包含多种倾向时，请结合完整上下文自行选择最合适的模式。
- 例如“使用正常模式巡视，今天天气不好，注意安全”同时包含显式模式与风险语义；请权衡两者，如果认为风险更重要，可以选择safe并在理由中说明。
"""


def normalize_catalog_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def match_catalog_name(value: str, entries: list[tuple[str, list[str]]]) -> str:
    """Resolve exact/index/format variants to one canonical annotated name."""
    raw = str(value).strip()
    exact = [canonical for canonical, aliases in entries if raw == canonical or raw in aliases]
    if len(set(exact)) == 1:
        return exact[0]
    key = normalize_catalog_name(raw)
    normalized_matches = [
        canonical
        for canonical, aliases in entries
        if key and key in {normalize_catalog_name(canonical), *(normalize_catalog_name(alias) for alias in aliases)}
    ]
    if len(set(normalized_matches)) == 1:
        return normalized_matches[0]
    return raw


def canonicalize_catalog_references(
    payload: dict[str, Any],
    available_start_points: list[dict[str, Any]],
    available_equipment: list[dict[str, Any]],
) -> dict[str, Any]:
    start_entries = [
        (
            str(item["start_point_name"]),
            [str(item.get("start_point_index", ""))],
        )
        for item in available_start_points
    ]
    equipment_entries = [
        (
            str(item["equipment_name"]),
            [str(item.get("equipment_index", ""))],
        )
        for item in available_equipment
    ]
    if payload.get("start_point") is not None:
        raw_start = str(payload["start_point"])
        if raw_start.casefold() != "random":
            payload["start_point"] = match_catalog_name(raw_start, start_entries)
    if "target_point" in payload:
        payload["target_point"] = match_catalog_name(
            str(payload["target_point"]), equipment_entries
        )
    if "intermediate_points" in payload:
        payload["intermediate_points"] = [
            match_catalog_name(str(name), start_entries)
            for name in payload.get("intermediate_points", [])
        ]
    return payload


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("DeepSeek response does not contain a JSON object") from None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"DeepSeek returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek response root must be a JSON object")
    return payload


class DeepSeekRouteParser:
    def __init__(
        self,
        *,
        api_key_env: str = "DEEPSEEK_API_KEY",
        api_key_file: str | Path | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key_env = api_key_env
        self.api_key_file = Path(api_key_file).expanduser().resolve() if api_key_file else None
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    def parse(
        self,
        instruction: str,
        *,
        available_start_points: list[dict[str, Any]] | None = None,
        available_equipment: list[dict[str, Any]] | None = None,
    ) -> RoutePlan:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Inspection instruction cannot be empty")
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key and self.api_key_file is not None and self.api_key_file.exists():
            api_key = self.api_key_file.read_text(encoding="utf-8").strip()
        if not api_key:
            file_hint = f" or configure api_key_file={self.api_key_file}" if self.api_key_file else ""
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set. "
                f"Run: export {self.api_key_env}='your-api-key'{file_hint}"
            )
        catalog = {
            "robot_start_points": available_start_points or [],
            "inspection_equipment": available_equipment or [],
        }
        user_content = (
            f"标注语义目录：{json.dumps(catalog, ensure_ascii=False)}\n"
            f"用户指令：{instruction}"
        )
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek API connection failed: {exc.reason}") from exc
        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek API response has no assistant message content") from exc
        parsed = extract_json_object(str(content))
        parsed = canonicalize_catalog_references(
            parsed,
            available_start_points or [],
            available_equipment or [],
        )
        return RoutePlan.from_model_response(
            parsed,
            raw_instruction=instruction,
            provider="deepseek",
            model=self.model,
        )
