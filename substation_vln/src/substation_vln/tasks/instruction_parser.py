"""Remote DeepSeek parser for operation tickets and natural-language inspection instructions."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .schema import InspectionPlan


SYSTEM_PROMPT = """你是变电站机器人巡视任务解析器。把用户给出的操作票或自然语言指令拆分为按顺序执行的拍摄任务。
你的职责只包括解析需要拍摄的设备和部件，不判断设备正常、异常或故障。
只输出一个JSON对象，不要输出Markdown或解释。格式必须为：
{"tasks":[{"task_id":"task_001","sequence":1,"equipment_name":"设备唯一名称","equipment_type":"设备类型或unknown_device","inspection_part":"需要拍摄的部件或区域","action":"capture","image_count":1,"requested_views":[],"notes":""}]}
规则：
1. action只能是capture；
2. 保留原指令规定的先后顺序；
3. 不得虚构指令中不存在的设备编号、部件、拍摄数量或视角；
4. 未明确设备类型时填写unknown_device；
5. 未明确拍摄视角时requested_views为空列表；
6. 一项任务只对应一个设备的一个待拍摄部件。
"""


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


class DeepSeekInspectionTaskParser:
    def __init__(
        self,
        *,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    def parse(self, instruction: str) -> InspectionPlan:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Inspection instruction cannot be empty")
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set. "
                f"Run: export {self.api_key_env}='your-api-key'"
            )
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
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
        return InspectionPlan.from_model_response(
            parsed,
            raw_instruction=instruction,
            provider="deepseek",
            model=self.model,
        )
