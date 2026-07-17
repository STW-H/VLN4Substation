"""Validated result of natural-language route parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_MOVEMENT_MODES = {"normal", "fast", "safe"}


@dataclass(frozen=True)
class RoutePlan:
    """Minimal route semantics returned by the language model."""

    raw_instruction: str
    target_point: str
    start_point: str | None = None
    intermediate_points: list[str] = field(default_factory=list)
    movement_mode: str = "normal"
    movement_mode_reason: str = ""
    parser_provider: str = "deepseek"
    parser_model: str = "deepseek-chat"

    @classmethod
    def from_model_response(
        cls,
        payload: dict[str, Any],
        *,
        raw_instruction: str,
        provider: str,
        model: str,
    ) -> "RoutePlan":
        raw_start = payload.get("start_point")
        start_text = str(raw_start).strip() if raw_start is not None else ""
        start_point = None if start_text.casefold() in {"", "random"} else start_text
        target_point = str(payload.get("target_point", "")).strip()
        if not target_point:
            raise ValueError("target_point cannot be empty")

        raw_intermediate = payload.get("intermediate_points", [])
        if not isinstance(raw_intermediate, list):
            raise ValueError("intermediate_points must be a list")
        intermediate_points = [
            str(item).strip() for item in raw_intermediate if str(item).strip()
        ]

        movement_mode = str(payload.get("movement_mode", "normal")).strip().lower()
        if movement_mode not in ALLOWED_MOVEMENT_MODES:
            raise ValueError(f"Unsupported movement_mode: {movement_mode!r}")

        return cls(
            raw_instruction=raw_instruction.strip(),
            start_point=start_point,
            intermediate_points=intermediate_points,
            target_point=target_point,
            movement_mode=movement_mode,
            movement_mode_reason=str(payload.get("movement_mode_reason", "")).strip(),
            parser_provider=provider,
            parser_model=model,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_kind": "natural_language_route_plan",
            "raw_instruction": self.raw_instruction,
            "parser": {"provider": self.parser_provider, "model": self.parser_model},
            "start_point": self.start_point or "random",
            "intermediate_points": self.intermediate_points,
            "target_point": self.target_point,
            "movement_mode": self.movement_mode,
            "movement_mode_reason": self.movement_mode_reason,
        }
