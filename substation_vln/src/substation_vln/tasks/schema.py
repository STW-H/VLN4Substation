"""Validated structures for capture-only substation inspection tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ALLOWED_ACTIONS = {"capture"}


@dataclass(frozen=True)
class InspectionTask:
    task_id: str
    sequence: int
    equipment_name: str
    equipment_type: str
    inspection_part: str
    action: str = "capture"
    image_count: int = 1
    requested_views: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "InspectionTask":
        task = cls(
            task_id=str(data.get("task_id") or f"task_{index:03d}").strip(),
            sequence=int(data.get("sequence", index)),
            equipment_name=str(data.get("equipment_name", "")).strip(),
            equipment_type=str(data.get("equipment_type", "unknown_device")).strip() or "unknown_device",
            inspection_part=str(data.get("inspection_part", "")).strip(),
            action=str(data.get("action", "capture")).strip().lower(),
            image_count=int(data.get("image_count", 1)),
            requested_views=[str(item).strip() for item in data.get("requested_views", []) if str(item).strip()],
            notes=str(data.get("notes", "")).strip(),
        )
        task.validate()
        return task

    def validate(self) -> None:
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if self.sequence <= 0:
            raise ValueError(f"{self.task_id}: sequence must be positive")
        if not self.equipment_name:
            raise ValueError(f"{self.task_id}: equipment_name cannot be empty")
        if not self.inspection_part:
            raise ValueError(f"{self.task_id}: inspection_part cannot be empty")
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(f"{self.task_id}: unsupported action {self.action!r}; only capture is allowed")
        if self.image_count <= 0:
            raise ValueError(f"{self.task_id}: image_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InspectionPlan:
    raw_instruction: str
    tasks: list[InspectionTask]
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
    ) -> "InspectionPlan":
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("Model response must contain a non-empty tasks list")
        tasks = [InspectionTask.from_dict(item, index) for index, item in enumerate(raw_tasks, start=1)]
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id values must be unique")
        sequences = [task.sequence for task in tasks]
        if len(sequences) != len(set(sequences)):
            raise ValueError("task sequence values must be unique")
        return cls(
            raw_instruction=raw_instruction.strip(),
            tasks=sorted(tasks, key=lambda task: task.sequence),
            parser_provider=provider,
            parser_model=model,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_kind": "capture_only_inspection_plan",
            "raw_instruction": self.raw_instruction,
            "parser": {"provider": self.parser_provider, "model": self.parser_model},
            "tasks": [task.to_dict() for task in self.tasks],
        }
