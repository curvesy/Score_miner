"""Element configuration and class-order guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ElementSpec:
    element_id: str
    slug: str
    objects: tuple[str, ...]
    max_model_size_mb: int | None
    priority: int | None = None
    track: str = "public"

    @property
    def class_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.objects)}

    def class_id(self, class_name: str) -> int:
        key = class_name.strip()
        mapping = self.class_to_id
        if key not in mapping:
            raise ValueError(
                f"class {class_name!r} is not in manifest order for {self.element_id}: "
                f"{list(self.objects)}"
            )
        return mapping[key]

    def assert_objects_match(self, objects: list[str] | tuple[str, ...]) -> None:
        got = tuple(objects)
        if got != self.objects:
            raise ValueError(
                f"object order mismatch for {self.element_id}: expected "
                f"{self.objects}, got {got}"
            )


def _require_str_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def _objects_from_config(data: dict[str, Any]) -> tuple[str, ...]:
    if "objects" in data:
        return _require_str_list(data["objects"], "objects")
    classes = data.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("element config must contain objects list or classes mapping")
    ordered = []
    for key in sorted(classes, key=lambda item: int(item)):
        value = classes[key]
        if not isinstance(value, str):
            raise ValueError("classes values must be strings")
        ordered.append(value)
    return tuple(ordered)


def load_element_spec(path: str | Path) -> ElementSpec:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"invalid element config: {path}")
    return ElementSpec(
        element_id=str(data["element_id"]),
        slug=str(data.get("slug") or Path(path).stem),
        objects=_objects_from_config(data),
        max_model_size_mb=data.get("max_model_size_mb"),
        priority=data.get("priority"),
        track=str(data.get("track") or "public"),
    )
