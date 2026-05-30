from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DEFAULT_CANONICAL_LABELS = ("player", "goalkeeper", "referee", "ball")


class ClassMappingError(ValueError):
    """Raised when detector labels cannot be safely mapped to a manifest."""


@dataclass(frozen=True)
class ClassMapping:
    """Maps detector labels to TurboVision manifest cls_id values."""

    manifest_objects: tuple[str, ...]
    detector_to_manifest: Mapping[str, str]

    @classmethod
    def from_manifest(
        cls,
        manifest_objects: list[str] | tuple[str, ...],
        detector_to_manifest: Mapping[str, str] | None = None,
    ) -> ClassMapping:
        objects = tuple(_normalize_label(name) for name in manifest_objects)
        if not objects:
            raise ClassMappingError("Manifest has no object labels.")

        mapping = detector_to_manifest or _identity_mapping(objects)
        normalized = {
            _normalize_label(detector_label): _normalize_label(manifest_label)
            for detector_label, manifest_label in mapping.items()
        }
        result = cls(manifest_objects=objects, detector_to_manifest=normalized)
        result.validate()
        return result

    def validate(self) -> None:
        missing_targets = sorted(
            {
                manifest_label
                for manifest_label in self.detector_to_manifest.values()
                if manifest_label not in self.manifest_objects
            }
        )
        if missing_targets:
            raise ClassMappingError(
                "Detector mapping targets labels missing from manifest: "
                + ", ".join(missing_targets)
            )

    def cls_id_for_detector_label(self, detector_label: str) -> int | None:
        manifest_label = self.detector_to_manifest.get(_normalize_label(detector_label))
        if manifest_label is None:
            return None
        try:
            return self.manifest_objects.index(manifest_label)
        except ValueError as exc:
            raise ClassMappingError(
                f"Mapped manifest label '{manifest_label}' is not present."
            ) from exc

    def manifest_label_for_cls_id(self, cls_id: int) -> str:
        if cls_id < 0 or cls_id >= len(self.manifest_objects):
            raise ClassMappingError(
                f"cls_id={cls_id} outside manifest range 0..{len(self.manifest_objects) - 1}"
            )
        return self.manifest_objects[cls_id]


def _identity_mapping(objects: tuple[str, ...]) -> dict[str, str]:
    canonical = {label: label for label in objects}
    for label in DEFAULT_CANONICAL_LABELS:
        if label in objects:
            canonical[label] = label
    return canonical


def _normalize_label(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")
