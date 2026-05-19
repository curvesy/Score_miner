from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv


COCO_PERSON_CLASS_ID = 1
COCO_SPORTS_BALL_CLASS_ID = 37


@dataclass(frozen=True)
class DetectionClassIdMapper:
    """Map detector-native class IDs into TurboVision class IDs."""

    class_id_map: dict[int, int]
    drop_unmapped: bool = True

    @classmethod
    def coco_to_turbovision(
        cls,
        *,
        player_cls_id: int = 0,
        ball_cls_id: int | None = None,
    ) -> DetectionClassIdMapper:
        mapping = {COCO_PERSON_CLASS_ID: player_cls_id}
        if ball_cls_id is not None:
            mapping[COCO_SPORTS_BALL_CLASS_ID] = ball_cls_id
        return cls(class_id_map=mapping)

    def apply(self, detections: sv.Detections) -> sv.Detections:
        if detections.class_id is None:
            return detections

        keep_indices: list[int] = []
        mapped_class_ids: list[int] = []
        for idx, class_id in enumerate(detections.class_id.tolist()):
            mapped = self.class_id_map.get(int(class_id))
            if mapped is None:
                if self.drop_unmapped:
                    continue
                mapped = int(class_id)
            keep_indices.append(idx)
            mapped_class_ids.append(mapped)

        if not keep_indices:
            return sv.Detections.empty()

        remapped = detections[np.array(keep_indices, dtype=int)]
        remapped.class_id = np.array(mapped_class_ids, dtype=int)
        return remapped
