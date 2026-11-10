"""Data augmentation utilities for dense insect object detection."""

from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    degrees: float = 20.0
    flipud: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 0.5
    mixup: float = 0.0

    def to_yolo_args(self) -> dict[str, float]:
        return {
            "degrees": self.degrees,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
        }
