"""Data augmentation utilities for dense insect object detection."""

from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    degrees: float = 180.0
    flipud: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.15

    def to_yolo_args(self) -> dict[str, float]:
        return {
            "degrees": self.degrees,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
        }
