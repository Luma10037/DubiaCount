"""SAHI (Slicing Aided Hyper Inference) module for high-density small object detection."""

import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class SahiDubiaDetector:
    """Wrapper around SAHI for sliced inference on dense Dubia roach images."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
    ):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    def predict(self, image_input: str | np.ndarray):
        """Perform sliced inference on an image path or numpy image array."""
        result = get_sliced_prediction(
            image_input,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )
        count = len(result.object_prediction_list)
        return {
            "count": count,
            "prediction": result,
            "visualized_image": result.export_visuals()["image"],
        }
