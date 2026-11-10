"""SAHI (Slicing Aided Hyper Inference) module for high-density small object detection."""

import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions


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
        postprocess_match_threshold: float = 0.5,
        image_size: int = 512,
    ):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.postprocess_match_threshold = postprocess_match_threshold

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            image_size=image_size,
        )

    def predict(self, image_input: str | np.ndarray):
        """Run sliced inference and merge boxes that the same insect produced on two tiles."""
        result = get_sliced_prediction(
            image_input,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=self.postprocess_match_threshold,
            postprocess_class_agnostic=True,
            auto_slice_resolution=False,
            verbose=0,
        )
        count = len(result.object_prediction_list)
        # Labels cover the insects once a tile contains more than a few dozen boxes.
        hide_text = count > 40
        visual = visualize_object_predictions(
            image=np.ascontiguousarray(result.image),
            object_prediction_list=result.object_prediction_list,
            hide_labels=hide_text,
            hide_conf=hide_text,
        )
        return {
            "count": count,
            "prediction": result,
            "visualized_image": visual["image"],
        }
