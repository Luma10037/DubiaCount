"""Evaluation module for mAP and Counting Metrics (MAE/RMSE)."""

from ultralytics import YOLO


def evaluate_model(
    model_path: str,
    data_path: str = "configs/data.yaml",
    split: str = "val",
):
    """Evaluate YOLO model on validation/test set and compute detection metrics."""
    model = YOLO(model_path)
    metrics = model.val(data=data_path, split=split)

    mAP50 = metrics.box.map50
    mAP50_95 = metrics.box.map

    print(f"--- Evaluation Results ({split}) ---")
    print(f"mAP@0.50:      {mAP50:.4f}")
    print(f"mAP@0.50:0.95: {mAP50_95:.4f}")

    return {
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "fitness": metrics.fitness,
    }
