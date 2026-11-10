"""Evaluation module for mAP and counting metrics (MAE/RMSE)."""

from pathlib import Path

import yaml
from ultralytics import YOLO

from src.utils.metrics import compute_counting_metrics

_SPLIT_KEY = {"train": "train", "val": "val", "valid": "val", "test": "test"}


def _split_dirs(data_path: str, split: str) -> tuple[Path, Path]:
    with open(data_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config.get("path") or Path(data_path).parent)
    image_dir = root / config[_SPLIT_KEY[split]]
    label_dir = image_dir.parent / "labels"
    return image_dir, label_dir


def _ground_truth_count(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def evaluate_counting(
    model: YOLO,
    data_path: str = "configs/data.yaml",
    split: str = "val",
    imgsz: int = 512,
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict[str, float]:
    """Compare predicted insect counts with the number of labeled boxes per image."""
    image_dir, label_dir = _split_dirs(data_path, split)
    images = sorted(
        path
        for path in image_dir.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not images:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "images": 0.0}

    actual = [_ground_truth_count(label_dir / f"{image.stem}.txt") for image in images]
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    predicted = [len(result.boxes) for result in results]

    metrics = compute_counting_metrics(actual, predicted)
    metrics["images"] = float(len(images))
    return metrics


def evaluate_model(
    model_path: str,
    data_path: str = "configs/data.yaml",
    split: str = "val",
    imgsz: int = 512,
    conf: float = 0.25,
    iou: float = 0.45,
):
    """Evaluate detection mAP and per-image counting error."""
    model = YOLO(model_path)
    metrics = model.val(data=data_path, split=split, imgsz=imgsz, conf=conf, iou=iou)
    counting = evaluate_counting(
        model,
        data_path=data_path,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )

    mAP50 = metrics.box.map50
    mAP50_95 = metrics.box.map

    print(f"--- Evaluation Results ({split}) ---")
    print(f"mAP@0.50:      {mAP50:.4f}")
    print(f"mAP@0.50:0.95: {mAP50_95:.4f}")
    print(f"Count MAE:     {counting['mae']:.4f}")
    print(f"Count RMSE:    {counting['rmse']:.4f}")
    print(f"Count MAPE:    {counting['mape']:.2f}%")

    return {
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "fitness": metrics.fitness,
        **counting,
    }
