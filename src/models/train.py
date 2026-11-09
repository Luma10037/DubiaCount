"""Training pipeline entrypoint for Dubia Roach YOLO models."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def train(
    config_path: str = "configs/train_config.yaml",
    data_path: str = "configs/data.yaml",
):
    """Run model training using configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.pop("model", "yolov8n.pt")
    model = YOLO(model_name)

    print(f"Starting training with model: {model_name}")
    print(f"Hyperparameters: {cfg}")

    results = model.train(data=data_path, **cfg)
    print("Training completed.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dubia Roach Detection Model")
    parser.add_argument(
        "--config", default="configs/train_config.yaml", help="Path to training config"
    )
    parser.add_argument("--data", default="configs/data.yaml", help="Path to data config")
    args = parser.parse_args()

    train(config_path=args.config, data_path=args.data)
