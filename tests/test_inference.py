"""Unit tests for inference and basic configurations."""

from pathlib import Path

import yaml

from src.dataset.augment import AugmentationConfig


def test_train_config_file_valid():
    config_path = Path("configs/train_config.yaml")
    assert config_path.exists(), "configs/train_config.yaml should exist"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "epochs" in cfg
    assert "imgsz" in cfg
    assert "batch" in cfg


def test_augmentation_config():
    aug = AugmentationConfig(degrees=90.0, mosaic=0.8)
    args = aug.to_yolo_args()
    assert args["degrees"] == 90.0
    assert args["mosaic"] == 0.8
    assert args["flipud"] == 0.5
