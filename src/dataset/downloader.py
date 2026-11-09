"""Dataset downloader and preparation helpers."""

from pathlib import Path


def prepare_dataset_structure(base_dir: str | Path = "datasets/dubia") -> dict[str, Path]:
    """Create the standard YOLO dataset folder hierarchy."""
    base_path = Path(base_dir)
    subdirs = {
        "train_images": base_path / "images" / "train",
        "val_images": base_path / "images" / "val",
        "test_images": base_path / "images" / "test",
        "train_labels": base_path / "labels" / "train",
        "val_labels": base_path / "labels" / "val",
        "test_labels": base_path / "labels" / "test",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


if __name__ == "__main__":
    dirs = prepare_dataset_structure()
    print(f"Dataset directories ready: {dirs['train_images'].parent.parent}")
