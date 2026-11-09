"""Dataset downloader, extraction, and validation helpers."""

import zipfile
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


def unpack_dataset_zip(zip_path: str | Path, target_dir: str | Path = "datasets/dubia") -> Path:
    """Extract a downloaded Roboflow YOLOv8 zip directly into the target dataset directory."""
    zip_p = Path(zip_path)
    target_p = Path(target_dir)

    if not zip_p.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_p}")

    target_p.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_p.name} to {target_p}...")
    with zipfile.ZipFile(zip_p, "r") as zip_ref:
        zip_ref.extractall(target_p)

    print("Extraction complete. Running dataset verification...")
    verify_dataset(target_p)
    return target_p


def verify_dataset(dataset_dir: str | Path = "datasets/dubia") -> dict[str, int]:
    """Verify that dataset contains train, val, and test splits with images and labels."""
    base_path = Path(dataset_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    splits = ["train", "val", "test"]
    stats = {}

    for split in splits:
        if (base_path / split / "images").exists():
            img_dir = base_path / split / "images"
            lbl_dir = base_path / split / "labels"
        else:
            img_dir = base_path / "images" / split
            lbl_dir = base_path / "labels" / split

        img_count = (
            sum(1 for f in img_dir.glob("*") if f.suffix.lower() in valid_exts)
            if img_dir.exists()
            else 0
        )
        lbl_count = sum(1 for f in lbl_dir.glob("*.txt")) if lbl_dir.exists() else 0

        stats[f"{split}_images"] = img_count
        stats[f"{split}_labels"] = lbl_count

        print(f"[{split.upper()}] Images: {img_count} | Label files: {lbl_count}")

    total_images = sum(v for k, v in stats.items() if "images" in k)
    if total_images == 0:
        print("[WARNING] No images detected in dataset splits.")
    else:
        print(f"[SUCCESS] Dataset verified with {total_images} total images.")

    return stats


if __name__ == "__main__":
    verify_dataset()
