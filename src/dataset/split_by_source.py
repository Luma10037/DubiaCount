"""Keep every tile from the same original photo inside one dataset split."""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def original_source(filename: str) -> str:
    """Recover the source photo id from a tile or Roboflow export name.

    ``PXL_..._tile_r2_c1_jpg.rf.<hash>.jpg`` and ``PXL_..._tile_r2_c1.jpg``
    both map back to ``PXL_...``.
    """
    stem = Path(filename).stem
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    for suffix in ("_jpg", "_jpeg", "_png", "_webp", "_bmp"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    marker = "_tile_"
    if marker in stem:
        return stem.split(marker)[0]
    return stem


def assign_source_splits(sources_largest_first: list[str]) -> dict[str, str]:
    """Assign each original photo to exactly one split.

    ``sources_largest_first`` must already be ordered by descending image count.
    One photo stays entirely in train. Two photos become train and valid.
    Three or more keep the largest photos for train, the next for valid,
    and the smallest for test.
    """
    sources = list(sources_largest_first)
    if len(sources) <= 1:
        return {source: "train" for source in sources}
    if len(sources) == 2:
        return {sources[0]: "train", sources[1]: "valid"}

    mapping = {source: "train" for source in sources[:-2]}
    mapping[sources[-2]] = "valid"
    mapping[sources[-1]] = "test"
    return mapping


def regroup_dataset(dataset_dir: str | Path = "datasets/dubia") -> dict[str, str]:
    """Move tiles so no original photo appears in more than one split.

    Returns the source-to-split mapping that was applied.
    """
    root = Path(dataset_dir)
    grouped: dict[str, list[tuple[Path, Path | None]]] = defaultdict(list)

    for split in SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            grouped[original_source(image_path.name)].append(
                (image_path, label_path if label_path.exists() else None)
            )

    if not grouped:
        raise FileNotFoundError(f"No images found under {root}")

    ordered = sorted(grouped, key=lambda source: (-len(grouped[source]), source))
    mapping = assign_source_splits(ordered)

    stage = root / "_regroup_stage"
    if stage.exists():
        shutil.rmtree(stage)
    for split in SPLITS:
        (stage / split / "images").mkdir(parents=True)
        (stage / split / "labels").mkdir(parents=True)

    for source, pairs in grouped.items():
        dest_split = mapping[source]
        for image_path, label_path in pairs:
            shutil.move(str(image_path), stage / dest_split / "images" / image_path.name)
            if label_path is not None and label_path.exists():
                shutil.move(str(label_path), stage / dest_split / "labels" / label_path.name)

    for split in SPLITS:
        for kind in ("images", "labels"):
            target = root / split / kind
            target.mkdir(parents=True, exist_ok=True)
            for child in target.iterdir():
                if child.is_file():
                    child.unlink()
            for staged in (stage / split / kind).iterdir():
                shutil.move(str(staged), target / staged.name)

    shutil.rmtree(stage)
    for cache_file in root.rglob("*.cache"):
        cache_file.unlink()

    print("Split by original photo:")
    for source in ordered:
        print(f"  {source} -> {mapping[source]} ({len(grouped[source])} tiles)")
    return mapping


if __name__ == "__main__":
    regroup_dataset()
