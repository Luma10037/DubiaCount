"""Image tiling utility for slicing high-resolution dense images into manageable patches."""

import argparse
from pathlib import Path

from PIL import Image


def slice_image(
    image_path: str | Path,
    output_dir: str | Path,
    tile_width: int = 640,
    tile_height: int = 640,
    overlap_ratio: float = 0.15,
) -> list[Path]:
    """Slice a single large image into smaller tiles with overlapping boundaries.

    Args:
        image_path: Path to the source image.
        output_dir: Directory where sliced tiles will be saved.
        tile_width: Width of each output tile.
        tile_height: Height of each output tile.
        overlap_ratio: Overlap percentage between adjacent tiles (0.0 ~ 0.5).

    Returns:
        List of Path objects for the saved tile images.
    """
    img_p = Path(image_path)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    with Image.open(img_p) as img:
        img_w, img_h = img.size

        # If image is smaller than tile dimensions, copy or save as-is
        if img_w <= tile_width and img_h <= tile_height:
            dest = out_p / f"{img_p.stem}_tile_0_0{img_p.suffix}"
            img.save(dest)
            return [dest]

        stride_x = max(1, int(tile_width * (1.0 - overlap_ratio)))
        stride_y = max(1, int(tile_height * (1.0 - overlap_ratio)))

        x_starts = list(range(0, max(1, img_w - tile_width + 1), stride_x))
        if x_starts[-1] + tile_width < img_w:
            x_starts.append(img_w - tile_width)

        y_starts = list(range(0, max(1, img_h - tile_height + 1), stride_y))
        if y_starts[-1] + tile_height < img_h:
            y_starts.append(img_h - tile_height)

        # Remove duplicates while preserving order
        x_starts = sorted(set(x_starts))
        y_starts = sorted(set(y_starts))

        saved_tiles: list[Path] = []
        for row_idx, y in enumerate(y_starts):
            for col_idx, x in enumerate(x_starts):
                box = (x, y, x + tile_width, y + tile_height)
                tile = img.crop(box)
                tile_filename = f"{img_p.stem}_tile_r{row_idx}_c{col_idx}.jpg"
                tile_dest = out_p / tile_filename
                tile.save(tile_dest, quality=95)
                saved_tiles.append(tile_dest)

    return saved_tiles


def slice_directory(
    input_dir: str | Path,
    output_dir: str | Path = "datasets/tiles",
    tile_size: int = 640,
    overlap_ratio: float = 0.15,
) -> list[Path]:
    """Slice all images in input directory into tiles."""
    in_p = Path(input_dir)
    out_p = Path(output_dir)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in in_p.glob("*") if f.suffix.lower() in valid_exts]

    if not images:
        print(f"No valid images found in {in_p}")
        return []

    total_tiles: list[Path] = []
    print(f"Found {len(images)} images in {in_p}. Starting slicing...")

    for img_path in images:
        tiles = slice_image(
            image_path=img_path,
            output_dir=out_p,
            tile_width=tile_size,
            tile_height=tile_size,
            overlap_ratio=overlap_ratio,
        )
        print(f"  -> {img_path.name}: generated {len(tiles)} tiles")
        total_tiles.extend(tiles)

    print(f"[COMPLETED] Total {len(total_tiles)} tiles saved to: {out_p.resolve()}")
    return total_tiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slice high-resolution images into smaller tiles for easy annotation."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a single image or a directory containing images",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="datasets/tiles",
        help="Directory to save generated tiles (default: datasets/tiles)",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=640,
        help="Width and height of square tiles in pixels (default: 640)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.15,
        help="Overlap ratio between adjacent tiles (default: 0.15 = 15%%)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        saved = slice_image(
            image_path=input_path,
            output_dir=args.output,
            tile_width=args.size,
            tile_height=args.size,
            overlap_ratio=args.overlap,
        )
        print(f"Generated {len(saved)} tiles in {args.output}")
    elif input_path.is_dir():
        slice_directory(
            input_dir=input_path,
            output_dir=args.output,
            tile_size=args.size,
            overlap_ratio=args.overlap,
        )
    else:
        print(f"Error: Input path '{args.input}' does not exist.")
