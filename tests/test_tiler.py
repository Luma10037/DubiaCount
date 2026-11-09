"""Unit tests for image slicing utility."""

from pathlib import Path

from PIL import Image

from src.dataset.tiler import slice_image


def test_slice_image(tmp_path: Path):
    # Create a dummy image of 1200x800
    test_img = Image.new("RGB", (1200, 800), color=(100, 150, 200))
    img_path = tmp_path / "test_dubia.jpg"
    test_img.save(img_path)

    out_dir = tmp_path / "tiles"
    tiles = slice_image(
        image_path=img_path,
        output_dir=out_dir,
        tile_width=640,
        tile_height=640,
        overlap_ratio=0.2,
    )

    assert len(tiles) > 1
    for tile_p in tiles:
        assert tile_p.exists()
        with Image.open(tile_p) as tile:
            assert tile.size == (640, 640)
