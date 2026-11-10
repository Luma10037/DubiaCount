"""Tests for grouping tiles by the original photograph."""

from src.dataset.split_by_source import assign_source_splits, original_source


def test_original_source_from_roboflow_tile_name():
    name = "PXL_20261106_131543920_tile_r2_c1_jpg.rf.bf121b400e9e567cf63c5103929aff79.jpg"
    assert original_source(name) == "PXL_20261106_131543920"


def test_original_source_from_plain_tile_name():
    assert original_source("bin_a_tile_r0_c3.jpg") == "bin_a"


def test_two_photos_stay_in_separate_splits():
    mapping = assign_source_splits(["photo_a", "photo_b"])
    assert mapping == {"photo_a": "train", "photo_b": "valid"}


def test_three_photos_reserve_valid_and_test():
    mapping = assign_source_splits(["big", "mid", "small"])
    assert mapping == {"big": "train", "mid": "valid", "small": "test"}


def test_single_photo_stays_in_train():
    assert assign_source_splits(["only"]) == {"only": "train"}
