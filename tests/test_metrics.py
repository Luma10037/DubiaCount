"""Unit tests for counting metrics (MAE, RMSE, MAPE)."""

import pytest

from src.utils.metrics import compute_counting_metrics


def test_perfect_prediction():
    actual = [10, 20, 30]
    pred = [10, 20, 30]
    res = compute_counting_metrics(actual, pred)
    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
    assert res["mape"] == 0.0


def test_standard_prediction_error():
    actual = [10, 20, 30]
    pred = [12, 18, 33]  # errors: +2, -2, +3
    res = compute_counting_metrics(actual, pred)
    # MAE = (2 + 2 + 3) / 3 = 2.3333
    assert res["mae"] == pytest.approx(2.3333, abs=1e-3)
    # RMSE = sqrt((4 + 4 + 9) / 3) = sqrt(5.6667) = 2.3805
    assert res["rmse"] == pytest.approx(2.3805, abs=1e-3)


def test_mismatched_length():
    with pytest.raises(ValueError):
        compute_counting_metrics([1, 2], [1])


def test_empty_input():
    res = compute_counting_metrics([], [])
    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
