"""Metrics and evaluation utilities for counting tasks."""

from collections.abc import Sequence

import numpy as np


def compute_counting_metrics(
    actual: Sequence[float | int],
    predicted: Sequence[float | int],
) -> dict[str, float]:
    """Calculate crowd counting evaluation metrics: MAE, RMSE, and MAPE.

    Args:
        actual: Ground truth counts per sample.
        predicted: Predicted counts per sample.

    Returns:
        Dictionary containing 'mae', 'rmse', and 'mape'.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("Ground truth and prediction lengths must match.")

    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    # Safe MAPE calculation to avoid division by zero
    non_zero = y_true != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs(errors[non_zero] / y_true[non_zero])) * 100.0)
    else:
        mape = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
    }
