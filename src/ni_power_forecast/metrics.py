from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred, reference=None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }
    denom = np.maximum(np.abs(y_true), 10.0)
    out["mape_clipped"] = float(np.mean(np.abs(err) / denom) * 100)

    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        actual_move = y_true - reference
        pred_move = y_pred - reference
        nonzero = np.abs(actual_move) > 1e-9
        out["directional_accuracy"] = float(
            np.mean(np.sign(actual_move[nonzero]) == np.sign(pred_move[nonzero]))
        )
        out["signal_value_proxy"] = float(np.mean(np.sign(pred_move) * actual_move))
    return out
