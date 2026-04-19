"""
Evaluation metrics for the USV EKF baseline.

All inputs are expected to be aligned on the same time grid.
Orientation comparison uses wrapped-angle differences so the metrics behave
correctly near +/- pi.
"""

from __future__ import annotations

import numpy as np


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def position_rmse(est_pos: np.ndarray, true_pos: np.ndarray) -> float:
    """Root-mean-square error of 3D position over the whole trajectory."""
    err = np.linalg.norm(est_pos - true_pos, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def orientation_rmse(est_eul: np.ndarray, true_eul: np.ndarray) -> dict:
    """Per-axis RMSE of Euler angles (roll, pitch, yaw) with wrapping."""
    diff = _wrap(est_eul - true_eul)
    return {
        "roll_rmse_rad": float(np.sqrt(np.mean(diff[:, 0] ** 2))),
        "pitch_rmse_rad": float(np.sqrt(np.mean(diff[:, 1] ** 2))),
        "yaw_rmse_rad": float(np.sqrt(np.mean(diff[:, 2] ** 2))),
    }


def final_drift(est_pos: np.ndarray, true_pos: np.ndarray) -> float:
    """Position error at the final timestep (meters)."""
    return float(np.linalg.norm(est_pos[-1] - true_pos[-1]))


def error_over_time(est_pos: np.ndarray, true_pos: np.ndarray) -> np.ndarray:
    """Per-timestep position error norm (meters)."""
    return np.linalg.norm(est_pos - true_pos, axis=1)


def summarize(est_state: np.ndarray, true_state: np.ndarray) -> dict:
    """
    Compute the full metric bundle used in the report.

    Expects est/true of shape (N, 6) = [x, y, z, roll, pitch, yaw] OR (N, 9)
    in which case only position (cols 0:3) and Euler (cols 6:9) are used.
    """
    if est_state.shape[1] == 9:
        est_pos = est_state[:, 0:3]
        est_eul = est_state[:, 6:9]
    else:
        est_pos = est_state[:, 0:3]
        est_eul = est_state[:, 3:6]

    if true_state.shape[1] == 9:
        true_pos = true_state[:, 0:3]
        true_eul = true_state[:, 6:9]
    else:
        true_pos = true_state[:, 0:3]
        true_eul = true_state[:, 3:6]

    metrics = {
        "position_rmse_m": position_rmse(est_pos, true_pos),
        "final_drift_m": final_drift(est_pos, true_pos),
    }
    metrics.update(orientation_rmse(est_eul, true_eul))
    return metrics
