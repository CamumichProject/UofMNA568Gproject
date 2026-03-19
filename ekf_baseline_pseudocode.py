"""
Baseline EKF pseudocode for a 2D USV.

Goal:
- Provide a simple, understandable EKF baseline for comparison with In-EKF.
- State: [px, py, v, psi, omega]
    px, py   : position (m)
    v        : forward speed (m/s)
    psi      : heading (rad)
    omega    : yaw rate (rad/s)

This file is intentionally simple and can be converted into production code.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def process_model(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Motion model:
      px_{k+1} = px + v*cos(psi)*dt
      py_{k+1} = py + v*sin(psi)*dt
      v_{k+1}  = v
      psi_{k+1}= psi + omega*dt
      omega_{k+1}= omega
    """
    px, py, v, psi, omega = x

    x_next = np.array(
        [
            px + v * np.cos(psi) * dt,
            py + v * np.sin(psi) * dt,
            v,
            wrap_angle(psi + omega * dt),
            omega,
        ],
        dtype=float,
    )
    return x_next


def jacobian_F(x: np.ndarray, dt: float) -> np.ndarray:
    """Jacobian of process model w.r.t. state x."""
    _, _, v, psi, _ = x
    F = np.eye(5)

    F[0, 2] = np.cos(psi) * dt
    F[0, 3] = -v * np.sin(psi) * dt
    F[1, 2] = np.sin(psi) * dt
    F[1, 3] = v * np.cos(psi) * dt
    F[3, 4] = dt
    return F


def predict(x: np.ndarray, P: np.ndarray, Q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """EKF prediction step."""
    F = jacobian_F(x, dt)
    x_pred = process_model(x, dt)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def ekf_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    h_fn,
    H: np.ndarray,
    R: np.ndarray,
    angle_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generic EKF measurement update.
    angle_indices: indices in innovation vector to wrap as angles.
    """
    z_pred = h_fn(x)
    y = z - z_pred

    if angle_indices is not None:
        for idx in angle_indices:
            y[idx] = wrap_angle(y[idx])

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x_upd = x + K @ y
    x_upd[3] = wrap_angle(x_upd[3])  # keep heading bounded

    # Joseph form for numerical stability.
    I = np.eye(len(x))
    P_upd = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

    return x_upd, P_upd


def h_gps(x: np.ndarray) -> np.ndarray:
    """GPS measures position [px, py]."""
    return np.array([x[0], x[1]], dtype=float)


def h_heading(x: np.ndarray) -> np.ndarray:
    """Compass/IMU heading measurement [psi]."""
    return np.array([x[3]], dtype=float)


def run_ekf(data_stream: list[dict], dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    data_stream item example:
    {
      "gps": np.array([px, py]) or None,
      "heading": np.array([psi]) or None,
      "truth": np.array([px, py, v, psi, omega]) or None
    }
    """
    # Initial state and covariance.
    x = np.array([0.0, 0.0, 0.5, 0.0, 0.0], dtype=float)
    P = np.diag([2.0, 2.0, 1.0, 0.4, 0.2]) ** 2

    # Tune Q/R with your team data.
    Q = np.diag([0.05, 0.05, 0.20, 0.05, 0.10]) ** 2
    R_gps = np.diag([1.5, 1.5]) ** 2
    R_heading = np.diag([0.10]) ** 2

    # Constant measurement Jacobians for these sensors.
    H_gps = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    H_heading = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=float)

    xs = []
    Ps = []

    for sample in data_stream:
        # 1) Predict
        x, P = predict(x, P, Q, dt)

        # 2) Update with GPS if available
        if sample.get("gps") is not None:
            x, P = ekf_update(
                x=x,
                P=P,
                z=sample["gps"],
                h_fn=h_gps,
                H=H_gps,
                R=R_gps,
                angle_indices=None,
            )

        # 3) Update with heading if available
        if sample.get("heading") is not None:
            x, P = ekf_update(
                x=x,
                P=P,
                z=sample["heading"],
                h_fn=h_heading,
                H=H_heading,
                R=R_heading,
                angle_indices=[0],
            )

        xs.append(x.copy())
        Ps.append(P.copy())

    return np.array(xs), np.array(Ps)


def compute_metrics(est: np.ndarray, truth: np.ndarray) -> dict:
    """Return simple error metrics for report tables."""
    pos_err = np.linalg.norm(est[:, 0:2] - truth[:, 0:2], axis=1)
    heading_err = np.array([wrap_angle(a - b) for a, b in zip(est[:, 3], truth[:, 3])])

    return {
        "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading_err**2))),
        "final_position_drift_m": float(pos_err[-1]),
    }


if __name__ == "__main__":
    # Replace this with simulator/mocked data loader.
    # Keep it simple for week 1: make fake data that includes gps + heading.
    fake_data = []
    dt = 0.1
    n = 200

    # Simple synthetic truth trajectory for smoke test.
    truth = []
    x_true = np.array([0.0, 0.0, 1.0, 0.0, 0.02], dtype=float)
    for _ in range(n):
        x_true = process_model(x_true, dt)
        truth.append(x_true.copy())

        gps_noise = np.random.randn(2) * 1.5
        heading_noise = np.random.randn(1) * 0.10
        fake_data.append(
            {
                "gps": x_true[0:2] + gps_noise,
                "heading": np.array([wrap_angle(x_true[3] + heading_noise[0])]),
                "truth": x_true.copy(),
            }
        )

    truth = np.array(truth)
    est, _ = run_ekf(fake_data, dt=dt)
    metrics = compute_metrics(est, truth)

    print("Baseline EKF smoke-test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
