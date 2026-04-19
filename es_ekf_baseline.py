"""
Error-State Extended Kalman Filter (ES-EKF) baseline for USV state estimation.

Structure:
    - Nominal state x_nom (9-D) is integrated with raw IMU acceleration input.
    - Error state dx (9-D) is what the filter actually estimates:
        dx = [dp(3), dv(3), dtheta(3)]
      where dtheta is a small-angle rotation correction.
    - At each measurement the filter solves for dx, injects it into x_nom,
      then resets dx to zero. The covariance is reset accordingly.

This is the "strong" traditional baseline in our comparison against In-EKF.

Conventions match ekf_baseline.py so run_comparison can swap filters trivially.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from load_data import _euler_zyx_to_rotation, _wrap_euler


STATE_DIM = 9
ERROR_DIM = 9
MEAS_DIM = 6


@dataclass
class ESEKFConfig:
    """Noise parameters for the error-state filter."""

    sigma_accel: float = 0.2
    sigma_gyro: float = 0.05

    sigma_body_linvel: float = 0.10
    sigma_body_angvel: float = 0.02

    init_sigma_pos: float = 0.5
    init_sigma_vel: float = 0.2
    init_sigma_theta: float = 0.05


@dataclass
class ESEKFState:
    """Nominal state + error covariance. Error itself is kept zero after reset."""

    x_nom: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM))
    P: np.ndarray = field(default_factory=lambda: np.eye(ERROR_DIM))


def make_initial_state(cfg: ESEKFConfig) -> ESEKFState:
    P0 = np.diag(
        [cfg.init_sigma_pos] * 3
        + [cfg.init_sigma_vel] * 3
        + [cfg.init_sigma_theta] * 3
    ) ** 2
    return ESEKFState(x_nom=np.zeros(STATE_DIM), P=P0)


def build_process_noise(cfg: ESEKFConfig, dt: float) -> np.ndarray:
    """
    Error-state process noise. Scales with dt because noise accumulates during
    integration of the IMU inputs.
    """
    q_vel = (cfg.sigma_accel * dt) ** 2
    q_pos = (0.5 * cfg.sigma_accel * dt * dt) ** 2
    q_theta = (cfg.sigma_gyro * dt) ** 2

    return np.diag([q_pos] * 3 + [q_vel] * 3 + [q_theta] * 3)


def build_measurement_noise(cfg: ESEKFConfig) -> np.ndarray:
    return np.diag(
        [cfg.sigma_body_linvel] * 3 + [cfg.sigma_body_angvel] * 3
    ) ** 2


def predict(state: ESEKFState, u: np.ndarray, dt: float, cfg: ESEKFConfig) -> ESEKFState:
    """
    Propagate nominal with raw IMU; propagate error-state covariance separately.

    Nominal dynamics (Euler integration):
        p_nom_dot = v_nom
        v_nom_dot = R(theta_nom) @ a_body
        theta_nom_dot = omega_body  (approximated as small-angle update)

    Error dynamics linearized about the nominal (standard strapdown ES-EKF form):
        dp_dot     = dv
        dv_dot     = -R @ [a_body]_x @ dtheta   (tilt-induced specific-force error)
        dtheta_dot = -[omega_body]_x @ dtheta   (small-angle rate coupling)
    """
    x_nom = state.x_nom
    p = x_nom[0:3]
    v = x_nom[3:6]
    roll, pitch, yaw = x_nom[6:9]

    a_body = u[0:3]
    omega_body = u[3:6]

    R_bw = _euler_zyx_to_rotation(roll, pitch, yaw)
    a_world = R_bw @ a_body

    p_new = p + v * dt + 0.5 * a_world * dt * dt
    v_new = v + a_world * dt
    eul_new = _wrap_euler(np.array([roll, pitch, yaw]) + omega_body * dt)

    x_nom_new = np.concatenate([p_new, v_new, eul_new])

    F = _error_jacobian(R_bw, a_body, omega_body, dt)
    Q = build_process_noise(cfg, dt)
    P_new = F @ state.P @ F.T + Q

    return ESEKFState(x_nom=x_nom_new, P=P_new)


def update(state: ESEKFState, z: np.ndarray, cfg: ESEKFConfig) -> ESEKFState:
    """
    Compute the error-state correction from the measurement, then inject it
    into the nominal state and reset the error to zero.
    """
    x_nom = state.x_nom
    p = x_nom[0:3]
    v = x_nom[3:6]
    roll, pitch, yaw = x_nom[6:9]

    R_bw = _euler_zyx_to_rotation(roll, pitch, yaw)
    R_wb = R_bw.T

    z_pred = np.zeros(MEAS_DIM)
    z_pred[0:3] = R_wb @ v
    z_pred[3:6] = np.zeros(3)  # no angular-rate state; treat as zero-mean prior

    H = np.zeros((MEAS_DIM, ERROR_DIM))
    H[0:3, 3:6] = R_wb
    H[0:3, 6:9] = _skew(R_wb @ v)

    R = build_measurement_noise(cfg)
    innovation = z - z_pred
    innovation[3:6] = _wrap_euler(innovation[3:6])

    S = H @ state.P @ H.T + R
    K = state.P @ H.T @ np.linalg.inv(S)
    dx = K @ innovation

    x_nom_new = _inject(x_nom, dx)

    I = np.eye(ERROR_DIM)
    P_new = (I - K @ H) @ state.P @ (I - K @ H).T + K @ R @ K.T

    # Error is immediately reset to zero because it was folded into x_nom.
    return ESEKFState(x_nom=x_nom_new, P=P_new)


def run(
    traj_time: np.ndarray,
    inputs: np.ndarray,
    measurements: np.ndarray,
    dt: float,
    cfg: ESEKFConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the ES-EKF over a full scenario; same signature as ekf_baseline.run."""
    cfg = cfg or ESEKFConfig()
    state = make_initial_state(cfg)

    n = len(traj_time)
    xs = np.zeros((n, STATE_DIM))
    Ps = np.zeros((n, ERROR_DIM, ERROR_DIM))

    for k in range(n):
        state = predict(state, inputs[k], dt, cfg)
        state = update(state, measurements[k], cfg)
        xs[k] = state.x_nom
        Ps[k] = state.P

    return xs, Ps


def _inject(x_nom: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """
    Fold the error correction into the nominal state.

    - Position / velocity: additive.
    - Orientation: compose the small rotation dtheta with current Euler angles.
      We convert dtheta -> small rotation matrix, multiply R_nom @ R_delta,
      then extract ZYX Euler angles back out.
    """
    p_new = x_nom[0:3] + dx[0:3]
    v_new = x_nom[3:6] + dx[3:6]

    roll, pitch, yaw = x_nom[6:9]
    R_nom = _euler_zyx_to_rotation(roll, pitch, yaw)
    R_delta = _small_rotation(dx[6:9])
    R_new = R_nom @ R_delta
    eul_new = _wrap_euler(_rotation_to_euler_zyx(R_new))

    return np.concatenate([p_new, v_new, eul_new])


def _error_jacobian(
    R_bw: np.ndarray, a_body: np.ndarray, omega_body: np.ndarray, dt: float
) -> np.ndarray:
    """Discrete-time linearized dynamics of the error state."""
    F = np.eye(ERROR_DIM)

    F[0:3, 3:6] = np.eye(3) * dt
    F[3:6, 6:9] = -R_bw @ _skew(a_body) * dt
    F[6:9, 6:9] = np.eye(3) - _skew(omega_body) * dt

    return F


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix of a 3-vector."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _small_rotation(theta: np.ndarray) -> np.ndarray:
    """
    Rodrigues' formula for a rotation vector; safe when theta is small.
    Reduces to I + [theta]_x for tiny angles (which is what ES-EKF assumes).
    """
    angle = float(np.linalg.norm(theta))
    if angle < 1e-12:
        return np.eye(3) + _skew(theta)

    axis = theta / angle
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotation_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """Extract (roll, pitch, yaw) from a rotation matrix in ZYX convention."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))

    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal-lock fallback; rare for USVs but safe to handle.
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return np.array([roll, pitch, yaw])
