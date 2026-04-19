"""
Standard 6-DOF Extended Kalman Filter baseline for USV state estimation.

State (9-D):
    x = [px, py, pz, vx, vy, vz, roll, pitch, yaw]
    - position in world frame
    - velocity in world frame
    - Euler angles (ZYX) describing body-to-world rotation

Input (6-D) from IMU-like noisy accelerations (Campbell's CSV):
    u = [ax_b, ay_b, az_b, alpha_roll, alpha_pitch, alpha_yaw]
    - first three are body-frame linear accelerations
    - last three are body-frame angular accelerations

Measurement (6-D) from the noisy velocities CSV:
    z = [surge, sway, heave, roll_rate, pitch_rate, yaw_rate]

Prediction integrates the IMU accelerations forward; update corrects against
the measured body-frame rates. This mirrors a strapdown INS pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from load_data import _euler_zyx_to_rotation, _wrap_euler


STATE_DIM = 9
MEAS_DIM = 6


@dataclass
class EKFConfig:
    """Tunable noise parameters. Tune against Campbell's noise spec."""

    # Process noise on accelerations treated as inputs (std dev).
    sigma_accel: float = 0.2
    sigma_alpha: float = 0.05

    # Extra process noise on position/velocity/orientation for unmodeled effects.
    sigma_pos: float = 0.01
    sigma_vel: float = 0.05
    sigma_eul: float = 0.01

    # Measurement noise (std dev) on body-frame velocities and rates.
    sigma_body_linvel: float = 0.10
    sigma_body_angvel: float = 0.02

    # Initial covariance (std dev).
    init_sigma_pos: float = 0.5
    init_sigma_vel: float = 0.2
    init_sigma_eul: float = 0.05


@dataclass
class EKFState:
    x: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM))
    P: np.ndarray = field(default_factory=lambda: np.eye(STATE_DIM))


def make_initial_state(cfg: EKFConfig) -> EKFState:
    P0 = np.diag(
        [cfg.init_sigma_pos] * 3
        + [cfg.init_sigma_vel] * 3
        + [cfg.init_sigma_eul] * 3
    ) ** 2
    return EKFState(x=np.zeros(STATE_DIM), P=P0)


def build_process_noise(cfg: EKFConfig, dt: float) -> np.ndarray:
    """Diagonal Q scaled by timestep, reflecting input accel + unmodeled slop."""
    q_pos = (cfg.sigma_pos + 0.5 * cfg.sigma_accel * dt * dt) ** 2
    q_vel = (cfg.sigma_vel + cfg.sigma_accel * dt) ** 2
    q_eul = (cfg.sigma_eul + cfg.sigma_alpha * dt) ** 2

    return np.diag([q_pos] * 3 + [q_vel] * 3 + [q_eul] * 3)


def build_measurement_noise(cfg: EKFConfig) -> np.ndarray:
    return np.diag(
        [cfg.sigma_body_linvel] * 3 + [cfg.sigma_body_angvel] * 3
    ) ** 2


def predict(state: EKFState, u: np.ndarray, dt: float, cfg: EKFConfig) -> EKFState:
    """
    Propagate state forward using the IMU-like acceleration input.

    Continuous dynamics:
        p_dot   = v
        v_dot   = R_bw(euler) @ a_body
        eul_dot = alpha_body * dt   (treated as d(euler)/dt for simplicity)

    We use a first-order Euler integrator for the baseline.
    """
    px, py, pz, vx, vy, vz, roll, pitch, yaw = state.x

    a_body = u[0:3]
    alpha_body = u[3:6]

    R_bw = _euler_zyx_to_rotation(roll, pitch, yaw)
    a_world = R_bw @ a_body

    p_new = np.array([px, py, pz]) + np.array([vx, vy, vz]) * dt + 0.5 * a_world * dt * dt
    v_new = np.array([vx, vy, vz]) + a_world * dt
    eul_new = _wrap_euler(np.array([roll, pitch, yaw]) + alpha_body * dt)

    x_new = np.concatenate([p_new, v_new, eul_new])

    F = _jacobian_F(state.x, u, dt)
    Q = build_process_noise(cfg, dt)
    P_new = F @ state.P @ F.T + Q

    return EKFState(x=x_new, P=P_new)


def update(state: EKFState, z: np.ndarray, cfg: EKFConfig) -> EKFState:
    """
    Correct the state using body-frame velocity + angular rate measurements.

    Measurement model:
        z_linvel = R_bw^T @ v_world
        z_angvel = d(eul)/dt   (approximated here by the current state's implied rate)

    For a simple baseline, we directly measure body-frame linear velocity via
    the rotation of world velocity, and we assume angular rates are consistent
    with the current Euler derivatives.
    """
    px, py, pz, vx, vy, vz, roll, pitch, yaw = state.x
    R_bw = _euler_zyx_to_rotation(roll, pitch, yaw)
    R_wb = R_bw.T

    z_pred = np.zeros(MEAS_DIM)
    z_pred[0:3] = R_wb @ np.array([vx, vy, vz])
    z_pred[3:6] = np.zeros(3)  # angular velocity not in state; treat as zero-mean prior

    H = _jacobian_H(state.x)
    R = build_measurement_noise(cfg)

    innovation = z - z_pred
    innovation[3:6] = _wrap_euler(innovation[3:6])

    S = H @ state.P @ H.T + R
    K = state.P @ H.T @ np.linalg.inv(S)

    x_upd = state.x + K @ innovation
    x_upd[6:9] = _wrap_euler(x_upd[6:9])

    I = np.eye(STATE_DIM)
    P_upd = (I - K @ H) @ state.P @ (I - K @ H).T + K @ R @ K.T

    return EKFState(x=x_upd, P=P_upd)


def run(
    traj_time: np.ndarray,
    inputs: np.ndarray,
    measurements: np.ndarray,
    dt: float,
    cfg: EKFConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the EKF on a full scenario and return (estimated_states, covariances)."""
    cfg = cfg or EKFConfig()

    state = make_initial_state(cfg)
    n = len(traj_time)

    xs = np.zeros((n, STATE_DIM))
    Ps = np.zeros((n, STATE_DIM, STATE_DIM))

    for k in range(n):
        state = predict(state, inputs[k], dt, cfg)
        state = update(state, measurements[k], cfg)

        xs[k] = state.x
        Ps[k] = state.P

    return xs, Ps


def _jacobian_F(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Numerical Jacobian of the process model w.r.t. the state.

    For a first baseline we use a finite-difference approximation; it's easy to
    verify and robust to tweaks in the dynamics. Replace with an analytic form
    later if runtime matters.
    """
    eps = 1e-6
    F = np.zeros((STATE_DIM, STATE_DIM))
    fx = _propagate_mean(x, u, dt)

    for i in range(STATE_DIM):
        x_pert = x.copy()
        x_pert[i] += eps
        fx_pert = _propagate_mean(x_pert, u, dt)
        F[:, i] = (fx_pert - fx) / eps

    return F


def _jacobian_H(x: np.ndarray) -> np.ndarray:
    """Numerical Jacobian of the measurement model w.r.t. the state."""
    eps = 1e-6
    H = np.zeros((MEAS_DIM, STATE_DIM))
    hx = _measurement_mean(x)

    for i in range(STATE_DIM):
        x_pert = x.copy()
        x_pert[i] += eps
        hx_pert = _measurement_mean(x_pert)
        diff = hx_pert - hx
        diff[3:6] = _wrap_euler(diff[3:6])
        H[:, i] = diff / eps

    return H


def _propagate_mean(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    px, py, pz, vx, vy, vz, roll, pitch, yaw = x
    a_body = u[0:3]
    alpha_body = u[3:6]

    R_bw = _euler_zyx_to_rotation(roll, pitch, yaw)
    a_world = R_bw @ a_body

    p_new = np.array([px, py, pz]) + np.array([vx, vy, vz]) * dt + 0.5 * a_world * dt * dt
    v_new = np.array([vx, vy, vz]) + a_world * dt
    eul_new = _wrap_euler(np.array([roll, pitch, yaw]) + alpha_body * dt)

    return np.concatenate([p_new, v_new, eul_new])


def _measurement_mean(x: np.ndarray) -> np.ndarray:
    vx, vy, vz = x[3], x[4], x[5]
    roll, pitch, yaw = x[6], x[7], x[8]

    R_bw = _euler_zyx_to_rotation(roll, pitch, yaw)
    body_linvel = R_bw.T @ np.array([vx, vy, vz])

    z = np.zeros(MEAS_DIM)
    z[0:3] = body_linvel
    z[3:6] = np.zeros(3)
    return z
