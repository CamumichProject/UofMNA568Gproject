"""
Left-Invariant Extended Kalman Filter (In-EKF) on SE_2(3) for USV state estimation.

State lives on the matrix Lie group SE_2(3):
    X = (R, v, p),
    R in SO(3)  : body-to-world rotation
    v in R^3    : world-frame linear velocity
    p in R^3    : world-frame position

Error is parameterized in the Lie algebra in the LEFT-invariant convention,
    eta = X_hat^{-1} X,      xi = log(eta) = [xi_R (3), xi_v (3), xi_p (3)] in R^9.

Driving inputs (matching ekf_baseline / es_ekf_baseline so run_comparison.py
plugs this filter in without any other changes):
    u = [a_body (3), omega_body (3)]
The second triple is consumed as a body-frame angular rate, matching the
ES-EKF's use of the same channel.

Observation: body-frame linear velocity (DVL-style),
    y = R^T v + n,     n ~ N(0, R_meas).
The baselines also expose an angular-rate channel on z, but it is inert in
both EKF/ES-EKF (H is zero in those rows because no angular-rate state
exists), so we simply drop it here and feed the gyro as a prediction input
where it belongs.

Key property that motivates the In-EKF: under body-frame IMU dynamics the
LI error dynamics are independent of the current state estimate
(Barrau & Bonnabel's log-linear property). The body-velocity observation
loses strict log-linearity because the Jacobian depends on v_hat, but the
formulation still has the convergence/consistency advantages identified by
Barrau & Bonnabel for inertial navigation.

API mirrors ekf_baseline.run and es_ekf_baseline.run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from es_ekf_baseline import _rotation_to_euler_zyx, _skew, _small_rotation
from load_data import _wrap_euler


STATE_OUT_DIM = 9   # export [p, v, eul] to keep run_comparison / metrics unchanged
ERROR_DIM = 9       # xi = [xi_R, xi_v, xi_p]
MEAS_DIM = 3        # body-frame linear velocity only


@dataclass
class InEKFConfig:
    """Noise parameters for the In-EKF. Defaults mirror ES-EKF for fairness."""

    sigma_accel: float = 0.2
    sigma_gyro: float = 0.05

    sigma_body_linvel: float = 0.10

    init_sigma_R: float = 0.05
    init_sigma_v: float = 0.2
    init_sigma_p: float = 0.5


@dataclass
class InEKFState:
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(default_factory=lambda: np.eye(ERROR_DIM))


def make_initial_state(cfg: InEKFConfig) -> InEKFState:
    P0 = np.diag(
        [cfg.init_sigma_R] * 3
        + [cfg.init_sigma_v] * 3
        + [cfg.init_sigma_p] * 3
    ) ** 2
    return InEKFState(R=np.eye(3), v=np.zeros(3), p=np.zeros(3), P=P0)


def build_process_noise(cfg: InEKFConfig, dt: float) -> np.ndarray:
    q_R = (cfg.sigma_gyro * dt) ** 2
    q_v = (cfg.sigma_accel * dt) ** 2
    q_p = (0.5 * cfg.sigma_accel * dt * dt) ** 2
    return np.diag([q_R] * 3 + [q_v] * 3 + [q_p] * 3)


def build_measurement_noise(cfg: InEKFConfig) -> np.ndarray:
    return np.eye(MEAS_DIM) * (cfg.sigma_body_linvel ** 2)


def predict(state: InEKFState, u: np.ndarray, dt: float, cfg: InEKFConfig) -> InEKFState:
    """
    Propagate SE_2(3) mean through body-frame IMU dynamics, then push the
    left-invariant error covariance through the (state-independent) linear map.

    Continuous dynamics:
        R_dot = R [omega]_x
        v_dot = R a_body
        p_dot = v

    LI error dynamics (xi = [xi_R, xi_v, xi_p]):
        d(xi_R)/dt = -[omega]_x xi_R
        d(xi_v)/dt = -[a_body]_x xi_R - [omega]_x xi_v
        d(xi_p)/dt =               xi_v - [omega]_x xi_p
    Note the absence of the state estimate in these right-hand sides: this
    is the log-linear property that the standard EKF does not share.
    """
    a_body = u[0:3]
    omega = u[3:6]

    R_delta = _small_rotation(omega * dt)
    a_world = state.R @ a_body

    R_new = state.R @ R_delta
    v_new = state.v + a_world * dt
    p_new = state.p + state.v * dt + 0.5 * a_world * dt * dt

    # Rotation blocks use the proper exponential map rather than I - [omega]_x dt
    # so they are norm-preserving; the naive first-order form has eigenvalues
    # sqrt(1 + |omega|^2 dt^2) > 1 and compounds to overflow over ~10^4 steps.
    Phi_R = _small_rotation(-omega * dt)
    A_body = _skew(a_body)

    F = np.zeros((ERROR_DIM, ERROR_DIM))
    F[0:3, 0:3] = Phi_R
    F[3:6, 0:3] = -A_body * dt
    F[3:6, 3:6] = Phi_R
    F[6:9, 3:6] = np.eye(3) * dt
    F[6:9, 6:9] = Phi_R

    Q = build_process_noise(cfg, dt)
    P_new = F @ state.P @ F.T + Q
    P_new = 0.5 * (P_new + P_new.T)

    return InEKFState(R=R_new, v=v_new, p=p_new, P=P_new)


def update(state: InEKFState, z_lin: np.ndarray, cfg: InEKFConfig) -> InEKFState:
    """
    Body-frame velocity update.

    Predicted observation:
        y_hat = R_hat^T v_hat
    First-order expansion of the residual in the left-invariant error:
        y - y_hat ~= [y_hat]_x xi_R + xi_v + n
    so the Jacobian block for linear velocity is:
        H = [ [y_hat]_x,  I_3,  0 ] (shape 3 x 9)
    State-independent column structure; only y_hat scales the rotation block.
    """
    y_hat = state.R.T @ state.v
    r = z_lin - y_hat

    H = np.zeros((MEAS_DIM, ERROR_DIM))
    H[0:3, 0:3] = _skew(y_hat)
    H[0:3, 3:6] = np.eye(3)

    R_meas = build_measurement_noise(cfg)
    S = H @ state.P @ H.T + R_meas
    K = state.P @ H.T @ np.linalg.inv(S)
    xi = K @ r

    xi_R = xi[0:3]
    xi_v = xi[3:6]
    xi_p = xi[6:9]

    # Left-invariant injection: X_new = X_hat * Exp(xi).
    # For small xi the SE_2(3) exponential reduces to rotating by Exp(xi_R) and
    # adding R_hat @ xi_{v,p}; we use the small-angle form because K @ r is
    # already tiny relative to the state after the first few steps.
    R_new = state.R @ _small_rotation(xi_R)
    v_new = state.v + state.R @ xi_v
    p_new = state.p + state.R @ xi_p

    I = np.eye(ERROR_DIM)
    P_new = (I - K @ H) @ state.P @ (I - K @ H).T + K @ R_meas @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    return InEKFState(R=R_new, v=v_new, p=p_new, P=P_new)


def run(
    traj_time: np.ndarray,
    inputs: np.ndarray,
    measurements: np.ndarray,
    dt: float,
    cfg: InEKFConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the In-EKF over a full scenario.

    Returns
    -------
    xs : (N, 9) array of [p(3), v(3), euler_zyx(3)] per timestep. This layout
        matches ekf_baseline.run / es_ekf_baseline.run so that metrics.summarize
        and run_comparison._dump_csv work without modification.
    Ps : (N, 9, 9) covariance history in the LI error coordinates.
    """
    cfg = cfg or InEKFConfig()
    state = make_initial_state(cfg)

    n = len(traj_time)
    xs = np.zeros((n, STATE_OUT_DIM))
    Ps = np.zeros((n, ERROR_DIM, ERROR_DIM))

    for k in range(n):
        state = predict(state, inputs[k], dt, cfg)
        state = update(state, measurements[k, 0:3], cfg)

        xs[k, 0:3] = state.p
        xs[k, 3:6] = state.v
        xs[k, 6:9] = _wrap_euler(_rotation_to_euler_zyx(state.R))
        Ps[k] = state.P

    return xs, Ps
