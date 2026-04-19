"""
Load Campbell's simulation CSVs for the USV baseline EKF.

Data format (10 Hz sampling):
    Velocities file columns:    Surge, Sway, Heave, Roll, Pitch, Yaw
        - Surge/Sway/Heave : body-frame linear velocities (m/s)
        - Roll/Pitch/Yaw   : body-frame angular velocities (rad/s)
    Accelerations file columns: dxx, dyy, dzz, drr, dpp, dyy
        - time derivatives of the above (linear accel + angular accel)

Assumptions to confirm with Campbell:
    - Frame convention: body-fixed (surge = forward, sway = starboard, heave = down).
    - Angular rates are in rad/s.
    - A paired noise-free file will be published; we auto-detect by filename.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DT = 0.1  # 10 Hz per Discord confirmation from Campbell.


@dataclass
class USVTrajectory:
    """Container for one simulation scenario."""

    dt: float
    t: np.ndarray                  # (N,)
    velocity_body: np.ndarray      # (N, 6) surge, sway, heave, p, q, r
    acceleration_body: np.ndarray  # (N, 6) linear accel + angular accel

    @property
    def linear_velocity(self) -> np.ndarray:
        return self.velocity_body[:, 0:3]

    @property
    def angular_velocity(self) -> np.ndarray:
        return self.velocity_body[:, 3:6]

    @property
    def linear_acceleration(self) -> np.ndarray:
        return self.acceleration_body[:, 0:3]

    @property
    def angular_acceleration(self) -> np.ndarray:
        return self.acceleration_body[:, 3:6]


def load_trajectory(
    velocities_csv: str | Path,
    accelerations_csv: str | Path,
    dt: float = DEFAULT_DT,
) -> USVTrajectory:
    """Load a paired velocities + accelerations CSV scenario."""
    vel_df = pd.read_csv(velocities_csv)
    acc_df = pd.read_csv(accelerations_csv)

    # The accelerations file has a duplicate 'dyy' column name; pandas disambiguates
    # by appending ".1". We use positional indexing to stay robust.
    vel = vel_df.iloc[:, 0:6].to_numpy(dtype=float)
    acc = acc_df.iloc[:, 0:6].to_numpy(dtype=float)

    n = min(len(vel), len(acc))
    vel = vel[:n]
    acc = acc[:n]

    t = np.arange(n) * dt

    return USVTrajectory(dt=dt, t=t, velocity_body=vel, acceleration_body=acc)


def load_velocity_scenario(
    clean_csv: str | Path,
    noise_csv: str | Path,
    dt: float = DEFAULT_DT,
) -> tuple[USVTrajectory, np.ndarray, np.ndarray]:
    """
    Load a velocity-only scenario (clean + team-provided noise samples).

    Only velocity CSVs are published for the three sea-state scenarios. We
    derive a clean body-frame acceleration input by numerical differentiation
    of the clean velocity signal so both EKF baselines can still be driven by
    an IMU-like input.

    Parameters
    ----------
    clean_csv : path to the 6-column clean velocity CSV (surge, sway, heave,
        roll-rate, pitch-rate, yaw-rate).
    noise_csv : path to the matching "Noise added" CSV. Columns are zero-mean
        noise samples, not the already-corrupted signal, so we form the
        measurement as clean + noise.
    dt : sample period (default 10 Hz).

    Returns
    -------
    clean_traj        : USVTrajectory with clean velocities and derived
        clean body accelerations. Use this for ground-truth integration.
    noisy_velocity    : (N, 6) clean + provided noise; used as the EKF
        body-frame velocity measurement.
    noisy_acceleration: (N, 6) noisy body acceleration; used as the EKF input.
        We numerically differentiate the NOISY velocity, which automatically
        injects realistic high-frequency IMU-style noise without requiring a
        synthetic model.
    """
    clean_df = pd.read_csv(clean_csv)
    noise_df = pd.read_csv(noise_csv)

    clean_vel = clean_df.iloc[:, 0:6].to_numpy(dtype=float)
    noise = noise_df.iloc[:, 0:6].to_numpy(dtype=float)

    n = min(len(clean_vel), len(noise))
    clean_vel = clean_vel[:n]
    noise = noise[:n]

    noisy_vel = clean_vel + noise

    clean_acc = _finite_diff(clean_vel, dt)
    noisy_acc = _finite_diff(noisy_vel, dt)

    t = np.arange(n) * dt

    clean_traj = USVTrajectory(
        dt=dt,
        t=t,
        velocity_body=clean_vel,
        acceleration_body=clean_acc,
    )
    return clean_traj, noisy_vel, noisy_acc


def _finite_diff(signal: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference derivative; endpoints use forward/backward diff."""
    d = np.zeros_like(signal)
    d[1:-1] = (signal[2:] - signal[:-2]) / (2.0 * dt)
    d[0] = (signal[1] - signal[0]) / dt
    d[-1] = (signal[-1] - signal[-2]) / dt
    return d


def integrate_truth(traj: USVTrajectory) -> np.ndarray:
    """
    Integrate noise-free body velocities to produce a 'ground truth' world-frame
    position and Euler orientation trajectory.

    Returns array of shape (N, 6): [x, y, z, roll, pitch, yaw] in world frame.

    Euler-integration assumption (simple for baseline):
        - body-to-world rotation built from current Euler angles (ZYX convention).
        - angular velocity is treated as d(euler)/dt (valid for small tilt; good
          enough for a baseline comparator).
    """
    n = len(traj.t)
    dt = traj.dt

    pos_w = np.zeros((n, 3))
    eul = np.zeros((n, 3))  # roll, pitch, yaw

    for k in range(1, n):
        roll, pitch, yaw = eul[k - 1]
        R_bw = _euler_zyx_to_rotation(roll, pitch, yaw)

        v_body = traj.linear_velocity[k - 1]
        pos_w[k] = pos_w[k - 1] + R_bw @ v_body * dt

        w_body = traj.angular_velocity[k - 1]
        eul[k] = _wrap_euler(eul[k - 1] + w_body * dt)

    return np.hstack([pos_w, eul])


def _euler_zyx_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation matrix using ZYX (yaw-pitch-roll) convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _wrap_euler(eul: np.ndarray) -> np.ndarray:
    """Wrap each Euler angle to [-pi, pi)."""
    return (eul + np.pi) % (2.0 * np.pi) - np.pi
