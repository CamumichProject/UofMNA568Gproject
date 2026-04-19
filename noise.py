"""
Deterministic IMU-style noise injection for the clean simulator CSVs.

Design goals:
    - Everyone on the team (EKF / ES-EKF / In-EKF) can call the same function
      and get the exact same noisy signals for a given seed.
    - Easy to sweep noise levels for sensitivity plots in the final paper.
    - Noise model matches a standard strapdown IMU:
        noisy = true + bias + white_gaussian_noise

The acceleration channels are treated as accelerometer-like (linear) and
angular-rate-like (gyro). The velocity channels are noised more lightly and
are intended to mimic a DVL / body-velocity estimate that the filter uses as
a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IMUNoiseConfig:
    """Noise + bias parameters for accelerometer, gyro, and body-velocity sensor."""

    # Accelerometer (linear acceleration channels: dxx, dyy, dzz).
    accel_sigma: float = 0.10        # m/s^2, white noise std dev
    accel_bias_sigma: float = 0.02   # m/s^2, constant bias drawn per run

    # Gyro / angular acceleration (drr, dpp, dyy).
    gyro_sigma: float = 0.010        # rad/s^2, white noise std dev
    gyro_bias_sigma: float = 0.002   # rad/s^2, constant bias drawn per run

    # Body-frame velocity sensor (surge, sway, heave).
    linvel_sigma: float = 0.05       # m/s
    # Angular-rate sensor (roll, pitch, yaw columns if they are rates).
    angvel_sigma: float = 0.01       # rad/s

    seed: int = 0


@dataclass
class NoisyData:
    acceleration_body: np.ndarray   # (N, 6) noisy: linear accel + angular accel
    velocity_body: np.ndarray       # (N, 6) noisy: body lin vel + body ang rate
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))


def add_imu_noise(
    clean_accel: np.ndarray,
    clean_vel: np.ndarray,
    cfg: IMUNoiseConfig | None = None,
) -> NoisyData:
    """
    Inject IMU-style noise + bias into clean simulator data.

    Parameters
    ----------
    clean_accel : (N, 6) array. Columns = linear accel (3) + angular accel (3).
    clean_vel   : (N, 6) array. Columns = body linear velocity (3) + body angular (3).
    cfg         : noise configuration. Use the default for a reasonable first pass.

    Returns
    -------
    NoisyData containing noisy copies plus the sampled biases (useful for debugging
    and for letting the filter tune `R` against known ground truth).
    """
    cfg = cfg or IMUNoiseConfig()
    rng = np.random.default_rng(cfg.seed)

    n = clean_accel.shape[0]

    accel_bias = rng.normal(0.0, cfg.accel_bias_sigma, size=3)
    gyro_bias = rng.normal(0.0, cfg.gyro_bias_sigma, size=3)

    accel_white = rng.normal(0.0, cfg.accel_sigma, size=(n, 3))
    gyro_white = rng.normal(0.0, cfg.gyro_sigma, size=(n, 3))

    noisy_accel = clean_accel.copy()
    noisy_accel[:, 0:3] += accel_bias + accel_white
    noisy_accel[:, 3:6] += gyro_bias + gyro_white

    linvel_white = rng.normal(0.0, cfg.linvel_sigma, size=(n, 3))
    angvel_white = rng.normal(0.0, cfg.angvel_sigma, size=(n, 3))

    noisy_vel = clean_vel.copy()
    noisy_vel[:, 0:3] += linvel_white
    noisy_vel[:, 3:6] += angvel_white

    return NoisyData(
        acceleration_body=noisy_accel,
        velocity_body=noisy_vel,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )
