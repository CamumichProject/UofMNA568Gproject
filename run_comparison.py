"""
End-to-end runner: load velocity CSVs, run both baselines per sea state, report.

Pipeline per scenario:
    1. Load clean body-frame velocity CSV + matching noise-sample CSV.
    2. Integrate the clean velocity to obtain a ground-truth world trajectory.
    3. Build the EKF measurement  = clean_velocity + provided_noise.
       Build the EKF input        = numerically differentiated noisy velocity.
    4. Run the standard EKF and the Error-State EKF on the SAME noisy signals.
    5. Print a per-scenario metric table and save comparison plots to figures/.

Run:
    MPLBACKEND=Agg python run_comparison.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ekf_baseline
import es_ekf_baseline
from load_data import integrate_truth, load_velocity_scenario
from metrics import error_over_time, summarize


ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"


@dataclass
class Scenario:
    name: str
    slug: str          # filesystem-safe identifier used in figure filenames
    clean_csv: Path
    noise_csv: Path


SCENARIOS: list[Scenario] = [
    Scenario(
        name="Calm sea",
        slug="calm",
        clean_csv=ROOT / "Velocities calm" / "T1 Calm.csv",
        noise_csv=ROOT / "Velocities calm" / "T1 Calm Noise added.csv",
    ),
    Scenario(
        name="Medium sea (10y)",
        slug="medium",
        clean_csv=ROOT / "Velocities medium" / "10y sea data.csv",
        noise_csv=ROOT / "Velocities medium" / "10y sea data Noise added.csv",
    ),
    Scenario(
        name="High seas (Test1)",
        slug="high",
        clean_csv=ROOT / "Velocities High seas" / "Velocities Test1 High seas.csv",
        noise_csv=ROOT / "Velocities High seas" / "Velocities Test1 High seas Noise added.csv",
    ),
]


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    all_metrics: list[tuple[str, dict, dict]] = []

    for sc in SCENARIOS:
        print(f"\n=== {sc.name} ===")
        print(f"  clean: {sc.clean_csv.relative_to(ROOT)}")
        print(f"  noise: {sc.noise_csv.relative_to(ROOT)}")

        clean, noisy_vel, noisy_acc = load_velocity_scenario(
            sc.clean_csv, sc.noise_csv
        )
        truth_state = integrate_truth(clean)

        ekf_states, _ = ekf_baseline.run(
            traj_time=clean.t,
            inputs=noisy_acc,
            measurements=noisy_vel,
            dt=clean.dt,
        )
        es_ekf_states, _ = es_ekf_baseline.run(
            traj_time=clean.t,
            inputs=noisy_acc,
            measurements=noisy_vel,
            dt=clean.dt,
        )

        ekf_metrics = summarize(ekf_states, truth_state)
        es_metrics = summarize(es_ekf_states, truth_state)

        _print_table(ekf_metrics, es_metrics)
        all_metrics.append((sc.name, ekf_metrics, es_metrics))

        _plot_trajectory(sc, ekf_states, es_ekf_states, truth_state)
        _plot_error_over_time(sc, clean.t, ekf_states, es_ekf_states, truth_state)
        _plot_orientation(sc, clean.t, ekf_states, es_ekf_states, truth_state)

    _print_summary(all_metrics)


def _print_table(ekf_metrics: dict, es_metrics: dict) -> None:
    keys = list(ekf_metrics.keys())
    header = f"{'metric':<22}{'EKF':>12}{'ES-EKF':>12}"
    print(header)
    print("-" * len(header))
    for k in keys:
        print(f"{k:<22}{ekf_metrics[k]:>12.4f}{es_metrics[k]:>12.4f}")


def _print_summary(results: list[tuple[str, dict, dict]]) -> None:
    print("\n\n=== Summary across scenarios ===")
    header = f"{'scenario':<22}{'filter':<10}{'pos RMSE (m)':>14}{'final drift (m)':>18}{'yaw RMSE (rad)':>18}"
    print(header)
    print("-" * len(header))
    for name, ekf, es in results:
        print(
            f"{name:<22}{'EKF':<10}{ekf['position_rmse_m']:>14.4f}"
            f"{ekf['final_drift_m']:>18.4f}{ekf['yaw_rmse_rad']:>18.4f}"
        )
        print(
            f"{'':<22}{'ES-EKF':<10}{es['position_rmse_m']:>14.4f}"
            f"{es['final_drift_m']:>18.4f}{es['yaw_rmse_rad']:>18.4f}"
        )


def _plot_trajectory(
    sc: Scenario, ekf: np.ndarray, es: np.ndarray, truth: np.ndarray
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(truth[:, 0], truth[:, 1], label="truth", linewidth=2)
    ax.plot(ekf[:, 0], ekf[:, 1], label="EKF", linestyle="--")
    ax.plot(es[:, 0], es[:, 1], label="ES-EKF", linestyle=":")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"USV XY trajectory — {sc.name}")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"trajectory_xy_{sc.slug}.png", dpi=150)
    plt.close(fig)


def _plot_error_over_time(
    sc: Scenario, t: np.ndarray, ekf: np.ndarray, es: np.ndarray, truth: np.ndarray
) -> None:
    err_ekf = error_over_time(ekf[:, 0:3], truth[:, 0:3])
    err_es = error_over_time(es[:, 0:3], truth[:, 0:3])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t, err_ekf, label="EKF")
    ax.plot(t, err_es, label="ES-EKF")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (m)")
    ax.set_title(f"Position error over time — {sc.name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"position_error_{sc.slug}.png", dpi=150)
    plt.close(fig)


def _plot_orientation(
    sc: Scenario, t: np.ndarray, ekf: np.ndarray, es: np.ndarray, truth: np.ndarray
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    labels = ("roll", "pitch", "yaw")
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, truth[:, 3 + i], label="truth", linewidth=2)
        ax.plot(t, ekf[:, 6 + i], label="EKF", linestyle="--")
        ax.plot(t, es[:, 6 + i], label="ES-EKF", linestyle=":")
        ax.set_ylabel(f"{label} (rad)")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Orientation — {sc.name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"orientation_{sc.slug}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
