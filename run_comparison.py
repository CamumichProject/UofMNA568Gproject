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
import inekf_baseline
from load_data import integrate_truth, load_velocity_scenario
from metrics import error_over_time, summarize


ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
CSV_DIR = ROOT / "filter_outputs"


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
    CSV_DIR.mkdir(exist_ok=True)

    all_metrics: list[tuple[str, dict, dict, dict]] = []

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
        inekf_states, _ = inekf_baseline.run(
            traj_time=clean.t,
            inputs=noisy_acc,
            measurements=noisy_vel,
            dt=clean.dt,
        )

        ekf_metrics = summarize(ekf_states, truth_state)
        es_metrics = summarize(es_ekf_states, truth_state)
        inekf_metrics = summarize(inekf_states, truth_state)

        _print_table(ekf_metrics, es_metrics, inekf_metrics)
        all_metrics.append((sc.name, ekf_metrics, es_metrics, inekf_metrics))

        _plot_trajectory(sc, ekf_states, es_ekf_states, inekf_states, truth_state)
        _plot_error_over_time(
            sc, clean.t, ekf_states, es_ekf_states, inekf_states, truth_state
        )
        _plot_orientation(
            sc, clean.t, ekf_states, es_ekf_states, inekf_states, truth_state
        )
        _dump_csv(sc, clean.t, truth_state, ekf_states, es_ekf_states, inekf_states)

    _print_summary(all_metrics)


def _print_table(ekf_metrics: dict, es_metrics: dict, inekf_metrics: dict) -> None:
    keys = list(ekf_metrics.keys())
    header = f"{'metric':<22}{'EKF':>12}{'ES-EKF':>12}{'In-EKF':>12}"
    print(header)
    print("-" * len(header))
    for k in keys:
        print(
            f"{k:<22}{ekf_metrics[k]:>12.4f}{es_metrics[k]:>12.4f}"
            f"{inekf_metrics[k]:>12.4f}"
        )


def _print_summary(results: list[tuple[str, dict, dict, dict]]) -> None:
    print("\n\n=== Summary across scenarios ===")
    header = (
        f"{'scenario':<22}{'filter':<10}{'pos RMSE (m)':>14}"
        f"{'final drift (m)':>18}{'yaw RMSE (rad)':>18}"
    )
    print(header)
    print("-" * len(header))
    for name, ekf, es, inekf in results:
        print(
            f"{name:<22}{'EKF':<10}{ekf['position_rmse_m']:>14.4f}"
            f"{ekf['final_drift_m']:>18.4f}{ekf['yaw_rmse_rad']:>18.4f}"
        )
        print(
            f"{'':<22}{'ES-EKF':<10}{es['position_rmse_m']:>14.4f}"
            f"{es['final_drift_m']:>18.4f}{es['yaw_rmse_rad']:>18.4f}"
        )
        print(
            f"{'':<22}{'In-EKF':<10}{inekf['position_rmse_m']:>14.4f}"
            f"{inekf['final_drift_m']:>18.4f}{inekf['yaw_rmse_rad']:>18.4f}"
        )


def _plot_trajectory(
    sc: Scenario,
    ekf: np.ndarray,
    es: np.ndarray,
    inekf: np.ndarray,
    truth: np.ndarray,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(truth[:, 0], truth[:, 1], label="truth", linewidth=2)
    ax.plot(ekf[:, 0], ekf[:, 1], label="EKF", linestyle="--")
    ax.plot(es[:, 0], es[:, 1], label="ES-EKF", linestyle=":")
    ax.plot(inekf[:, 0], inekf[:, 1], label="In-EKF", linestyle="-.")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"USV XY trajectory — {sc.name}")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"trajectory_xy_{sc.slug}.png", dpi=150)
    plt.close(fig)


def _plot_error_over_time(
    sc: Scenario,
    t: np.ndarray,
    ekf: np.ndarray,
    es: np.ndarray,
    inekf: np.ndarray,
    truth: np.ndarray,
) -> None:
    err_ekf = error_over_time(ekf[:, 0:3], truth[:, 0:3])
    err_es = error_over_time(es[:, 0:3], truth[:, 0:3])
    err_inekf = error_over_time(inekf[:, 0:3], truth[:, 0:3])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t, err_ekf, label="EKF")
    ax.plot(t, err_es, label="ES-EKF")
    ax.plot(t, err_inekf, label="In-EKF")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (m)")
    ax.set_title(f"Position error over time — {sc.name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"position_error_{sc.slug}.png", dpi=150)
    plt.close(fig)


def _plot_orientation(
    sc: Scenario,
    t: np.ndarray,
    ekf: np.ndarray,
    es: np.ndarray,
    inekf: np.ndarray,
    truth: np.ndarray,
) -> None:
    """Rolling-RMS Euler-angle error per axis.

    Raw instantaneous error is dominated by the measurement model's zero-mean
    prior on angular rate and saturates near +/- pi for all three filters, so
    the overlay is unreadable. A rolling RMS over a 10 s window condenses the
    noise into a single smooth curve per filter and makes aggregate differences
    (if any) visible.
    """
    def _wrap(a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    err_ekf = _wrap(ekf[:, 6:9] - truth[:, 3:6])
    err_es = _wrap(es[:, 6:9] - truth[:, 3:6])
    err_inekf = _wrap(inekf[:, 6:9] - truth[:, 3:6])

    dt = float(t[1] - t[0]) if len(t) > 1 else 0.1
    window = max(1, int(round(10.0 / dt)))  # 10 s rolling window

    def _rolling_rms(err: np.ndarray) -> np.ndarray:
        sq = err ** 2
        kernel = np.ones(window) / window
        out = np.empty_like(sq)
        for j in range(sq.shape[1]):
            out[:, j] = np.sqrt(np.convolve(sq[:, j], kernel, mode="same"))
        return out

    rms_ekf = _rolling_rms(err_ekf)
    rms_es = _rolling_rms(err_es)
    rms_inekf = _rolling_rms(err_inekf)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    labels = ("roll", "pitch", "yaw")
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, rms_ekf[:, i], label="EKF", linewidth=1.4)
        ax.plot(t, rms_es[:, i], label="ES-EKF", linewidth=1.4, linestyle="--")
        ax.plot(t, rms_inekf[:, i], label="In-EKF", linewidth=1.4, linestyle="-.")
        ax.set_ylabel(f"{label} RMS error (rad)")
        ax.set_ylim(0.0, np.pi)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Orientation error (10 s rolling RMS) — {sc.name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"orientation_{sc.slug}.png", dpi=150)
    plt.close(fig)


def _dump_csv(
    sc: Scenario,
    t: np.ndarray,
    truth: np.ndarray,
    ekf: np.ndarray,
    es: np.ndarray,
    inekf: np.ndarray,
) -> None:
    """Write per-timestep truth + filter positions/Euler so teammates can do side-by-sides."""
    header = (
        "t,"
        "truth_x,truth_y,truth_z,truth_roll,truth_pitch,truth_yaw,"
        "ekf_x,ekf_y,ekf_z,ekf_vx,ekf_vy,ekf_vz,ekf_roll,ekf_pitch,ekf_yaw,"
        "esekf_x,esekf_y,esekf_z,esekf_vx,esekf_vy,esekf_vz,esekf_roll,esekf_pitch,esekf_yaw,"
        "inekf_x,inekf_y,inekf_z,inekf_vx,inekf_vy,inekf_vz,inekf_roll,inekf_pitch,inekf_yaw"
    )
    data = np.hstack([t.reshape(-1, 1), truth, ekf, es, inekf])
    out_path = CSV_DIR / f"filter_states_{sc.slug}.csv"
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"  wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
