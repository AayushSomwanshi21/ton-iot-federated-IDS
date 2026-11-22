import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "outputs" / "benchmarks"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots" / "timing"

TRANSPORTS = ("no_tls", "tls")
TIMING_METRICS = (
    "local_training_seconds",
    "communication_seconds",
    "aggregation_seconds",
    "validation_seconds",
)


def load_runs(transport):
    directory = BENCHMARK_ROOT / f"edgeiiot_fedavg_{transport}"
    runs = []

    if not directory.exists():
        print(f"Missing benchmark directory: {directory}")
        return runs

    for metrics_path in sorted(directory.rglob("metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        if not metrics.get("timing_enabled"):
            continue

        timing = metrics["timing"]
        runs.append({
            "timestamp": metrics_path.parent.name,
            "metrics_path": str(metrics_path),
            **timing,
        })

    print(f"{transport}: loaded {len(runs)} benchmark run(s)")
    return runs


def summarize(runs):
    if not runs:
        return None

    values = {
        name: np.asarray([run[name] for run in runs], dtype=float)
        for name in (
            *TIMING_METRICS,
            "fedavg_seconds",
            "experiment_total_seconds",
        )
    }

    return {
        name: {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values, ddof=1)) if len(runs) > 1 else 0.0,
            "min": float(np.min(metric_values)),
            "median": float(np.median(metric_values)),
            "max": float(np.max(metric_values)),
        }
        for name, metric_values in values.items()
    }


def save_figure(fig, name):
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def write_summary(runs_by_transport, summaries):
    csv_path = PLOTS_DIR / "timing_summary.csv"
    columns = [
        "transport",
        "run_count",
        "metric",
        "mean_seconds",
        "std_seconds",
        "min_seconds",
        "median_seconds",
        "max_seconds",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for transport in TRANSPORTS:
            for metric, values in summaries[transport].items():
                writer.writerow({
                    "transport": transport,
                    "run_count": len(runs_by_transport[transport]),
                    "metric": metric,
                    **{
                        "mean_seconds": values["mean"],
                        "std_seconds": values["std"],
                        "min_seconds": values["min"],
                        "median_seconds": values["median"],
                        "max_seconds": values["max"],
                    },
                })

    json_path = PLOTS_DIR / "timing_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({
            "run_counts": {
                transport: len(runs_by_transport[transport])
                for transport in TRANSPORTS
            },
            "summaries": summaries,
        }, file, indent=2)

    print(f"Saved summary: {csv_path}")
    print(f"Saved summary: {json_path}")


def plot_component_breakdown(summaries):
    labels = [
        "Local training",
        "Communication",
        "Aggregation",
        "Validation",
    ]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, transport in zip((-width / 2, width / 2), TRANSPORTS):
        means = [summaries[transport][metric]["mean"] for metric in TIMING_METRICS]
        stds = [summaries[transport][metric]["std"] for metric in TIMING_METRICS]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label="TLS" if transport == "tls" else "Without TLS",
        )

    ax.set_title("FedAvg Timing Component Comparison")
    ax.set_xlabel("Timing component")
    ax.set_ylabel("Mean time (seconds)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, "tls_vs_no_tls_timing_components.png")


def plot_total_times(runs_by_transport, summaries, metric, title, filename):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(TRANSPORTS))

    means = []
    stds = []
    for transport in TRANSPORTS:
        runs = runs_by_transport[transport]
        values = [run[metric] for run in runs]
        positions = np.full(len(values), len(means))
        ax.scatter(
            positions,
            values,
            s=45,
            alpha=0.75,
            color=["tab:blue", "tab:orange"][len(means)],
        )

        means.append(summaries[transport][metric]["mean"])
        stds.append(summaries[transport][metric]["std"])

    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="D",
        markersize=8,
        capsize=6,
        color="black",
        ecolor="black",
        linewidth=1.5,
        label="Mean ± SD",
    )

    ax.set_title(title)
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x, ["Without TLS", "With TLS"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, filename)


def plot_individual_runs(runs_by_transport, metric, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for index, transport in enumerate(TRANSPORTS):
        runs = runs_by_transport[transport]
        values = [run[metric] for run in runs]
        ax.plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            linewidth=1.8,
            label="TLS" if transport == "tls" else "Without TLS",
        )

    ax.set_title(title)
    ax.set_xlabel("Benchmark run")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(range(1, 6))
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, filename)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_by_transport = {
        transport: load_runs(transport)
        for transport in TRANSPORTS
    }

    missing = [
        transport
        for transport in TRANSPORTS
        if not runs_by_transport[transport]
    ]
    if missing:
        raise RuntimeError(
            "No timed benchmark runs found for: " + ", ".join(missing)
        )

    summaries = {
        transport: summarize(runs_by_transport[transport])
        for transport in TRANSPORTS
    }

    write_summary(runs_by_transport, summaries)
    plot_component_breakdown(summaries)
    plot_total_times(
        runs_by_transport,
        summaries,
        "fedavg_seconds",
        "FedAvg Total Time: With vs Without TLS",
        "tls_vs_no_tls_fedavg_total_time.png",
    )
    plot_total_times(
        runs_by_transport,
        summaries,
        "experiment_total_seconds",
        "Experiment Total Time: With vs Without TLS",
        "tls_vs_no_tls_experiment_total_time.png",
    )
    plot_individual_runs(
        runs_by_transport,
        "fedavg_seconds",
        "FedAvg Time Across Repeated Runs",
        "tls_vs_no_tls_fedavg_runs.png",
    )
    plot_individual_runs(
        runs_by_transport,
        "communication_seconds",
        "Communication Time Across Repeated Runs",
        "tls_vs_no_tls_communication_runs.png",
    )

    print(f"\nPlots and summaries written to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
