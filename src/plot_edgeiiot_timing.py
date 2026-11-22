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


def read_metrics(metrics_path):
    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_runs(transport):
    """Load timed FedAvg runs for one transport setting."""
    directory = BENCHMARK_ROOT / f"edgeiiot_fedavg_{transport}"
    runs = []

    if not directory.exists():
        print(f"Missing benchmark directory: {directory}")
        return runs

    for metrics_path in sorted(directory.rglob("metrics.json")):
        metrics = read_metrics(metrics_path)
        if not metrics.get("timing_enabled") or "timing" not in metrics:
            continue

        runs.append({
            "timestamp": metrics_path.parent.name,
            "metrics_path": str(metrics_path),
            **metrics["timing"],
        })

    print(f"FedAvg {transport}: loaded {len(runs)} timed run(s)")
    return runs


def infer_tls(metrics, metrics_path):
    """Infer transport from saved metadata, falling back to directory naming."""
    for key in ("use_tls", "tls_enabled"):
        if key in metrics:
            return bool(metrics[key])
    transport = str(metrics.get("transport", "")).lower()
    if transport in ("tls", "no_tls"):
        return transport == "tls"
    return "_tls" in str(metrics_path.parent).lower() or "with_tls" in str(metrics_path.parent).lower()


def load_fedprox_runs():
    """Discover timed FedProx runs, grouped later by mu and TLS setting."""
    runs = []
    if not BENCHMARK_ROOT.exists():
        return runs

    # Search recursively so both TLS and non-TLS output directory layouts work.
    for metrics_path in sorted(BENCHMARK_ROOT.rglob("metrics.json")):
        if "fedprox" not in str(metrics_path).lower():
            continue

        metrics = read_metrics(metrics_path)
        if not metrics.get("timing_enabled") or "timing" not in metrics:
            continue

        mu = metrics.get("proximal_mu", metrics.get("mu"))
        if mu is None:
            # Expected directory format: edgeiiot_fedprox_mu_0.001
            for part in metrics_path.parts:
                if "fedprox_mu_" in part:
                    try:
                        mu = float(part.split("fedprox_mu_", 1)[1].split("_")[0])
                    except ValueError:
                        pass
                    break
        if mu is None:
            print(f"Skipping FedProx run with no mu value: {metrics_path}")
            continue

        runs.append({
            "mu": float(mu),
            "tls": infer_tls(metrics, metrics_path),
            "timestamp": metrics_path.parent.name,
            "metrics_path": str(metrics_path),
            **metrics["timing"],
        })

    print(f"FedProx: loaded {len(runs)} timed run(s)")
    return runs


def summarize(runs, metric_names):
    if not runs:
        return None

    values = {
        name: np.asarray([run[name] for run in runs if name in run], dtype=float)
        for name in metric_names
    }
    return {
        name: {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else 0.0,
            "min": float(np.min(metric_values)),
            "median": float(np.median(metric_values)),
            "max": float(np.max(metric_values)),
        }
        for name, metric_values in values.items()
        if len(metric_values)
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
        "transport", "run_count", "metric",
        "mean_seconds", "std_seconds", "min_seconds",
        "median_seconds", "max_seconds",
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
                    "mean_seconds": values["mean"],
                    "std_seconds": values["std"],
                    "min_seconds": values["min"],
                    "median_seconds": values["median"],
                    "max_seconds": values["max"],
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
    labels = ["Local training", "Communication", "Aggregation", "Validation"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, transport in zip((-width / 2, width / 2), TRANSPORTS):
        means = [summaries[transport][metric]["mean"] for metric in TIMING_METRICS]
        stds = [summaries[transport][metric]["std"] for metric in TIMING_METRICS]
        ax.bar(
            x + offset, means, width, yerr=stds, capsize=4,
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

    means, stds = [], []
    for index, transport in enumerate(TRANSPORTS):
        values = [run[metric] for run in runs_by_transport[transport] if metric in run]
        ax.scatter(
            np.full(len(values), index), values, s=45, alpha=0.75,
            color=["tab:blue", "tab:orange"][index],
        )
        means.append(summaries[transport][metric]["mean"])
        stds.append(summaries[transport][metric]["std"])

    ax.errorbar(
        x, means, yerr=stds, fmt="D", markersize=8, capsize=6,
        color="black", ecolor="black", linewidth=1.5, label="Mean ± SD",
    )
    ax.set_title(title)
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x, ["Without TLS", "With TLS"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, filename)


def plot_individual_runs(runs_by_transport, metric, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for transport in TRANSPORTS:
        values = [run[metric] for run in runs_by_transport[transport] if metric in run]
        ax.plot(
            range(1, len(values) + 1), values, marker="o", linewidth=1.8,
            label="TLS" if transport == "tls" else "Without TLS",
        )

    ax.set_title(title)
    ax.set_xlabel("Benchmark run")
    ax.set_ylabel("Time (seconds)")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, filename)


def plot_fedprox_mu(fedprox_runs):
    """Plot FedProx runtime against proximal mu; include TLS/non-TLS separately."""
    if not fedprox_runs:
        print("No timed FedProx runs found; skipping FedProx mu plots.")
        return

    metric_options = (
        ("fedprox_seconds", "FedProx training time"),
        ("experiment_total_seconds", "Total experiment time"),
        ("communication_seconds", "Communication time"),
    )
    mus = sorted({run["mu"] for run in fedprox_runs})

    for metric, title in metric_options:
        available = [run for run in fedprox_runs if metric in run]
        if not available:
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for tls_value, label, marker in (
            (False, "Without TLS", "o"),
            (True, "With TLS", "s"),
        ):
            group = [run for run in available if run["tls"] == tls_value]
            if not group:
                continue
            grouped_means, grouped_stds, grouped_mus = [], [], []
            for mu in mus:
                vals = [run[metric] for run in group if run["mu"] == mu]
                if vals:
                    grouped_mus.append(mu)
                    grouped_means.append(float(np.mean(vals)))
                    grouped_stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                    ax.scatter([mu] * len(vals), vals, alpha=0.7, marker=marker)

            ax.errorbar(
                grouped_mus, grouped_means, yerr=grouped_stds,
                marker=marker, capsize=4, linewidth=1.8, label=label,
            )

        ax.set_xscale("log")
        ax.set_title(f"{title} Across FedProx μ Values")
        ax.set_xlabel("FedProx proximal μ (log scale)")
        ax.set_ylabel("Time (seconds)")
        ax.grid(alpha=0.25)
        ax.legend()
        save_figure(fig, f"fedprox_mu_{metric}.png")

    # Component breakdown for each mu, with TLS status distinguished.
    component_names = [name for name in TIMING_METRICS if any(name in r for r in fedprox_runs)]
    if component_names:
        labels = {
            "local_training_seconds": "Local training",
            "communication_seconds": "Communication",
            "aggregation_seconds": "Aggregation",
            "validation_seconds": "Validation",
        }
        for mu in mus:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            subset = [run for run in fedprox_runs if run["mu"] == mu]
            plotted = False
            for tls_value, label in ((False, "Without TLS"), (True, "With TLS")):
                group = [run for run in subset if run["tls"] == tls_value]
                if not group:
                    continue
                means = [
                    float(np.mean([run[name] for run in group if name in run]))
                    if any(name in run for run in group) else 0.0
                    for name in component_names
                ]
                ax.plot(
                    [labels[name] for name in component_names], means,
                    marker="o", linewidth=1.8, label=label,
                )
                plotted = True
            if plotted:
                ax.set_title(f"FedProx Timing Components (μ={mu:g})")
                ax.set_ylabel("Mean time (seconds)")
                ax.grid(axis="y", alpha=0.25)
                ax.legend()
                save_figure(fig, f"fedprox_mu_{mu:g}_timing_components.png")

    # Export the raw timed FedProx observations for traceability.
    csv_path = PLOTS_DIR / "fedprox_timing_runs.csv"
    columns = [
        "mu", "transport", "timestamp", "fedprox_seconds",
        "experiment_total_seconds", *TIMING_METRICS, "metrics_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for run in fedprox_runs:
            row = dict(run)
            row["transport"] = "tls" if run["tls"] else "no_tls"
            writer.writerow(row)
    print(f"Saved FedProx timing data: {csv_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    runs_by_transport = {
        transport: load_runs(transport)
        for transport in TRANSPORTS
    }

    if all(runs_by_transport[transport] for transport in TRANSPORTS):
        summaries = {
            transport: summarize(
                runs_by_transport[transport],
                (*TIMING_METRICS, "fedavg_seconds", "experiment_total_seconds"),
            )
            for transport in TRANSPORTS
        }
        write_summary(runs_by_transport, summaries)
        plot_component_breakdown(summaries)
        plot_total_times(
            runs_by_transport, summaries, "fedavg_seconds",
            "FedAvg Total Time: With vs Without TLS",
            "tls_vs_no_tls_fedavg_total_time.png",
        )
        plot_total_times(
            runs_by_transport, summaries, "experiment_total_seconds",
            "Experiment Total Time: With vs Without TLS",
            "tls_vs_no_tls_experiment_total_time.png",
        )
        plot_individual_runs(
            runs_by_transport, "fedavg_seconds",
            "FedAvg Time Across Repeated Runs",
            "tls_vs_no_tls_fedavg_runs.png",
        )
        plot_individual_runs(
            runs_by_transport, "communication_seconds",
            "Communication Time Across Repeated Runs",
            "tls_vs_no_tls_communication_runs.png",
        )
    else:
        print(
            "Skipping FedAvg TLS comparison plots: timed runs are required "
            "for both no_tls and tls."
        )

    fedprox_runs = load_fedprox_runs()
    plot_fedprox_mu(fedprox_runs)

    print(f"\nPlots and summaries written to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
