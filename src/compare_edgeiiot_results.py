import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_ROOT / "plots"

EXPERIMENTS = {
    ("FedAvg", 10, 1): OUTPUT_ROOT / "edgeiiot_fedavg_10_rounds_local_epoch_1",
    ("FedAvg", 10, 3): OUTPUT_ROOT / "edgeiiot_fedavg_10_rounds_local_epochs_3",
    ("FedAvg", 10, 5): OUTPUT_ROOT / "edgeiiot_fedavg_10_rounds_epochs_5",
    ("FedAvg", 20, 1): OUTPUT_ROOT / "edgeiiot_fedavg_20_rounds_local_epoch_1",
    ("FedAvg", 20, 3): OUTPUT_ROOT / "edgeiiot_fedavg_20_rounds_local_epoch_3",
    ("FedAvg", 20, 5): OUTPUT_ROOT / "edgeiiot_fedavg_20_rounds_local_epoch_5",
    ("Centralized", 10, 10): OUTPUT_ROOT / "edgeiiot_centralized_10_rounds_epochs_10",
    ("Centralized", 10, 30): OUTPUT_ROOT / "edgeiiot_centralized_10_rounds_epochs_30",
    ("Centralized", 10, 50): OUTPUT_ROOT / "edgeiiot_centralized_10_rounds_epochs_50",
    ("Centralized", 20, 20): OUTPUT_ROOT / "edgeiiot_centralized_20_rounds_epochs_20",
    ("Centralized", 20, 60): OUTPUT_ROOT / "edgeiiot_centralized_20_rounds_epochs_60",
    ("Centralized", 20, 100): OUTPUT_ROOT / "edgeiiot_centralized_20_rounds_local_epoch_100",
    ("Local", 10, 10): OUTPUT_ROOT / "edgeiiot_local_10_rounds_epochs_10",
    ("Local", 10, 30): OUTPUT_ROOT / "edgeiiot_local_10_rounds_epochs_30",
    ("Local", 10, 50): OUTPUT_ROOT / "edgeiiot_local_10_rounds_epochs_50",
    ("Local", 20, 20): OUTPUT_ROOT / "edgeiiot_local_20_rounds_epochs_20",
    ("Local", 20, 60): OUTPUT_ROOT / "edgeiiot_local_20_rounds_epochs_60",
    ("Local", 20, 100): OUTPUT_ROOT / "edgeiiot_local_20_rounds_epochs_100",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_figure(fig, name):
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def extract_result(approach, rounds, epoch_level, directory):
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        print(f"Missing: {metrics_path}")
        return None

    metrics = load_json(metrics_path)
    result = {
        "approach": approach,
        "rounds": rounds,
        "configured_epochs": epoch_level,
        "total_training_epochs": (
            rounds * epoch_level if approach == "FedAvg" else epoch_level
        ),
    }

    if approach == "FedAvg":
        result["test_f1"] = metrics["test_f1"]
        result["test_accuracy"] = metrics["test_accuracy"]
        result["test_precision"] = metrics["test_precision"]
        result["test_recall"] = metrics["test_recall"]
        result["best_validation_f1"] = metrics["best_validation_f1"]
        result["validation_history"] = metrics["validation_history"]
    elif approach == "Centralized":
        result["test_f1"] = metrics["test_f1"]
        result["test_accuracy"] = metrics["test_accuracy"]
        result["test_precision"] = metrics["test_precision"]
        result["test_recall"] = metrics["test_recall"]
        result["best_validation_f1"] = metrics["best_validation_f1"]
        result["validation_history"] = metrics["validation_history"]
    else:
        result["test_f1"] = metrics["mean_test_f1"]
        result["test_f1_std"] = metrics["std_test_f1"]
        accuracies = [
            client["test_accuracy"] for client in metrics["clients"]
        ]
        result["test_accuracy"] = float(np.mean(accuracies))
        result["test_accuracy_std"] = float(np.std(accuracies))
        result["best_validation_f1"] = float(np.mean([
            client["best_validation_f1"] for client in metrics["clients"]
        ]))

    return result


def write_summary_csv(results):
    columns = [
        "approach",
        "rounds",
        "configured_epochs",
        "total_training_epochs",
        "test_f1",
        "test_f1_std",
        "test_accuracy",
        "test_accuracy_std",
        "test_precision",
        "test_recall",
        "best_validation_f1",
    ]
    path = PLOTS_DIR / "experiment_comparison_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({
                key: result.get(key, "")
                for key in columns
            })
    print(f"Saved summary: {path}")


def plot_fedavg_rounds(results, metric="test_f1"):
    fedavg = [r for r in results if r["approach"] == "FedAvg"]
    epoch_levels = [1, 3, 5]
    x = np.arange(len(epoch_levels))
    width = 0.35
    metric_label = "Test F1" if metric == "test_f1" else "Test Accuracy"
    slug = "test_f1" if metric == "test_f1" else "test_accuracy"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, rounds in zip((-width / 2, width / 2), (10, 20)):
        scores = []
        for epoch_level in epoch_levels:
            match = next(
                (r for r in fedavg
                 if r["rounds"] == rounds and r["configured_epochs"] == epoch_level),
                None,
            )
            scores.append(match[metric] if match else np.nan)
        ax.bar(
            x + offset,
            scores,
            width,
            label=f"{rounds} rounds",
        )

    ax.set_title(f"FedAvg {metric_label} by Communication Rounds and Local Epochs")
    ax.set_xlabel("Local epochs")
    ax.set_ylabel(metric_label)
    ax.set_xticks(x, epoch_levels)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, f"fedavg_rounds_and_local_epochs_{slug}.png")


def plot_fedavg_validation_history(results):
    fedavg = [r for r in results if r["approach"] == "FedAvg"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for rounds in (10, 20):
        for epoch_level in (1, 3, 5):
            match = next(
                (r for r in fedavg
                 if r["rounds"] == rounds and r["configured_epochs"] == epoch_level),
                None,
            )
            if not match:
                continue
            history = match["validation_history"]
            ax.plot(
                [item["round"] for item in history],
                [item["val_f1"] for item in history],
                marker="o",
                linewidth=1.5,
                label=f"{rounds} rounds, {epoch_level} local epoch(s)",
            )

    ax.set_title("FedAvg Validation F1 Across Communication Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, "fedavg_validation_f1_by_round.png")


def plot_approach_comparison(results, rounds, metric="test_f1"):
    epoch_configs = {
        10: [(1, 10), (3, 30), (5, 50)],
        20: [(1, 20), (3, 60), (5, 100)],
    }[rounds]
    metric_label = "Test F1" if metric == "test_f1" else "Test Accuracy"
    slug = "test_f1" if metric == "test_f1" else "test_accuracy"
    std_key = "test_f1_std" if metric == "test_f1" else "test_accuracy_std"

    labels = [f"{local_epochs} local epoch(s)" for local_epochs, _ in epoch_configs]
    x = np.arange(len(labels))
    width = 0.25
    offset = {"FedAvg": -width, "Centralized": 0, "Local": width}

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for approach in ("FedAvg", "Centralized", "Local"):
        scores = []
        errors = []
        for local_epochs, total_epochs in epoch_configs:
            configured_epochs = local_epochs if approach == "FedAvg" else total_epochs
            match = next(
                (r for r in results
                 if r["approach"] == approach
                 and r["rounds"] == rounds
                 and r["configured_epochs"] == configured_epochs),
                None,
            )
            scores.append(match[metric] if match else np.nan)
            errors.append(match.get(std_key, 0) if match else 0)

        ax.bar(
            x + offset[approach],
            scores,
            width,
            yerr=errors if approach == "Local" else None,
            capsize=4,
            label=approach,
        )

    ax.set_title(
        f"Local, Centralized, and FedAvg {metric_label} ({rounds} Rounds)"
    )
    ax.set_xlabel("Equivalent training configuration")
    ax.set_ylabel(metric_label)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(
        fig,
        f"approach_comparison_{rounds}_rounds_{slug}.png",
    )


def plot_centralized_and_local(results, metric="test_f1"):
    metric_label = "Test F1" if metric == "test_f1" else "Test Accuracy"
    slug = "test_f1" if metric == "test_f1" else "test_accuracy"

    for approach in ("Centralized", "Local"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for rounds in (10, 20):
            subset = [
                r for r in results
                if r["approach"] == approach and r["rounds"] == rounds
            ]
            subset.sort(key=lambda r: r["configured_epochs"])
            if not subset:
                continue
            ax.plot(
                [r["configured_epochs"] for r in subset],
                [r[metric] for r in subset],
                marker="o",
                linewidth=2,
                label=f"{rounds} rounds",
            )

        ax.set_title(f"{approach} {metric_label} by Total Training Epochs")
        ax.set_xlabel("Total training epochs")
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        save_figure(
            fig,
            f"{approach.lower()}_total_epochs_{slug}.png",
        )


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for (approach, rounds, epoch_level), directory in EXPERIMENTS.items():
        result = extract_result(
            approach,
            rounds,
            epoch_level,
            directory,
        )
        if result:
            results.append(result)

    if not results:
        raise RuntimeError("No metrics.json experiment files were found.")

    write_summary_csv(results)
    plot_fedavg_rounds(results)
    plot_fedavg_rounds(results, metric="test_accuracy")
    plot_fedavg_validation_history(results)
    plot_approach_comparison(results, 10)
    plot_approach_comparison(results, 20)
    plot_approach_comparison(results, 10, metric="test_accuracy")
    plot_approach_comparison(results, 20, metric="test_accuracy")
    plot_centralized_and_local(results)
    plot_centralized_and_local(results, metric="test_accuracy")

    print(f"\nFound {len(results)} experiment result(s).")
    print(f"Graphs and summary written to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
