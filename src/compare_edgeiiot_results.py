import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "edgeiiot_comparison"

RESULT_FILES = {
    "FedAvg": PROJECT_ROOT / "outputs" / "edgeiiot_fedavg" / "metrics.json",
    "Centralized": PROJECT_ROOT / "outputs" / "edgeiiot_centralized" / "metrics.json",
    "Local clients": PROJECT_ROOT / "outputs" / "edgeiiot_local" / "metrics.json",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    fedavg = load_json(RESULT_FILES["FedAvg"])
    centralized = load_json(RESULT_FILES["Centralized"])
    local = load_json(RESULT_FILES["Local clients"])

    # One summary row per approach, plus one row per local client.
    rows = [
        {
            "model": "FedAvg",
            "client_id": "",
            "test_f1": fedavg["test_f1"],
            "test_precision": fedavg["test_precision"],
            "test_recall": fedavg["test_recall"],
            "test_accuracy": fedavg["test_accuracy"],
        },
        {
            "model": "Centralized",
            "client_id": "",
            "test_f1": centralized["test_f1"],
            "test_precision": centralized["test_precision"],
            "test_recall": centralized["test_recall"],
            "test_accuracy": centralized["test_accuracy"],
        },
        {
            "model": "Local clients (mean)",
            "client_id": "",
            "test_f1": local["mean_test_f1"],
            "test_precision": "",
            "test_recall": "",
            "test_accuracy": "",
        },
        {
            "model": "Local clients (F1 std, population)",
            "client_id": "",
            "test_f1": local["std_test_f1"],
            "test_precision": "",
            "test_recall": "",
            "test_accuracy": "",
        },
    ]

    for client in local["clients"]:
        rows.append({
            "model": "Local client",
            "client_id": client["client_id"],
            "test_f1": client["test_f1"],
            "test_precision": client["test_precision"],
            "test_recall": client["test_recall"],
            "test_accuracy": client["test_accuracy"],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "test_split": "Same deterministic Edge-IIoTset split from load_and_prepare_edgeiiot",
        "threshold": 0.5,
        "fedavg": {
            "test_f1": fedavg["test_f1"],
            "test_precision": fedavg["test_precision"],
            "test_recall": fedavg["test_recall"],
            "test_accuracy": fedavg["test_accuracy"],
            "selected_round": fedavg["selected_round"],
        },
        "centralized": {
            "test_f1": centralized["test_f1"],
            "test_precision": centralized["test_precision"],
            "test_recall": centralized["test_recall"],
            "test_accuracy": centralized["test_accuracy"],
            "selected_epoch": centralized["selected_epoch"],
        },
        "local_clients": {
            "mean_test_f1": local["mean_test_f1"],
            "std_test_f1_population": local["std_test_f1"],
            "std_test_f1_sample": local["std_test_f1_ddof1"],
            "per_client": local["clients"],
        },
    }

    json_path = OUTPUT_DIR / "comparison.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
