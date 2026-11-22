from src.server.fedavg import train_fedavg
from src.models.binary_classifier import BinaryClassifier
from src.dataset.edgeiiot import load_and_prepare_edgeiiot
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import torch
import numpy as np
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------
# Configuration
# --------------------
NUM_ROUNDS = 10
LOCAL_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.001
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FedAvg for Edge-IIoTset."
    )
    parser.add_argument(
        "--tls",
        action="store_true",
        help="Send each client model update through the TLS server.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Measure training and communication time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_start = time.perf_counter()
    print("Using device:", device)
    print("TLS:", "enabled" if args.tls else "disabled")
    print("Timing:", "enabled" if args.timing else "disabled")

    transport_name = "tls" if args.tls else "no_tls"
    if args.timing:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / "benchmarks"
            / f"edgeiiot_fedavg_{transport_name}"
            / run_stamp
        )
    else:
        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / f"edgeiiot_fedavg_{transport_name}"
        )

    checkpoint_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "metrics.json"

    # Load, split, scale, and partition the data.
    data = load_and_prepare_edgeiiot()
    print("Number of features:", len(data["feature_names"]))
    print("Features:", data["feature_names"])
    clients = data["clients"]
    X_val = data["X_val_scaled"]
    y_val = data["y_val"]
    X_test = data["X_test_scaled"]
    y_test = data["y_test"]

    input_dim = X_val.shape[1]
    global_model = BinaryClassifier(input_dim=input_dim).to(device)

    # Preserve the class-weighting approach used in the earlier experiment.
    all_client_labels = np.concatenate([
        np.asarray(client["y_train"]).reshape(-1)
        for client in clients.values()
    ])
    num_positive = int((all_client_labels == 1).sum())
    num_negative = int((all_client_labels == 0).sum())

    if num_positive == 0 or num_negative == 0:
        raise ValueError("Training data must contain both label classes.")

    pos_weight = torch.tensor(
        [num_negative / num_positive],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train and select the checkpoint using validation F1.
    result = train_fedavg(
        global_model=global_model,
        clients=clients,
        X_val=X_val,
        y_val=y_val,
        criterion=criterion,
        device=device,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        threshold=THRESHOLD,
        checkpoint_path=checkpoint_path,
        use_tls=args.tls,
        measure_timing=args.timing,
    )

    best_model = result["model"]
    best_model.eval()

    # Evaluate test data once, after selecting the best validation checkpoint.
    X_test_tensor = torch.tensor(
        np.asarray(X_test).copy(),
        dtype=torch.float32,
        device=device,
    )
    y_test_array = np.asarray(y_test).reshape(-1)

    with torch.no_grad():
        test_logits = best_model(X_test_tensor).reshape(-1)
        test_probs = torch.sigmoid(test_logits)
        test_preds = (
            test_probs >= THRESHOLD
        ).int().cpu().numpy().reshape(-1)

    metrics = {
        "selected_round": int(result["best_round"]),
        "best_validation_f1": float(result["best_val_f1"]),
        "test_accuracy": float(accuracy_score(y_test_array, test_preds)),
        "test_precision": float(
            precision_score(y_test_array, test_preds, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test_array, test_preds, zero_division=0)
        ),
        "test_f1": float(f1_score(y_test_array, test_preds, zero_division=0)),
        "threshold": THRESHOLD,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "pos_weight": float(pos_weight.item()),
        "confusion_matrix": confusion_matrix(
            y_test_array, test_preds
        ).tolist(),
        "validation_history": result["history"],
        "transport": "tls" if args.tls else "no_tls",
        "timing_enabled": args.timing,
    }

    if args.timing:
        metrics["timing"] = {
            **result["timing"],
            "experiment_total_seconds": time.perf_counter() - experiment_start,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nBest validation round:", result["best_round"])
    print(f"Best validation F1: {result['best_val_f1']:.4f}")
    print("\nTest metrics:")
    for name, value in metrics.items():
        if name != "validation_history":
            print(f"{name}: {value}")

    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print(f"Metrics saved to:    {metrics_path}")


if __name__ == "__main__":
    main()
