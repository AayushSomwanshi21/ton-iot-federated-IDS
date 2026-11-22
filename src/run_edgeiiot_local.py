import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data.edgeiiot import load_and_prepare_edgeiiot
from src.models.binary_classifier import BinaryClassifier


# Match the centralized baseline.
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001
THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "edgeiiot_local"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Using device:", device)
    data = load_and_prepare_edgeiiot()

    X_val = torch.tensor(
        np.asarray(data["X_val_scaled"]).copy(),
        dtype=torch.float32,
        device=device,
    )
    y_val = np.asarray(data["y_val"]).reshape(-1)

    X_test = torch.tensor(
        np.asarray(data["X_test_scaled"]).copy(),
        dtype=torch.float32,
        device=device,
    )
    y_test = np.asarray(data["y_test"]).reshape(-1)

    # Use the same class weight as the FedAvg and centralized experiments.
    all_train_labels = np.concatenate([
        np.asarray(client["y_train"]).reshape(-1)
        for client in data["clients"].values()
    ])
    num_positive = int((all_train_labels == 1).sum())
    num_negative = int((all_train_labels == 0).sum())

    pos_weight = torch.tensor(
        [num_negative / num_positive],
        dtype=torch.float32,
        device=device,
    )

    client_results = []

    for client_id, client in data["clients"].items():
        print(f"\nTraining local model for client {client_id}")

        X_train = torch.tensor(
            np.asarray(client["X_train_scaled"]).copy(),
            dtype=torch.float32,
        )
        y_train = torch.tensor(
            np.asarray(client["y_train"]).reshape(-1).copy(),
            dtype=torch.float32,
        )

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        model = BinaryClassifier(input_dim=X_train.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        best_val_f1 = -1.0
        best_epoch = 0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            model.train()

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_X).reshape(-1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(
                    model(X_val).reshape(-1)
                ).cpu().numpy()
                val_preds = (val_probs >= THRESHOLD).astype(int)

            val_f1 = f1_score(y_val, val_preds, zero_division=0)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        # Restore this client's best validation checkpoint.
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            test_probs = torch.sigmoid(
                model(X_test).reshape(-1)
            ).cpu().numpy()
            test_preds = (test_probs >= THRESHOLD).astype(int)

        result = {
            "client_id": int(client_id),
            "train_size": int(len(y_train)),
            "selected_epoch": best_epoch,
            "best_validation_f1": float(best_val_f1),
            "test_accuracy": float(accuracy_score(y_test, test_preds)),
            "test_precision": float(
                precision_score(y_test, test_preds, zero_division=0)
            ),
            "test_recall": float(
                recall_score(y_test, test_preds, zero_division=0)
            ),
            "test_f1": float(
                f1_score(y_test, test_preds, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                y_test, test_preds
            ).tolist(),
        }
        client_results.append(result)

        print(
            f"Client {client_id}: "
            f"val F1={best_val_f1:.4f}, "
            f"test F1={result['test_f1']:.4f}"
        )

    test_f1_scores = [result["test_f1"] for result in client_results]

    summary = {
        "threshold": THRESHOLD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "pos_weight": float(pos_weight.item()),
        "mean_test_f1": float(np.mean(test_f1_scores)),
        "std_test_f1": float(np.std(test_f1_scores)),
        "std_test_f1_ddof1": float(np.std(test_f1_scores, ddof=1)),
        "clients": client_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nLocal-client summary")
    print(f"Mean test F1: {summary['mean_test_f1']:.4f}")
    print(f"Test F1 std (population): {summary['std_test_f1']:.4f}")
    print(f"Test F1 std (sample): {summary['std_test_f1_ddof1']:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
