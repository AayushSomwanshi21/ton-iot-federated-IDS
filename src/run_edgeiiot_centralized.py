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


# Match the FedAvg experiment.
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001
THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "edgeiiot_centralized"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Using device:", device)

    data = load_and_prepare_edgeiiot()

    X_train = torch.tensor(
        np.asarray(data["X_train_scaled"]).copy(),
        dtype=torch.float32,
    )
    y_train = torch.tensor(
        np.asarray(data["y_train"]).reshape(-1).copy(),
        dtype=torch.float32,
    )

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

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    num_positive = int((y_train == 1).sum().item())
    num_negative = int((y_train == 0).sum().item())
    pos_weight = torch.tensor(
        [num_negative / num_positive],
        dtype=torch.float32,
        device=device,
    )

    model = BinaryClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    history = []

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
            val_logits = model(X_val).reshape(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= THRESHOLD).astype(int)

        val_f1 = f1_score(y_val, val_preds, zero_division=0)
        history.append({"epoch": epoch, "validation_f1": float(val_f1)})

        print(f"Epoch {epoch}/{EPOCHS} — validation F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model(X_test).reshape(-1)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs >= THRESHOLD).astype(int)

    metrics = {
        "selected_epoch": best_epoch,
        "best_validation_f1": float(best_val_f1),
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "threshold": THRESHOLD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "pos_weight": float(pos_weight.item()),
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "validation_history": history,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CHECKPOINT_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nTest metrics:")
    for name, value in metrics.items():
        if name != "validation_history":
            print(f"{name}: {value}")

    print(f"\nCheckpoint saved to: {CHECKPOINT_PATH}")
    print(f"Metrics saved to:    {METRICS_PATH}")


if __name__ == "__main__":
    main()
