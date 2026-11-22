import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from src.security.tls_update_client import tls_round_trip_state_dict
from src.clients.train import train_client


def _to_feature_tensor(values, device):
    """Convert a feature array or DataFrame to a float tensor."""
    return torch.tensor(
        np.asarray(values).copy(),
        dtype=torch.float32,
        device=device,
    )


def _to_label_array(values):
    """Convert a Series or array of labels to a 1D NumPy array."""
    if hasattr(values, "to_numpy"):
        values = values.to_numpy(copy=True)

    return np.asarray(values).reshape(-1)


def _weighted_average_states(client_states, client_sizes):
    """Average client state dicts, weighted by each client's row count."""
    total_rows = sum(client_sizes)
    if total_rows <= 0:
        raise ValueError("Client training sets must contain at least one row.")

    averaged_state = {}

    for key in client_states[0]:
        first_tensor = client_states[0][key]

        # Model parameters and floating buffers are averaged.
        if torch.is_floating_point(first_tensor):
            averaged_state[key] = sum(
                client_states[i][key] * (client_sizes[i] / total_rows)
                for i in range(len(client_states))
            )
        else:
            # Keep non-floating buffers from the first client state.
            averaged_state[key] = first_tensor.clone()

    return averaged_state


def train_fedavg(
    global_model,
    clients,
    X_val,
    y_val,
    criterion,
    device,
    num_rounds=5,
    local_epochs=1,
    batch_size=256,
    learning_rate=0.001,
    threshold=0.5,
    checkpoint_path=None,
):
    """
    Run FedAvg and select the model with the highest validation F1.

    Args:
        global_model: Initialized PyTorch model.
        clients: Dict keyed by client ID. Each value must contain
            'X_train_scaled' and 'y_train'.
        X_val, y_val: Validation features and binary labels.
        criterion: Loss function, e.g. BCEWithLogitsLoss.
        device: torch.device.
        checkpoint_path: Optional path to save the best validation checkpoint.

    Returns:
        A dict with the restored best model, selected round, validation
        metrics, and per-round history.
    """
    if not clients:
        raise ValueError("No clients were provided.")

    device = torch.device(device)
    global_model = global_model.to(device)

    X_val_tensor = _to_feature_tensor(X_val, device)
    y_val_array = _to_label_array(y_val)

    if len(X_val_tensor) != len(y_val_array):
        raise ValueError(
            "Validation features and labels have different lengths.")

    best_val_f1 = -1.0
    best_round = None
    best_state = None
    history = []

    for round_num in range(1, num_rounds + 1):
        client_states = []
        client_sizes = []

        for client_id, client in clients.items():
            if "X_train_scaled" not in client or "y_train" not in client:
                raise KeyError(
                    f"Client {client_id} must contain "
                    "'X_train_scaled' and 'y_train'."
                )

            # Each client starts from the same current global model.
            local_model = copy.deepcopy(global_model).to(device)

            client_state, client_size = train_client(
                model=local_model,
                X_train=client["X_train_scaled"],
                y_train=client["y_train"],
                criterion=criterion,
                device=device,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            # Send the trained client update through the TLS server before aggregation.
            client_state = tls_round_trip_state_dict(client_state)
            client_states.append(client_state)
            client_sizes.append(client_size)

            del local_model

        # Aggregate client models using their training-set sizes.
        aggregated_state = _weighted_average_states(
            client_states,
            client_sizes,
        )
        global_model.load_state_dict(aggregated_state)
        global_model.to(device)

        # Validation is used to select the checkpoint.
        global_model.eval()
        with torch.no_grad():
            val_logits = global_model(X_val_tensor).reshape(-1)
            val_probs = torch.sigmoid(val_logits)
            val_preds = (
                (val_probs >= threshold).int().cpu().numpy().reshape(-1)
            )

        val_f1 = f1_score(y_val_array, val_preds, zero_division=0)
        val_precision = precision_score(
            y_val_array, val_preds, zero_division=0
        )
        val_recall = recall_score(
            y_val_array, val_preds, zero_division=0
        )

        round_metrics = {
            "round": round_num,
            "val_f1": float(val_f1),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
        }
        history.append(round_metrics)

        print(
            f"Round {round_num}/{num_rounds} | "
            f"Val F1: {val_f1:.4f} | "
            f"Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_round = round_num
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in global_model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training finished without selecting a checkpoint.")

    # Restore the best validation checkpoint.
    global_model.load_state_dict(best_state)
    global_model.to(device)
    global_model.eval()

    best_metrics = next(
        item for item in history if item["round"] == best_round
    )

    # Optionally save the checkpoint.
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "round": best_round,
                "best_val_f1": best_val_f1,
                "model_state_dict": best_state,
                "history": history,
            },
            checkpoint_path,
        )

    return {
        "model": global_model,
        "best_round": best_round,
        "best_val_f1": best_val_f1,
        "best_validation_metrics": best_metrics,
        "history": history,
    }
