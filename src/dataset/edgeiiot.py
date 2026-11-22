from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "ML-EdgeIIoT-dataset.csv"


def load_and_prepare_edgeiiot(
    csv_path=DEFAULT_CSV_PATH,
    num_clients=7,
    alpha=5,
    random_state=42,
):
    """
    Load Edge-IIoTset, make duplicate-group train/validation/test splits,
    partition the training data among clients, and scale features.

    Returns a dictionary containing the splits, scaled arrays, clients,
    scaler, and feature/label metadata.
    """
    # Load data
    df = pd.read_csv(csv_path)

    required_columns = {"Attack_label", "Attack_type"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}")

    # Target and numeric input features.
    y = df["Attack_label"].astype("int64")

    X_numeric = df.drop(columns=["Attack_label", "Attack_type"]).select_dtypes(
        include=[np.number]
    )

    # Remove constant columns.
    constant_cols = X_numeric.columns[
        X_numeric.nunique(dropna=False) <= 1
    ].tolist()
    X = X_numeric.drop(columns=constant_cols).copy()

    if X.empty:
        raise ValueError(
            "No numeric feature columns remain after removing constants.")

    # Keep exact duplicate rows together across the splits.
    groups = pd.util.hash_pandas_object(df, index=False)

    # First split: 70% train, 30% temporary.
    split1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state,
    )
    train_idx, temp_idx = next(split1.split(X, y, groups=groups))

    # Second split: split temporary data equally into validation and test.
    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state,
    )
    val_rel_idx, test_rel_idx = next(
        split2.split(
            X.iloc[temp_idx],
            y.iloc[temp_idx],
            groups=groups.iloc[temp_idx],
        )
    )

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    X_test, y_test = X.iloc[test_idx].copy(), y.iloc[test_idx].copy()

    # Dirichlet partition of the training set among clients.
    rng = np.random.default_rng(random_state)
    y_train_array = y_train.to_numpy()
    client_positions = [[] for _ in range(num_clients)]

    for label in np.unique(y_train_array):
        label_positions = np.where(y_train_array == label)[0]
        rng.shuffle(label_positions)

        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = rng.multinomial(len(label_positions), proportions)

        start = 0
        for client_id, count in enumerate(counts):
            client_positions[client_id].extend(
                label_positions[start: start + count]
            )
            start += count

    clients = {}
    for client_id, positions in enumerate(client_positions):
        positions = np.asarray(positions, dtype=int)
        rng.shuffle(positions)

        clients[client_id] = {
            "X_train": X_train.iloc[positions].copy(),
            "y_train": y_train.iloc[positions].copy(),
        }

    # Fit scaler on training data only, then transform all splits and clients.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    for client in clients.values():
        client["X_train_scaled"] = scaler.transform(client["X_train"])

    # Verify no exact duplicate group crosses split boundaries.
    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    test_groups = set(groups.iloc[test_idx])

    duplicate_group_overlap = {
        "train_validation": len(train_groups & val_groups),
        "train_test": len(train_groups & test_groups),
        "validation_test": len(val_groups & test_groups),
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "clients": clients,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
        "constant_columns_removed": constant_cols,
        "duplicate_group_overlap": duplicate_group_overlap,
        "alpha": alpha,
        "num_clients": num_clients,
    }
