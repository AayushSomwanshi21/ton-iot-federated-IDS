import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_client(
    model,
    X_train,
    y_train,
    criterion,
    device,
    local_epochs=1,
    batch_size=256,
    learning_rate=0.001,
):
    """
    Train one client's copy of the model and return its state dict
    and number of training examples.
    """
    X_tensor = torch.tensor(
        X_train.copy(),
        dtype=torch.float32,
    )
    y_tensor = torch.tensor(
        y_train.to_numpy(copy=True).reshape(-1, 1),
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    for _ in range(local_epochs):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    # Return CPU copies so client models can be aggregated safely.
    client_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    return client_state, len(X_tensor)
