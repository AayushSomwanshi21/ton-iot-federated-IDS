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
    proximal_mu=0.0,
):
    """
    Train one client's copy of the global model.

    If proximal_mu > 0, use the FedProx proximal regularization term.
    If proximal_mu = 0, training is equivalent to the existing FedAvg
    local training procedure.

    Returns:
        client_state: CPU model state dictionary
        num_examples: number of local training examples
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

    # Save the global model parameters before local training.
    # These remain fixed throughout this client's local training.
    global_params = [
        param.detach().clone()
        for param in model.parameters()
    ]

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

            # Original classification loss
            loss = criterion(logits, batch_y)

            # FedProx proximal regularization
            if proximal_mu > 0:
                proximal_term = torch.zeros(
                    1,
                    device=device,
                )

                for local_param, global_param in zip(
                    model.parameters(),
                    global_params,
                ):
                    proximal_term += torch.sum(
                        (local_param - global_param) ** 2
                    )

                loss = loss + (
                    proximal_mu / 2.0
                ) * proximal_term

            loss.backward()
            optimizer.step()

    # Return CPU copies so client models can be aggregated safely.
    client_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    return client_state, len(X_tensor)
