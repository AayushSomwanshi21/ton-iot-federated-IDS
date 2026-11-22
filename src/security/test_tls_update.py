import torch
from torch import nn

from src.security.tls_update_client import tls_round_trip_state_dict


def main():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

    original_state = model.state_dict()
    returned_state = tls_round_trip_state_dict(original_state)

    assert original_state.keys() == returned_state.keys()

    for name in original_state:
        assert torch.equal(original_state[name], returned_state[name]), (
            f"Tensor mismatch: {name}"
        )

    print("TLS update round-trip passed; all model tensors match.")


if __name__ == "__main__":
    main()
