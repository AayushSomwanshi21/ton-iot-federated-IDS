import torch
from torch import nn

from src.security.model_payload import (
    deserialize_state_dict,
    serialize_state_dict,
)


def main():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

    original_state = model.state_dict()
    payload = serialize_state_dict(original_state)
    restored_state = deserialize_state_dict(payload)

    assert original_state.keys() == restored_state.keys()

    for name in original_state:
        assert torch.equal(original_state[name], restored_state[name]), (
            f"Tensor mismatch: {name}"
        )

    print(f"Round-trip passed: {len(payload)} bytes; all tensors match.")


if __name__ == "__main__":
    main()
