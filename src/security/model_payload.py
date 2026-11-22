import io

import torch


def serialize_state_dict(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Convert a model state_dict into bytes for transport."""
    cpu_state = {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
    }

    buffer = io.BytesIO()
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def deserialize_state_dict(payload: bytes) -> dict[str, torch.Tensor]:
    """Load model weights from bytes onto CPU."""
    buffer = io.BytesIO(payload)

    state_dict = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state_dict, dict):
        raise TypeError("Expected a state_dict dictionary")

    if not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state_dict.items()
    ):
        raise TypeError("Payload contains unexpected state_dict values")

    return state_dict
