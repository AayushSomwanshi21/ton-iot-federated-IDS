import socket
import ssl

import torch
from torch import nn

from src.security.model_payload import serialize_state_dict
from src.security.tls_transport import recv_bytes, send_bytes

HOST = "127.0.0.1"
PORT = 8443
CA_CERT = "local_certs/server-cert.pem"

# Sample model for testing the transport; not the project model yet.
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

payload = serialize_state_dict(model.state_dict())
print(f"Sending model payload: {len(payload)} bytes")

context = ssl.create_default_context(cafile=CA_CERT)

with socket.create_connection((HOST, PORT)) as connection:
    with context.wrap_socket(connection, server_hostname="localhost") as tls_connection:
        send_bytes(tls_connection, payload)
        response = recv_bytes(tls_connection)
        print("Server response:", response.decode("utf-8"))
