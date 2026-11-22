import socket
import ssl

from src.security.model_payload import (
    deserialize_state_dict,
    serialize_state_dict,
)
from src.security.tls_transport import recv_bytes, send_bytes

HOST = "127.0.0.1"
PORT = 8443
CA_CERT = "local_certs/server-cert.pem"


def tls_round_trip_state_dict(state_dict):
    """Send a model state_dict over TLS and return the server's echoed copy."""
    payload = serialize_state_dict(state_dict)

    context = ssl.create_default_context(cafile=CA_CERT)

    with socket.create_connection((HOST, PORT)) as connection:
        with context.wrap_socket(
            connection,
            server_hostname="localhost",
        ) as tls_connection:
            send_bytes(tls_connection, payload)
            returned_payload = recv_bytes(tls_connection)

    return deserialize_state_dict(returned_payload)
