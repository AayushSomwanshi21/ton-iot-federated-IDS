import socket
import ssl

from src.security.model_payload import deserialize_state_dict
from src.security.tls_transport import recv_bytes, send_bytes

HOST = "127.0.0.1"
PORT = 8443

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(
    certfile="local_certs/server-cert.pem",
    keyfile="local_certs/server-key.pem",
)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()
    print(f"TLS update server listening on {HOST}:{PORT}")

    while True:
        connection, address = server.accept()

        with connection:
            with context.wrap_socket(connection, server_side=True) as tls_connection:
                try:
                    payload = recv_bytes(tls_connection)
                    state_dict = deserialize_state_dict(payload)

                    print(
                        f"Received update from {address}: "
                        f"{len(payload)} bytes, "
                        f"{len(state_dict)} tensors"
                    )

                    # Echo the validated payload so the sender can confirm
                    # the update made the round trip over TLS.
                    send_bytes(tls_connection, payload)

                except Exception as error:
                    print(f"Rejected update from {address}: {error}")
