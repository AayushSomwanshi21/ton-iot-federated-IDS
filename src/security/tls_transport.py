import struct
import socket


def send_bytes(connection: socket.socket, payload: bytes) -> None:
    """Send one byte payload, preceded by its 8-byte length."""
    header = struct.pack("!Q", len(payload))
    connection.sendall(header + payload)


def recv_exact(connection: socket.socket, size: int) -> bytes:
    """Receive exactly size bytes or raise if the connection closes early."""
    chunks = []
    remaining = size

    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError(
                "Connection closed before payload was complete")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def recv_bytes(connection: socket.socket, max_size: int = 100_000_000) -> bytes:
    """Receive one length-prefixed payload, with a size limit."""
    header = recv_exact(connection, 8)
    (payload_size,) = struct.unpack("!Q", header)

    if payload_size > max_size:
        raise ValueError(f"Payload too large: {payload_size} bytes")

    return recv_exact(connection, payload_size)
