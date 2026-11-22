from datetime import datetime, timedelta, timezone
from pathlib import Path
import ipaddress

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


CERT_DIR = Path("local_certs")
CERT_DIR.mkdir(exist_ok=True)

# Generate a private key for this local test server.
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
])

now = datetime.now(timezone.utc)

certificate = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(private_key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now)
    .not_valid_after(now + timedelta(days=365))
    .add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]),
        critical=False,
    )
    .sign(private_key, hashes.SHA256())
)

(CERT_DIR / "server-key.pem").write_bytes(
    private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
)

(CERT_DIR / "server-cert.pem").write_bytes(
    certificate.public_bytes(serialization.Encoding.PEM)
)

print("Created local_certs/server-cert.pem")
print("Created local_certs/server-key.pem")
