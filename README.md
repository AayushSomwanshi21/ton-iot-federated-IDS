# TON IoT Federated IDS

Project workspace for an IoT intrusion detection system using federated learning.

## Project layout

- `data/`: local datasets (not committed)
- `notebooks/`: exploration and experiments
- `src/data/`: data loading and preprocessing
- `src/models/`: device encoders and shared classifier
- `src/clients/`: client-side training
- `src/server/`: coordination and aggregation
- `src/security/`: TLS, secure aggregation, differential privacy, and robustness
- `tests/`: automated tests
- `outputs/`: local results (not committed)

## Getting started

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
