# Quantum CoughFreq

Quantum CoughFreq is a hybrid quantum-classical scaffold for respiratory cough screening. The pipeline extracts MFCC and PSD features from cough audio, converts normalized acoustic cues into a 4-qubit quantum state, and now includes a neon dashboard UI for single-file analysis plus dataset-level TB-risk screening.

## Included modules

- `quantum_coughfreq/audio.py`: audio loading, MFCC extraction, PSD analysis, supported-format handling, and frequency-range heuristics.
- `quantum_coughfreq/quantum.py`: Qiskit feature mapping with `ZZFeatureMap` or angle encoding and a `RealAmplitudes` variational circuit.
- `quantum_coughfreq/hybrid.py`: PyTorch CNN + Qiskit hybrid classifier scaffold with cross-entropy training support.
- `quantum_coughfreq/cli.py`: terminal commands for prediction and dataset inspection.
- `quantum_coughfreq/api.py`: FastAPI app exposing the dashboard, single-file analysis API, and dataset-screening API.
- `quantum_coughfreq/web/`: dashboard HTML, CSS, and JavaScript assets.
- `main.py`: local development entrypoint.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py predict sample.wav
```

## Terminal commands

```bash
python main.py predict sample.wav
python main.py predict sample.wav --json
python main.py dataset-summary ./dataset
python main.py classify-dataset "./dataset/QMl data/public_dataset_v3/coughvid_20211012"
python main.py classify-dataset "./dataset/QMl data/public_dataset_v3/coughvid_20211012" --workers 1 --top-k 20
```

Expected dataset layout when you add data later:

```text
dataset/
  Viral/
  Bacterial/
  TB/
```

## Start API later

```bash
python main.py serve --port 8011
```

Then open:

```text
http://127.0.0.1:8011/dashboard
```

## Example request

```bash
curl -X POST "http://127.0.0.1:8011/predict-cough" \
  -F "audio_file=@sample.wav"
```

## Dashboard APIs

```bash
curl http://127.0.0.1:8011/api/dashboard-config

curl -X POST "http://127.0.0.1:8011/api/analyze-cough" \
  -F "audio_file=@sample.wav"
```

## Notes

- The diagnostic response is a prototype workflow for research and demos, not a clinical decision system.
- The hybrid model class is ready for supervised training, while the CLI/API screening path currently uses deterministic acoustic heuristics plus the quantum TB probability for inference.
- Supported dataset audio formats: `.wav`, `.ogg`, `.webm`, `.mp3`, `.flac`, `.m4a`.
