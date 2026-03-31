from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from quantum_coughfreq.audio import SUPPORTED_AUDIO_EXTENSIONS, extract_audio_features
from quantum_coughfreq.cli import classify_dataset

if TYPE_CHECKING:
    from quantum_coughfreq.hybrid import CoughDiagnosticService


WEB_DIR = Path(__file__).resolve().parent / "web"


class PredictionResponse(BaseModel):
    detected_frequency_range: str
    infection_type: str
    confidence_score: float
    tb_probability: float
    tb_risk_level: str
    acoustic_signature: str


class DatasetClassificationRequest(BaseModel):
    dataset_dir: str
    top_k: int = 10
    workers: int = 1
    max_files: Optional[int] = None
    detailed: bool = False


def _normalize_matrix(values: np.ndarray) -> list[list[float]]:
    matrix = np.asarray(values, dtype=np.float32)
    min_value = float(np.min(matrix))
    max_value = float(np.max(matrix))
    if np.isclose(min_value, max_value):
        normalized = np.zeros_like(matrix, dtype=np.float32)
    else:
        normalized = (matrix - min_value) / (max_value - min_value)
    return normalized.round(4).tolist()


def _downsample_curve(values: np.ndarray, points: int = 96) -> list[float]:
    curve = np.asarray(values, dtype=np.float32).reshape(-1)
    if curve.size == 0:
        return [0.0] * points
    indices = np.linspace(0, curve.size - 1, points).astype(np.int32)
    return curve[indices].astype(np.float32).round(4).tolist()


def _build_pipeline_log(
    audio_name: str,
    file_suffix: str,
    sample_rate: int,
    duration_seconds: float,
    dominant_frequency_hz: float,
    tb_probability: float,
    infection_type: str,
) -> list[str]:
    return [
        f"Receiving audio file: {audio_name}",
        f"Validating format: {file_suffix or 'unknown'} accepted",
        "Decoding cough sample into mono buffer",
        f"Resampling stream to {sample_rate} Hz",
        f"Windowing signal across {duration_seconds:.2f} seconds",
        "Computing MFCC and Mel-spectrum maps",
        "Injecting 4 acoustic features into quantum state",
        f"Estimating TB amplitude from quantum circuit: {tb_probability:.4f}",
        f"Dominant cough frequency detected: {dominant_frequency_hz:.1f} Hz",
        f"Final respiratory class selected: {infection_type}",
    ]


def _build_quantum_circuit_summary(service: CoughDiagnosticService) -> Dict[str, Any]:
    estimator = service.quantum_estimator
    columns = [
        {"label": "H", "kind": "prep"},
        {"label": "Rz(x)", "kind": "encode"},
        {"label": "Ry(x)", "kind": "encode"},
        {"label": "ZZ", "kind": "entangle"},
        {"label": "Rx(w)", "kind": "rotation"},
        {"label": "Ry(w)", "kind": "rotation"},
        {"label": "Rz(w)", "kind": "rotation"},
        {"label": "M", "kind": "measure"},
    ]
    rows = []
    for qubit_index in range(estimator.num_qubits):
        rows.append(
            {
                "qubit": f"q{qubit_index}",
                "gates": columns,
            }
        )

    return {
        "qubits": estimator.num_qubits,
        "layers": 1 + estimator.ansatz.reps,
        "encoding": estimator.encoding,
        "rows": rows,
    }


def _build_projection_payload(
    infection_type: str,
    tb_probability: float,
    quantum_vector: np.ndarray,
) -> Dict[str, Any]:
    rng = np.random.default_rng(42)
    centers = {
        "Viral": (-0.85, -0.15),
        "Bacterial": (0.0, 0.05),
        "TB": (0.85, 0.2),
    }
    colors = {
        "Viral": "#38d8ff",
        "Bacterial": "#90ff78",
        "TB": "#ff5e88",
    }

    points = []
    for label, center in centers.items():
        for _ in range(24):
            points.append(
                {
                    "label": label,
                    "x": round(float(center[0] + rng.normal(0.0, 0.16)), 4),
                    "y": round(float(center[1] + rng.normal(0.0, 0.12)), 4),
                    "color": colors[label],
                }
            )

    sample_center = centers[infection_type]
    sample_x = float(sample_center[0] + np.mean(quantum_vector) / np.pi * 0.25)
    sample_y = float((tb_probability - 0.5) * 1.5)
    sample = {
        "label": infection_type,
        "x": round(sample_x, 4),
        "y": round(sample_y, 4),
        "color": colors[infection_type],
    }
    return {"points": points, "sample": sample}


def _build_analysis_payload(
    file_path: str,
    original_name: str,
    service: CoughDiagnosticService,
) -> Dict[str, Any]:
    features = extract_audio_features(file_path)
    prediction = service.predict_from_features(features)

    band_total = features.low_band_power + features.mid_band_power + features.high_band_power + 1e-8
    mfcc_heatmap = np.repeat(features.mfcc_mean[:, None], 24, axis=1)

    return {
        "audio_file": original_name,
        "audio_format": Path(original_name).suffix.lower() or ".wav",
        "duration_seconds": round(features.duration_seconds, 2),
        "sample_rate": features.sample_rate,
        "sample_count": features.sample_count,
        "waveform_points": np.asarray(features.waveform_preview, dtype=np.float32).round(4).tolist(),
        "mel_spectrogram": _normalize_matrix(features.mel_spectrogram),
        "mfcc_heatmap": _normalize_matrix(mfcc_heatmap),
        "psd_curve": _downsample_curve(features.psd_values, points=96),
        "frequency_bands": [
            {"label": "Low", "value": round(float(features.low_band_power / band_total), 4), "color": "#35d6ff"},
            {"label": "Mid", "value": round(float(features.mid_band_power / band_total), 4), "color": "#84ff71"},
            {"label": "High", "value": round(float(features.high_band_power / band_total), 4), "color": "#ff7d4d"},
        ],
        "dominant_frequency_hz": round(features.dominant_frequency_hz, 2),
        "detected_frequency_range": prediction.detected_frequency_range,
        "infection_type": prediction.infection_type,
        "confidence_score": round(prediction.confidence_score, 2),
        "tb_probability": round(prediction.tb_probability, 4),
        "tb_risk_level": prediction.tb_risk_level,
        "acoustic_signature": prediction.acoustic_signature,
        "class_probabilities": {
            label: round(probability, 4)
            for label, probability in prediction.class_probabilities.items()
        },
        "quantum_vector": np.asarray(features.quantum_vector, dtype=np.float32).round(4).tolist(),
        "quantum_circuit": _build_quantum_circuit_summary(service),
        "pipeline_log": _build_pipeline_log(
            audio_name=original_name,
            file_suffix=Path(original_name).suffix.lower(),
            sample_rate=features.sample_rate,
            duration_seconds=features.duration_seconds,
            dominant_frequency_hz=features.dominant_frequency_hz,
            tb_probability=prediction.tb_probability,
            infection_type=prediction.infection_type,
        ),
        "latent_projection": _build_projection_payload(
            infection_type=prediction.infection_type,
            tb_probability=prediction.tb_probability,
            quantum_vector=features.quantum_vector,
        ),
        "sphere": {
            "radius": round(0.45 + float(prediction.confidence_score / 100.0) * 0.35, 4),
            "glow": round(0.35 + prediction.tb_probability * 0.5, 4),
            "energy": round(float(np.max(np.abs(features.waveform_preview))), 4),
        },
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Quantum CoughFreq", version="0.2.0")
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

    service = None  # type: Optional[CoughDiagnosticService]
    service_error = None  # type: Optional[str]

    def get_service() -> CoughDiagnosticService:
        nonlocal service, service_error
        if service is not None:
            return service
        if service_error is not None:
            raise RuntimeError(service_error)

        try:
            from quantum_coughfreq.hybrid import CoughDiagnosticService as _CoughDiagnosticService
        except ModuleNotFoundError as exc:
            service_error = f"Missing runtime dependency: {exc.name}"
            raise RuntimeError(service_error) from exc

        service = _CoughDiagnosticService()
        return service

    async def save_upload(audio_file: UploadFile) -> tuple[str, str]:
        suffix = Path(audio_file.filename or "sample.wav").suffix or ".wav"
        if suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}",
            )

        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await audio_file.read())
            return temp_file.name, (audio_file.filename or f"sample{suffix}")

    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=302)

    @app.get("/dashboard")
    async def dashboard() -> FileResponse:
        return FileResponse(WEB_DIR / "dashboard.html")

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/dashboard-config")
    async def dashboard_config() -> Dict[str, Any]:
        default_dataset = Path.cwd() / "dataset" / "QMl data" / "public_dataset_v3" / "coughvid_20211012"
        return {
            "supported_formats": list(SUPPORTED_AUDIO_EXTENSIONS),
            "default_dataset_dir": default_dataset.as_posix(),
        }

    @app.post("/predict-cough", response_model=PredictionResponse)
    async def predict_cough(audio_file: UploadFile = File(...)) -> PredictionResponse:
        temp_path, original_name = await save_upload(audio_file)

        try:
            payload = _build_analysis_payload(temp_path, original_name, get_service())
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return PredictionResponse(
            detected_frequency_range=str(payload["detected_frequency_range"]),
            infection_type=str(payload["infection_type"]),
            confidence_score=float(payload["confidence_score"]),
            tb_probability=float(payload["tb_probability"]),
            tb_risk_level=str(payload["tb_risk_level"]),
            acoustic_signature=str(payload["acoustic_signature"]),
        )

    @app.post("/api/analyze-cough")
    async def analyze_cough(audio_file: UploadFile = File(...)) -> Dict[str, Any]:
        temp_path, original_name = await save_upload(audio_file)

        try:
            return _build_analysis_payload(temp_path, original_name, get_service())
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @app.post("/api/classify-dataset")
    async def classify_dataset_endpoint(request: DatasetClassificationRequest) -> Dict[str, Any]:
        try:
            return classify_dataset(
                dataset_dir=request.dataset_dir,
                top_k=request.top_k,
                detailed=request.detailed,
                workers=request.workers,
                max_files=request.max_files,
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Dataset classification failed: {exc}") from exc

    return app


app = create_app()
