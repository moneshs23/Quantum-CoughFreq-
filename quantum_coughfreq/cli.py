from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

import uvicorn

from quantum_coughfreq.audio import SUPPORTED_AUDIO_EXTENSIONS, is_supported_audio_file
from quantum_coughfreq.hybrid import CoughDiagnosticService


EXPECTED_LABELS = ("Viral", "Bacterial", "TB")
EXPECTED_RISK_LEVELS = ("High", "Moderate", "Low")
DEFAULT_WORKERS = 1
_THREAD_LOCAL = threading.local()


def _get_service() -> CoughDiagnosticService:
    service = getattr(_THREAD_LOCAL, "service", None)
    if service is None:
        service = CoughDiagnosticService()
        _THREAD_LOCAL.service = service
    return service


def _predict_audio_file(audio_path: str) -> Dict[str, object]:
    try:
        prediction = _get_service().predict(audio_path)
    except Exception as exc:  # noqa: BLE001
        return {"audio_file": audio_path, "error": str(exc)}

    return {
        "audio_file": audio_path,
        "infection_type": prediction.infection_type,
        "confidence_score": round(prediction.confidence_score, 2),
        "tb_probability": round(prediction.tb_probability, 4),
        "tb_risk_level": prediction.tb_risk_level,
        "detected_frequency_range": prediction.detected_frequency_range,
        "acoustic_signature": prediction.acoustic_signature,
    }


def predict_command(audio_file: str, json_output: bool = False) -> int:
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}")
        return 1
    if not is_supported_audio_file(audio_path):
        print(f"Error: unsupported audio format. Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")
        return 1

    service = CoughDiagnosticService()
    print(f"\n[INFO] Starting Quantum CoughFreq Prediction...")
    print(f"[INFO] Analyzing audio file: {audio_path.name}")
    print(f"[INFO] Extracting features and running quantum estimator...\n")
    prediction = service.predict(audio_path.as_posix())

    payload = {
        "audio_file": audio_path.as_posix(),
        "detected_frequency_range": prediction.detected_frequency_range,
        "infection_type": prediction.infection_type,
        "confidence_score": round(prediction.confidence_score, 2),
        "tb_probability": round(prediction.tb_probability, 4),
        "tb_risk_level": prediction.tb_risk_level,
        "acoustic_signature": prediction.acoustic_signature,
        "class_probabilities": {
            label: round(probability, 4) for label, probability in prediction.class_probabilities.items()
        },
    }

    if json_output:
        print(json.dumps(payload, indent=2))
        return 0

    print("Quantum CoughFreq Prediction")
    print(f"audio_file: {payload['audio_file']}")
    print(f"detected_frequency_range: {payload['detected_frequency_range']}")
    print(f"infection_type: {payload['infection_type']}")
    print(f"confidence_score: {payload['confidence_score']}%")
    print(f"tb_probability: {payload['tb_probability']}")
    print(f"tb_risk_level: {payload['tb_risk_level']}")
    print(f"acoustic_signature: {payload['acoustic_signature']}")
    return 0


def classify_dataset_command(
    dataset_dir: str,
    json_output: bool = False,
    detailed: bool = False,
    top_k: int = 10,
    workers: int = DEFAULT_WORKERS,
    max_files: Optional[int] = None,
) -> int:
    print(f"\n[INFO] Initializing dataset classification...")
    print(f"[INFO] Target Directory: {dataset_dir}")
    print(f"[INFO] Workers: {workers}")
    print(f"[INFO] Searching and analyzing files, please wait...\n")
    try:
        payload = classify_dataset(
            dataset_dir=dataset_dir,
            top_k=top_k,
            detailed=detailed or json_output,
            workers=workers,
            max_files=max_files,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except NotADirectoryError as exc:
        print(f"Error: {exc}")
        return 1
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if json_output:
        print(json.dumps(payload, indent=2))
        return 0

    summary = payload["summary"]
    sorted_results = payload["results"]
    skipped = payload["skipped"]

    print("Quantum CoughFreq Dataset Classification")
    print(f"dataset_dir: {summary['dataset_dir']}")
    print(f"total_audio_files: {summary['total_audio_files']}")
    print(f"processed_audio_files: {summary['processed_audio_files']}")
    print(f"skipped_audio_files: {summary['skipped_audio_files']}")
    print(f"workers_used: {summary['workers_used']}")
    print(f"predicted_tb_cases: {summary['predicted_tb_cases']}")
    print(f"predicted_other_cases: {summary['predicted_other_cases']}")
    print(f"high_tb_risk: {summary['high_tb_risk']}")
    print(f"moderate_tb_risk: {summary['moderate_tb_risk']}")
    print(f"low_tb_risk: {summary['low_tb_risk']}")
    print(f"viral_predictions: {summary['viral_predictions']}")
    print(f"bacterial_predictions: {summary['bacterial_predictions']}")
    print(f"tb_predictions: {summary['tb_predictions']}")

    if sorted_results:
        print("\nTop TB-risk files:")
        for result in sorted_results[:top_k]:
            print(
                f"- {result['audio_file']} | risk={result['tb_risk_level']} | "
                f"tb_probability={result['tb_probability']} | predicted={result['infection_type']}"
            )

    if detailed and sorted_results:
        print("\nDetailed results:")
        for result in sorted_results:
            print(
                f"- {result['audio_file']} | predicted={result['infection_type']} | "
                f"confidence={result['confidence_score']}% | tb_probability={result['tb_probability']} | "
                f"risk={result['tb_risk_level']} | signature={result['acoustic_signature']}"
            )

    if skipped:
        print("\nSkipped files:")
        for item in skipped[:top_k]:
            print(f"- {item['audio_file']} | error={item['error']}")
    return 0


def classify_dataset(
    dataset_dir: str,
    top_k: int = 10,
    detailed: bool = False,
    workers: int = DEFAULT_WORKERS,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset path is not a directory: {root}")

    audio_files = _collect_audio_files(root)
    if not audio_files:
        raise ValueError(
            "no supported audio files found. "
            f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    if max_files is not None and max_files > 0:
        audio_files = audio_files[:max_files]

    class_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    results: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []
    worker_count = max(1, workers)
    audio_paths = [audio_file.as_posix() for audio_file in audio_files]

    if worker_count == 1:
        prediction_rows = [_predict_audio_file(audio_path) for audio_path in audio_paths]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            prediction_rows = list(executor.map(_predict_audio_file, audio_paths, chunksize=16))

    for row in prediction_rows:
        if "error" in row:
            skipped.append({"audio_file": str(row["audio_file"]), "error": str(row["error"])})
            continue

        class_counts[str(row["infection_type"])] += 1
        risk_counts[str(row["tb_risk_level"])] += 1
        results.append(row)

    sorted_results = sorted(results, key=lambda item: float(item["tb_probability"]), reverse=True)
    summary = {
        "dataset_dir": root.as_posix(),
        "total_audio_files": len(audio_files),
        "processed_audio_files": len(results),
        "skipped_audio_files": len(skipped),
        "workers_used": worker_count,
        "predicted_tb_cases": class_counts["TB"],
        "predicted_other_cases": class_counts["Viral"] + class_counts["Bacterial"],
        "high_tb_risk": risk_counts["High"],
        "moderate_tb_risk": risk_counts["Moderate"],
        "low_tb_risk": risk_counts["Low"],
        "viral_predictions": class_counts["Viral"],
        "bacterial_predictions": class_counts["Bacterial"],
        "tb_predictions": class_counts["TB"],
    }

    payload = {
        "summary": summary,
        "top_tb_risk_files": sorted_results[:top_k],
        "results": sorted_results if detailed else [],
        "skipped": skipped,
    }
    return payload


def dataset_summary_command(dataset_dir: str) -> int:
    root = Path(dataset_dir)
    if not root.exists():
        print(f"Error: dataset directory not found: {root}")
        return 1
    if not root.is_dir():
        print(f"Error: dataset path is not a directory: {root}")
        return 1

    summary = summarize_dataset(root)
    print("Quantum CoughFreq Dataset Summary")
    print(f"dataset_dir: {root.as_posix()}")
    print(f"total_wav_files: {sum(summary.values())}")
    for label in EXPECTED_LABELS:
        print(f"{label}: {summary[label]}")

    missing_labels = [label for label, count in summary.items() if count == 0]
    if missing_labels:
        print(f"warning: missing or empty label folders: {', '.join(missing_labels)}")
    else:
        print("dataset_status: ready_for_training_scaffold")
    return 0


def summarize_dataset(root: Path) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for label in EXPECTED_LABELS:
        label_dir = root / label
        summary[label] = len(list(label_dir.rglob("*.wav"))) if label_dir.exists() else 0
    return summary


def _collect_audio_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and is_supported_audio_file(path))


def serve_command(host: str = "0.0.0.0", port: int = 8011) -> int:
    from quantum_coughfreq.api import app

    uvicorn.run(app, host=host, port=port)
    return 0


def tabular_summary_command(csv_file: str) -> int:
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        return 1

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print("Quantum CoughFreq Tabular Dataset Summary")
        print(f"csv_file: {csv_path.as_posix()}")
        print(f"total_samples: {len(df)}")

        if "status" in df.columns:
            print("\nLabel Distribution:")
            print(df["status"].value_counts().to_string())

        mfcc_cols = [col for col in df.columns if "MFCC" in col]
        if mfcc_cols:
            print(f"\nFeatures detected: {len(mfcc_cols)} MFCCs")

        print("\ndataset_status: ready_for_quantum_analysis")
        return 0
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return 1
