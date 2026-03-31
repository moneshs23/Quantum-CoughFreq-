from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quantum_coughfreq.audio import AudioFeatures, extract_audio_features
from quantum_coughfreq.quantum import QuantumTBEstimator, TorchQuantumLayer


CLASS_LABELS = ["Viral", "Bacterial", "TB"]
QUANTUM_SCREENING_WEIGHTS = np.array(
    [0.35, -0.55, 0.82, 0.24, -0.48, 0.73, -0.66, 0.51, 0.18, -0.31, 0.64, -0.42],
    dtype=np.float32,
)


@dataclass
class PredictionResult:
    detected_frequency_range: str
    infection_type: str
    confidence_score: float
    tb_probability: float
    tb_risk_level: str
    acoustic_signature: str
    class_probabilities: Dict[str, float]


class SpectrogramCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(spectrogram)
        return self.projection(encoded)


class HybridCoughClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = SpectrogramCNN()
        self.quantum = TorchQuantumLayer(num_qubits=4, encoding="zz")
        self.classifier = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU(),
            nn.Linear(16, len(CLASS_LABELS)),
        )

    def forward(self, spectrogram: torch.Tensor, handcrafted_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cnn_embedding = self.cnn(spectrogram)
        quantum_inputs = (cnn_embedding + handcrafted_features[:, :4]) / 2.0
        tb_probability = self.quantum(quantum_inputs)
        classifier_input = torch.cat([cnn_embedding, handcrafted_features[:, :4], tb_probability], dim=1)
        logits = self.classifier(classifier_input)
        return logits, tb_probability

    def compute_loss(
        self,
        spectrogram: torch.Tensor,
        handcrafted_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward(spectrogram, handcrafted_features)
        return F.cross_entropy(logits, labels)


class CoughDiagnosticService:
    def __init__(self) -> None:
        self.quantum_estimator = QuantumTBEstimator(num_qubits=4, encoding="zz")
        self.quantum_weights = QUANTUM_SCREENING_WEIGHTS[: self.quantum_estimator.parameter_count]

    def _heuristic_scores(self, features: AudioFeatures, tb_probability: float) -> np.ndarray:
        total_band_power = features.low_band_power + features.mid_band_power + features.high_band_power + 1e-8
        low_share = features.low_band_power / total_band_power
        mid_share = features.mid_band_power / total_band_power
        high_share = features.high_band_power / total_band_power
        high_range = features.detected_frequency_range == "High"
        sharp_cough = features.acoustic_signature == "Sharp/Hacking"
        tb_band_hit = 900.0 <= features.dominant_frequency_hz <= 3200.0

        viral_score = (
            0.18
            + (0.32 * high_share)
            + (0.12 if sharp_cough else 0.0)
            + (0.12 if high_range else 0.0)
            + (0.18 * (1.0 - tb_probability))
        )
        bacterial_score = (
            0.18
            + (0.34 * low_share)
            + (0.12 if not sharp_cough else 0.0)
            + (0.12 if not high_range else 0.0)
            + (0.16 * (1.0 - tb_probability))
        )
        tb_score = (
            0.16
            + (0.50 * tb_probability)
            + (0.12 * mid_share)
            + (0.08 * high_share)
            + (0.10 if tb_band_hit else 0.0)
            + (0.06 if sharp_cough else 0.0)
        )

        scores = np.array([viral_score, bacterial_score, tb_score], dtype=np.float32)
        scores = scores / np.sum(scores)
        return scores

    def _quantum_tb_probability(self, features: AudioFeatures) -> float:
        return float(
            self.quantum_estimator.tb_probability(
                features=features.quantum_vector,
                weights=self.quantum_weights,
            )
        )

    def _tb_risk_level(self, tb_probability: float, tb_class_probability: float) -> str:
        if tb_probability >= 0.72 or (tb_probability >= 0.60 and tb_class_probability >= 0.40):
            return "High"
        if tb_probability >= 0.52 or tb_class_probability >= 0.34:
            return "Moderate"
        return "Low"

    def predict_from_features(self, features: AudioFeatures) -> PredictionResult:
        tb_probability = self._quantum_tb_probability(features)
        probabilities = self._heuristic_scores(features, tb_probability)
        label_index = int(np.argmax(probabilities))
        class_probabilities = {
            label: float(probability)
            for label, probability in zip(CLASS_LABELS, probabilities)
        }
        tb_risk_level = self._tb_risk_level(tb_probability, class_probabilities["TB"])

        return PredictionResult(
            detected_frequency_range=features.detected_frequency_range,
            infection_type=CLASS_LABELS[label_index],
            confidence_score=float(probabilities[label_index] * 100.0),
            tb_probability=tb_probability,
            tb_risk_level=tb_risk_level,
            acoustic_signature=features.acoustic_signature,
            class_probabilities=class_probabilities,
        )

    def predict(self, file_path: str) -> PredictionResult:
        features = extract_audio_features(file_path)
        return self.predict_from_features(features)
