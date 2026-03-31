from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import audioread
import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly, stft, welch


HIGH_FREQUENCY_THRESHOLD_HZ = 2500.0
LOW_FREQUENCY_THRESHOLD_HZ = 1000.0
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".ogg", ".webm", ".mp3", ".flac", ".m4a")
MFCC_COUNT = 13
MEL_BANDS = 64
FFT_SIZE = 2048
HOP_LENGTH = 512


@dataclass
class AudioFeatures:
    sample_rate: int
    duration_seconds: float
    sample_count: int
    mfcc_mean: np.ndarray
    psd_frequencies: np.ndarray
    psd_values: np.ndarray
    dominant_frequency_hz: float
    low_band_power: float
    mid_band_power: float
    high_band_power: float
    detected_frequency_range: str
    acoustic_signature: str
    quantum_vector: np.ndarray
    mel_spectrogram: np.ndarray
    waveform_preview: np.ndarray


def _band_power(frequencies: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.mean(psd[mask]))


def classify_frequency_range(
    dominant_frequency_hz: float,
    low_band_power: float,
    high_band_power: float,
) -> Tuple[str, str]:
    if dominant_frequency_hz > HIGH_FREQUENCY_THRESHOLD_HZ or high_band_power >= low_band_power:
        return "High", "Sharp/Hacking"
    if dominant_frequency_hz < LOW_FREQUENCY_THRESHOLD_HZ:
        return "Low", "Deep/Productive"
    return "Low", "Deep/Productive"


def normalize_for_quantum_injection(values: np.ndarray, output_dim: int = 4) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    if flattened.size < output_dim:
        flattened = np.pad(flattened, (0, output_dim - flattened.size))
    elif flattened.size > output_dim:
        flattened = flattened[:output_dim]

    value_min = float(np.min(flattened))
    value_max = float(np.max(flattened))
    if np.isclose(value_min, value_max):
        scaled = np.zeros_like(flattened, dtype=np.float32)
    else:
        scaled = 2.0 * (flattened - value_min) / (value_max - value_min) - 1.0
    return scaled * np.pi


def is_supported_audio_file(file_path: Union[str, Path]) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def _waveform_preview(signal: np.ndarray, target_points: int = 512) -> np.ndarray:
    if signal.size == 0:
        return np.zeros(target_points, dtype=np.float32)
    if signal.size == target_points:
        return signal.astype(np.float32)

    indices = np.linspace(0, signal.size - 1, target_points).astype(np.int32)
    return signal[indices].astype(np.float32)


def _load_audio_signal(audio_path: Path, target_sample_rate: int) -> Tuple[np.ndarray, int]:
    try:
        signal, sample_rate = sf.read(audio_path.as_posix(), dtype="float32", always_2d=False)
        if np.ndim(signal) > 1:
            signal = np.mean(signal, axis=1)
    except Exception:
        chunks = []
        with audioread.audio_open(audio_path.as_posix()) as source:
            sample_rate = int(source.samplerate)
            channels = int(source.channels)
            for frame in source:
                decoded = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    decoded = decoded.reshape(-1, channels).mean(axis=1)
                chunks.append(decoded)
        signal = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.float32)

    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {audio_path}")

    if sample_rate != target_sample_rate:
        signal = resample_poly(signal, target_sample_rate, sample_rate).astype(np.float32)
        sample_rate = target_sample_rate

    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal = signal / peak
    return signal, sample_rate


def _hz_to_mel(frequency_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (frequency_hz / 700.0))


def _mel_to_hz(mel_values: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel_values / 2595.0) - 1.0)


def _mel_filter_bank(sample_rate: int, n_fft: int, n_mels: int, fmax: float) -> np.ndarray:
    fft_bins = (n_fft // 2) + 1
    mel_points = np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, fft_bins - 1)

    filter_bank = np.zeros((n_mels, fft_bins), dtype=np.float32)
    for index in range(1, n_mels + 1):
        left = bins[index - 1]
        center = bins[index]
        right = bins[index + 1]
        if center <= left:
            center = min(left + 1, fft_bins - 1)
        if right <= center:
            right = min(center + 1, fft_bins)

        for freq_bin in range(left, center):
            filter_bank[index - 1, freq_bin] = (freq_bin - left) / max(center - left, 1)
        for freq_bin in range(center, right):
            filter_bank[index - 1, freq_bin] = (right - freq_bin) / max(right - center, 1)

    return filter_bank


def _power_to_db(values: np.ndarray) -> np.ndarray:
    safe_values = np.maximum(values, 1e-10)
    reference = np.max(safe_values)
    return (10.0 * np.log10(safe_values) - 10.0 * np.log10(reference)).astype(np.float32)


def _fix_frame_count(matrix: np.ndarray, frame_count: int) -> np.ndarray:
    if matrix.shape[1] == frame_count:
        return matrix.astype(np.float32)
    if matrix.shape[1] > frame_count:
        return matrix[:, :frame_count].astype(np.float32)

    pad_width = frame_count - matrix.shape[1]
    padding = np.repeat(matrix[:, -1:], pad_width, axis=1) if matrix.shape[1] else np.zeros(
        (matrix.shape[0], pad_width),
        dtype=np.float32,
    )
    return np.concatenate([matrix, padding], axis=1).astype(np.float32)


def _compute_mfcc_and_mel(signal: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    _, _, stft_matrix = stft(
        signal,
        fs=sample_rate,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_LENGTH,
        nfft=FFT_SIZE,
        padded=True,
        boundary="zeros",
    )
    power_spectrogram = np.abs(stft_matrix).astype(np.float32) ** 2
    mel_filters = _mel_filter_bank(sample_rate=sample_rate, n_fft=FFT_SIZE, n_mels=MEL_BANDS, fmax=4000.0)
    mel_spectrogram = np.matmul(mel_filters, power_spectrogram)
    mel_spectrogram_db = _fix_frame_count(_power_to_db(mel_spectrogram), frame_count=64)

    log_mel = np.log(np.maximum(mel_spectrogram, 1e-10))
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:MFCC_COUNT]
    mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
    return mfcc_mean, mel_spectrogram_db


def extract_audio_features(file_path: Union[str, Path], target_sample_rate: int = 16000) -> AudioFeatures:
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not is_supported_audio_file(audio_path):
        raise ValueError(
            f"Unsupported audio format: {audio_path.suffix or '<no extension>'}. "
            f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    signal, sample_rate = _load_audio_signal(audio_path, target_sample_rate=target_sample_rate)
    mfcc_mean, mel_spectrogram_db = _compute_mfcc_and_mel(signal, sample_rate)
    waveform_preview = _waveform_preview(signal)

    frequencies, psd = welch(signal, fs=sample_rate, nperseg=1024)
    dominant_frequency = float(frequencies[np.argmax(psd)])

    low_band_power = _band_power(frequencies, psd, 0.0, LOW_FREQUENCY_THRESHOLD_HZ)
    mid_band_power = _band_power(frequencies, psd, LOW_FREQUENCY_THRESHOLD_HZ, HIGH_FREQUENCY_THRESHOLD_HZ)
    high_band_power = _band_power(frequencies, psd, HIGH_FREQUENCY_THRESHOLD_HZ, 4000.0)

    detected_range, acoustic_signature = classify_frequency_range(
        dominant_frequency_hz=dominant_frequency,
        low_band_power=low_band_power,
        high_band_power=high_band_power,
    )

    handcrafted_vector = np.array(
        [
            dominant_frequency / max(sample_rate, 1),
            low_band_power,
            mid_band_power,
            high_band_power,
        ],
        dtype=np.float32,
    )
    quantum_vector = normalize_for_quantum_injection(handcrafted_vector, output_dim=4)

    return AudioFeatures(
        sample_rate=sample_rate,
        duration_seconds=float(signal.size / max(sample_rate, 1)),
        sample_count=int(signal.size),
        mfcc_mean=mfcc_mean,
        psd_frequencies=frequencies.astype(np.float32),
        psd_values=psd.astype(np.float32),
        dominant_frequency_hz=dominant_frequency,
        low_band_power=low_band_power,
        mid_band_power=mid_band_power,
        high_band_power=high_band_power,
        detected_frequency_range=detected_range,
        acoustic_signature=acoustic_signature,
        quantum_vector=quantum_vector.astype(np.float32),
        mel_spectrogram=mel_spectrogram_db,
        waveform_preview=waveform_preview,
    )
