"""Paper §3.1 audio preprocessing for English→Indic ASR.

Order: load → mono → 16 kHz → normalize [-1, 1] → LPF 3 kHz → +10 dB
(soft-clip to [-1, 1] after gain for WAV safety).
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as F

from indic_speech_translation.config import PreprocessConfig
from indic_speech_translation.utils import PathLike, ensure_path, require_file

DB_TO_AMPLITUDE = 10.0 ** (10.0 / 20.0)  # +10 dB ≈ 3.162


@dataclass
class ProcessedAudio:
    waveform: torch.Tensor  # (1, T)
    sample_rate: int
    source_path: Optional[Path] = None


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def normalize_peak(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.numel() == 0:
        return waveform
    peak = waveform.abs().max()
    if peak.item() > 0:
        waveform = waveform / peak
    return waveform


def apply_gain_db(waveform: torch.Tensor, gain_db: float = 10.0) -> torch.Tensor:
    return waveform * (10.0 ** (gain_db / 20.0))


def soft_clip(waveform: torch.Tensor, limit: float = 1.0) -> torch.Tensor:
    """Soft-clip to [-limit, limit] after amplification."""
    return limit * torch.tanh(waveform / limit)


def preprocess_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    cfg: Optional[PreprocessConfig] = None,
) -> ProcessedAudio:
    cfg = cfg or PreprocessConfig()
    waveform = _to_mono(waveform.float())
    if waveform.numel() == 0:
        raise ValueError("Audio became empty after preprocessing")

    if sample_rate != cfg.target_sr:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=cfg.target_sr
        )(waveform)
        sample_rate = cfg.target_sr

    waveform = normalize_peak(waveform)
    waveform = F.lowpass_biquad(
        waveform, sample_rate=sample_rate, cutoff_freq=cfg.lowpass_hz
    )
    waveform = apply_gain_db(waveform, gain_db=cfg.gain_db)
    waveform = soft_clip(waveform, limit=1.0)

    if waveform.numel() == 0:
        raise ValueError("Audio became empty after preprocessing")

    return ProcessedAudio(waveform=waveform, sample_rate=sample_rate)


def load_and_preprocess(
    path: PathLike,
    cfg: Optional[PreprocessConfig] = None,
) -> ProcessedAudio:
    audio_path = require_file(path, "audio file")
    waveform, sample_rate = torchaudio.load(str(audio_path))
    processed = preprocess_waveform(waveform, sample_rate, cfg=cfg)
    processed.source_path = audio_path
    return processed


def save_processed_wav(
    processed: ProcessedAudio,
    output_path: Optional[PathLike] = None,
) -> Path:
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / f"processed_{uuid.uuid4().hex}.wav"
    out = ensure_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), processed.waveform.cpu(), processed.sample_rate)
    return out


def preprocess_to_tempfile(
    path: PathLike,
    cfg: Optional[PreprocessConfig] = None,
) -> Tuple[Path, ProcessedAudio]:
    processed = load_and_preprocess(path, cfg=cfg)
    return save_processed_wav(processed), processed
