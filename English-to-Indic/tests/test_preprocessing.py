from __future__ import annotations

import math

import torch
import torchaudio

from indic_speech_translation.config import PreprocessConfig
from indic_speech_translation.preprocessing import (
    DB_TO_AMPLITUDE,
    apply_gain_db,
    load_and_preprocess,
    normalize_peak,
    preprocess_waveform,
    soft_clip,
)


def test_normalize_peak_range():
    wave = torch.tensor([[0.0, 2.0, -4.0]])
    out = normalize_peak(wave)
    assert torch.isclose(out.abs().max(), torch.tensor(1.0))


def test_gain_db_scale():
    wave = torch.ones(1, 4)
    out = apply_gain_db(wave, gain_db=10.0)
    assert torch.allclose(out, wave * DB_TO_AMPLITUDE)


def test_soft_clip_bounds():
    wave = torch.tensor([[-5.0, 0.0, 5.0]])
    out = soft_clip(wave, limit=1.0)
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_preprocess_order_and_sr(tmp_path):
    sr = 8000
    t = torch.linspace(0, 1, sr)
    wave = 0.5 * torch.sin(2 * math.pi * 300 * t) + 0.5 * torch.sin(2 * math.pi * 3500 * t)
    wave = wave.unsqueeze(0)
    path = tmp_path / "tone.wav"
    torchaudio.save(str(path), wave, sr)

    cfg = PreprocessConfig()
    processed = load_and_preprocess(path, cfg=cfg)
    assert processed.sample_rate == 16000
    assert processed.waveform.ndim == 2
    assert processed.waveform.size(0) == 1
    assert processed.waveform.abs().max() <= 1.0 + 1e-5


def test_preprocess_empty_raises():
    wave = torch.zeros(1, 0)
    with __import__("pytest").raises(ValueError, match="empty"):
        preprocess_waveform(wave, 16000)


def test_mono_mixdown():
    wave = torch.stack([torch.ones(100), -torch.ones(100)], dim=0)
    out = preprocess_waveform(wave, 16000)
    assert out.waveform.size(0) == 1
