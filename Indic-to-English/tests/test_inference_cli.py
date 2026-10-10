from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio

from indic_speech_translation.inference import build_parser, run_inference
from indic_speech_translation.pipeline import CascadeResult


def test_cli_help():
    parser = build_parser()
    help_text = parser.format_help()
    assert "--audio" in help_text
    assert "--config" in help_text


def test_run_inference_mocked(monkeypatch, configs_dir: Path, tmp_path: Path):
    wav = tmp_path / "x.wav"
    torchaudio.save(str(wav), torch.zeros(1, 1600), 16000)
    out = tmp_path / "out.json"

    class FakePipeline:
        def translate_audio(self, audio_path):
            return CascadeResult(
                audio_path=str(audio_path),
                asr_text="src",
                translation="tgt",
                source_lang="ta",
                target_lang="en",
                mt_src_code="tam_Taml",
                mt_tgt_code="eng_Latn",
            )

    monkeypatch.setattr(
        "indic_speech_translation.inference.CascadePipeline.from_config",
        lambda *a, **k: FakePipeline(),
    )
    payload = run_inference(wav, configs_dir / "ta_en.yaml", output_json=out)
    assert payload["translation"] == "tgt"
    assert json.loads(out.read_text())["asr_text"] == "src"
