from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from indic_speech_translation.evaluation import (
    EvalResult,
    evaluate_asr,
    evaluate_cascade,
    evaluate_mt,
    load_manifest,
    run_evaluation,
)
from indic_speech_translation.pipeline import CascadeResult


class _FakePipeline:
    def transcribe(self, audio_path):
        return "hello world"

    def translate_text(self, text: str):
        return "good morning"

    def translate_audio(self, audio_path):
        return CascadeResult(
            audio_path=str(audio_path),
            asr_text="hello world",
            translation="good morning",
            source_lang="ta",
            target_lang="en",
            mt_src_code="tam_Taml",
            mt_tgt_code="eng_Latn",
        )


def test_load_manifest_requires_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.csv")


def test_protocol_column_validation():
    df = pd.DataFrame({"audio_path": ["a.wav"]})
    with pytest.raises(ValueError, match="requires columns"):
        evaluate_asr(_FakePipeline(), df, split_name="dev")


def test_asr_protocol_metrics():
    df = pd.DataFrame(
        {
            "audio_path": ["a.wav", "b.wav"],
            "source_text": ["hello world", "hello world"],
        }
    )
    result = evaluate_asr(_FakePipeline(), df, split_name="dev")
    assert isinstance(result, EvalResult)
    assert result.protocol == "asr"
    assert result.num_examples == 2
    assert "WER" in result.metrics
    assert result.metrics["WER"] == 0.0


def test_mt_and_cascade_protocols():
    df_mt = pd.DataFrame(
        {
            "source_text": ["வணக்கம்"],
            "reference_translation": ["good morning"],
        }
    )
    mt = evaluate_mt(_FakePipeline(), df_mt, split_name="dev")
    assert mt.protocol == "mt"
    assert "BLEU" in mt.metrics
    assert "chrF++" in mt.metrics

    df_c = pd.DataFrame(
        {
            "audio_path": ["a.wav"],
            "reference_translation": ["good morning"],
            "source_text": ["hello world"],
        }
    )
    casc = evaluate_cascade(_FakePipeline(), df_c, split_name="dev")
    assert casc.protocol == "cascade"
    assert casc.details["asr_WER"] == 0.0


def test_run_evaluation_unknown_protocol(configs_dir: Path, tmp_path: Path):
    manifest = tmp_path / "m.csv"
    pd.DataFrame(
        {"audio_path": ["a.wav"], "source_text": ["x"], "reference_translation": ["y"]}
    ).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Unknown protocol"):
        run_evaluation(configs_dir / "ta_en.yaml", protocol="foo", manifest=manifest)
