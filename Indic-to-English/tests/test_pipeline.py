from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio

from indic_speech_translation.asr.indicconformer import ASRHypothesis, IndicConformerASR
from indic_speech_translation.config import load_config
from indic_speech_translation.pipeline import CascadePipeline, CascadeResult
from indic_speech_translation.translation.indictrans2 import (
    IndicTrans2Translator,
    TranslationOutput,
)
from indic_speech_translation.utils import flatten_to_str


class _FakeASR:
    def __init__(self, text: str = "வணக்கம்"):
        self.text = text

    def transcribe_file(self, audio_path):
        return ASRHypothesis(text=self.text, audio_path=str(audio_path), language_id="ta")


class _FakeMT:
    def translate(self, text: str):
        return TranslationOutput(
            source_text=text,
            translation=f"EN({text})",
            src_lang="tam_Taml",
            tgt_lang="eng_Latn",
        )


def test_flatten_to_str_nested():
    assert flatten_to_str([["hello", "world"]]) == "hello world"
    assert flatten_to_str(("a", ["b"])) == "a b"
    assert flatten_to_str({"text": "hi"}) == "hi"


def test_flatten_to_str_hypothesis_object():
    hyp = SimpleNamespace(text="नमस्ते", score=-1.2)
    assert flatten_to_str(hyp) == "नमस्ते"
    assert flatten_to_str([[hyp]]) == "नमस्ते"


def test_cascade_pipeline_mocked(configs_dir: Path, tmp_path: Path):
    cfg = load_config(configs_dir / "ta_en.yaml")
    wav = tmp_path / "a.wav"
    torchaudio.save(str(wav), torch.zeros(1, 1600), 16000)

    pipeline = CascadePipeline(
        config=cfg,
        asr=_FakeASR("வணக்கம்"),
        translator=_FakeMT(),
    )
    result = pipeline.translate_audio(wav)
    assert isinstance(result, CascadeResult)
    assert result.asr_text == "வணக்கம்"
    assert result.translation == "EN(வணக்கம்)"
    assert result.source_lang == "ta"
    assert result.target_lang == "en"
    assert result.metadata["preprocess"]["lowpass_hz"] == 3000


def test_independent_asr_mt_calls(configs_dir: Path, tmp_path: Path):
    cfg = load_config(configs_dir / "bn_en.yaml")
    pipeline = CascadePipeline(config=cfg, asr=_FakeASR("বাংলা"), translator=_FakeMT())
    wav = tmp_path / "b.wav"
    torchaudio.save(str(wav), torch.zeros(1, 1600), 16000)
    assert pipeline.transcribe(wav) == "বাংলা"
    assert pipeline.translate_text("বাংলা") == "EN(বাংলা)"


def test_indicconformer_uses_injected_model(configs_dir: Path, tmp_path: Path):
    cfg = load_config(configs_dir / "hi_en.yaml")

    class DummyModel:
        def __init__(self):
            self.cfg = SimpleNamespace(
                decoding=SimpleNamespace(strategy="greedy", beam=SimpleNamespace(beam_size=1))
            )
            self.cur_decoder = "rnnt"
            self._changed = None
            self.last_kwargs = None

        def freeze(self):
            return None

        def to(self, device):
            return self

        def change_decoding_strategy(self, decoding_cfg):
            self._changed = decoding_cfg

        def transcribe(self, paths, language_id=None, batch_size=None):
            self.last_kwargs = {
                "language_id": language_id,
                "batch_size": batch_size,
                "n_paths": len(paths),
            }
            return [["नमस्ते"]]

    model = DummyModel()
    asr = IndicConformerASR(config=cfg.asr, preprocess=cfg.preprocess, model=model)
    # Injected models skip _load_model; apply decoding explicitly like load path
    asr._apply_decoding_strategy(model)
    assert model.cfg.decoding.strategy == "beam"
    assert model.cfg.decoding.beam.beam_size == 5
    assert model._changed is model.cfg.decoding

    wav = tmp_path / "c.wav"
    torchaudio.save(str(wav), torch.randn(1, 16000) * 0.1, 16000)
    hyp = asr.transcribe_file(wav)
    assert hyp.text == "नमस्ते"
    assert hyp.language_id == "hi"
    assert model.last_kwargs["language_id"] == "hi"
    assert model.last_kwargs["batch_size"] == 1


def test_indicconformer_tuple_hyp_return(configs_dir: Path, tmp_path: Path):
    cfg = load_config(configs_dir / "bn_en.yaml")

    class DummyModel:
        def transcribe(self, paths, language_id=None, batch_size=None):
            return ([SimpleNamespace(text="বাংলা")], {"meta": 1})

    asr = IndicConformerASR(config=cfg.asr, preprocess=cfg.preprocess, model=DummyModel())
    wav = tmp_path / "d.wav"
    torchaudio.save(str(wav), torch.randn(1, 1600) * 0.1, 16000)
    hyp = asr.transcribe_file(wav)
    assert hyp.text == "বাংলা"
