from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio

from indic_speech_translation.asr.whisper import ASRHypothesis, WhisperASR
from indic_speech_translation.config import load_config
from indic_speech_translation.pipeline import CascadePipeline, CascadeResult
from indic_speech_translation.translation.nllb import NLLBTranslator, TranslationOutput
from indic_speech_translation.utils import flatten_to_str


class _FakeASR:
    def __init__(self, text: str = "hello"):
        self.text = text

    def transcribe_file(self, audio_path):
        return ASRHypothesis(text=self.text, audio_path=str(audio_path), language_id="en")


class _FakeMT:
    def translate(self, text: str):
        return TranslationOutput(
            source_text=text,
            translation=f"HI({text})",
            src_lang="eng_Latn",
            tgt_lang="hin_Deva",
        )


def test_flatten_to_str_nested():
    assert flatten_to_str([["hello", "world"]]) == "hello world"
    assert flatten_to_str(("a", ["b"])) == "a b"
    assert flatten_to_str({"text": "hi"}) == "hi"


def test_flatten_to_str_hypothesis_object():
    hyp = SimpleNamespace(text="hello", score=-1.2)
    assert flatten_to_str(hyp) == "hello"
    assert flatten_to_str([[hyp]]) == "hello"


def test_cascade_pipeline_mocked(configs_dir: Path, tmp_path: Path):
    cfg = load_config(configs_dir / "en_hi.yaml")
    wav = tmp_path / "a.wav"
    torchaudio.save(str(wav), torch.zeros(1, 1600), 16000)

    pipeline = CascadePipeline(
        config=cfg,
        asr=_FakeASR("hello"),
        translator=_FakeMT(),
    )
    result = pipeline.translate_audio(wav)
    assert isinstance(result, CascadeResult)
    assert result.asr_text == "hello"
    assert result.translation == "HI(hello)"
    assert result.source_lang == "en"
    assert result.target_lang == "hi"
    assert result.metadata["preprocess"]["lowpass_hz"] == 3000


def test_independent_asr_mt_calls(configs_dir: Path, tmp_path: Path):
    cfg = load_config(configs_dir / "en_bn.yaml")
    pipeline = CascadePipeline(config=cfg, asr=_FakeASR("hello"), translator=_FakeMT())
    wav = tmp_path / "b.wav"
    torchaudio.save(str(wav), torch.zeros(1, 1600), 16000)
    assert pipeline.transcribe(wav) == "hello"
    assert pipeline.translate_text("hello") == "HI(hello)"


def test_whisper_uses_injected_model(configs_dir: Path, tmp_path: Path):
    cfg = load_config(configs_dir / "en_hi.yaml")

    class DummyModel:
        def __init__(self):
            self.last_kwargs = None

        def transcribe(self, waveform, language=None, task=None, sample_rate=None):
            self.last_kwargs = {
                "language": language,
                "task": task,
                "sample_rate": sample_rate,
            }
            return {"text": "hello world"}

    model = DummyModel()
    asr = WhisperASR(config=cfg.asr, preprocess=cfg.preprocess, model=model)
    wav = tmp_path / "c.wav"
    torchaudio.save(str(wav), torch.randn(1, 16000) * 0.1, 16000)
    hyp = asr.transcribe_file(wav)
    assert hyp.text == "hello world"
    assert hyp.language_id == "en"
    assert model.last_kwargs["language"] == "en"
    assert model.last_kwargs["task"] == "transcribe"


def test_nllb_uses_injected_model(configs_dir: Path):
    cfg = load_config(configs_dir / "en_ta.yaml")

    class DummyTok:
        pad_token_id = 0

        def __call__(self, texts, truncation=True, padding=True, max_length=256, return_tensors="pt"):
            import torch

            return {
                "input_ids": torch.ones(len(texts), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
            }

        def convert_tokens_to_ids(self, token):
            return 42 if token == "tam_Taml" else 1

        def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
            return ["வணக்கம்"] * len(ids)

    class DummyModel:
        def __init__(self):
            self.last_kwargs = None

        def generate(self, **kwargs):
            self.last_kwargs = kwargs
            import torch

            return torch.ones(1, 3, dtype=torch.long)

    model = DummyModel()
    translator = NLLBTranslator(
        config=cfg.mt,
        src_lang=cfg.mt_src_code,
        tgt_lang=cfg.mt_tgt_code,
        generation=cfg.generation,
        device="cpu",
        model=model,
        tokenizer=DummyTok(),
    )
    out = translator.translate("hello")
    assert out.translation == "வணக்கம்"
    assert out.tgt_lang == "tam_Taml"
    assert model.last_kwargs["forced_bos_token_id"] == 42
    assert model.last_kwargs["num_beams"] == 5
