from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from indic_speech_translation.config import load_config, validate_config


@pytest.mark.parametrize("name", ["en_ta.yaml", "en_bn.yaml", "en_hi.yaml"])
def test_load_language_configs(configs_dir: Path, name: str):
    cfg = load_config(configs_dir / name)
    assert cfg.source_lang == "en"
    assert cfg.target_lang in {"ta", "bn", "hi"}
    assert cfg.preprocess.lowpass_hz == 3000.0
    assert cfg.preprocess.gain_db == 10.0
    assert cfg.preprocess.target_sr == 16000
    assert cfg.train.learning_rate == 2e-5
    assert cfg.train.per_device_batch_size == 2
    assert cfg.train.num_train_epochs == 5
    assert cfg.train.weight_decay == 0.01
    assert cfg.train.max_samples == 20000
    assert cfg.train.metric_for_best_model == "BLEU"
    assert cfg.train.greater_is_better is True
    assert cfg.train.max_steps is None
    assert cfg.generation.num_beams == 5
    assert cfg.mt.base_model_id == "facebook/nllb-200-distilled-600M"
    assert cfg.asr.model_id == "openai/whisper-small"
    assert cfg.asr.language_id == "en"
    assert cfg.asr.decoding_strategy == "beam"
    assert cfg.asr.beam_size == 5


def test_reject_unknown_target(tmp_path: Path, configs_dir: Path):
    raw = yaml.safe_load((configs_dir / "en_ta.yaml").read_text())
    raw["target_lang"] = "mr"
    raw["mt_tgt_code"] = "mar_Deva"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported target_lang"):
        load_config(path)


def test_reject_non_english_source(tmp_path: Path, configs_dir: Path):
    raw = yaml.safe_load((configs_dir / "en_bn.yaml").read_text())
    raw["source_lang"] = "hi"
    raw["mt_src_code"] = "hin_Deva"
    path = tmp_path / "bad_src.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported source_lang"):
        load_config(path)


def test_reject_wrong_preprocess(tmp_path: Path, configs_dir: Path):
    raw = yaml.safe_load((configs_dir / "en_hi.yaml").read_text())
    raw["preprocess"]["lowpass_hz"] = 5000
    path = tmp_path / "bad_prep.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="lowpass_hz"):
        load_config(path)


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_incomplete_asr(configs_dir: Path):
    cfg = load_config(configs_dir / "en_ta.yaml")
    cfg.asr.model_id = ""
    with pytest.raises(ValueError, match="Incomplete ASR"):
        validate_config(cfg)
