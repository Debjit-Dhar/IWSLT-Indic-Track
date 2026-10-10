from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from indic_speech_translation.config import load_config, validate_config


@pytest.mark.parametrize("name", ["ta_en.yaml", "bn_en.yaml", "hi_en.yaml"])
def test_load_language_configs(configs_dir: Path, name: str):
    cfg = load_config(configs_dir / name)
    assert cfg.target_lang == "en"
    assert cfg.source_lang in {"ta", "bn", "hi"}
    assert cfg.preprocess.lowpass_hz == 3000.0
    assert cfg.preprocess.gain_db == 10.0
    assert cfg.preprocess.target_sr == 16000
    assert cfg.train.learning_rate == 2e-5
    assert cfg.train.per_device_batch_size == 4
    assert cfg.train.lora_r == 4
    assert cfg.train.lora_alpha == 16
    assert cfg.train.num_train_epochs == 5
    assert cfg.train.weight_decay == 0.01
    assert cfg.train.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg.train.metric_for_best_model == "BLEU"
    assert cfg.train.greater_is_better is True
    assert cfg.train.max_steps is None
    assert cfg.generation.num_beams == 5
    assert cfg.mt.base_model_id == "ai4bharat/indictrans2-indic-en-1B"
    assert cfg.asr.decoding_strategy == "beam"
    assert cfg.asr.beam_size == 5


def test_reject_unknown_source(tmp_path: Path, configs_dir: Path):
    raw = yaml.safe_load((configs_dir / "ta_en.yaml").read_text())
    raw["source_lang"] = "mr"
    raw["mt_src_code"] = "mar_Deva"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported source_lang"):
        load_config(path)


def test_reject_non_english_target(tmp_path: Path, configs_dir: Path):
    raw = yaml.safe_load((configs_dir / "bn_en.yaml").read_text())
    raw["target_lang"] = "bn"
    raw["mt_tgt_code"] = "ben_Beng"
    path = tmp_path / "bad_tgt.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported target_lang"):
        load_config(path)


def test_reject_wrong_preprocess(tmp_path: Path, configs_dir: Path):
    raw = yaml.safe_load((configs_dir / "hi_en.yaml").read_text())
    raw["preprocess"]["lowpass_hz"] = 5000
    path = tmp_path / "bad_prep.yaml"
    path.write_text(yaml.dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="lowpass_hz"):
        load_config(path)


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_incomplete_asr(configs_dir: Path):
    cfg = load_config(configs_dir / "ta_en.yaml")
    cfg.asr.model_id = ""
    with pytest.raises(ValueError, match="Incomplete ASR"):
        validate_config(cfg)
