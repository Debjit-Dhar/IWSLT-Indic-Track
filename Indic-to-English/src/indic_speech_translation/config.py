"""Load and validate experiment YAML configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from indic_speech_translation.utils import (
    MT_LANG_CODES,
    SUPPORTED_SOURCE_LANGS,
    SUPPORTED_TARGET_LANGS,
    ensure_path,
)

PathLike = Union[str, Path]


@dataclass
class ASRConfig:
    model_id: str
    language_id: str
    decoder: str = "rnnt"
    decoding_strategy: str = "beam"
    beam_size: int = 5


@dataclass
class MTConfig:
    base_model_id: str = "ai4bharat/indictrans2-indic-en-1B"
    adapter_path: Optional[str] = None
    max_length: int = 256
    num_beams: int = 5


@dataclass
class PreprocessConfig:
    """Paper §3.1 acoustic preprocessing (Indic→English)."""

    target_sr: int = 16000
    lowpass_hz: float = 3000.0
    gain_db: float = 10.0


@dataclass
class TrainConfig:
    """Paper §4.2 LoRA schedule; Trainer knobs match the research notebook."""

    data_dir: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "dev"
    output_dir: str = "outputs/lora"
    learning_rate: float = 2e-5
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 5
    weight_decay: float = 0.01
    fp16: bool = True
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    label_smoothing: float = 0.1
    dropout: float = 0.1
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    lr_scheduler_type: str = "polynomial"
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    metric_for_best_model: str = "BLEU"
    greater_is_better: bool = True
    generation_num_beams: int = 5
    generation_max_length: int = 256
    num_workers: int = 2
    seed: int = 42
    max_steps: Optional[int] = None  # epochs are primary (§4.2)


@dataclass
class GenerationConfig:
    num_beams: int = 5
    max_length: int = 256


@dataclass
class DataConfig:
    audio_dir: Optional[str] = None
    manifest_csv: Optional[str] = None
    split_name: str = "dev"


@dataclass
class ExperimentConfig:
    source_lang: str
    target_lang: str
    mt_src_code: str
    mt_tgt_code: str
    asr: ASRConfig
    mt: MTConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _require_mapping(raw: Any, name: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return raw


def _build_asr(raw: Dict[str, Any]) -> ASRConfig:
    section = _require_mapping(raw.get("asr"), "asr")
    for key in ("model_id", "language_id"):
        if not section.get(key):
            raise ValueError(f"asr.{key} is required")
    return ASRConfig(
        model_id=str(section["model_id"]),
        language_id=str(section["language_id"]),
        decoder=str(section.get("decoder", "rnnt")),
        decoding_strategy=str(section.get("decoding_strategy", "beam")),
        beam_size=int(section.get("beam_size", 5)),
    )


def _build_mt(raw: Dict[str, Any]) -> MTConfig:
    section = _require_mapping(raw.get("mt"), "mt")
    if not section.get("base_model_id"):
        raise ValueError("mt.base_model_id is required")
    adapter = section.get("adapter_path")
    if adapter in ("", "null", "None"):
        adapter = None
    return MTConfig(
        base_model_id=str(section["base_model_id"]),
        adapter_path=str(adapter) if adapter is not None else None,
        max_length=int(section.get("max_length", 256)),
        num_beams=int(section.get("num_beams", 5)),
    )


def _build_preprocess(raw: Dict[str, Any]) -> PreprocessConfig:
    section = raw.get("preprocess") or {}
    section = _require_mapping(section, "preprocess") if section else {}
    cfg = PreprocessConfig(
        target_sr=int(section.get("target_sr", 16000)),
        lowpass_hz=float(section.get("lowpass_hz", 3000.0)),
        gain_db=float(section.get("gain_db", 10.0)),
    )
    if cfg.target_sr != 16000:
        raise ValueError("preprocess.target_sr must be 16000 (paper §3.1 / model input)")
    if abs(cfg.lowpass_hz - 3000.0) > 1e-6:
        raise ValueError("preprocess.lowpass_hz must be 3000 (paper §3.1)")
    if abs(cfg.gain_db - 10.0) > 1e-6:
        raise ValueError("preprocess.gain_db must be 10.0 (paper §3.1)")
    return cfg


def _build_train(raw: Dict[str, Any]) -> TrainConfig:
    section = raw.get("train") or {}
    section = _require_mapping(section, "train") if section else {}
    targets = section.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    max_steps = section.get("max_steps", None)
    return TrainConfig(
        data_dir=section.get("data_dir"),
        train_split=str(section.get("train_split", "train")),
        eval_split=str(section.get("eval_split", "dev")),
        output_dir=str(section.get("output_dir", "outputs/lora")),
        learning_rate=float(section.get("learning_rate", 2e-5)),
        per_device_batch_size=int(section.get("per_device_batch_size", 4)),
        gradient_accumulation_steps=int(section.get("gradient_accumulation_steps", 1)),
        num_train_epochs=int(section.get("num_train_epochs", 5)),
        weight_decay=float(section.get("weight_decay", 0.01)),
        fp16=bool(section.get("fp16", True)),
        lora_r=int(section.get("lora_r", 4)),
        lora_alpha=int(section.get("lora_alpha", 16)),
        lora_dropout=float(section.get("lora_dropout", 0.1)),
        lora_target_modules=list(targets),
        label_smoothing=float(section.get("label_smoothing", 0.1)),
        dropout=float(section.get("dropout", 0.1)),
        warmup_ratio=float(section.get("warmup_ratio", 0.1)),
        warmup_steps=int(section.get("warmup_steps", 0)),
        lr_scheduler_type=str(section.get("lr_scheduler_type", "polynomial")),
        optim=str(section.get("optim", "adamw_torch")),
        adam_beta1=float(section.get("adam_beta1", 0.9)),
        adam_beta2=float(section.get("adam_beta2", 0.98)),
        max_grad_norm=float(section.get("max_grad_norm", 1.0)),
        early_stopping_patience=int(section.get("early_stopping_patience", 5)),
        early_stopping_threshold=float(section.get("early_stopping_threshold", 0.0)),
        metric_for_best_model=str(section.get("metric_for_best_model", "BLEU")),
        greater_is_better=bool(section.get("greater_is_better", True)),
        generation_num_beams=int(section.get("generation_num_beams", 5)),
        generation_max_length=int(section.get("generation_max_length", 256)),
        num_workers=int(section.get("num_workers", 2)),
        seed=int(section.get("seed", 42)),
        max_steps=int(max_steps) if max_steps is not None else None,
    )


def validate_config(cfg: ExperimentConfig) -> ExperimentConfig:
    src = cfg.source_lang.lower()
    tgt = cfg.target_lang.lower()
    if src not in SUPPORTED_SOURCE_LANGS:
        raise ValueError(
            f"Unsupported source_lang={cfg.source_lang!r}; "
            f"supported Indic→English sources: {sorted(SUPPORTED_SOURCE_LANGS)}"
        )
    if tgt not in SUPPORTED_TARGET_LANGS:
        raise ValueError(
            f"Unsupported target_lang={cfg.target_lang!r}; only English (en) is supported"
        )
    expected_src = MT_LANG_CODES[src]
    expected_tgt = MT_LANG_CODES[tgt]
    if cfg.mt_src_code != expected_src:
        raise ValueError(f"mt_src_code must be {expected_src} for source_lang={src}")
    if cfg.mt_tgt_code != expected_tgt:
        raise ValueError(f"mt_tgt_code must be {expected_tgt} for target_lang={tgt}")
    if not cfg.asr.model_id or not cfg.asr.language_id:
        raise ValueError("Incomplete ASR config: model_id and language_id are required")
    if not cfg.mt.base_model_id:
        raise ValueError("Incomplete MT config: base_model_id is required")
    if cfg.asr.language_id.lower() != src:
        raise ValueError(
            f"asr.language_id={cfg.asr.language_id!r} must match source_lang={src!r}"
        )
    return cfg


def load_config(path: PathLike) -> ExperimentConfig:
    config_path = ensure_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    for key in ("source_lang", "target_lang", "mt_src_code", "mt_tgt_code"):
        if key not in raw:
            raise ValueError(f"Missing required config key: {key}")

    gen_raw = raw.get("generation") or {}
    data_raw = raw.get("data") or {}
    cfg = ExperimentConfig(
        source_lang=str(raw["source_lang"]).lower(),
        target_lang=str(raw["target_lang"]).lower(),
        mt_src_code=str(raw["mt_src_code"]),
        mt_tgt_code=str(raw["mt_tgt_code"]),
        asr=_build_asr(raw),
        mt=_build_mt(raw),
        preprocess=_build_preprocess(raw),
        train=_build_train(raw),
        generation=GenerationConfig(
            num_beams=int(gen_raw.get("num_beams", 5)),
            max_length=int(gen_raw.get("max_length", 256)),
        ),
        data=DataConfig(
            audio_dir=data_raw.get("audio_dir"),
            manifest_csv=data_raw.get("manifest_csv"),
            split_name=str(data_raw.get("split_name", "dev")),
        ),
        config_path=str(config_path),
    )
    return validate_config(cfg)
