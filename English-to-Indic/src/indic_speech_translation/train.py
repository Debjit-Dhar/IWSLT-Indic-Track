"""Full fine-tuning for NLLB-200 (paper §4.1; research notebook)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from indic_speech_translation.config import ExperimentConfig, load_config
from indic_speech_translation.utils import get_logger, require_dir, require_file, write_json

logger = get_logger(__name__)


def _pair_file_paths(data_dir: Path, split: str, src_code: str, tgt_code: str) -> tuple[Path, Path]:
    """Resolve parallel text files.

    Layouts:
      data_dir/{split}/{src}-{tgt}/{split}.{src|tgt}
      data_dir/{split}.{src|tgt}
    """
    nested_dir = data_dir / split / f"{src_code}-{tgt_code}"
    nested_src = nested_dir / f"{split}.{src_code}"
    nested_tgt = nested_dir / f"{split}.{tgt_code}"
    if nested_src.is_file() and nested_tgt.is_file():
        return nested_src, nested_tgt

    flat_src = data_dir / f"{split}.{src_code}"
    flat_tgt = data_dir / f"{split}.{tgt_code}"
    if flat_src.is_file() and flat_tgt.is_file():
        return flat_src, flat_tgt

    raise FileNotFoundError(
        f"Could not find parallel files for split={split} "
        f"({src_code}->{tgt_code}) under {data_dir}"
    )


def _clean_lines(lines: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for ln in lines:
        text = ln.rstrip("\n")
        if str(text).strip().lower() == "nan":
            cleaned.append("")
        else:
            cleaned.append(text)
    return cleaned


def load_parallel_lines(
    data_dir: Path,
    split: str,
    src_code: str,
    tgt_code: str,
    max_samples: Optional[int] = None,
) -> Dict[str, List[str]]:
    src_path, tgt_path = _pair_file_paths(data_dir, split, src_code, tgt_code)
    with src_path.open(encoding="utf-8") as sf, tgt_path.open(encoding="utf-8") as tf:
        src_lines = _clean_lines([ln for ln in sf])
        tgt_lines = _clean_lines([ln for ln in tf])
    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Line count mismatch for {split}: {len(src_lines)} src vs {len(tgt_lines)} tgt"
        )
    if max_samples is not None:
        src_lines = src_lines[:max_samples]
        tgt_lines = tgt_lines[:max_samples]
    return {"sentence_SRC": src_lines, "sentence_TGT": tgt_lines}


def load_csv_pairs(
    csv_path: Path,
    source_column: str,
    target_column: str,
    max_samples: Optional[int] = None,
) -> Dict[str, List[str]]:
    import pandas as pd

    path = require_file(csv_path, "CSV")
    df = pd.read_csv(path)
    required = {source_column, target_column}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {sorted(required)}; found {list(df.columns)}"
        )
    if max_samples is not None:
        df = df.iloc[:max_samples, :]
    src = _clean_lines(df[source_column].astype(str).tolist())
    tgt = _clean_lines(df[target_column].astype(str).tolist())
    return {"sentence_SRC": src, "sentence_TGT": tgt}


def _load_split_pairs(cfg: ExperimentConfig, split_role: str) -> Dict[str, List[str]]:
    train_cfg = cfg.train
    csv_path = train_cfg.train_csv if split_role == "train" else train_cfg.eval_csv
    split_name = train_cfg.train_split if split_role == "train" else train_cfg.eval_split
    if csv_path:
        return load_csv_pairs(
            Path(csv_path),
            source_column=train_cfg.source_column,
            target_column=train_cfg.target_column,
            max_samples=train_cfg.max_samples,
        )
    if not train_cfg.data_dir:
        raise ValueError("Set train.data_dir or train.train_csv / train.eval_csv")
    data_dir = require_dir(train_cfg.data_dir, "train.data_dir")
    return load_parallel_lines(
        data_dir,
        split_name,
        cfg.mt_src_code,
        cfg.mt_tgt_code,
        max_samples=train_cfg.max_samples,
    )


def _preprocess_fn(example: Dict[str, Any], tokenizer: Any, max_length: int = 256) -> Dict[str, Any]:
    model_inputs = tokenizer(
        example["sentence_SRC"],
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    labels = tokenizer(
        text_target=example["sentence_TGT"],
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def _compute_metrics_factory(tokenizer: Any, print_samples: bool = False):
    from sacrebleu.metrics import BLEU, CHRF

    metric_dict = {"BLEU": BLEU(), "chrF": CHRF()}

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels[labels == -100] = tokenizer.pad_token_id
        preds[preds == -100] = tokenizer.pad_token_id

        decoded_preds = [
            x.strip()
            for x in tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        ]
        decoded_labels = [
            x.strip()
            for x in tokenizer.batch_decode(
                labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        ]

        if print_samples and decoded_preds:
            for pred, label in zip(decoded_preds[:3], decoded_labels[:3]):
                logger.info("Prediction: %s", pred)
                logger.info("Reference: %s", label)

        return {
            name: metric.corpus_score(decoded_preds, [decoded_labels]).score
            for name, metric in metric_dict.items()
        }

    return compute_metrics


def build_datasets(cfg: ExperimentConfig, tokenizer: Any):
    from datasets import Dataset

    train_cfg = cfg.train

    def _prepare(split_role: str):
        raw = _load_split_pairs(cfg, split_role)
        ds = Dataset.from_dict(raw).shuffle(seed=train_cfg.seed)
        return ds.map(
            lambda ex: _preprocess_fn(ex, tokenizer=tokenizer, max_length=cfg.mt.max_length),
            batched=True,
            num_proc=max(1, train_cfg.num_workers),
        )

    train_ds = _prepare("train")
    eval_ds = _prepare("eval")
    logger.info("Train size: %d | Eval size: %d", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


def train_translation(
    config_path: Path | str,
    resume_from: Optional[str] = None,
) -> Path:
    """Fine-tune NLLB-200 (§4.1). Returns checkpoint output dir."""
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    cfg = load_config(config_path)
    train_cfg = cfg.train
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg.mt.checkpoint_path or cfg.mt.base_model_id
    logger.info("Loading NLLB model %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.mt.base_model_id,
        src_lang=cfg.mt_src_code,
        tgt_lang=cfg.mt_tgt_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    train_ds, eval_ds = build_datasets(cfg, tokenizer=tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    training_kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        fp16=train_cfg.fp16,
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=train_cfg.save_total_limit,
        predict_with_generate=True,
        load_best_model_at_end=True,
        per_device_train_batch_size=train_cfg.per_device_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        eval_accumulation_steps=train_cfg.gradient_accumulation_steps,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        optim=train_cfg.optim,
        learning_rate=train_cfg.learning_rate,
        num_train_epochs=train_cfg.num_train_epochs,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        warmup_steps=train_cfg.warmup_steps,
        dataloader_num_workers=train_cfg.num_workers,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        generation_max_length=train_cfg.generation_max_length,
        generation_num_beams=train_cfg.generation_num_beams,
        report_to="none",
        seed=train_cfg.seed,
    )
    try:
        training_args = Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            **training_kwargs,
        )
    except TypeError:
        training_args = Seq2SeqTrainingArguments(
            eval_strategy="epoch",
            save_strategy="epoch",
            **training_kwargs,
        )
    if train_cfg.max_steps is not None:
        training_args.max_steps = train_cfg.max_steps

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_compute_metrics_factory(tokenizer, print_samples=True),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg.early_stopping_patience,
                early_stopping_threshold=train_cfg.early_stopping_threshold,
            )
        ],
    )

    logger.info("Starting NLLB fine-tuning (epochs=%s)", train_cfg.num_train_epochs)
    trainer.train(resume_from_checkpoint=resume_from)

    ckpt_dir = output_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    metadata = {
        "base_model_id": cfg.mt.base_model_id,
        "source_lang": cfg.source_lang,
        "target_lang": cfg.target_lang,
        "mt_src_code": cfg.mt_src_code,
        "mt_tgt_code": cfg.mt_tgt_code,
        "train": cfg.train.__dict__,
        "config_path": cfg.config_path,
        "checkpoint_dir": str(ckpt_dir),
    }
    write_json(output_dir / "training_metadata.json", metadata)
    logger.info("Saved NLLB checkpoint to %s", ckpt_dir)
    return ckpt_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune NLLB-200 for English→Indic (paper §4.1)"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Checkpoint directory to resume from",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    train_translation(args.config, resume_from=args.resume_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
