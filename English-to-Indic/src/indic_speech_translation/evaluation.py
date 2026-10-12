"""Eval protocols: ASR WER, MT-on-refs, end-to-end cascade."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from indic_speech_translation.config import load_config
from indic_speech_translation.pipeline import CascadePipeline
from indic_speech_translation.utils import PathLike, get_logger, require_file, write_json

logger = get_logger(__name__)

PROTOCOLS = ("asr", "mt", "cascade")


@dataclass
class EvalResult:
    protocol: str
    split_name: str
    metrics: Dict[str, float]
    num_examples: int
    config_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _corpus_bleu(hyps: Sequence[str], refs: Sequence[str]) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_bleu(list(hyps), [list(refs)]).score)


def _corpus_chrf(hyps: Sequence[str], refs: Sequence[str]) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_chrf(list(hyps), [list(refs)]).score)


def _corpus_wer(hyps: Sequence[str], refs: Sequence[str]) -> float:
    import jiwer

    if not hyps:
        return 0.0
    return float(jiwer.wer(list(refs), list(hyps)))


def load_manifest(path: PathLike) -> pd.DataFrame:
    """Load eval CSV. Required columns depend on protocol (see docs)."""
    csv_path = require_file(path, "manifest CSV")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Manifest is empty: {csv_path}")
    return df


def _validate_columns(df: pd.DataFrame, required: Sequence[str], protocol: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Protocol '{protocol}' requires columns {list(required)}; missing {missing}"
        )


def evaluate_asr(
    pipeline: CascadePipeline,
    df: pd.DataFrame,
    split_name: str,
    config_path: Optional[str] = None,
) -> EvalResult:
    _validate_columns(df, ["audio_path", "source_text"], "asr")
    hyps: List[str] = []
    refs: List[str] = []
    for _, row in df.iterrows():
        hyps.append(pipeline.transcribe(row["audio_path"]))
        refs.append(str(row["source_text"]))
    return EvalResult(
        protocol="asr",
        split_name=split_name,
        metrics={"WER": _corpus_wer(hyps, refs)},
        num_examples=len(hyps),
        config_path=config_path,
        details={"hypotheses": hyps, "references": refs},
    )


def evaluate_mt(
    pipeline: CascadePipeline,
    df: pd.DataFrame,
    split_name: str,
    config_path: Optional[str] = None,
) -> EvalResult:
    """MT on reference English transcripts (no ASR)."""
    _validate_columns(df, ["source_text", "reference_translation"], "mt")
    hyps: List[str] = []
    refs: List[str] = []
    for _, row in df.iterrows():
        hyps.append(pipeline.translate_text(str(row["source_text"])))
        refs.append(str(row["reference_translation"]))
    return EvalResult(
        protocol="mt",
        split_name=split_name,
        metrics={
            "BLEU": _corpus_bleu(hyps, refs),
            "chrF++": _corpus_chrf(hyps, refs),
        },
        num_examples=len(hyps),
        config_path=config_path,
        details={"hypotheses": hyps, "references": refs},
    )


def evaluate_cascade(
    pipeline: CascadePipeline,
    df: pd.DataFrame,
    split_name: str,
    config_path: Optional[str] = None,
) -> EvalResult:
    """Full cascade vs Indic references."""
    _validate_columns(df, ["audio_path", "reference_translation"], "cascade")
    translations: List[str] = []
    asr_texts: List[str] = []
    refs: List[str] = []
    for _, row in df.iterrows():
        result = pipeline.translate_audio(row["audio_path"])
        translations.append(result.translation)
        asr_texts.append(result.asr_text)
        refs.append(str(row["reference_translation"]))
    details: Dict[str, Any] = {
        "hypotheses": translations,
        "asr_texts": asr_texts,
        "references": refs,
    }
    if "source_text" in df.columns:
        details["asr_WER"] = _corpus_wer(asr_texts, [str(x) for x in df["source_text"]])
    return EvalResult(
        protocol="cascade",
        split_name=split_name,
        metrics={
            "BLEU": _corpus_bleu(translations, refs),
            "chrF++": _corpus_chrf(translations, refs),
        },
        num_examples=len(translations),
        config_path=config_path,
        details=details,
    )


def run_evaluation(
    config_path: PathLike,
    protocol: str,
    manifest: Optional[PathLike] = None,
    output_json: Optional[PathLike] = None,
    device: Optional[str] = None,
    include_details: bool = False,
) -> EvalResult:
    protocol = protocol.lower()
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; choose from {PROTOCOLS}")

    cfg = load_config(config_path)
    manifest_path = manifest or cfg.data.manifest_csv
    if not manifest_path:
        raise ValueError("manifest CSV required via --manifest or data.manifest_csv")

    df = load_manifest(manifest_path)
    split_name = cfg.data.split_name
    pipeline = CascadePipeline.from_config(cfg, device=device)

    if protocol == "asr":
        result = evaluate_asr(pipeline, df, split_name, config_path=str(config_path))
    elif protocol == "mt":
        result = evaluate_mt(pipeline, df, split_name, config_path=str(config_path))
    else:
        result = evaluate_cascade(pipeline, df, split_name, config_path=str(config_path))

    payload = result.to_dict()
    if not include_details:
        payload.pop("details", None)
    if output_json:
        write_json(output_json, payload)
        logger.info("Wrote evaluation to %s", output_json)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ASR / MT / cascade")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--protocol",
        required=True,
        choices=list(PROTOCOLS),
        help="asr | mt | cascade",
    )
    parser.add_argument("--manifest", default=None, help="CSV manifest path")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Keep per-example hypotheses in JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_evaluation(
        config_path=args.config,
        protocol=args.protocol,
        manifest=args.manifest,
        output_json=args.output,
        device=args.device,
        include_details=args.include_details,
    )
    print(
        json.dumps(
            {
                "protocol": result.protocol,
                "split_name": result.split_name,
                "num_examples": result.num_examples,
                "metrics": result.metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
