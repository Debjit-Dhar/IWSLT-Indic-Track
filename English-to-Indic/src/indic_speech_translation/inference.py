"""CLI helpers for cascaded inference."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from indic_speech_translation.config import load_config
from indic_speech_translation.pipeline import CascadePipeline
from indic_speech_translation.utils import PathLike, get_logger, write_json

logger = get_logger(__name__)


def run_inference(
    audio: PathLike,
    config_path: PathLike,
    output_json: Optional[PathLike] = None,
    device: Optional[str] = None,
) -> dict:
    cfg = load_config(config_path)
    pipeline = CascadePipeline.from_config(cfg, device=device)
    result = pipeline.translate_audio(audio)
    payload = result.to_dict()
    if output_json:
        write_json(output_json, payload)
        logger.info("Wrote %s", output_json)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="English→Indic cascaded ST inference")
    parser.add_argument("--audio", required=True, help="Input WAV path")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_inference(
        audio=args.audio,
        config_path=args.config,
        output_json=args.output,
        device=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
