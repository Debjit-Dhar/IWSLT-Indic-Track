"""Path, logging, and ASR output helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Union

PathLike = Union[str, Path]

SUPPORTED_SOURCE_LANGS = frozenset({"bn", "hi", "ta"})
SUPPORTED_TARGET_LANGS = frozenset({"en"})

MT_LANG_CODES = {
    "bn": "ben_Beng",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "en": "eng_Latn",
}


def get_logger(name: str = "indic_speech_translation") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def require_file(path: PathLike, label: str = "file") -> Path:
    p = ensure_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def require_dir(path: PathLike, label: str = "directory") -> Path:
    p = ensure_path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def write_json(path: PathLike, payload: Any) -> Path:
    p = ensure_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def flatten_to_str(value: Any) -> str:
    """Collapse nested NeMo / HF transcription returns into one string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, dict):
        for key in ("text", "pred_text", "transcription", "hypothesis"):
            if key in value:
                return flatten_to_str(value[key])
        return " ".join(flatten_to_str(v) for v in value.values()).strip()
    # NeMo Hypothesis-like objects
    if not isinstance(value, (list, tuple, set)):
        for attr in ("text", "pred_text", "transcript"):
            if hasattr(value, attr):
                attr_val = getattr(value, attr)
                if attr_val is value or callable(attr_val):
                    continue
                return flatten_to_str(attr_val)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        parts = [flatten_to_str(item) for item in value]
        return " ".join(p for p in parts if p).strip()
    return str(value).strip()
