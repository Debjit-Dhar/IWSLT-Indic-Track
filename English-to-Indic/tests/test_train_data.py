from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from indic_speech_translation.train import load_csv_pairs, load_parallel_lines


def test_load_csv_pairs_notebook_columns(tmp_path: Path):
    csv_path = tmp_path / "en-hi train.csv"
    pd.DataFrame(
        {
            "Source text": ["Hello", "nan"],
            "Translated text": ["नमस्ते", "दुनिया"],
        }
    ).to_csv(csv_path, index=False)
    pairs = load_csv_pairs(csv_path, "Source text", "Translated text", max_samples=1)
    assert pairs["sentence_SRC"] == ["Hello"]
    assert pairs["sentence_TGT"] == ["नमस्ते"]


def test_load_csv_pairs_missing_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["a"]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="must contain columns"):
        load_csv_pairs(csv_path, "Source text", "Translated text")


def test_load_parallel_lines_flat(tmp_path: Path):
    (tmp_path / "train.eng_Latn").write_text("one\ntwo\n", encoding="utf-8")
    (tmp_path / "train.hin_Deva").write_text("एक\nदो\n", encoding="utf-8")
    pairs = load_parallel_lines(tmp_path, "train", "eng_Latn", "hin_Deva")
    assert pairs["sentence_SRC"] == ["one", "two"]
    assert pairs["sentence_TGT"] == ["एक", "दो"]
