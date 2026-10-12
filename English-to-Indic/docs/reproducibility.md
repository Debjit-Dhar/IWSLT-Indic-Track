# Reproducibility

Checklist for the English→Indic cascade from the IWSLT 2025 JU-CSE-NLP paper.

## Prerequisites

1. Python ≥ 3.9; CUDA GPU recommended
2. IWSLT 2025 Indic Track data (train/dev/test WAV + YAML + text)
3. Whisper Small: `openai/whisper-small`
4. NLLB base: `facebook/nllb-200-distilled-600M`
5. Optional fine-tuned NLLB checkpoint via `mt.checkpoint_path` (not committed)

## Paper settings

### Preprocessing (§3.1)

- [ ] Peak-normalize to `[-1, 1]`
- [ ] Biquad low-pass **3000 Hz**
- [ ] Amplify **+10 dB** (repo soft-clips afterward for WAV safety)
- [ ] 16 kHz mono

### ASR

- [ ] `openai/whisper-small`
- [ ] No ASR fine-tuning
- [ ] English `language_id: en`, `task: transcribe`
- [ ] Beam decode: `decoding_strategy: beam`, `beam_size: 5`

### MT fine-tune (§4.1)

- [ ] Base: NLLB-200-distilled-600M
- [ ] Separate checkpoint per target (Bn / Hi / Ta)
- [ ] Incremental slices of **20000** train rows (`train.max_samples`)
- [ ] `lr=2e-5`, batch `2`, `weight_decay=0.01`
- [ ] **5 epochs** (primary schedule)
- [ ] Generation beam `5`
- [ ] Hugging Face `Seq2SeqTrainer`
- [ ] Train on IWSLT train text; eval on dev
- [ ] Explicit `train.data_dir` **or** notebook-style `train.train_csv` / `train.eval_csv`

### Evaluation

- [ ] ASR: WER vs source English text
- [ ] MT: BLEU / chrF++ on reference Indic translations
- [ ] Cascade: BLEU / chrF++ vs Indic refs
- [ ] Record split name in outputs

## Training data layout

Nested:

```text
data_dir/
  train/{src}-{tgt}/train.{src}
  train/{src}-{tgt}/train.{tgt}
  dev/{src}-{tgt}/dev.{src}
  dev/{src}-{tgt}/dev.{tgt}
```

or flat:

```text
data_dir/
  train.{src}
  train.{tgt}
  dev.{src}
  dev.{tgt}
```

`{src}` / `{tgt}` are NLLB FLORES-200 codes (e.g. `eng_Latn`, `hin_Deva`).

Notebook-style CSVs (as in `notebooks/english-hindi-nllb-finetuning.ipynb`) use columns `Source text` and `Translated text`. Set `train.train_csv` and `train.eval_csv`.

## Eval manifest CSV

| Protocol | Columns |
|---|---|
| `asr` | `audio_path`, `source_text` |
| `mt` | `source_text`, `reference_translation` |
| `cascade` | `audio_path`, `reference_translation` (optional `source_text`) |

## Hardware

Paper fine-tuning: Kaggle T4×2. Single-GPU works; raise `gradient_accumulation_steps` if needed. Full pair revalidation is local (data + checkpoints required).

## Limits

- Weights, datasets, and checkpoints are not shipped
- Indic→English is out of scope
- README numbers are quoted from the paper tables only
