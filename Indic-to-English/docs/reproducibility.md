# Reproducibility

Checklist for the Indic→English cascade from the IWSLT 2025 JU-CSE-NLP paper.

## Prerequisites

1. Python ≥ 3.9; CUDA GPU recommended
2. IWSLT 2025 Indic Track data (train/dev/test WAV + YAML + text)
3. IndicConformer hybrid RNNT checkpoints:
   - `ai4bharat/indicconformer_stt_bn_hybrid_rnnt_large`
   - `ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large`
   - `ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large`
4. IndicTrans2 base: `ai4bharat/indictrans2-indic-en-1B`
5. NeMo (nemo-v2) for ASR — see [AI4Bharat IndicConformer / NeMo](https://github.com/AI4Bharat/IndicConformerASR) (`[asr]` extra is docs-only; install NeMo from source)
6. IndicTransToolkit (`IndicProcessor` / `IndicDataCollator`) for training
7. LoRA adapters — train here or set `mt.adapter_path` (not committed)

## Paper settings

### Preprocessing (§3.1)

- [ ] Peak-normalize to `[-1, 1]`
- [ ] Biquad low-pass **3000 Hz**
- [ ] Amplify **+10 dB** (repo soft-clips afterward for WAV safety)
- [ ] 16 kHz mono

### ASR

- [ ] Language-specific IndicConformer checkpoint
- [ ] No ASR fine-tuning
- [ ] Beam decode: `decoding_strategy: beam`, `beam_size: 5`

### MT LoRA (§4.2)

- [ ] Base: IndicTrans2 Indic→En
- [ ] Targets: `q_proj,k_proj,v_proj,o_proj`
- [ ] `r=4`, `alpha=16`
- [ ] `lr=2e-5`, batch `4`, `weight_decay=0.01`
- [ ] **5 epochs** (primary schedule)
- [ ] fp16 + `Seq2SeqTrainer`
- [ ] Generation beam `5`
- [ ] Early stopping on validation (default metric: BLEU)
- [ ] Train on IWSLT train text; eval on dev
- [ ] Explicit `train.data_dir`, `train_split`, `eval_split`

### Evaluation

- [ ] ASR: WER vs source Indic text
- [ ] MT: BLEU / chrF++ on reference transcripts
- [ ] Cascade: BLEU / chrF++ vs English refs
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

`{src}` / `{tgt}` are IndicTrans2 codes (e.g. `tam_Taml`, `eng_Latn`).

## Eval manifest CSV

| Protocol | Columns |
|---|---|
| `asr` | `audio_path`, `source_text` |
| `mt` | `source_text`, `reference_translation` |
| `cascade` | `audio_path`, `reference_translation` (optional `source_text`) |

## Hardware

Paper fine-tuning: Kaggle T4×2. Single-GPU works; raise `gradient_accumulation_steps` if needed. Full pair revalidation is local (data + adapters required).

## Limits

- Weights, datasets, and adapters are not shipped
- English→Indic is out of scope
- README numbers are quoted from the paper tables only
