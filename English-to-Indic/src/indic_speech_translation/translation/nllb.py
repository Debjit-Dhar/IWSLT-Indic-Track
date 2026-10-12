"""NLLB-200 translation (+ optional fine-tuned checkpoint)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from indic_speech_translation.config import GenerationConfig, MTConfig
from indic_speech_translation.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TranslationOutput:
    source_text: str
    translation: str
    src_lang: str
    tgt_lang: str


class NLLBTranslator:
    """NLLB-200 base or fine-tuned checkpoint (paper §3.2 / §4.1)."""

    def __init__(
        self,
        config: MTConfig,
        src_lang: str,
        tgt_lang: str,
        generation: Optional[GenerationConfig] = None,
        device: Optional[str] = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.config = config
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.generation = generation or GenerationConfig(
            num_beams=config.num_beams, max_length=config.max_length
        )
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        if self.model is None or self.tokenizer is None:
            self._load()

    @classmethod
    def from_config(
        cls,
        mt_config: MTConfig,
        src_lang: str,
        tgt_lang: str,
        generation: Optional[GenerationConfig] = None,
        device: Optional[str] = None,
    ) -> "NLLBTranslator":
        return cls(
            config=mt_config,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            generation=generation,
            device=device,
        )

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for NLLB. Install package dependencies."
            ) from exc

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model_path = self.config.checkpoint_path or self.config.base_model_id
        tokenizer_id = self.config.base_model_id
        logger.info("Loading NLLB model: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def _forced_bos_token_id(self) -> Optional[int]:
        tok = self.tokenizer
        if tok is None:
            return None
        if hasattr(tok, "convert_tokens_to_ids"):
            token_id = tok.convert_tokens_to_ids(self.tgt_lang)
            if token_id is not None and token_id != getattr(tok, "unk_token_id", None):
                return token_id
        lang_code_to_id = getattr(tok, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict):
            return lang_code_to_id.get(self.tgt_lang)
        return None

    def translate(self, text: str) -> TranslationOutput:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: Sequence[str]) -> List[TranslationOutput]:
        if not texts:
            return []

        import torch

        clean = [t.strip() if t is not None else "" for t in texts]
        encoded = self.tokenizer(
            clean,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        generate_kwargs: dict[str, Any] = dict(
            num_beams=self.generation.num_beams,
            max_length=self.generation.max_length,
        )
        bos_id = self._forced_bos_token_id()
        if bos_id is not None:
            generate_kwargs["forced_bos_token_id"] = bos_id

        with torch.no_grad():
            generated = self.model.generate(**encoded, **generate_kwargs)

        decoded = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [
            TranslationOutput(
                source_text=src,
                translation=hyp.strip(),
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
            )
            for src, hyp in zip(clean, decoded)
        ]
