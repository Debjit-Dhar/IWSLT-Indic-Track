"""IndicTrans2 translation (+ optional LoRA adapter)."""

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


class IndicTrans2Translator:
    """IndicTrans2 base model with optional PEFT adapter."""

    def __init__(
        self,
        config: MTConfig,
        src_lang: str,
        tgt_lang: str,
        generation: Optional[GenerationConfig] = None,
        device: Optional[str] = None,
        model: Any = None,
        tokenizer: Any = None,
        processor: Any = None,
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
        self.processor = processor
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
    ) -> "IndicTrans2Translator":
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
                "transformers is required for IndicTrans2. Install package dependencies."
            ) from exc

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading IndicTrans2 base: %s", self.config.base_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.base_model_id,
            trust_remote_code=True,
        )

        if self.config.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("peft is required to load LoRA adapters") from exc
            logger.info("Loading LoRA adapter: %s", self.config.adapter_path)
            self.model = PeftModel.from_pretrained(self.model, self.config.adapter_path)

        self.model.to(device)
        self.model.eval()

        try:
            from IndicTransToolkit import IndicProcessor

            self.processor = IndicProcessor(inference=True)
        except ImportError:
            logger.warning(
                "IndicTransToolkit not installed; falling back to raw tokenizer input"
            )
            self.processor = None

    def translate(self, text: str) -> TranslationOutput:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: Sequence[str]) -> List[TranslationOutput]:
        if not texts:
            return []

        import torch

        clean = [t.strip() for t in texts]
        if self.processor is not None:
            batch = self.processor.preprocess_batch(
                clean, src_lang=self.src_lang, tgt_lang=self.tgt_lang
            )
        else:
            batch = clean

        encoded = self.tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                num_beams=self.generation.num_beams,
                max_length=self.generation.max_length,
            )

        decoded = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if self.processor is not None and hasattr(self.processor, "postprocess_batch"):
            decoded = self.processor.postprocess_batch(decoded, lang=self.tgt_lang)

        return [
            TranslationOutput(
                source_text=src,
                translation=hyp.strip(),
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
            )
            for src, hyp in zip(clean, decoded)
        ]
