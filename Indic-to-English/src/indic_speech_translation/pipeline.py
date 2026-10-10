"""IndicConformer → IndicTrans2 cascade."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from indic_speech_translation.asr.indicconformer import IndicConformerASR
from indic_speech_translation.config import ExperimentConfig, load_config
from indic_speech_translation.translation.indictrans2 import IndicTrans2Translator
from indic_speech_translation.utils import PathLike, get_logger

logger = get_logger(__name__)


@dataclass
class CascadeResult:
    audio_path: str
    asr_text: str
    translation: str
    source_lang: str
    target_lang: str
    mt_src_code: str
    mt_tgt_code: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CascadePipeline:
    """ASR then MT for Indic→English."""

    def __init__(
        self,
        config: ExperimentConfig,
        asr: Optional[IndicConformerASR] = None,
        translator: Optional[IndicTrans2Translator] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.asr = asr or IndicConformerASR.from_config(
            config.asr, preprocess=config.preprocess, device=device
        )
        self.translator = translator or IndicTrans2Translator.from_config(
            config.mt,
            src_lang=config.mt_src_code,
            tgt_lang=config.mt_tgt_code,
            generation=config.generation,
            device=device,
        )

    @classmethod
    def from_config(
        cls,
        config: PathLike | ExperimentConfig,
        device: Optional[str] = None,
        asr: Optional[IndicConformerASR] = None,
        translator: Optional[IndicTrans2Translator] = None,
    ) -> "CascadePipeline":
        if not isinstance(config, ExperimentConfig):
            config = load_config(config)
        return cls(config=config, asr=asr, translator=translator, device=device)

    def transcribe(self, audio_path: PathLike) -> str:
        return self.asr.transcribe_file(audio_path).text

    def translate_text(self, text: str) -> str:
        return self.translator.translate(text).translation

    def translate_audio(self, audio_path: PathLike) -> CascadeResult:
        hyp = self.asr.transcribe_file(audio_path)
        mt = self.translator.translate(hyp.text)
        logger.info("ASR: %s", hyp.text)
        logger.info("MT: %s", mt.translation)
        return CascadeResult(
            audio_path=str(audio_path),
            asr_text=hyp.text,
            translation=mt.translation,
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang,
            mt_src_code=self.config.mt_src_code,
            mt_tgt_code=self.config.mt_tgt_code,
            metadata={
                "asr_model_id": self.config.asr.model_id,
                "mt_base_model_id": self.config.mt.base_model_id,
                "mt_adapter_path": self.config.mt.adapter_path,
                "preprocess": {
                    "target_sr": self.config.preprocess.target_sr,
                    "lowpass_hz": self.config.preprocess.lowpass_hz,
                    "gain_db": self.config.preprocess.gain_db,
                },
            },
        )
