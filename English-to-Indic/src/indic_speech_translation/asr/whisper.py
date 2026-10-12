"""Whisper ASR (paper §3.2: Whisper Small, frozen)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from indic_speech_translation.config import ASRConfig, PreprocessConfig
from indic_speech_translation.preprocessing import preprocess_to_tempfile
from indic_speech_translation.utils import PathLike, flatten_to_str, get_logger

logger = get_logger(__name__)


@dataclass
class ASRHypothesis:
    text: str
    audio_path: str
    language_id: str


class WhisperASR:
    """Frozen OpenAI Whisper checkpoint; transcribe WAV paths."""

    def __init__(
        self,
        config: ASRConfig,
        preprocess: Optional[PreprocessConfig] = None,
        device: Optional[str] = None,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        self.config = config
        self.preprocess = preprocess or PreprocessConfig()
        self.device = device
        self._model = model
        self._processor = processor
        if self._model is None:
            self._model, self._processor = self._load_model()

    @classmethod
    def from_config(
        cls,
        asr_config: ASRConfig,
        preprocess: Optional[PreprocessConfig] = None,
        device: Optional[str] = None,
    ) -> "WhisperASR":
        return cls(config=asr_config, preprocess=preprocess, device=device)

    def _load_model(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for Whisper. Install package dependencies."
            ) from exc

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading Whisper: %s", self.config.model_id)
        processor = WhisperProcessor.from_pretrained(self.config.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.config.model_id)
        model.to(device)
        model.eval()
        return model, processor

    def transcribe_file(self, audio_path: PathLike, cleanup_temp: bool = True) -> ASRHypothesis:
        path = Path(audio_path)
        temp_path, processed = preprocess_to_tempfile(path, cfg=self.preprocess)
        try:
            text = self._transcribe_waveform(processed.waveform, processed.sample_rate)
        finally:
            if cleanup_temp and temp_path.exists():
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        return ASRHypothesis(
            text=text,
            audio_path=str(path),
            language_id=self.config.language_id,
        )

    def transcribe_files(
        self, audio_paths: Sequence[PathLike], cleanup_temp: bool = True
    ) -> List[ASRHypothesis]:
        return [self.transcribe_file(p, cleanup_temp=cleanup_temp) for p in audio_paths]

    def _transcribe_waveform(self, waveform: Any, sample_rate: int) -> str:
        if hasattr(self._model, "transcribe"):
            raw = self._call_transcribe_api(waveform, sample_rate)
            return flatten_to_str(raw)

        import torch

        audio = waveform.squeeze()
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        inputs = self._processor(
            audio.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        generate_kwargs: dict[str, Any] = {
            "language": self.config.language_id,
            "task": self.config.task,
        }
        if (self.config.decoding_strategy or "beam").lower() == "beam":
            generate_kwargs["num_beams"] = int(self.config.beam_size)

        with torch.no_grad():
            predicted_ids = self._model.generate(input_features, **generate_kwargs)
        decoded = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return flatten_to_str(decoded)

    def _call_transcribe_api(
        self, waveform: Any, sample_rate: int
    ) -> Any:
        kwargs = {
            "language": self.config.language_id,
            "task": self.config.task,
            "sample_rate": sample_rate,
        }
        try:
            return self._model.transcribe(waveform, **kwargs)
        except TypeError:
            try:
                return self._model.transcribe(waveform, language=self.config.language_id)
            except TypeError:
                return self._model.transcribe(waveform)
