"""IndicConformer ASR (NeMo hybrid RNNT)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from indic_speech_translation.config import ASRConfig, PreprocessConfig
from indic_speech_translation.preprocessing import preprocess_to_tempfile
from indic_speech_translation.utils import PathLike, flatten_to_str, get_logger

logger = get_logger(__name__)


@dataclass
class ASRHypothesis:
    text: str
    audio_path: str
    language_id: str


class IndicConformerASR:
    """Frozen IndicConformer checkpoint; transcribe WAV paths."""

    def __init__(
        self,
        config: ASRConfig,
        preprocess: Optional[PreprocessConfig] = None,
        device: Optional[str] = None,
        model: Any = None,
    ) -> None:
        self.config = config
        self.preprocess = preprocess or PreprocessConfig()
        self.device = device
        self._model = model
        if self._model is None:
            self._model = self._load_model()

    @classmethod
    def from_config(
        cls,
        asr_config: ASRConfig,
        preprocess: Optional[PreprocessConfig] = None,
        device: Optional[str] = None,
    ) -> "IndicConformerASR":
        return cls(config=asr_config, preprocess=preprocess, device=device)

    def _load_model(self) -> Any:
        try:
            import torch
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ImportError(
                "NeMo required for IndicConformer. Install AI4Bharat NeMo "
                "(nemo-v2); see docs/reproducibility.md."
            ) from exc

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading IndicConformer: %s", self.config.model_id)
        model = nemo_asr.models.ASRModel.from_pretrained(self.config.model_id)
        model.freeze()
        model = model.to(device)
        if hasattr(model, "cur_decoder"):
            model.cur_decoder = self.config.decoder
        self._apply_decoding_strategy(model)
        return model

    def _apply_decoding_strategy(self, model: Any) -> None:
        """Match notebook wiring: mutate cfg.decoding, then change_decoding_strategy."""
        strategy = (self.config.decoding_strategy or "greedy").lower()
        if not hasattr(model, "cfg") or not hasattr(model.cfg, "decoding"):
            logger.warning("Model has no cfg.decoding; skipping decoding setup")
            return

        model.cfg.decoding.strategy = strategy
        if strategy == "beam":
            beam_cfg = getattr(model.cfg.decoding, "beam", None)
            if beam_cfg is not None and hasattr(beam_cfg, "beam_size"):
                beam_cfg.beam_size = int(self.config.beam_size)

        if hasattr(model, "change_decoding_strategy"):
            try:
                model.change_decoding_strategy(model.cfg.decoding)
            except Exception as exc:  # pragma: no cover - NeMo version dependent
                logger.warning("change_decoding_strategy failed: %s", exc)

    def transcribe_file(self, audio_path: PathLike, cleanup_temp: bool = True) -> ASRHypothesis:
        path = Path(audio_path)
        temp_path, _ = preprocess_to_tempfile(path, cfg=self.preprocess)
        try:
            text = self._transcribe_paths([temp_path])[0]
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

    def _transcribe_paths(self, paths: Sequence[Union[str, Path]]) -> List[str]:
        path_strs = [str(p) for p in paths]
        raw = self._call_transcribe(path_strs)

        # Hybrid NeMo builds sometimes return (hyps, …)
        if isinstance(raw, tuple) and raw:
            raw = raw[0]

        if isinstance(raw, (list, tuple)) and len(raw) == len(path_strs):
            return [flatten_to_str(item) for item in raw]

        text = flatten_to_str(raw)
        if len(path_strs) == 1:
            return [text]
        logger.warning(
            "Unexpected ASR batch return (n_paths=%d); repeating flattened text",
            len(path_strs),
        )
        return [text] * len(path_strs)

    def _call_transcribe(self, path_strs: List[str]) -> Any:
        """Call model.transcribe with notebook-compatible kwargs."""
        kwargs = {"language_id": self.config.language_id, "batch_size": len(path_strs)}
        try:
            return self._model.transcribe(path_strs, **kwargs)
        except TypeError:
            # Some NeMo builds reject batch_size
            try:
                return self._model.transcribe(path_strs, language_id=self.config.language_id)
            except TypeError:
                return self._model.transcribe(path_strs)
