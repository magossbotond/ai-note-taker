"""Local speech-to-text transcription using faster-whisper.

Runs on your machine: no API keys, and audio never leaves it. Uses an NVIDIA GPU
(CUDA) when available for speed and falls back to the CPU otherwise. The model is
loaded from the local cache (see setup.bat); it is only downloaded if missing.
"""
from __future__ import annotations

import gc
import glob
import logging
import os
import site
from typing import Callable

from app.core.errors import OperationCancelled


def _register_cuda_dlls() -> None:
    """The CUDA runtime (cuBLAS + cuDNN) comes from the pip wheels `nvidia-cublas-cu12`
    and `nvidia-cudnn-cu12`, which put their DLLs under site-packages/nvidia/*/bin.
    Windows won't find them unless those folders are registered first, and this
    must happen before CTranslate2 first touches CUDA."""
    for site_dir in site.getsitepackages():
        for bin_dir in glob.glob(os.path.join(site_dir, "nvidia", "*", "bin")):
            os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dlls()

import ctranslate2  # noqa: E402  (must come after the DLL directories are registered)
from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.utils import download_model  # noqa: E402

logger = logging.getLogger(__name__)

# Asked of CTranslate2 directly, so torch isn't needed just to detect the GPU.
_CUDA_AVAILABLE = ctranslate2.get_cuda_device_count() > 0

# Decoding settings. Whisper is prone to repeating itself or inventing text on
# long recordings and on silence (e.g. an unused microphone channel):
# - vad_filter skips the non-speech parts before they reach the model;
# - condition_on_previous_text=False stops one bad segment from derailing the next.
BEAM_SIZE = 5
VAD_FILTER = True
CONDITION_ON_PREVIOUS_TEXT = False


def _resolve_model_path(model_size: str) -> str:
    """Path of the model in the local cache. Only downloads it (needs internet)
    if it isn't cached yet - a plain WhisperModel(name) would contact
    huggingface.co on every load just to check for updates."""
    try:
        return download_model(model_size, local_files_only=True)
    except ValueError:
        raise  # not a valid model name
    except Exception:
        logger.info("Model '%s' is not cached locally; downloading it...", model_size)
        return download_model(model_size)


class LocalAudioTranscriber:
    """Wraps a faster-whisper model for local audio transcription."""

    def __init__(self, model_size: str = "medium") -> None:
        """Load the whisper model, picking the best hardware config available.

        Args:
            model_size: faster-whisper model name (e.g. "medium", "large-v3").
                "medium" is the default because it gives the best accuracy/speed
                tradeoff for non-English languages like Hungarian and Romanian.
        """
        self.model_size = model_size
        self._model_path = _resolve_model_path(model_size)
        self.model = None
        if _CUDA_AVAILABLE:
            try:
                self._load("cuda", "float16")
            except Exception as exc:
                logger.warning("GPU load failed (%s); using CPU.", exc)
        if self.model is None:
            self._load("cpu", "int8")

    def _load(self, device: str, compute_type: str) -> None:
        logger.info(
            "Loading '%s' model on %s (compute_type=%s)...",
            self.model_size, device.upper(), compute_type,
        )
        self.model = WhisperModel(self._model_path, device=device, compute_type=compute_type)
        self.device = device
        self.compute_type = compute_type
        logger.info("Model loaded successfully.")

    def unload(self) -> None:
        """Release the Whisper model (and its GPU memory) so other models, e.g.
        Ollama's LLM, can use it. The instance can't transcribe afterwards."""
        self.model = None
        gc.collect()  # drop the CTranslate2 model so it frees its VRAM now
        logger.info("Model unloaded.")

    def _run(
        self,
        audio_file_path: str,
        language: str,
        should_cancel: Callable[[], bool] | None,
    ) -> str:
        segments, info = self.model.transcribe(
            audio_file_path,
            language=language,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        )

        logger.info(
            "Detected language '%s' with probability %.2f",
            info.language, info.language_probability,
        )

        # segments is a lazy generator: the audio is decoded as we iterate, so
        # this loop is also where a cancel request can take effect.
        texts = []
        for segment in segments:
            if should_cancel is not None and should_cancel():
                raise OperationCancelled("transcription cancelled")
            texts.append(segment.text.strip())
        return " ".join(texts).strip()

    def transcribe(
        self,
        audio_file_path: str,
        language: str = "hu",
        should_cancel: Callable[[], bool] | None = None,
    ) -> str:
        """Transcribe a .wav file into plain text.

        Args:
            audio_file_path: Path to the audio file to transcribe.
            language: ISO 639-1 language code (e.g. "hu" for Hungarian,
                "ro" for Romanian).
            should_cancel: Optional callable polled between segments; when it
                returns True the transcription stops with OperationCancelled.

        Returns:
            The full transcribed text as a single concatenated string.

        Raises:
            FileNotFoundError: If audio_file_path does not exist.
            OperationCancelled: If should_cancel() returned True.
            RuntimeError: If transcription failed for any other reason.
        """
        if not os.path.isfile(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        if self.model is None:
            raise RuntimeError("The model was unloaded; create a new LocalAudioTranscriber.")

        try:
            try:
                return self._run(audio_file_path, language, should_cancel)
            except OperationCancelled:
                raise
            except Exception as gpu_exc:
                if self.device != "cuda":
                    raise
                # CUDA can fail lazily (missing cuBLAS/cuDNN DLL, or out of VRAM
                # because something else holds the GPU). Reload on CPU and retry.
                logger.warning("GPU transcription failed (%s); retrying on CPU...", gpu_exc)
                self.model = None
                gc.collect()
                self._load("cpu", "int8")
                return self._run(audio_file_path, language, should_cancel)

        except OperationCancelled:
            raise
        except Exception as exc:
            # Broad catch is intentional here: this is the top-level entry point
            # for transcription and we want a clear, actionable error instead of
            # a raw traceback bubbling up into the GUI layer.
            raise RuntimeError(f"Transcription failed for '{audio_file_path}': {exc}") from exc


if __name__ == "__main__":
    # Manual check: python -m app.transcription.transcriber path\to\audio.wav [language]
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python -m app.transcription.transcriber <audio.wav> [language]")
    logging.basicConfig(level=logging.INFO)
    try:
        transcriber = LocalAudioTranscriber(model_size="medium")
        print(transcriber.transcribe(sys.argv[1], language=sys.argv[2] if len(sys.argv) > 2 else "hu"))
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"[ERROR] {exc}")
