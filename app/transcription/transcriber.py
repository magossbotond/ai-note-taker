"""Local speech-to-text transcription using faster-whisper.

Runs fully offline (no API keys). Automatically uses an NVIDIA GPU (CUDA)
when available for speed, and falls back to CPU otherwise so it still
works on machines without a dedicated GPU.
"""
from __future__ import annotations

import gc
import glob
import logging
import os
import site

# The CUDA runtime (cuBLAS + cuDNN) comes from the pip wheels `nvidia-cublas-cu12`
# and `nvidia-cudnn-cu12`, which put their DLLs under site-packages/nvidia/*/bin.
# Windows won't find them unless those folders are registered first, and this
# must happen before CTranslate2 first touches CUDA.
for _site_dir in site.getsitepackages():
    for _bin_dir in glob.glob(os.path.join(_site_dir, "nvidia", "*", "bin")):
        os.add_dll_directory(_bin_dir)
        os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

import ctranslate2  # noqa: E402  (must come after the DLL directories are registered)
from faster_whisper import WhisperModel  # noqa: E402

logger = logging.getLogger(__name__)

# Asked of CTranslate2 directly, so torch isn't needed just to detect the GPU.
_CUDA_AVAILABLE = ctranslate2.get_cuda_device_count() > 0


class LocalAudioTranscriber:
    """Wraps a faster-whisper model for local, offline audio transcription."""

    def __init__(self, model_size: str = "medium") -> None:
        """Load the whisper model, picking the best hardware config available.

        Args:
            model_size: faster-whisper model name (e.g. "medium", "large-v3").
                "medium" is the default because it gives the best accuracy/speed
                tradeoff for non-English languages like Hungarian and Romanian.
        """
        self.model_size = model_size
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
        self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        self.device = device
        self.compute_type = compute_type
        logger.info("Model loaded successfully.")

    def unload(self) -> None:
        """Release the Whisper model (and its GPU memory) so other models, e.g.
        Ollama's LLM, can use it. The instance can't transcribe afterwards."""
        self.model = None
        gc.collect()  # drop the CTranslate2 model so it frees its VRAM now
        logger.info("Model unloaded.")

    def _run(self, audio_file_path: str, language: str) -> str:
        segments, info = self.model.transcribe(
            audio_file_path,
            language=language,
            beam_size=5,
        )

        logger.info(
            "Detected language '%s' with probability %.2f",
            info.language, info.language_probability,
        )

        # segments is a generator; each segment holds a chunk of decoded text.
        full_text = " ".join(segment.text.strip() for segment in segments)
        return full_text.strip()

    def transcribe(self, audio_file_path: str, language: str = "hu") -> str:
        """Transcribe a .wav file into plain text.

        Args:
            audio_file_path: Path to the audio file to transcribe.
            language: ISO 639-1 language code (e.g. "hu" for Hungarian,
                "ro" for Romanian).

        Returns:
            The full transcribed text as a single concatenated string.

        Raises:
            FileNotFoundError: If audio_file_path does not exist.
        """
        if not os.path.isfile(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            try:
                return self._run(audio_file_path, language)
            except Exception as gpu_exc:
                if self.device != "cuda":
                    raise
                # CUDA can fail lazily (missing cuBLAS/cuDNN DLL, or out of VRAM
                # because something else holds the GPU). Reload on CPU and retry.
                logger.warning("GPU transcription failed (%s); retrying on CPU...", gpu_exc)
                self.model = None
                gc.collect()
                self._load("cpu", "int8")
                return self._run(audio_file_path, language)

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
