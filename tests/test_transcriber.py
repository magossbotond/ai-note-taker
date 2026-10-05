"""Unit tests for app.transcription.transcriber with a fake WhisperModel
(no model files, GPU or audio decoding needed)."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from app.core.errors import OperationCancelled
from app.transcription import transcriber as tm


class FakeWhisperModel:
    """Records how it was built/called; `fail_on_cuda` mimics a lazy CUDA failure."""

    instances = []
    fail_on_cuda = False

    def __init__(self, path, device, compute_type):
        self.path, self.device, self.compute_type = path, device, compute_type
        self.transcribe_kwargs = None
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs = kwargs
        if self.device == "cuda" and FakeWhisperModel.fail_on_cuda:
            raise RuntimeError("Library cublas64_12.dll is not found")
        segments = (SimpleNamespace(text=f" part {i} ") for i in range(3))
        return segments, SimpleNamespace(language="hu", language_probability=1.0)


class TranscriberTestCase(unittest.TestCase):
    def setUp(self):
        FakeWhisperModel.instances = []
        FakeWhisperModel.fail_on_cuda = False
        for patcher in (
            mock.patch.object(tm, "WhisperModel", FakeWhisperModel),
            mock.patch.object(tm, "download_model", return_value="/models/medium"),
            mock.patch.object(tm, "_CUDA_AVAILABLE", False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        handle, self.wav = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        self.addCleanup(os.remove, self.wav)


class ResolveModelPathTests(unittest.TestCase):
    def test_uses_the_local_cache_without_downloading(self):
        with mock.patch.object(tm, "download_model", return_value="/cache/medium") as dl:
            self.assertEqual(tm._resolve_model_path("medium"), "/cache/medium")
        dl.assert_called_once_with("medium", local_files_only=True)

    def test_downloads_only_when_not_cached(self):
        with mock.patch.object(tm, "download_model", side_effect=[OSError("not cached"), "/new/medium"]) as dl:
            self.assertEqual(tm._resolve_model_path("medium"), "/new/medium")
        self.assertEqual(dl.call_args_list[1], mock.call("medium"))

    def test_invalid_model_name_is_not_retried_as_a_download(self):
        with mock.patch.object(tm, "download_model", side_effect=ValueError("Invalid model size")) as dl:
            with self.assertRaises(ValueError):
                tm._resolve_model_path("nonsense")
        self.assertEqual(dl.call_count, 1)


class TranscribeTests(TranscriberTestCase):
    def test_joins_segments_and_passes_the_anti_hallucination_settings(self):
        transcriber = tm.LocalAudioTranscriber()
        self.assertEqual(transcriber.transcribe(self.wav, language="hu"), "part 0 part 1 part 2")
        kwargs = FakeWhisperModel.instances[0].transcribe_kwargs
        self.assertEqual(kwargs["language"], "hu")
        self.assertTrue(kwargs["vad_filter"])
        self.assertFalse(kwargs["condition_on_previous_text"])

    def test_model_is_loaded_from_the_resolved_local_path(self):
        tm.LocalAudioTranscriber()
        self.assertEqual(FakeWhisperModel.instances[0].path, "/models/medium")

    def test_missing_audio_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            tm.LocalAudioTranscriber().transcribe("does-not-exist.wav")

    def test_cancel_stops_with_operation_cancelled_not_a_wrapped_error(self):
        seen = []
        def cancel_after_first():
            seen.append(1)
            return len(seen) > 1
        with self.assertRaises(OperationCancelled):
            tm.LocalAudioTranscriber().transcribe(self.wav, should_cancel=cancel_after_first)

    def test_unloaded_transcriber_gives_a_clear_error(self):
        transcriber = tm.LocalAudioTranscriber()
        transcriber.unload()
        with self.assertRaisesRegex(RuntimeError, "unloaded"):
            transcriber.transcribe(self.wav)


class GpuFallbackTests(TranscriberTestCase):
    def test_lazy_cuda_failure_falls_back_to_cpu_and_still_returns_text(self):
        FakeWhisperModel.fail_on_cuda = True
        with mock.patch.object(tm, "_CUDA_AVAILABLE", True):
            transcriber = tm.LocalAudioTranscriber()
            self.assertEqual(transcriber.device, "cuda")
            self.assertEqual(transcriber.transcribe(self.wav), "part 0 part 1 part 2")
        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(transcriber.compute_type, "int8")


if __name__ == "__main__":
    unittest.main()
