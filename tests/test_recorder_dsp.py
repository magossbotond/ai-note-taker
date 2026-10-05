"""Unit tests for app.audio.recorder.AudioRecorder: the disk spooling and the
block-wise assembly of the final WAV. No audio device is needed - segments are
filled by hand instead of by PortAudio callbacks."""
import os
import queue
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf

from app.audio import recorder as recorder_module
from app.audio.recorder import AudioRecorder, _Segment

RATE = 1000  # tiny target rate keeps the tests fast; BLOCK_SECONDS=1 -> 1000-frame blocks


def sine(frequency: float, rate: int, seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class RecorderTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="recorder_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.recorder = AudioRecorder(samplerate=RATE)
        self.recorder.BLOCK_SECONDS = 1
        self.recorder._tmp_dir = self.tmp
        self.counter = 0

    def segment(self, audio: np.ndarray, rate: int = RATE) -> _Segment:
        """A finished segment holding `audio` (shape (frames,) or (frames, channels))."""
        audio = audio.reshape(len(audio), -1).astype(np.float32)
        self.counter += 1
        seg = _Segment(os.path.join(self.tmp, f"seg{self.counter}.f32"), rate, audio.shape[1])
        seg.append(audio)
        seg.close()
        return seg

    def record(self, mic_segments, speaker_segments, filename="out.wav"):
        self.recorder._mic_segments = list(mic_segments)
        self.recorder._speaker_segments = list(speaker_segments)
        path = os.path.join(self.tmp, filename)
        self.assertEqual(self.recorder._write_recording(path), path)
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        self.assertEqual(rate, RATE)
        return audio


class MonoTests(unittest.TestCase):
    def test_to_mono_averages_channels(self):
        stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(AudioRecorder._to_mono(stereo), [0.5, 0.5])

    def test_to_mono_keeps_single_channel(self):
        mono = np.array([[0.1], [0.2]], dtype=np.float32)
        np.testing.assert_allclose(AudioRecorder._to_mono(mono), [0.1, 0.2])

    def test_invalid_mix_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            AudioRecorder(mix_mode="nonsense")


class ResampleTests(RecorderTestCase):
    def reference(self, mono: np.ndarray, rate: int) -> np.ndarray:
        """Whole-array linear resampling, the behaviour the block-wise version must match."""
        n_out = int(round(len(mono) * RATE / rate))
        return np.interp(np.linspace(0, len(mono) - 1, n_out), np.arange(len(mono)), mono)

    def test_blockwise_resampling_matches_whole_array_resampling(self):
        rng = np.random.default_rng(1)
        for rate in (1500, 800, 3000):  # up- and downsampling, several blocks each
            mono = rng.standard_normal(rate * 5).astype(np.float32)
            blocks = list(self.recorder._iter_mono_blocks(self.segment(mono, rate)))
            self.assertGreater(len(blocks), 1)
            np.testing.assert_allclose(np.concatenate(blocks), self.reference(mono, rate), atol=1e-5)

    def test_same_rate_passes_audio_through_unchanged(self):
        mono = np.random.default_rng(2).standard_normal(2500).astype(np.float32)
        out = np.concatenate(list(self.recorder._iter_mono_blocks(self.segment(mono))))
        np.testing.assert_array_equal(out, mono)

    def test_stereo_segments_are_downmixed(self):
        stereo = np.stack([np.ones(100), np.zeros(100)], axis=1)
        out = np.concatenate(list(self.recorder._iter_mono_blocks(self.segment(stereo))))
        np.testing.assert_allclose(out, 0.5)


class WriteRecordingTests(RecorderTestCase):
    def test_separate_mode_puts_mic_left_and_system_right(self):
        audio = self.record([self.segment(np.full(2500, 0.25))], [self.segment(np.full(2500, -0.5))])
        self.assertEqual(audio.shape, (2500, 2))
        np.testing.assert_allclose(audio[:, 0], 0.25, atol=1e-3)
        np.testing.assert_allclose(audio[:, 1], -0.5, atol=1e-3)

    def test_segments_from_different_devices_are_joined_in_order(self):
        # 1 s recorded at 1500 Hz, then 1 s at 800 Hz (e.g. headset switched profiles)
        audio = self.record(
            [self.segment(np.full(1500, 0.2), 1500), self.segment(np.full(800, 0.4), 800)], []
        )
        self.assertEqual(len(audio), 2000)
        np.testing.assert_allclose(audio[:900, 0], 0.2, atol=1e-3)
        np.testing.assert_allclose(audio[1100:, 0], 0.4, atol=1e-3)

    def test_missing_system_audio_leaves_the_right_channel_silent(self):
        audio = self.record([self.segment(np.full(1500, 0.3))], [])
        self.assertEqual(audio.shape, (1500, 2))
        np.testing.assert_allclose(audio[:, 1], 0.0)

    def test_shorter_track_is_padded_with_silence(self):
        audio = self.record([self.segment(np.full(3000, 0.3))], [self.segment(np.full(1200, 0.3))])
        self.assertEqual(len(audio), 3000)
        np.testing.assert_allclose(audio[1300:, 1], 0.0)

    def test_sum_mode_gives_mono(self):
        self.recorder.mix_mode = "sum"
        audio = self.record([self.segment(np.full(1000, 0.25))], [self.segment(np.full(1000, 0.25))])
        self.assertEqual(audio.shape, (1000, 1))
        np.testing.assert_allclose(audio, 0.5, atol=1e-3)

    def test_too_loud_mix_is_scaled_to_full_scale_instead_of_clipping(self):
        self.recorder.mix_mode = "sum"
        audio = self.record([self.segment(np.full(1000, 0.9))], [self.segment(np.full(1000, 0.9))])
        self.assertAlmostEqual(float(np.max(np.abs(audio))), 1.0, places=2)

    def test_no_audio_at_all_returns_none(self):
        self.assertIsNone(self.recorder._write_recording(os.path.join(self.tmp, "out.wav")))

    def test_silent_source_is_reported(self):
        with self.assertLogs("app.audio.recorder", level="WARNING") as logs:
            self.record([self.segment(np.zeros(1000))], [self.segment(sine(5, RATE))])
        self.assertTrue(any("Microphone is essentially silent" in line for line in logs.output))
        self.assertFalse(any("System audio is essentially silent" in line for line in logs.output))


class SpoolingTests(RecorderTestCase):
    def test_spool_moves_queued_audio_to_disk(self):
        mic_queue, speaker_queue = queue.Queue(), queue.Queue()
        self.recorder._mic_queue, self.recorder._speaker_queue = mic_queue, speaker_queue
        self.recorder._mic_segment = _Segment(os.path.join(self.tmp, "m.f32"), RATE, 1)
        self.recorder._speaker_segment = _Segment(os.path.join(self.tmp, "s.f32"), RATE, 2)
        self.addCleanup(self.recorder._mic_segment.close)
        self.addCleanup(self.recorder._speaker_segment.close)
        for _ in range(3):
            mic_queue.put(np.ones((100, 1), np.float32))
        speaker_queue.put(np.ones((50, 2), np.float32))

        self.recorder._spool_active()

        self.assertEqual(mic_queue.qsize(), 0)  # nothing is left piling up in RAM
        self.assertEqual(self.recorder._mic_segment.frames, 300)
        self.assertEqual(self.recorder._speaker_segment.frames, 50)

    def test_flush_keeps_only_segments_that_contain_audio(self):
        self.recorder._mic_queue = queue.Queue()
        self.recorder._mic_queue.put(np.ones((10, 1), np.float32))
        self.recorder._mic_segment = _Segment(os.path.join(self.tmp, "m.f32"), RATE, 1)
        self.recorder._speaker_segment = _Segment(os.path.join(self.tmp, "s.f32"), RATE, 1)  # stays empty

        self.recorder._flush_active_segment()

        self.assertEqual(len(self.recorder._mic_segments), 1)
        self.assertEqual(self.recorder._speaker_segments, [])
        self.assertIsNone(self.recorder._mic_segment)

    def test_discard_deletes_the_temp_folder(self):
        self.recorder._mic_segments = [self.segment(np.ones(10))]
        self.recorder._discard_spool()
        self.assertFalse(os.path.exists(self.tmp))
        self.assertIsNone(self.recorder._tmp_dir)

    def test_old_leftover_folders_are_cleaned_up_but_fresh_ones_are_kept(self):
        base = tempfile.gettempdir()
        old = tempfile.mkdtemp(prefix=recorder_module._SPOOL_PREFIX, dir=base)
        fresh = tempfile.mkdtemp(prefix=recorder_module._SPOOL_PREFIX, dir=base)
        self.addCleanup(shutil.rmtree, fresh, ignore_errors=True)
        two_days_ago = time.time() - 2 * 24 * 3600
        os.utime(old, (two_days_ago, two_days_ago))

        self.recorder._cleanup_stale_spools()

        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(fresh))


class StopAndAbortTests(RecorderTestCase):
    def make_recording(self):
        self.recorder._is_recording = True
        self.recorder._watch_thread = SimpleNamespace(join=lambda: None)
        self.recorder._pa = mock.Mock()

    def test_disk_error_while_recording_is_reported_by_stop(self):
        self.make_recording()
        self.recorder._spool_error = OSError("No space left on device")
        with self.assertRaisesRegex(RuntimeError, "disk error"):
            self.recorder.stop(os.path.join(self.tmp, "out.wav"))
        self.assertFalse(os.path.exists(self.tmp))  # temp files are cleaned up even on failure
        self.assertFalse(self.recorder.is_recording)

    def test_abort_discards_everything_and_writes_no_file(self):
        self.make_recording()
        self.recorder._mic_segments = [self.segment(np.ones(10))]
        self.recorder.abort()
        self.assertFalse(os.path.exists(self.tmp))
        self.assertFalse(self.recorder.is_recording)

    def test_abort_when_idle_does_nothing(self):
        self.recorder.abort()
        self.assertTrue(os.path.exists(self.tmp))


if __name__ == "__main__":
    unittest.main()
