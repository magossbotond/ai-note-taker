"""Unit tests for the audio-processing helpers of app.audio.recorder.AudioRecorder.
They need no audio device: only the pure resample/mix/assemble functions run."""
import unittest

import numpy as np

from app.audio.recorder import AudioRecorder


def sine(frequency: float, rate: int, seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class ResampleTests(unittest.TestCase):
    def test_same_rate_returns_input_unchanged(self):
        audio = sine(440, 44100)
        self.assertIs(AudioRecorder._resample(audio, 44100, 44100), audio)

    def test_none_passes_through(self):
        self.assertIsNone(AudioRecorder._resample(None, 48000, 44100))

    def test_output_length_matches_target_rate(self):
        out = AudioRecorder._resample(sine(440, 48000), 48000, 44100)
        self.assertEqual(len(out), 44100)
        self.assertEqual(out.dtype, np.float32)

    def test_signal_shape_is_preserved(self):
        out = AudioRecorder._resample(sine(440, 16000), 16000, 44100)
        self.assertAlmostEqual(float(np.max(np.abs(out))), 0.5, delta=0.01)


class MonoAndPaddingTests(unittest.TestCase):
    def test_to_mono_averages_channels(self):
        stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(AudioRecorder._to_mono(stereo), [0.5, 0.5])

    def test_to_mono_keeps_single_channel(self):
        mono = np.array([[0.1], [0.2]], dtype=np.float32)
        np.testing.assert_allclose(AudioRecorder._to_mono(mono), [0.1, 0.2])

    def test_pad_to_same_length_pads_the_shorter_one(self):
        a, b = AudioRecorder._pad_to_same_length(np.ones(5), np.ones(3))
        self.assertEqual((len(a), len(b)), (5, 5))
        np.testing.assert_array_equal(b, [1, 1, 1, 0, 0])

    def test_pad_to_same_length_replaces_missing_track_with_silence(self):
        a, b = AudioRecorder._pad_to_same_length(np.ones(4), None)
        np.testing.assert_array_equal(b, np.zeros(4))


class CombineTests(unittest.TestCase):
    def test_separate_mode_puts_mic_left_and_system_right(self):
        recorder = AudioRecorder(mix_mode="separate")
        out = recorder._combine(np.full(4, 0.25, np.float32), np.full(4, -0.5, np.float32))
        self.assertEqual(out.shape, (4, 2))
        np.testing.assert_allclose(out[:, 0], 0.25)
        np.testing.assert_allclose(out[:, 1], -0.5)

    def test_separate_mode_without_system_audio_leaves_right_channel_silent(self):
        out = AudioRecorder(mix_mode="separate")._combine(np.full(4, 0.3, np.float32), None)
        np.testing.assert_allclose(out[:, 1], 0.0)

    def test_sum_mode_is_mono_and_never_clips(self):
        out = AudioRecorder(mix_mode="sum")._combine(
            np.full(4, 0.9, np.float32), np.full(4, 0.9, np.float32)
        )
        self.assertEqual(out.shape, (4, 1))
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_loud_stereo_is_normalized_to_full_scale(self):
        out = AudioRecorder(mix_mode="separate")._combine(
            np.full(4, 2.0, np.float32), np.full(4, 1.0, np.float32)
        )
        self.assertAlmostEqual(float(np.max(np.abs(out))), 1.0)

    def test_invalid_mix_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            AudioRecorder(mix_mode="nonsense")


class AssembleSegmentsTests(unittest.TestCase):
    def test_segments_at_different_rates_are_joined_at_the_target_rate(self):
        recorder = AudioRecorder(samplerate=44100)
        # e.g. 1 s on speakers at 48 kHz, then 1 s on a Bluetooth headset at 16 kHz
        segments = [
            (sine(440, 48000)[:, None], 48000),
            (sine(440, 16000)[:, None], 16000),
        ]
        out = recorder._assemble_segments(segments)
        self.assertEqual(len(out), 2 * 44100)

    def test_empty_segments_give_none(self):
        self.assertIsNone(AudioRecorder()._assemble_segments([(None, 44100)]))


if __name__ == "__main__":
    unittest.main()
