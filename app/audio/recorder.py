"""Dual-source audio recording for Windows: microphone + system audio (WASAPI loopback).

Uses `pyaudiowpatch` (a WASAPI-patched PyAudio fork). Streams are opened via
PortAudio's callback API, so PortAudio drives capture on its own threads - we
don't hand-roll a blocking read-loop.

Devices are resolved fresh every time start() runs, so whichever mic/speaker
is default *at that moment* gets used. A background watcher thread polls for
a default-device change (e.g. Bluetooth or wired headphones being connected/
disconnected) WHILE a recording is in progress, and hot-swaps the live
streams onto the new device. A device switch may also change the sample rate
or channel count, so audio recorded before and after a swap is kept as
separate segments and resampled to a common rate before being stitched
together in stop().

Bluetooth headsets commonly switch into a "Hands-Free/AG Audio" profile
(especially once the mic is actively used, e.g. during a call), which often
only supports 16000 Hz or 8000 Hz rather than 44100 Hz, and sometimes doesn't
expose a WASAPI loopback stream at all. Two things handle this:

  1. Opening any stream tries a sequence of candidate sample rates - the
     device's own reported native rate first, then 44100 -> 16000 -> 8000 -
     instead of assuming a single hardcoded rate will always work.
  2. If system-audio loopback fails outright after exhausting every
     candidate rate (or no loopback device can be found at all), that is
     treated as non-fatal: a warning is printed and recording continues with
     the microphone alone. The device watcher keeps retrying in the
     background, so if the headset later drops out of Hands-Free mode (or is
     disconnected back to the regular speakers), loopback capture resumes
     automatically without restarting the recording.

Sources are kept on SEPARATE stereo channels (mic = left, system = right) by
default instead of being summed into mono. Without headphones, the mic
acoustically picks up the same audio the loopback is already capturing
directly - summing a direct digital copy with a delayed, reverberant acoustic
copy of the same signal is exactly what causes an audible echo. Keeping them
on separate channels avoids ever adding the duplicated content together.
"""
import logging
import threading
import queue

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Records microphone + system-audio loopback simultaneously into one WAV file."""

    DEVICE_POLL_INTERVAL = 2.0  # seconds between checks for a default-device change
    FALLBACK_RATES = (44100, 16000, 8000)  # tried, in order, after the device's own native rate

    def __init__(self, samplerate: int = 44100, blocksize: int = 1024, mix_mode: str = "separate"):
        """
        mix_mode:
            "separate" (default, recommended) - mic on the left channel, system
                audio on the right channel. No summing, so no echo, regardless
                of whether headphones are used.
            "sum" - legacy mono downmix (mic + system audio added together).
                Only use this if you are certain you'll always be wearing
                headphones; otherwise the mic's acoustic pickup of the speaker
                output will double up with the direct loopback copy and echo.
        """
        if mix_mode not in ("separate", "sum"):
            raise ValueError("mix_mode must be 'separate' or 'sum'")

        self.target_samplerate = samplerate
        self.blocksize = blocksize
        self.mix_mode = mix_mode

        self._pa = None
        self._mic_stream = None
        self._speaker_stream = None

        self._mic_queue = None
        self._speaker_queue = None
        self._mic_rate = None
        self._speaker_rate = None
        self._mic_name = None
        self._speaker_name = None  # None whenever loopback is currently unavailable

        # Audio recorded before each device swap, kept separate because it may
        # be at a different sample rate/channel count than later segments.
        self._mic_segments = []
        self._speaker_segments = []

        self._is_recording = False
        self._lock = threading.Lock()

        self._watch_thread = None
        self._watch_stop_event = threading.Event()

    # ---- device discovery ---------------------------------------------

    @staticmethod
    def _get_default_loopback_device(pa):
        """Finds the WASAPI loopback device that mirrors the current default speaker."""
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as exc:
            raise RuntimeError(
                "WASAPI is not available on this system. System-audio loopback "
                "recording requires Windows."
            ) from exc

        default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if default_speakers.get("isLoopbackDevice", False):
            return default_speakers

        for loopback in pa.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                return loopback

        raise RuntimeError(
            f"Could not find a WASAPI loopback device for the current default speaker "
            f"('{default_speakers['name']}'). Some Bluetooth/USB headsets do not expose "
            "a loopback stream while in call/headset mode."
        )

    @staticmethod
    def _make_callback(target_queue, channels):
        """Builds a callback bound to its own queue/channel count, independent of
        whatever the recorder's current device happens to be by the time it fires."""

        def _callback(in_data, frame_count, time_info, status):
            data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels)
            target_queue.put(data.copy())
            return (None, pyaudio.paContinue)

        return _callback

    def _open_single_stream(self, pa, device_info):
        """Opens an input stream on device_info, trying its native sample rate first
        and then falling back through FALLBACK_RATES. Raises RuntimeError only if
        every candidate rate fails."""
        channels = int(device_info["maxInputChannels"]) or 1
        native_rate = int(device_info.get("defaultSampleRate") or 0)

        candidate_rates = []
        for rate in (native_rate, *self.FALLBACK_RATES):
            if rate and rate not in candidate_rates:
                candidate_rates.append(rate)

        last_error = None
        for rate in candidate_rates:
            try:
                data_queue = queue.Queue()
                stream = pa.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device_info["index"],
                    frames_per_buffer=self.blocksize,
                    stream_callback=self._make_callback(data_queue, channels),
                )
                stream.start_stream()
                return stream, data_queue, rate, channels
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "'%s' failed to open at %s Hz (%s); trying next sample rate...",
                    device_info["name"], rate, exc,
                )

        raise RuntimeError(
            f"Could not open '{device_info['name']}' at any of {candidate_rates} Hz: {last_error}"
        )

    def _open_streams(self, pa):
        """Opens fresh mic + loopback streams against whatever is currently default.

        The microphone is required - if it can't be opened at any fallback rate,
        this raises. System-audio loopback is best-effort: if it fails entirely
        (common with Bluetooth headsets in Hands-Free mode), a warning is printed
        and recording proceeds with the microphone only.
        """
        mic_info = pa.get_default_input_device_info()
        try:
            mic_stream, mic_queue, mic_rate, mic_channels = self._open_single_stream(pa, mic_info)
        except Exception as exc:
            raise RuntimeError(f"Could not open microphone '{mic_info['name']}': {exc}") from exc

        logger.info("Microphone: %s (%s Hz)", mic_info["name"], mic_rate)

        speaker_stream = speaker_queue = speaker_rate = None
        speaker_name = None
        try:
            speaker_info = self._get_default_loopback_device(pa)
            speaker_stream, speaker_queue, speaker_rate, _ = self._open_single_stream(pa, speaker_info)
            speaker_name = speaker_info["name"]
            logger.info("System-audio loopback: %s (%s Hz)", speaker_name, speaker_rate)
        except Exception as exc:
            logger.warning(
                "System audio loopback unavailable - continuing with microphone only. Reason: %s",
                exc,
            )

        return {
            "mic_stream": mic_stream,
            "mic_queue": mic_queue,
            "mic_rate": mic_rate,
            "mic_name": mic_info["name"],
            "speaker_stream": speaker_stream,
            "speaker_queue": speaker_queue,
            "speaker_rate": speaker_rate,
            "speaker_name": speaker_name,
        }

    def _apply_stream_state(self, state):
        self._mic_stream = state["mic_stream"]
        self._speaker_stream = state["speaker_stream"]
        self._mic_queue = state["mic_queue"]
        self._speaker_queue = state["speaker_queue"]
        self._mic_rate = state["mic_rate"]
        self._speaker_rate = state["speaker_rate"]
        self._mic_name = state["mic_name"]
        self._speaker_name = state["speaker_name"]

    def _close_streams(self):
        for stream in (self._mic_stream, self._speaker_stream):
            if stream is None:
                continue
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass  # device may already be gone if it was just unplugged

    def _flush_active_segment(self):
        """Drains whatever is currently queued into a finished segment, guarding
        against system audio never having been available at all."""
        self._mic_segments.append((self._drain_queue(self._mic_queue), self._mic_rate))
        if self._speaker_queue is not None:
            self._speaker_segments.append((self._drain_queue(self._speaker_queue), self._speaker_rate))

    # ---- background device-change watcher ------------------------------

    def _probe_default_names(self, pa):
        """Returns (mic_name, speaker_name); speaker_name is None if loopback is
        currently unavailable (e.g. Bluetooth headset in Hands-Free mode)."""
        mic_name = pa.get_default_input_device_info()["name"]
        try:
            speaker_name = self._get_default_loopback_device(pa)["name"]
        except Exception:
            speaker_name = None
        return mic_name, speaker_name

    def _watch_devices_loop(self):
        """Detects a headphone connect/disconnect (default device change), OR
        system audio loopback becoming available/unavailable, during an active
        recording, and hot-swaps the streams to keep following it."""
        while not self._watch_stop_event.wait(self.DEVICE_POLL_INTERVAL):
            probe_pa = None
            try:
                probe_pa = pyaudio.PyAudio()
                mic_name, speaker_name = self._probe_default_names(probe_pa)
                changed = mic_name != self._mic_name or speaker_name != self._speaker_name
                if changed:
                    logger.info("Audio device state changed - switching streams...")
                    self._swap_streams(probe_pa)
                    probe_pa = None  # ownership transferred into self._pa
            except Exception as exc:
                logger.warning("Device watch check failed (will retry): %s", exc)
            finally:
                if probe_pa is not None:
                    probe_pa.terminate()

    def _swap_streams(self, new_pa):
        with self._lock:
            if not self._is_recording:
                new_pa.terminate()
                return

            self._flush_active_segment()

            self._close_streams()
            self._pa.terminate()

            self._pa = new_pa
            state = self._open_streams(self._pa)
            self._apply_stream_state(state)

    # ---- public API -----------------------------------------------------

    def start(self):
        """Starts recording both the microphone and system audio via WASAPI callbacks."""
        if self._is_recording:
            logger.warning("Already recording.")
            return

        self._pa = pyaudio.PyAudio()
        try:
            state = self._open_streams(self._pa)
        except Exception:
            self._pa.terminate()  # don't leak the PortAudio instance if no mic could be opened
            self._pa = None
            raise
        self._apply_stream_state(state)

        self._mic_segments = []
        self._speaker_segments = []

        self._is_recording = True
        self._watch_stop_event.clear()
        self._watch_thread = threading.Thread(target=self._watch_devices_loop, daemon=True)
        self._watch_thread.start()
        logger.info("Recording microphone + system audio...")

    def stop(self, filename: str):
        """Stops recording, stitches together every segment, and writes the WAV file.

        Returns the saved filename, or None if no audio was captured.
        """
        if not self._is_recording:
            logger.warning("Not currently recording.")
            return None

        self._is_recording = False
        self._watch_stop_event.set()
        self._watch_thread.join()

        with self._lock:
            self._close_streams()
            self._pa.terminate()
            self._flush_active_segment()

        mic_audio = self._assemble_segments(self._mic_segments)
        speaker_audio = self._assemble_segments(self._speaker_segments)

        if mic_audio is None and speaker_audio is None:
            logger.warning("No audio captured.")
            return None

        self._warn_if_silent("Microphone", mic_audio)
        self._warn_if_silent("System audio", speaker_audio)

        combined = self._combine(mic_audio, speaker_audio)
        sf.write(filename, combined, self.target_samplerate)
        logger.info("Recording saved to %s", filename)
        return filename

    # ---- audio assembly ---------------------------------------------------

    def _assemble_segments(self, segments):
        """Converts each segment to mono, resamples it to the target rate (segments
        recorded on different devices/rates - e.g. before and after a Bluetooth
        profile switch - won't match), and concatenates them in recording order."""
        pieces = []
        for audio, rate in segments:
            if audio is None:
                continue
            mono = self._to_mono(audio)
            pieces.append(self._resample(mono, rate, self.target_samplerate))
        if not pieces:
            return None
        return np.concatenate(pieces, axis=0)

    @staticmethod
    def _drain_queue(source_queue):
        chunks = []
        while not source_queue.empty():
            chunks.append(source_queue.get())
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _resample(audio, orig_rate, target_rate):
        """Simple linear-interpolation resampler (good enough for speech, no extra
        dependency). Works regardless of which fallback rate (44100/16000/8000/
        device-native) a given segment was actually recorded at."""
        if audio is None or orig_rate == target_rate:
            return audio

        n_samples_new = int(round(len(audio) * target_rate / orig_rate))
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, num=n_samples_new)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    @staticmethod
    def _warn_if_silent(label, audio, threshold: float = 1e-4):
        """Flags a captured track that is essentially silent, which usually means the
        wrong device was recorded (e.g. loopback on a device that doesn't support it)."""
        if audio is None:
            logger.warning("%s captured no data at all.", label)
            return
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < threshold:
            logger.warning(
                "%s is essentially silent (RMS=%.6f). If this is 'System audio', the "
                "current output device may not support WASAPI loopback (common with "
                "some Bluetooth/USB headsets in call mode).",
                label, rms,
            )

    @staticmethod
    def _to_mono(audio):
        if audio.shape[1] == 1:
            return audio[:, 0]
        return audio.mean(axis=1)

    @staticmethod
    def _pad_to_same_length(a, b):
        if a is None:
            a = np.zeros_like(b)
        if b is None:
            b = np.zeros_like(a)
        diff = len(a) - len(b)
        if diff > 0:
            b = np.pad(b, (0, diff))
        elif diff < 0:
            a = np.pad(a, (0, -diff))
        return a, b

    def _combine(self, mic_audio, speaker_audio):
        """Combines mic + system audio. Works the same way whether speaker_audio is
        a fully recorded track, a partial one (loopback dropped in/out mid-recording),
        or entirely None (loopback was never available) - the missing portion is
        just silence, and the output format doesn't change."""
        mic_audio, speaker_audio = self._pad_to_same_length(mic_audio, speaker_audio)

        if self.mix_mode == "sum":
            mixed = mic_audio + speaker_audio
            peak = np.max(np.abs(mixed))
            if peak > 1.0:
                mixed = mixed / peak
            return mixed[:, None].astype(np.float32)

        # "separate": mic -> left channel, system audio -> right channel.
        stereo = np.stack([mic_audio, speaker_audio], axis=1)
        peak = np.max(np.abs(stereo))
        if peak > 1.0:
            stereo = stereo / peak
        return stereo.astype(np.float32)

    @property
    def is_recording(self) -> bool:
        return self._is_recording
