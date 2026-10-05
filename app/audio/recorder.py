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
     treated as non-fatal: a warning is logged and recording continues with
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

Memory: captured audio is NOT kept in RAM for the whole recording. The watcher
thread moves it from the callback queues into raw temporary files every couple
of seconds, and stop() builds the final WAV from those files in small blocks,
so RAM use stays flat no matter how long the interview is. The temporary files
are always deleted again (and leftovers from a crashed session are cleaned up
on the next start).
"""
import glob
import logging
import os
import queue
import shutil
import tempfile
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf

logger = logging.getLogger(__name__)

_SPOOL_PREFIX = "notetaker_rec_"
_BYTES_PER_SAMPLE = 4  # float32


class _Segment:
    """A stretch of raw audio from ONE source on ONE device, spooled to disk as
    interleaved float32 frames. A device change starts a new segment because the
    sample rate / channel count may differ."""

    def __init__(self, path: str, rate: int, channels: int) -> None:
        self.path = path
        self.rate = rate
        self.channels = channels
        self.frames = 0
        self._file = open(path, "wb")

    def append(self, block: np.ndarray) -> None:
        """block: float32 array of shape (frames, channels)."""
        self._file.write(block.tobytes())
        self.frames += len(block)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def read(self, start: int, stop: int) -> np.ndarray:
        """Frames [start, stop) as a float32 array of shape (n, channels)."""
        with open(self.path, "rb") as f:
            f.seek(start * self.channels * _BYTES_PER_SAMPLE)
            raw = f.read((stop - start) * self.channels * _BYTES_PER_SAMPLE)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, self.channels)


class AudioRecorder:
    """Records microphone + system-audio loopback simultaneously into one WAV file."""

    DEVICE_POLL_INTERVAL = 2.0  # seconds between device checks (and between disk spools)
    FALLBACK_RATES = (44100, 16000, 8000)  # tried, in order, after the device's own native rate
    BLOCK_SECONDS = 10  # granularity of the disk-based processing in stop()
    STALE_SPOOL_AGE = 24 * 3600  # seconds after which a leftover temp folder is deleted

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
        self._mic_name = None
        self._speaker_name = None  # None whenever loopback is currently unavailable

        # Where the audio goes: one temp folder per recording. The *_segment
        # attributes are the files being written right now; finished segments
        # (one per device swap) move into the lists.
        self._tmp_dir = None
        self._segment_counter = 0
        self._mic_segment = None
        self._speaker_segment = None
        self._mic_segments = []
        self._speaker_segments = []
        self._spool_error = None  # first disk error hit while spooling, reported by stop()

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
        (common with Bluetooth headsets in Hands-Free mode), a warning is logged
        and recording proceeds with the microphone only.
        """
        mic_info = pa.get_default_input_device_info()
        try:
            mic_stream, mic_queue, mic_rate, mic_channels = self._open_single_stream(pa, mic_info)
        except Exception as exc:
            raise RuntimeError(f"Could not open microphone '{mic_info['name']}': {exc}") from exc

        logger.info("Microphone: %s (%s Hz)", mic_info["name"], mic_rate)

        speaker_stream = speaker_queue = speaker_rate = speaker_channels = None
        speaker_name = None
        try:
            speaker_info = self._get_default_loopback_device(pa)
            speaker_stream, speaker_queue, speaker_rate, speaker_channels = self._open_single_stream(
                pa, speaker_info
            )
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
            "mic_channels": mic_channels,
            "mic_name": mic_info["name"],
            "speaker_stream": speaker_stream,
            "speaker_queue": speaker_queue,
            "speaker_rate": speaker_rate,
            "speaker_channels": speaker_channels,
            "speaker_name": speaker_name,
        }

    def _apply_stream_state(self, state):
        self._mic_stream = state["mic_stream"]
        self._speaker_stream = state["speaker_stream"]
        self._mic_queue = state["mic_queue"]
        self._speaker_queue = state["speaker_queue"]
        self._mic_name = state["mic_name"]
        self._speaker_name = state["speaker_name"]

        # Every (re)opened stream records into a fresh segment file.
        self._mic_segment = self._new_segment("mic", state["mic_rate"], state["mic_channels"])
        self._speaker_segment = None
        if state["speaker_stream"] is not None:
            self._speaker_segment = self._new_segment(
                "spk", state["speaker_rate"], state["speaker_channels"]
            )

    def _close_streams(self):
        for stream in (self._mic_stream, self._speaker_stream):
            if stream is None:
                continue
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass  # device may already be gone if it was just unplugged

    # ---- disk spooling ----------------------------------------------------

    def _new_segment(self, prefix: str, rate: int, channels: int) -> _Segment:
        self._segment_counter += 1
        path = os.path.join(self._tmp_dir, f"{prefix}_{self._segment_counter:03d}.f32")
        return _Segment(path, rate, channels)

    @staticmethod
    def _drain_queue(source_queue):
        """Everything currently queued as one (frames, channels) array, or None."""
        chunks = []
        while True:
            try:
                chunks.append(source_queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    def _spool_active(self):
        """Moves the audio captured so far from the in-memory queues onto disk.
        Call with self._lock held."""
        for source_queue, segment in (
            (self._mic_queue, self._mic_segment),
            (self._speaker_queue, self._speaker_segment),
        ):
            if source_queue is None or segment is None:
                continue
            block = self._drain_queue(source_queue)
            if block is not None:
                segment.append(block)

    def _flush_active_segment(self):
        """Writes out whatever is still queued and finishes the current segment
        files (called before a device swap and at stop). Call with self._lock held."""
        self._spool_active()
        for segment, finished in (
            (self._mic_segment, self._mic_segments),
            (self._speaker_segment, self._speaker_segments),
        ):
            if segment is None:
                continue
            segment.close()
            if segment.frames:
                finished.append(segment)
        self._mic_segment = self._speaker_segment = None

    def _discard_spool(self):
        """Closes and deletes every temporary file of the current recording."""
        for segment in (self._mic_segment, self._speaker_segment, *self._mic_segments, *self._speaker_segments):
            if segment is not None:
                segment.close()
        self._mic_segment = self._speaker_segment = None
        self._mic_segments = []
        self._speaker_segments = []
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def _cleanup_stale_spools(self):
        """Deletes temp folders left behind by a recording that crashed or was
        killed, so raw interview audio doesn't pile up in the temp directory."""
        cutoff = time.time() - self.STALE_SPOOL_AGE
        for path in glob.glob(os.path.join(tempfile.gettempdir(), _SPOOL_PREFIX + "*")):
            try:
                if os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass

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

    def _spool_tick(self):
        """One periodic move of queued audio to disk; a disk error is remembered
        and reported by stop() instead of killing the watcher thread."""
        with self._lock:
            if not self._is_recording:
                return
            try:
                self._spool_active()
            except OSError as exc:
                if self._spool_error is None:
                    logger.error("Could not write the recording to disk: %s", exc)
                    self._spool_error = exc

    def _watch_devices_loop(self):
        """Every tick: spool queued audio to disk, then detect a headphone
        connect/disconnect (default device change), OR system audio loopback
        becoming available/unavailable, and hot-swap the streams to keep following it."""
        while not self._watch_stop_event.wait(self.DEVICE_POLL_INTERVAL):
            self._spool_tick()

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

        self._cleanup_stale_spools()
        self._tmp_dir = tempfile.mkdtemp(prefix=_SPOOL_PREFIX)
        self._segment_counter = 0
        self._mic_segments = []
        self._speaker_segments = []
        self._spool_error = None

        self._pa = pyaudio.PyAudio()
        try:
            state = self._open_streams(self._pa)
            self._apply_stream_state(state)
        except Exception:
            self._close_streams()  # a stream may be open if only the segment file failed
            self._pa.terminate()  # don't leak the PortAudio instance if no mic could be opened
            self._pa = None
            self._discard_spool()
            raise

        self._is_recording = True
        self._watch_stop_event.clear()
        self._watch_thread = threading.Thread(target=self._watch_devices_loop, daemon=True)
        self._watch_thread.start()
        logger.info("Recording microphone + system audio...")

    def _halt_capture(self):
        """Stops the watcher and the audio streams and finishes the segment files."""
        self._is_recording = False
        self._watch_stop_event.set()
        self._watch_thread.join()

        with self._lock:
            self._close_streams()
            self._pa.terminate()
            self._flush_active_segment()

    def stop(self, filename: str):
        """Stops recording, stitches together every segment, and writes the WAV file.

        Returns the saved filename, or None if no audio was captured.

        Raises:
            RuntimeError: If audio could not be written to disk while recording.
        """
        if not self._is_recording:
            logger.warning("Not currently recording.")
            return None

        try:
            self._halt_capture()
            if self._spool_error is not None:
                raise RuntimeError(
                    f"The recording was interrupted by a disk error: {self._spool_error}"
                ) from self._spool_error
            return self._write_recording(filename)
        finally:
            self._discard_spool()

    def abort(self):
        """Stops recording and throws the audio away (e.g. the app is closing)."""
        if not self._is_recording:
            return
        try:
            self._halt_capture()
        finally:
            self._discard_spool()
        logger.info("Recording aborted, audio discarded.")

    # ---- audio assembly (disk based, in small blocks) -----------------------

    @property
    def _block_frames(self) -> int:
        return self.BLOCK_SECONDS * self.target_samplerate

    def _write_recording(self, filename: str):
        """Builds the final WAV from the spooled segments. Each source is first
        converted into one mono track at the target rate (segments recorded on
        different devices/rates - e.g. before and after a Bluetooth profile
        switch - won't match), then the two tracks are mixed block by block."""
        mic_path = os.path.join(self._tmp_dir, "mic_mono.f32")
        speaker_path = os.path.join(self._tmp_dir, "speaker_mono.f32")
        mic_frames, mic_energy = self._render_track(self._mic_segments, mic_path)
        speaker_frames, speaker_energy = self._render_track(self._speaker_segments, speaker_path)

        if mic_frames == 0 and speaker_frames == 0:
            logger.warning("No audio captured.")
            return None

        self._warn_if_silent("Microphone", mic_frames, mic_energy)
        self._warn_if_silent("System audio", speaker_frames, speaker_energy)

        self._write_wav(filename, (mic_path, mic_frames), (speaker_path, speaker_frames))
        logger.info("Recording saved to %s", filename)
        return filename

    def _render_track(self, segments, out_path):
        """Concatenates all segments of one source into a single mono float32 file
        at the target rate. Returns (frames, sum of squared samples)."""
        frames = 0
        energy = 0.0
        with open(out_path, "wb") as out:
            for segment in segments:
                for block in self._iter_mono_blocks(segment):
                    out.write(block.tobytes())
                    frames += len(block)
                    energy += float(np.sum(np.square(block, dtype=np.float64)))
        return frames, energy

    def _iter_mono_blocks(self, segment):
        """Yields one segment as mono float32 blocks at the target rate. Uses linear
        interpolation (good enough for speech, no extra dependency), computed
        block by block with the same sample positions a whole-array resample would use."""
        n_in = segment.frames
        if segment.rate == self.target_samplerate:
            for start in range(0, n_in, self._block_frames):
                yield self._to_mono(segment.read(start, min(start + self._block_frames, n_in)))
            return

        n_out = int(round(n_in * self.target_samplerate / segment.rate))
        step = (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
        for start in range(0, n_out, self._block_frames):
            stop = min(start + self._block_frames, n_out)
            positions = np.arange(start, stop) * step
            first = int(positions[0])
            last = min(int(np.ceil(positions[-1])) + 1, n_in)
            mono = self._to_mono(segment.read(first, last))
            yield np.interp(positions - first, np.arange(last - first), mono).astype(np.float32)

    @staticmethod
    def _to_mono(audio):
        if audio.shape[1] == 1:
            return audio[:, 0]
        return audio.mean(axis=1)

    @staticmethod
    def _read_track(path, frames, start, count):
        """`count` frames of a mono float32 track from `start`; zero-padded past its end."""
        block = np.zeros(count, dtype=np.float32)
        available = min(count, frames - start)
        if available > 0:
            with open(path, "rb") as f:
                f.seek(start * _BYTES_PER_SAMPLE)
                block[:available] = np.frombuffer(f.read(available * _BYTES_PER_SAMPLE), dtype=np.float32)
        return block

    def _iter_mixed_blocks(self, mic, speaker):
        """Yields the final audio block by block. Works the same whether the system
        track is fully recorded, partial (loopback dropped in/out mid-recording) or
        empty (never available): the missing portion is just silence, and the
        output format doesn't change."""
        total = max(mic[1], speaker[1])
        for start in range(0, total, self._block_frames):
            count = min(self._block_frames, total - start)
            mic_block = self._read_track(*mic, start, count)
            speaker_block = self._read_track(*speaker, start, count)
            if self.mix_mode == "sum":
                yield (mic_block + speaker_block)[:, None]
            else:
                # "separate": mic -> left channel, system audio -> right channel.
                yield np.stack([mic_block, speaker_block], axis=1)

    def _write_wav(self, filename, mic, speaker):
        """mic/speaker are (path, frames) of the mono tracks. Two passes: the first
        finds the peak (so a too-loud mix can be scaled down instead of clipping),
        the second writes the file."""
        peak = max(
            (float(np.max(np.abs(block))) for block in self._iter_mixed_blocks(mic, speaker)),
            default=0.0,
        )
        scale = 1.0 / peak if peak > 1.0 else 1.0
        channels = 1 if self.mix_mode == "sum" else 2
        with sf.SoundFile(filename, "w", samplerate=self.target_samplerate, channels=channels) as out:
            for block in self._iter_mixed_blocks(mic, speaker):
                out.write(block * scale)

    @staticmethod
    def _warn_if_silent(label, frames, energy, threshold: float = 1e-4):
        """Flags a captured track that is essentially silent, which usually means the
        wrong device was recorded (e.g. loopback on a device that doesn't support it)."""
        if frames == 0:
            logger.warning("%s captured no data at all.", label)
            return
        rms = float(np.sqrt(energy / frames))
        if rms < threshold:
            logger.warning(
                "%s is essentially silent (RMS=%.6f). If this is 'System audio', the "
                "current output device may not support WASAPI loopback (common with "
                "some Bluetooth/USB headsets in call mode).",
                label, rms,
            )

    @property
    def is_recording(self) -> bool:
        return self._is_recording
