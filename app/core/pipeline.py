"""The interview workflow: record -> save -> transcribe -> summarize -> save notes.

`InterviewPipeline` owns the whole flow and the background threads; the GUI only
shows its state and forwards button clicks. Every step that can take long runs on
its own worker thread, one step at a time, and reports back to the main thread
through a single signal. The collaborators (recorder, transcriber, summarizer)
are injectable, so the flow can be tested without audio hardware, a GPU or a
running Ollama.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from app.audio.recorder import AudioRecorder
from app.core.config import AppConfig
from app.core.errors import OperationCancelled
from app.summarization.summarizer import InterviewSummarizer, SummaryResult, SummaryStatus
from app.transcription.transcriber import LocalAudioTranscriber

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"  # saving the audio, transcribing or summarizing


class _Step:
    """One background step: its worker thread and what to do with the outcome."""

    def __init__(self, on_success, failure_prefix: str) -> None:
        self.on_success = on_success
        self.failure_prefix = failure_prefix
        self.thread: threading.Thread | None = None


class _Outcome:
    """What a background step produced, handed back to the main thread."""

    def __init__(self, step: _Step, kind: str, value=None) -> None:
        self.step = step
        self.kind = kind  # "ok" | "error" | "cancelled"
        self.value = value


class InterviewPipeline(QObject):
    # Human-readable status lines for the user.
    log = pyqtSignal(str)
    state_changed = pyqtSignal(object)  # PipelineState
    session_changed = pyqtSignal(object)  # Path of the new interview folder
    transcript_ready = pyqtSignal(str)
    notes_ready = pyqtSignal(object)  # SummaryResult

    # Internal. Emitted from worker threads; because this object lives in the main
    # thread, Qt delivers them there (queued), so all handlers run on the GUI thread.
    _step_progress = pyqtSignal(str)
    _step_done = pyqtSignal(object)  # _Outcome

    def __init__(
        self,
        config: AppConfig | None = None,
        recorder: AudioRecorder | None = None,
        summarizer: InterviewSummarizer | None = None,
        transcriber_factory: Callable[[str], LocalAudioTranscriber] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config or AppConfig()
        # mix_mode="separate": mic on the left channel, system audio on the right
        # channel. Avoids the echo you'd get from summing both into mono when
        # recording without headphones.
        self._recorder = recorder or AudioRecorder(samplerate=self.config.samplerate, mix_mode="separate")
        self._summarizer = summarizer or InterviewSummarizer(
            model=self.config.llm_model, num_ctx=self.config.llm_context
        )
        # The Whisper model is created per transcription and unloaded right after,
        # so its VRAM is free for Ollama's summarization.
        self._transcriber_factory = transcriber_factory or (
            lambda model: LocalAudioTranscriber(model_size=model)
        )

        self._state = PipelineState.IDLE
        self._session_dir: Path | None = None
        self._last_notes = ""
        self._cancel = threading.Event()
        self._step: _Step | None = None

        self._step_progress.connect(self.log)
        self._step_done.connect(self._on_step_done)

    # ---- read-only view for the GUI ----------------------------------------

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def session_dir(self) -> Path | None:
        """Folder of the interview being (or last) processed."""
        return self._session_dir

    @property
    def last_notes(self) -> str:
        return self._last_notes

    # ---- commands ------------------------------------------------------------

    def start_recording(self) -> None:
        if self._state is not PipelineState.IDLE:
            return
        # Device problems (no mic, WASAPI failure) are expected here: report them
        # and stay usable instead of letting the exception escape into Qt.
        try:
            self._recorder.start()
        except Exception as exc:
            logger.warning("Could not start recording", exc_info=True)
            self.log.emit(f"Could not start recording: {exc}")
            return
        self._cancel.clear()
        self._set_state(PipelineState.RECORDING)
        self.log.emit("Recording started...")

    def stop_recording(self) -> None:
        if self._state is not PipelineState.RECORDING:
            return
        self._set_state(PipelineState.PROCESSING)
        session_dir = self.config.interviews_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        audio_path = session_dir / "recording.wav"

        def save(_progress):
            session_dir.mkdir(parents=True, exist_ok=True)
            return self._recorder.stop(str(audio_path))  # builds the WAV: can take seconds

        def on_saved(saved_path):
            if not saved_path:
                with contextlib.suppress(OSError):
                    session_dir.rmdir()  # nothing was recorded, don't leave an empty folder
                self.log.emit("Recording stopped, but no audio was captured.")
                self._finish()
                return
            self._session_dir = session_dir
            self._last_notes = ""
            self.session_changed.emit(session_dir)
            self.log.emit("Recording finished. Audio saved to: " + saved_path)
            if self._cancel.is_set():
                self._cancelled()
            else:
                self._start_transcription(saved_path)

        self._run_task(save, on_saved, "Could not save the recording", "save")

    def abort_recording(self) -> None:
        """Discards a recording in progress (the app is closing)."""
        if self._state is PipelineState.RECORDING:
            try:
                self._recorder.abort()
            except Exception:
                logger.exception("Could not abort the recording cleanly")
            self._set_state(PipelineState.IDLE)

    def cancel(self) -> None:
        """Asks the running step to stop soon; the pipeline then returns to IDLE.
        Audio and any transcript saved so far are kept."""
        if self._state is PipelineState.PROCESSING:
            self._cancel.set()

    # ---- steps -----------------------------------------------------------------

    def _start_transcription(self, audio_path: str) -> None:
        self.log.emit("Starting transcription... Please wait.")

        def transcribe(_progress):
            transcriber = None
            try:
                transcriber = self._transcriber_factory(self.config.whisper_model)
                return transcriber.transcribe(
                    audio_path, language=self.config.language, should_cancel=self._cancel.is_set
                )
            finally:
                # Runs on the worker thread BEFORE the outcome is reported, so the
                # Whisper model's VRAM is already free when the LLM starts.
                if transcriber is not None:
                    transcriber.unload()

        self._run_task(transcribe, self._on_transcribed, "", "transcribe")

    def _on_transcribed(self, text: str) -> None:
        self.log.emit("Transcription complete!")
        self._save_text("transcript.txt", text)
        self.transcript_ready.emit(text)
        if self._cancel.is_set():
            self._cancelled()
        else:
            self._start_summarization(text)

    def _start_summarization(self, transcript: str) -> None:
        self.log.emit("Generating interview notes with Ollama... Please wait.")

        def summarize(progress):
            return self._summarizer.generate_notes(
                transcript, on_progress=progress, should_cancel=self._cancel.is_set
            )

        self._run_task(summarize, self._on_summarized, "Summarization failed", "summarize")

    def _on_summarized(self, result: SummaryResult) -> None:
        if result.status is SummaryStatus.TOO_SHORT:
            self.log.emit("Transcript too short to summarize - it was not sent to the model.")
        elif result.status is SummaryStatus.NO_RELEVANT_CONTENT:
            self.log.emit(
                "The model found no interview-relevant content in the transcript "
                "(check transcript.txt - was the right audio recorded?)."
            )
        else:
            self.log.emit("Notes ready!")
        self._last_notes = result.notes
        self._save_text("notes.md", result.notes)
        self.notes_ready.emit(result)
        self._finish()

    # ---- plumbing ----------------------------------------------------------------

    def _run_task(self, fn, on_success, failure_prefix: str, name: str) -> None:
        """Runs fn(progress) on a worker thread. Afterwards exactly one handler runs
        on the main thread, once the worker thread has finished."""
        step = _Step(on_success, failure_prefix)

        def work() -> None:
            try:
                outcome = _Outcome(step, "ok", fn(self._step_progress.emit))
            except OperationCancelled:
                outcome = _Outcome(step, "cancelled")
            except Exception as exc:  # last-resort guard so a stray error can't hang the pipeline
                outcome = _Outcome(step, "error", exc)
            self._step_done.emit(outcome)  # the last thing the worker does

        step.thread = threading.Thread(target=work, name=f"pipeline-{name}")
        self._step = step
        step.thread.start()

    def _on_step_done(self, outcome: _Outcome) -> None:
        step = outcome.step
        step.thread.join()  # the worker only has to return now, so this is instant
        self._step = None
        try:
            if outcome.kind == "ok":
                step.on_success(outcome.value)
            elif outcome.kind == "cancelled":
                self._cancelled()
            else:
                self._failed(step.failure_prefix, outcome.value)
        except Exception as exc:  # a bug in a handler must not leave us stuck in PROCESSING
            logger.exception("Pipeline step handler failed")
            self.log.emit(f"Unexpected error: {exc}")
            self._finish()

    def _failed(self, prefix: str, exc: Exception) -> None:
        if isinstance(exc, (FileNotFoundError, RuntimeError)):
            self.log.emit(f"{prefix}: {exc}" if prefix else str(exc))
        else:
            logger.error("Unexpected pipeline error", exc_info=exc)
            self.log.emit(f"Unexpected error ({prefix or 'processing'}): {exc}")
        self._finish()

    def _cancelled(self) -> None:
        self.log.emit("Cancelled. The recording and anything saved so far are kept.")
        self._finish()

    def _finish(self) -> None:
        self._set_state(PipelineState.IDLE)

    def _set_state(self, state: PipelineState) -> None:
        self._state = state
        self.state_changed.emit(state)

    def _save_text(self, filename: str, text: str) -> None:
        """Writes into the current interview folder; a failure is logged, never
        raised, so a disk problem can't break the pipeline."""
        if self._session_dir is None:
            return
        try:
            (self._session_dir / filename).write_text(text, encoding="utf-8")
            self.log.emit(f"Saved {filename} to {self._session_dir}")
        except OSError as exc:
            self.log.emit(f"Could not save {filename}: {exc}")
