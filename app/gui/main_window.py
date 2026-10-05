import contextlib
import os
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.audio.recorder import AudioRecorder
from app.summarization.summarizer import (
    InterviewSummarizer,
    SummarizationError,
    SummaryResult,
    SummaryStatus,
)
from app.transcription.transcriber import LocalAudioTranscriber

# Every interview gets its own timestamped folder here (recording.wav,
# transcript.txt, notes.md). Anchored to the project root, not the CWD.
INTERVIEWS_DIR = Path(__file__).resolve().parents[2] / "data" / "interviews"


class TranscriptionWorker(QObject):
    """Runs transcription on a background QThread so the GUI never freezes."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, audio_file_path: str, language: str = "hu") -> None:
        super().__init__()
        self.audio_file_path = audio_file_path
        self.language = language

    def run(self) -> None:
        transcriber = None
        try:
            transcriber = LocalAudioTranscriber(model_size="medium")
            text = transcriber.transcribe(self.audio_file_path, language=self.language)
            # Free the model's VRAM before signalling completion, so the Ollama
            # summarization that starts next has the GPU to itself.
            transcriber.unload()
            transcriber = None
            self.finished.emit(text)
        except FileNotFoundError as exc:
            self.error.emit(f"Audio file missing: {exc}")
        except RuntimeError as exc:
            self.error.emit(f"Transcription failed: {exc}")
        except Exception as exc:  # last-resort guard so a stray error can't hang the thread
            self.error.emit(f"Unexpected error during transcription: {exc}")
        finally:
            if transcriber is not None:  # error path: still release the model
                transcriber.unload()


class SummarizationWorker(QObject):
    """Runs the Ollama summarization on a background QThread (LLM calls can take
    tens of seconds, which would otherwise freeze the GUI)."""

    finished = pyqtSignal(object)  # SummaryResult
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, summarizer: InterviewSummarizer, transcript: str) -> None:
        super().__init__()
        self._summarizer = summarizer
        self.transcript = transcript

    def run(self) -> None:
        try:
            self.finished.emit(
                self._summarizer.generate_notes(self.transcript, on_progress=self.progress.emit)
            )
        except SummarizationError as exc:
            self.error.emit(f"Summarization failed: {exc}")
        except Exception as exc:  # last-resort guard so a stray error can't hang the thread
            self.error.emit(f"Unexpected error during summarization: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Interview Notetaker")
        self.resize(900, 600)

        # --- Audio recorder (Phase 2) ---
        # mix_mode="separate": mic on the left channel, system audio on the
        # right channel. Avoids the echo you'd get from summing both into mono
        # when recording without headphones.
        self.recorder = AudioRecorder(samplerate=44100, mix_mode="separate")

        # --- Transcriber (Phase 3) ---
        # The Whisper model is loaded per transcription inside the worker and
        # unloaded right after, so its VRAM is free for Ollama's summarization.
        self._transcription_thread: QThread | None = None
        self._transcription_worker: TranscriptionWorker | None = None

        # --- Summarizer (Phase 4) ---
        # Cheap to construct (no model load happens until the first chat call).
        self._summarizer = InterviewSummarizer(model="llama3.1")
        self._summarization_thread: QThread | None = None
        self._summarization_worker: SummarizationWorker | None = None

        # --- Saved output ---
        self._session_dir: Path | None = None  # folder of the interview being processed
        self._last_notes = ""  # Markdown of the most recent notes, for "Copy notes"

        # --- Buttons: one per function ---
        self.start_button = QPushButton("Start Recording")
        self.start_button.setStyleSheet(self._button_style(color="#27ae60"))
        self.start_button.clicked.connect(self.start_recording)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setStyleSheet(self._button_style(color="#c0392b"))
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)  # nothing to stop until recording starts

        self.copy_button = QPushButton("Copy notes")
        self.copy_button.clicked.connect(self.copy_notes)
        self.copy_button.setEnabled(False)  # nothing to copy until notes exist

        self.open_button = QPushButton("Open folder")
        self.open_button.clicked.connect(self.open_folder)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.copy_button)
        button_row.addWidget(self.open_button)

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)

        # Structured HR notes; the model answers in Markdown, rendered via setMarkdown.
        self.notes_view = QTextEdit()
        self.notes_view.setReadOnly(True)
        self.notes_view.setPlaceholderText("Structured interview notes will appear here.")

        layout = QVBoxLayout()
        layout.addLayout(button_row)
        layout.addWidget(self.status_log, stretch=1)
        layout.addWidget(QLabel("Interview notes"))
        layout.addWidget(self.notes_view, stretch=2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    @staticmethod
    def _button_style(color: str) -> str:
        return f"background-color: {color}; color: white; font-weight: bold; padding: 8px;"

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] {message}")

    def start_recording(self) -> None:
        """Called when the 'Start Recording' button is clicked."""
        # Slots must not raise: PyQt6 aborts the whole app on an unhandled
        # exception inside one. Device problems (no mic, WASAPI failure) are
        # expected here, so report them in the log and stay usable.
        try:
            self.recorder.start()
        except Exception as exc:
            self._log(f"Could not start recording: {exc}")
            return
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._log("Recording started...")

    def stop_recording(self) -> None:
        """Called when the 'Stop Recording' button is clicked."""
        session_dir = INTERVIEWS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.stop_button.setEnabled(False)
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            saved_path = self.recorder.stop(str(session_dir / "recording.wav"))
        except Exception as exc:  # see start_recording: slots must not raise
            self._log(f"Could not save the recording: {exc}")
            self.start_button.setEnabled(True)
            return

        if saved_path:
            self._session_dir = session_dir
            self._last_notes = ""
            self.copy_button.setEnabled(False)
            self.notes_view.clear()
            self._log("Recording finished. Audio saved to: " + saved_path)
            self._start_transcription(saved_path)
        else:
            with contextlib.suppress(OSError):
                session_dir.rmdir()  # nothing was recorded, don't leave an empty folder
            self._log("Recording stopped, but no audio was captured.")
            self.start_button.setEnabled(True)

    # ---- saved output ----------------------------------------------------

    def _save_text(self, filename: str, text: str) -> None:
        """Writes text into the current interview folder; a failure is logged,
        never raised, so a disk problem can't break the pipeline."""
        if self._session_dir is None:
            return
        try:
            (self._session_dir / filename).write_text(text, encoding="utf-8")
            self._log(f"Saved {filename} to {self._session_dir}")
        except OSError as exc:
            self._log(f"Could not save {filename}: {exc}")

    def copy_notes(self) -> None:
        """Copies the generated notes (Markdown text) to the clipboard."""
        if self._last_notes:
            QApplication.clipboard().setText(self._last_notes)
            self._log("Notes copied to clipboard.")

    def open_folder(self) -> None:
        """Opens the current interview's folder, or the interviews root if none yet."""
        folder = self._session_dir or INTERVIEWS_DIR
        try:
            folder.mkdir(parents=True, exist_ok=True)
            os.startfile(folder)
        except OSError as exc:
            self._log(f"Could not open {folder}: {exc}")

    # ---- transcription (runs on a background QThread) ----------------

    def _start_transcription(self, audio_file_path: str) -> None:
        self._log("Starting transcription... Please wait.")
        # Keep the start button disabled until the whole pipeline (transcription
        # + summarization) finishes, so a new recording can't overlap it.
        self.start_button.setEnabled(False)

        self._transcription_thread = QThread(self)
        self._transcription_worker = TranscriptionWorker(audio_file_path, language="hu")
        self._transcription_worker.moveToThread(self._transcription_thread)

        self._transcription_thread.started.connect(self._transcription_worker.run)
        self._transcription_worker.finished.connect(self._on_transcription_finished)
        self._transcription_worker.error.connect(self._on_transcription_error)
        self._transcription_worker.finished.connect(self._transcription_thread.quit)
        self._transcription_worker.error.connect(self._transcription_thread.quit)
        self._transcription_thread.finished.connect(self._transcription_worker.deleteLater)
        self._transcription_thread.finished.connect(self._transcription_thread.deleteLater)

        self._transcription_thread.start()

    def _on_transcription_finished(self, text: str) -> None:
        self._log("Transcription complete!")
        self.status_log.append("\n--- TRANSCRIPTION ---\n" + text + "\n")
        self._save_text("transcript.txt", text)
        # Start stays disabled; it is re-enabled once summarization ends.
        self._start_summarization(text)

    def _on_transcription_error(self, message: str) -> None:
        self._log(message)
        self.start_button.setEnabled(True)

    # ---- summarization (runs on a background QThread) ----------------

    def _start_summarization(self, transcript: str) -> None:
        self._log("Generating interview notes with Ollama... Please wait.")
        self.notes_view.clear()

        self._summarization_thread = QThread(self)
        self._summarization_worker = SummarizationWorker(self._summarizer, transcript)
        self._summarization_worker.moveToThread(self._summarization_thread)

        self._summarization_thread.started.connect(self._summarization_worker.run)
        self._summarization_worker.progress.connect(self._log)
        self._summarization_worker.finished.connect(self._on_summarization_finished)
        self._summarization_worker.error.connect(self._on_summarization_error)
        self._summarization_worker.finished.connect(self._summarization_thread.quit)
        self._summarization_worker.error.connect(self._summarization_thread.quit)
        self._summarization_thread.finished.connect(self._summarization_worker.deleteLater)
        self._summarization_thread.finished.connect(self._summarization_thread.deleteLater)

        self._summarization_thread.start()

    def _on_summarization_finished(self, result: SummaryResult) -> None:
        if result.status is SummaryStatus.TOO_SHORT:
            self._log("Transcript too short to summarize - it was not sent to the model.")
        elif result.status is SummaryStatus.NO_RELEVANT_CONTENT:
            self._log(
                "The model found no interview-relevant content in the transcript "
                "(check transcript.txt - was the right audio recorded?)."
            )
        else:
            self._log("Notes ready!")
        self.notes_view.setMarkdown(result.notes)
        self._last_notes = result.notes
        self.copy_button.setEnabled(True)
        self._save_text("notes.md", result.notes)
        self.start_button.setEnabled(True)

    def _on_summarization_error(self, message: str) -> None:
        self._log(message)
        self.start_button.setEnabled(True)
