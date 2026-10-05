"""The main window: shows the pipeline's state and forwards button clicks to it.
All workflow logic (recording, transcription, summarization, saving) lives in
app.core.pipeline.InterviewPipeline."""
import os
from datetime import datetime

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.pipeline import InterviewPipeline, PipelineState


class MainWindow(QMainWindow):
    def __init__(self, pipeline: InterviewPipeline | None = None) -> None:
        super().__init__()
        self.setWindowTitle("AI Interview Notetaker")
        self.resize(900, 600)

        self._pipeline = pipeline or InterviewPipeline(parent=self)
        self._closing = False  # True while we wait for cancelled background work to end

        # --- Buttons: one per function ---
        self.start_button = QPushButton("Start Recording")
        self.start_button.setStyleSheet(self._button_style(color="#27ae60"))
        self.start_button.clicked.connect(self._pipeline.start_recording)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setStyleSheet(self._button_style(color="#c0392b"))
        self.stop_button.clicked.connect(self._pipeline.stop_recording)

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

        self._pipeline.log.connect(self._log)
        self._pipeline.state_changed.connect(self._on_state_changed)
        self._pipeline.session_changed.connect(self._on_session_changed)
        self._pipeline.transcript_ready.connect(self._show_transcript)
        self._pipeline.notes_ready.connect(self._show_notes)
        self._on_state_changed(self._pipeline.state)

    @staticmethod
    def _button_style(color: str) -> str:
        return f"background-color: {color}; color: white; font-weight: bold; padding: 8px;"

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] {message}")

    # ---- reacting to the pipeline --------------------------------------------

    def _on_state_changed(self, state: PipelineState) -> None:
        self.start_button.setEnabled(state is PipelineState.IDLE)
        self.stop_button.setEnabled(state is PipelineState.RECORDING)
        if self._closing and state is PipelineState.IDLE:
            self.close()  # the cancelled work has ended, finish quitting

    def _on_session_changed(self, _session_dir) -> None:
        self.notes_view.clear()
        self.copy_button.setEnabled(False)

    def _show_transcript(self, text: str) -> None:
        self.status_log.append("\n--- TRANSCRIPTION ---\n" + text + "\n")

    def _show_notes(self, result) -> None:
        self.notes_view.setMarkdown(result.notes)
        self.copy_button.setEnabled(True)

    # ---- buttons that don't touch the workflow ------------------------------------

    def copy_notes(self) -> None:
        """Copies the generated notes (Markdown text) to the clipboard."""
        if self._pipeline.last_notes:
            QApplication.clipboard().setText(self._pipeline.last_notes)
            self._log("Notes copied to clipboard.")

    def open_folder(self) -> None:
        """Opens the current interview's folder, or the interviews root if none yet."""
        folder = self._pipeline.session_dir or self._pipeline.config.interviews_dir
        try:
            folder.mkdir(parents=True, exist_ok=True)
            os.startfile(folder)
        except OSError as exc:
            self._log(f"Could not open {folder}: {exc}")

    # ---- closing the window ---------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        """Never destroy the window while a recording is open or a background thread
        is running - that crashes Qt. Ask first, then shut the work down cleanly."""
        state = self._pipeline.state
        if state is PipelineState.IDLE:
            event.accept()
        elif self._closing:
            event.ignore()  # already cancelling; we close ourselves when it is done
        elif state is PipelineState.RECORDING:
            if self._confirm("A recording is in progress.\n\nQuit and discard it?"):
                self._pipeline.abort_recording()
                event.accept()
            else:
                event.ignore()
        else:  # PROCESSING
            if self._confirm(
                "The recording is still being processed.\n\n"
                "Cancel it and quit? The audio and anything saved so far are kept."
            ):
                self._closing = True
                self._log("Cancelling - the window closes as soon as the current step ends...")
                self._pipeline.cancel()
            event.ignore()

    def _confirm(self, text: str) -> bool:
        answer = QMessageBox.question(
            self,
            "AI Interview Notetaker",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes
