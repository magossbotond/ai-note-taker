"""Tests for app.gui.main_window.MainWindow on Qt's offscreen platform, driven by a
real InterviewPipeline with fake recorder / Whisper / Ollama (see test_pipeline)."""
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tests.qt_app import get_app, wait_until

from PyQt6.QtWidgets import QMessageBox

from app.core.config import AppConfig
from app.core.pipeline import InterviewPipeline, PipelineState
from app.gui import main_window as main_window_module
from app.gui.main_window import MainWindow
from tests.test_pipeline import FakeRecorder, FakeSummarizer, FakeTranscriber

YES, NO = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No


class WindowTestCase(unittest.TestCase):
    def setUp(self):
        self.app = get_app()
        self.tmp = Path(tempfile.mkdtemp(prefix="window_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.events = []

    def make(self, transcriber=None, recorder=None):
        self.recorder = recorder or FakeRecorder()
        self.pipeline = InterviewPipeline(
            config=AppConfig(interviews_dir=self.tmp / "interviews"),
            recorder=self.recorder,
            summarizer=FakeSummarizer(self.events),
            transcriber_factory=lambda model: transcriber or FakeTranscriber(self.events),
        )
        self.window = MainWindow(self.pipeline)
        self.window.show()
        self.addCleanup(self.window.deleteLater)
        return self.window

    def idle(self):
        return self.pipeline.state is PipelineState.IDLE

    def ask(self, answer):
        return mock.patch.object(QMessageBox, "question", return_value=answer)


class ButtonStateTests(WindowTestCase):
    def test_buttons_follow_the_pipeline_state(self):
        win = self.make()
        self.assertTrue(win.start_button.isEnabled())
        self.assertFalse(win.stop_button.isEnabled())

        win.start_button.click()
        self.assertFalse(win.start_button.isEnabled())
        self.assertTrue(win.stop_button.isEnabled())

        win.stop_button.click()
        self.assertFalse(win.start_button.isEnabled())   # processing: neither works
        self.assertFalse(win.stop_button.isEnabled())

        self.assertTrue(wait_until(self.idle))
        self.assertTrue(win.start_button.isEnabled())
        self.assertFalse(win.stop_button.isEnabled())


class ResultDisplayTests(WindowTestCase):
    def record_once(self):
        self.window.start_button.click()
        self.window.stop_button.click()
        self.assertTrue(wait_until(self.idle))

    def test_transcript_notes_and_log_are_shown(self):
        win = self.make()
        self.record_once()
        self.assertIn("--- TRANSCRIPTION ---", win.status_log.toPlainText())
        self.assertIn("Notes ready!", win.status_log.toPlainText())
        self.assertIn("Készségek", win.notes_view.toPlainText())
        self.assertTrue(win.copy_button.isEnabled())

    def test_copy_notes_puts_them_on_the_clipboard(self):
        win = self.make()
        self.record_once()
        win.copy_button.click()
        self.assertEqual(self.app.clipboard().text(), self.pipeline.last_notes)

    def test_a_new_session_clears_the_previous_notes(self):
        win = self.make()
        self.record_once()
        self.pipeline.session_changed.emit(self.tmp)
        self.assertEqual(win.notes_view.toPlainText(), "")
        self.assertFalse(win.copy_button.isEnabled())

    def test_open_folder_opens_the_interview_folder(self):
        win = self.make()
        self.record_once()
        with mock.patch.object(main_window_module.os, "startfile", create=True) as startfile:
            win.open_button.click()
        startfile.assert_called_once_with(self.pipeline.session_dir)

    def test_open_folder_failure_is_logged_not_raised(self):
        win = self.make()
        with mock.patch.object(main_window_module.os, "startfile", side_effect=OSError("no explorer"), create=True):
            win.open_button.click()
        self.assertIn("Could not open", win.status_log.toPlainText())


class CloseWindowTests(WindowTestCase):
    def test_closing_when_idle_just_closes(self):
        win = self.make()
        with self.ask(YES) as question:
            self.assertTrue(win.close())
        question.assert_not_called()

    def test_closing_during_a_recording_asks_and_discards_it(self):
        win = self.make()
        win.start_button.click()
        with self.ask(YES):
            self.assertTrue(win.close())
        self.assertTrue(self.recorder.aborted)
        self.assertTrue(self.idle())

    def test_declining_to_quit_during_a_recording_keeps_recording(self):
        win = self.make()
        win.start_button.click()
        with self.ask(NO):
            self.assertFalse(win.close())
        self.assertFalse(self.recorder.aborted)
        self.assertIs(self.pipeline.state, PipelineState.RECORDING)
        self.assertTrue(win.isVisible())

    def test_closing_during_processing_cancels_and_closes_once_the_work_has_stopped(self):
        started = threading.Event()
        win = self.make(transcriber=FakeTranscriber(self.events, block=started))
        win.start_button.click()
        win.stop_button.click()
        self.assertTrue(wait_until(started.is_set))   # Whisper is "running" now

        with self.ask(YES):
            self.assertFalse(win.close())              # not closed yet: the thread must end first
        self.assertTrue(win.isVisible())
        self.assertTrue(wait_until(lambda: not win.isVisible()))   # closes itself afterwards

        self.assertTrue(self.idle())
        self.assertEqual(self.events, ["transcribe", "unload"])   # no summarization after cancel

    def test_declining_to_quit_during_processing_lets_it_finish(self):
        win = self.make()
        win.start_button.click()
        win.stop_button.click()
        with self.ask(NO):
            self.assertFalse(win.close())
        self.assertTrue(wait_until(self.idle))
        self.assertEqual(self.events, ["transcribe", "unload", "summarize"])
        self.assertTrue(win.isVisible())


if __name__ == "__main__":
    unittest.main()
