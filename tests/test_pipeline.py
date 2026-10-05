"""Tests for app.core.pipeline.InterviewPipeline with fake recorder / Whisper / Ollama.
Real QThreads and a real Qt event loop are used, so the threading behaviour
(ordering, hand-back to the main thread, cancellation) is what is being tested."""
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests.qt_app import get_app, wait_until

from app.core.config import AppConfig
from app.core.errors import OperationCancelled
from app.core.pipeline import InterviewPipeline, PipelineState
from app.summarization.summarizer import SummarizationError, SummaryResult, SummaryStatus

IDLE, RECORDING, PROCESSING = PipelineState.IDLE, PipelineState.RECORDING, PipelineState.PROCESSING


class FakeRecorder:
    def __init__(self, start_error=None, stop_error=None, produce_audio=True):
        self.start_error, self.stop_error, self.produce_audio = start_error, stop_error, produce_audio
        self.aborted = False
        self.stop_thread = None

    def start(self):
        if self.start_error:
            raise self.start_error

    def stop(self, path):
        self.stop_thread = threading.current_thread()
        if self.stop_error:
            raise self.stop_error
        if not self.produce_audio:
            return None
        Path(path).write_bytes(b"RIFF")
        return path

    def abort(self):
        self.aborted = True


class FakeTranscriber:
    def __init__(self, events, text="Öt éve dolgozom backend fejlesztőként.", error=None, block=None):
        self.events, self.text, self.error, self.block = events, text, error, block

    def transcribe(self, path, language="hu", should_cancel=None):
        self.events.append("transcribe")
        if self.block is not None:  # wait until cancelled
            self.block.set()
            deadline = time.monotonic() + 5
            while not should_cancel():
                self.assertion_guard(deadline)
                time.sleep(0.005)
            raise OperationCancelled("cancelled")
        if self.error:
            raise self.error
        return self.text

    @staticmethod
    def assertion_guard(deadline):
        if time.monotonic() > deadline:
            raise AssertionError("transcription was never cancelled")

    def unload(self):
        self.events.append("unload")


class FakeSummarizer:
    def __init__(self, events, result=None, error=None):
        self.events, self.error = events, error
        self.result = result or SummaryResult("**Készségek:**\n- Python", SummaryStatus.OK)
        self.progress_thread = None

    def generate_notes(self, text, on_progress=None, should_cancel=None):
        self.events.append("summarize")
        self.text = text
        if on_progress:
            on_progress("Summarizing part 1/2...")
        if self.error:
            raise self.error
        return self.result


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        get_app()
        self.tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.events = []

    def make(self, recorder=None, transcriber=None, summarizer=None):
        self.recorder = recorder or FakeRecorder()
        self.transcriber = transcriber or FakeTranscriber(self.events)
        self.summarizer = summarizer or FakeSummarizer(self.events)
        self.pipeline = InterviewPipeline(
            config=AppConfig(interviews_dir=self.tmp / "interviews"),
            recorder=self.recorder,
            summarizer=self.summarizer,
            transcriber_factory=lambda model: self.transcriber,
        )
        self.logs, self.states = [], []
        self.main_thread = threading.current_thread()
        self.log_threads = []
        self.pipeline.log.connect(self.logs.append)
        self.pipeline.log.connect(lambda _m: self.log_threads.append(threading.current_thread()))
        self.pipeline.state_changed.connect(self.states.append)
        return self.pipeline

    def run_to_idle(self):
        self.pipeline.start_recording()
        self.pipeline.stop_recording()
        self.assertTrue(wait_until(lambda: self.pipeline.state is IDLE), "pipeline never returned to IDLE")

    def logged(self, text):
        return any(text in line for line in self.logs)


class HappyPathTests(PipelineTestCase):
    def test_full_run_saves_all_three_files_and_reports_results(self):
        self.make()
        transcripts, notes, sessions = [], [], []
        self.pipeline.transcript_ready.connect(transcripts.append)
        self.pipeline.notes_ready.connect(notes.append)
        self.pipeline.session_changed.connect(sessions.append)

        self.run_to_idle()

        folder = self.pipeline.session_dir
        self.assertEqual(sessions, [folder])
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["notes.md", "recording.wav", "transcript.txt"])
        self.assertEqual((folder / "transcript.txt").read_text(encoding="utf-8"), transcripts[0])
        self.assertEqual((folder / "notes.md").read_text(encoding="utf-8"), "**Készségek:**\n- Python")
        self.assertEqual(self.pipeline.last_notes, notes[0].result if hasattr(notes[0], "result") else notes[0].notes)
        self.assertEqual(self.states, [RECORDING, PROCESSING, IDLE])
        self.assertTrue(self.logged("Notes ready!"))

    def test_whisper_is_unloaded_before_the_llm_starts(self):
        self.make()
        self.run_to_idle()
        self.assertEqual(self.events, ["transcribe", "unload", "summarize"])

    def test_the_summarizer_receives_the_transcript(self):
        self.make()
        self.run_to_idle()
        self.assertEqual(self.summarizer.text, self.transcriber.text)

    def test_heavy_work_runs_off_the_gui_thread_but_messages_arrive_on_it(self):
        self.make()
        self.run_to_idle()
        self.assertIsNot(self.recorder.stop_thread, self.main_thread)  # saving the WAV is off-thread
        self.assertTrue(self.logged("Summarizing part 1/2..."))        # progress from the worker thread
        self.assertTrue(all(t is self.main_thread for t in self.log_threads))

    def test_status_messages_for_special_results(self):
        for status, expected in (
            (SummaryStatus.TOO_SHORT, "too short to summarize"),
            (SummaryStatus.NO_RELEVANT_CONTENT, "no interview-relevant content"),
        ):
            with self.subTest(status=status):
                self.events.clear()
                self.make(summarizer=FakeSummarizer(self.events, result=SummaryResult("Nincs elegendő információ.", status)))
                self.run_to_idle()
                self.assertTrue(self.logged(expected))
                self.assertTrue((self.pipeline.session_dir / "notes.md").exists())


class FailureTests(PipelineTestCase):
    def test_recorder_that_cannot_start_leaves_the_pipeline_idle(self):
        self.make(recorder=FakeRecorder(start_error=RuntimeError("Could not open microphone 'X'")))
        self.pipeline.start_recording()
        self.assertIs(self.pipeline.state, IDLE)
        self.assertTrue(self.logged("Could not start recording: Could not open microphone"))

    def test_no_audio_captured_leaves_no_folder_behind(self):
        self.make(recorder=FakeRecorder(produce_audio=False))
        self.run_to_idle()
        self.assertTrue(self.logged("no audio was captured"))
        self.assertEqual(list((self.tmp / "interviews").iterdir()), [])

    def test_failure_while_saving_the_recording_is_reported(self):
        self.make(recorder=FakeRecorder(stop_error=RuntimeError("disk error")))
        self.run_to_idle()
        self.assertTrue(self.logged("Could not save the recording: disk error"))
        self.assertEqual(self.events, [])  # nothing was transcribed

    def test_transcription_error_stops_the_pipeline_and_still_unloads_whisper(self):
        self.make(transcriber=FakeTranscriber(self.events, error=RuntimeError("Transcription failed for 'x': boom")))
        self.run_to_idle()
        self.assertTrue(self.logged("Transcription failed for 'x': boom"))
        self.assertEqual(self.events, ["transcribe", "unload"])

    def test_summarization_error_keeps_the_transcript(self):
        self.make(summarizer=FakeSummarizer(self.events, error=SummarizationError("Could not reach Ollama")))
        self.run_to_idle()
        self.assertTrue(self.logged("Summarization failed: Could not reach Ollama"))
        self.assertTrue((self.pipeline.session_dir / "transcript.txt").exists())
        self.assertFalse((self.pipeline.session_dir / "notes.md").exists())

    def test_unexpected_exception_is_reported_not_swallowed_silently(self):
        self.make(summarizer=FakeSummarizer(self.events, error=KeyError("bug")))
        self.run_to_idle()
        self.assertTrue(self.logged("Unexpected error"))


class CancelAndAbortTests(PipelineTestCase):
    def test_cancelling_during_transcription_stops_before_summarizing(self):
        started = threading.Event()
        self.make(transcriber=FakeTranscriber(self.events, block=started))
        self.pipeline.start_recording()
        self.pipeline.stop_recording()
        self.assertTrue(wait_until(started.is_set))

        self.pipeline.cancel()
        self.assertTrue(wait_until(lambda: self.pipeline.state is IDLE))

        self.assertEqual(self.events, ["transcribe", "unload"])
        self.assertTrue(self.logged("Cancelled"))
        folder = self.pipeline.session_dir
        self.assertTrue((folder / "recording.wav").exists())      # the audio is kept
        self.assertFalse((folder / "transcript.txt").exists())

    def test_cancelling_while_the_recording_is_being_saved_skips_transcription(self):
        gate = threading.Event()
        class SlowRecorder(FakeRecorder):
            def stop(self, path):
                gate.wait(5)
                return super().stop(path)
        self.make(recorder=SlowRecorder())
        self.pipeline.start_recording()
        self.pipeline.stop_recording()
        self.pipeline.cancel()
        gate.set()
        self.assertTrue(wait_until(lambda: self.pipeline.state is IDLE))
        self.assertEqual(self.events, [])
        self.assertTrue(self.logged("Cancelled"))

    def test_abort_recording_discards_the_audio(self):
        self.make()
        self.pipeline.start_recording()
        self.pipeline.abort_recording()
        self.assertTrue(self.recorder.aborted)
        self.assertIs(self.pipeline.state, IDLE)

    def test_cancel_and_abort_do_nothing_when_idle(self):
        self.make()
        self.pipeline.cancel()
        self.pipeline.abort_recording()
        self.assertFalse(self.recorder.aborted)
        self.assertEqual(self.states, [])

    def test_a_new_recording_after_a_cancel_runs_normally(self):
        started = threading.Event()
        self.make(transcriber=FakeTranscriber(self.events, block=started))
        self.pipeline.start_recording()
        self.pipeline.stop_recording()
        wait_until(started.is_set)
        self.pipeline.cancel()
        wait_until(lambda: self.pipeline.state is IDLE)

        self.events.clear()
        self.pipeline._transcriber_factory = lambda model: FakeTranscriber(self.events)
        time.sleep(1.1)  # the next session folder is named by the second
        self.run_to_idle()
        self.assertEqual(self.events, ["transcribe", "unload", "summarize"])


class GuardTests(PipelineTestCase):
    def test_commands_in_the_wrong_state_are_ignored(self):
        self.make()
        self.pipeline.stop_recording()          # nothing is recording
        self.assertEqual(self.states, [])
        self.pipeline.start_recording()
        self.pipeline.start_recording()         # already recording
        self.assertEqual(self.states, [RECORDING])


if __name__ == "__main__":
    unittest.main()
