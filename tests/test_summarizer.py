"""Unit tests for app.summarization.summarizer, using a fake Ollama client
(no daemon or model needed). Run: python -m unittest discover -s tests -t ."""
import unittest

import ollama

from app.core.errors import OperationCancelled
from app.summarization.summarizer import (
    CHUNK_PROMPT,
    NO_CONTENT_MESSAGE,
    SYSTEM_PROMPT,
    InterviewSummarizer,
    SummarizationError,
    SummaryStatus,
    split_into_chunks,
    to_markdown,
)


class FakeClient:
    """Stands in for ollama.Client. `reply` maps (system_prompt, user) -> answer;
    `chat_errors` are raised, in order, by the first chat() calls."""

    def __init__(self, reply=None, chat_errors=()):
        self.reply = reply or (lambda system, user: "**Készségek:**\n• Python")
        self.chat_errors = list(chat_errors)
        self.chat_calls = []  # (system, user, options)
        self.generate_calls = []

    def chat(self, model, messages, options, keep_alive):
        system, user = messages[0]["content"], messages[1]["content"]
        self.chat_calls.append((system, user, options))
        if self.chat_errors:
            raise self.chat_errors.pop(0)
        return {"message": {"content": self.reply(system, user)}}

    def generate(self, model, keep_alive):
        self.generate_calls.append(keep_alive)


def make_transcript(sentences: int, words_per_sentence: int = 10) -> str:
    """Unique words, so tests can check that no part of a transcript got lost."""
    return " ".join(
        " ".join(f"w{i}x{j}" for j in range(words_per_sentence)) + "."
        for i in range(sentences)
    )


class SplitIntoChunksTests(unittest.TestCase):
    def test_respects_max_words_and_loses_nothing(self):
        text = make_transcript(sentences=50)
        chunks = split_into_chunks(text, max_words=35)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 35)
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_breaks_at_sentence_boundaries(self):
        chunks = split_into_chunks(make_transcript(sentences=10), max_words=25)
        for chunk in chunks:
            self.assertTrue(chunk.endswith("."))

    def test_text_without_punctuation_is_split_by_words(self):
        text = " ".join(f"w{i}" for i in range(100))
        chunks = split_into_chunks(text, max_words=30)
        self.assertEqual([len(c.split()) for c in chunks], [30, 30, 30, 10])

    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(split_into_chunks("  \n ", max_words=10), [])


class ToMarkdownTests(unittest.TestCase):
    def test_converts_bullets_and_keeps_bold(self):
        self.assertEqual(
            to_markdown("**Készségek:**\n• Python\n  * Django"),
            "**Készségek:**\n- Python\n- Django",
        )


class GenerateNotesTests(unittest.TestCase):
    def summarizer(self, client, num_ctx=2600):
        # num_ctx=2600 -> chunk_words floors at 200, small enough to force chunking.
        return InterviewSummarizer(model="test-model", num_ctx=num_ctx, client=client)

    def test_blank_or_tiny_transcript_is_too_short_and_skips_the_model(self):
        for text in ("", "   \n", None, "szia hogy vagy"):
            client = FakeClient()
            result = self.summarizer(client).generate_notes(text)
            self.assertIs(result.status, SummaryStatus.TOO_SHORT, repr(text))
            self.assertEqual(result.notes, NO_CONTENT_MESSAGE)
            self.assertEqual(client.chat_calls, [])

    def test_short_transcript_makes_one_call_and_unloads_the_model(self):
        client = FakeClient()
        summarizer = self.summarizer(client)
        result = summarizer.generate_notes(make_transcript(sentences=5))

        self.assertIs(result.status, SummaryStatus.OK)
        self.assertEqual(result.notes, "**Készségek:**\n- Python")  # bullets normalized
        self.assertEqual(len(client.chat_calls), 1)
        system, _user, options = client.chat_calls[0]
        self.assertEqual(system, SYSTEM_PROMPT)
        self.assertEqual(options["num_ctx"], summarizer.num_ctx)
        self.assertEqual(client.generate_calls, [0])

    def test_model_saying_there_is_nothing_is_reported_as_no_relevant_content(self):
        client = FakeClient(reply=lambda s, u: "Nincs elegendő információ.")
        result = self.summarizer(client).generate_notes(make_transcript(sentences=5))
        self.assertIs(result.status, SummaryStatus.NO_RELEVANT_CONTENT)

    def test_long_transcript_is_chunked_then_merged_without_losing_any_text(self):
        transcript = make_transcript(sentences=100)  # 1000 words, chunk_words = 200
        client = FakeClient(
            reply=lambda system, user: "• jegyzet" if system == CHUNK_PROMPT else "**Összegzés**"
        )
        progress = []
        result = self.summarizer(client).generate_notes(transcript, on_progress=progress.append)

        chunk_calls = [c for c in client.chat_calls if c[0] == CHUNK_PROMPT]
        final_calls = [c for c in client.chat_calls if c[0] == SYSTEM_PROMPT]
        self.assertGreaterEqual(len(chunk_calls), 5)
        self.assertEqual(len(final_calls), 1)
        self.assertIs(client.chat_calls[-1][0], final_calls[0][0])  # merge comes last
        seen_words = " ".join(user for _s, user, _o in chunk_calls).split()
        self.assertEqual(seen_words, transcript.split())  # nothing truncated
        self.assertIs(result.status, SummaryStatus.OK)
        self.assertEqual(result.notes, "**Összegzés**")
        self.assertIn(f"Summarizing part 1/{len(chunk_calls)}...", progress)
        self.assertEqual(progress[-1], "Combining the notes...")

    def test_long_transcript_where_every_part_is_irrelevant_skips_the_merge(self):
        client = FakeClient(reply=lambda s, u: "Nincs releváns információ.")
        result = self.summarizer(client).generate_notes(make_transcript(sentences=100))
        self.assertIs(result.status, SummaryStatus.NO_RELEVANT_CONTENT)
        self.assertTrue(all(call[0] == CHUNK_PROMPT for call in client.chat_calls))

    def test_gpu_crash_retries_on_cpu_and_stays_on_cpu(self):
        client = FakeClient(chat_errors=[ollama.ResponseError("runner crashed", 500)])
        summarizer = self.summarizer(client)
        summarizer.generate_notes(make_transcript(sentences=5))
        summarizer.generate_notes(make_transcript(sentences=5))

        options = [c[2] for c in client.chat_calls]
        self.assertNotIn("num_gpu", options[0])
        self.assertEqual(options[1]["num_gpu"], 0)  # the retry
        self.assertEqual(options[2]["num_gpu"], 0)  # next summary doesn't try the GPU again

    def test_missing_model_is_a_clear_error_and_is_not_retried(self):
        client = FakeClient(chat_errors=[ollama.ResponseError("model not found", 404)])
        with self.assertRaisesRegex(SummarizationError, "ollama pull test-model"):
            self.summarizer(client).generate_notes(make_transcript(sentences=5))
        self.assertEqual(len(client.chat_calls), 1)
        self.assertEqual(client.generate_calls, [0])  # still unloaded

    def test_cancelling_between_chunks_stops_the_remaining_model_calls(self):
        client = FakeClient(reply=lambda system, user: "• jegyzet")
        calls = []
        def cancel_after_two_calls():
            calls.append(1)
            return len(client.chat_calls) >= 2
        with self.assertRaises(OperationCancelled):
            self.summarizer(client).generate_notes(
                make_transcript(sentences=100), should_cancel=cancel_after_two_calls
            )
        self.assertEqual(len(client.chat_calls), 2)   # 5+ chunks were planned
        self.assertEqual(client.generate_calls, [0])  # the model is still unloaded

    def test_daemon_down_is_a_clear_error(self):
        client = FakeClient(chat_errors=[ConnectionError("refused")])
        with self.assertRaisesRegex(SummarizationError, "Could not reach Ollama"):
            self.summarizer(client).generate_notes(make_transcript(sentences=5))


if __name__ == "__main__":
    unittest.main()
