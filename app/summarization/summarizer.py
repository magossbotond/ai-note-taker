"""Local LLM summarization of interview transcripts using Ollama.

Runs fully offline (no API keys). Takes the raw Whisper transcript (which may
be in Hungarian, English, or mixed) and produces a structured Hungarian-language
summary for the HR reviewer.

Transcripts longer than the model's context window are summarized map-reduce
style: each chunk is condensed into short notes, then the notes are merged into
the final structured summary. Without this Ollama silently truncates the input,
and only the start of a long interview would be summarized.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import ollama

from app.core.errors import OperationCancelled

logger = logging.getLogger(__name__)

# Strong system prompt: pins the model's role, its job, and the output
# language/format so it doesn't drift into chit-chat or answer in the
# interview's original language.
SYSTEM_PROMPT = """\
Te egy profi HR asszisztens vagy, aki állásinterjúk nyers, gépi átirat \
(transzkript) szövegét dolgozza fel.

A feladatod:
1. Olvasd végig a teljes interjú-transzkriptumot.
2. Hagyd ki a köszönéseket, a kötetlen beszélgetést (small talk) és minden \
olyan részt, ami nem releváns a jelölt szakmai értékeléséhez.
3. Emeld ki és rendszerezd a lényeget strukturált, pontokba szedett \
(bullet point) összefoglalóban, a következő szekciókkal:
   - **Készségek** (technikai és soft skillek)
   - **Tapasztalat** (korábbi munkahelyek, projektek, szerepkörök)
   - **Kulcsfontosságú megállapítások** (erősségek, kockázatok, illeszkedés \
a pozícióhoz, bármi, amit egy döntéshozónak tudnia kell)

Szigorú szabályok:
- A végleges összefoglalót MINDIG magyar nyelven add meg, függetlenül attól, \
hogy az eredeti interjú milyen nyelven zajlott.
- Csak a transzkriptumban tényleg elmondott információkra hivatkozz, semmit \
ne találj ki.
- Legyél tömör, tényszerű és objektív - ne fűzz hozzá saját véleményt.
- SOHA ne kérj vissza a felhasználótól sem transzkriptumot, sem kiegészítő \
információt, és ne tégy fel kérdést. Bármilyen rövid, hiányos, vagy \
kizárólag köszönésből/kötetlen beszélgetésből álló szöveget kapsz, a választ \
MINDIG a lenti struktúrában add meg.
- Ha egy szekcióhoz nincs elég információ a transzkriptumban, írd oda: \
"Nincs elegendő információ." - ne hagyd ki a szekciót, és ne kérj pótlást.
- A válaszod kizárólag a strukturált összefoglaló legyen, felesleges \
bevezető vagy záró mondatok nélkül.
"""

# Used for the "map" step on long transcripts: condense ONE part of the
# interview into plain bullets (no sections - those come from the final merge).
CHUNK_PROMPT = """\
Te egy profi HR asszisztens vagy. Egy állásinterjú gépi átiratának EGY \
RÉSZLETÉT kapod.

Gyűjtsd ki belőle tömör, magyar nyelvű, pontokba szedett (bullet point) \
jegyzetekben mindazt, ami a jelölt szakmai értékeléséhez releváns: készségek, \
tapasztalat (munkahelyek, projektek, szerepkörök), erősségek, kockázatok, \
illeszkedés a pozícióhoz.

Szigorú szabályok:
- Hagyd ki a köszönéseket és a kötetlen beszélgetést.
- Csak a szövegben tényleg elhangzott információt írd le, semmit ne találj ki.
- SOHA ne tegyél fel kérdést és ne kérj további szöveget.
- Szekciók, bevezető és záró mondatok nélkül, kizárólag a pontok.
- Ha a részletben nincs releváns információ, ezt írd: "Nincs releváns \
információ."
"""

# Prefixed to the merged notes in the final "reduce" step, so the model knows
# its input is notes rather than a raw transcript.
MERGE_PREFIX = (
    "(Az alábbi szöveg az interjú részleteiből készült jegyzetek összessége, "
    "nem a nyers átirat. Ebből készítsd el a végleges összefoglalót.)\n\n"
)

NO_CONTENT_MESSAGE = "Nincs elegendő információ."

# Hungarian averages ~2.8 tokens per word with llama3.1 (measured); round up.
TOKENS_PER_WORD = 3.0
# Context budget kept free for the system prompt (~650 tokens), the model's
# answer, and slack - everything that isn't transcript text.
PROMPT_RESERVE_TOKENS = 2400
# Safety cap for re-condensing notes that are still too long for one call.
MAX_CONDENSE_PASSES = 3


class SummaryStatus(Enum):
    OK = "ok"
    TOO_SHORT = "too_short"  # not sent to the model: too few words
    NO_RELEVANT_CONTENT = "no_relevant_content"  # the model found nothing to summarize


@dataclass(frozen=True)
class SummaryResult:
    """Outcome of a summarization. `notes` is always Markdown, ready to display."""

    notes: str
    status: SummaryStatus


class SummarizationError(RuntimeError):
    """Summarization could not be completed (Ollama down, model missing, ...)."""


def to_markdown(notes: str) -> str:
    """llama3.1 often uses "•" bullets, which Markdown doesn't treat as a list
    (the lines would collapse into one paragraph). Convert them to "- "."""
    return re.sub(r"^[ \t]*[•*][ \t]+", "- ", notes, flags=re.MULTILINE)


def split_into_chunks(text: str, max_words: int) -> list[str]:
    """Splits text into chunks of at most `max_words` words, breaking at
    sentence/line boundaries where possible. Chunks are newline-joined sentences."""
    units = [u for u in re.split(r"(?<=[.!?])[ \t]+|\n+", text.strip()) if u.strip()]

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        words = unit.split()
        if len(words) > max_words:
            # A single "sentence" longer than a whole chunk (e.g. Whisper output
            # without punctuation): fall back to splitting by word count.
            if current:
                chunks.append("\n".join(current))
                current, count = [], 0
            chunks.extend(
                " ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)
            )
            continue
        if current and count + len(words) > max_words:
            chunks.append("\n".join(current))
            current, count = [], 0
        current.append(unit)
        count += len(words)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def _is_empty_verdict(text: str) -> bool:
    """True if the model's whole answer is just 'there is no (relevant) info'."""
    return _normalize(text) in {
        _normalize(NO_CONTENT_MESSAGE),
        "nincs releváns információ",
    }


class InterviewSummarizer:
    """Wraps a local Ollama chat model to turn a raw transcript into HR notes."""

    # Transcripts this short (e.g. a stopped-seconds-in recording, or just a
    # greeting) don't contain enough signal for the model to summarize, and
    # llama3.1 sometimes breaks character and asks for the transcript instead
    # of following the structured format. Skip the LLM call entirely in that
    # case rather than relying on prompt wording alone.
    MIN_WORDS = 5

    def __init__(
        self,
        model: str = "llama3.1",
        num_ctx: int = 8192,
        client: ollama.Client | None = None,
    ) -> None:
        """
        Args:
            model: Name of the local Ollama model to use (must already be
                pulled, e.g. via `ollama pull llama3.1`).
            num_ctx: Context window to request from Ollama. Ollama's own default
                is only 4096 tokens here, which is too small for interviews.
                Larger values cost more (V)RAM and slow the model down.
            client: Ollama client; injectable for tests. Defaults to the local daemon.
        """
        self.model = model
        self.num_ctx = num_ctx
        self.chunk_words = max(200, int((num_ctx - PROMPT_RESERVE_TOKENS) / TOKENS_PER_WORD))
        self._client = client or ollama.Client()
        self._force_cpu = False  # set after the GPU runner crashed once
        self._should_cancel: Callable[[], bool] | None = None  # set per generate_notes() call

    def generate_notes(
        self,
        raw_transcript: str,
        on_progress: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> SummaryResult:
        """Summarize a raw interview transcript into structured Hungarian HR notes.

        Args:
            raw_transcript: The raw text produced by the Whisper transcriber.
            on_progress: Optional callback receiving short status messages; long
                transcripts take several model calls.
            should_cancel: Optional callable polled before every model call; when
                it returns True the summary stops with OperationCancelled.

        Returns:
            A SummaryResult. Check `status` to tell a real summary (OK) from
            "transcript too short" and "model found no relevant content".

        Raises:
            SummarizationError: If the Ollama call fails (daemon not running,
                model not pulled, ...).
            OperationCancelled: If should_cancel() returned True.
        """
        transcript = (raw_transcript or "").strip()
        if len(transcript.split()) < self.MIN_WORDS:
            return SummaryResult(NO_CONTENT_MESSAGE, SummaryStatus.TOO_SHORT)

        progress = on_progress or (lambda _message: None)
        self._should_cancel = should_cancel
        try:
            notes = self._summarize(transcript, progress)
        except ConnectionError as exc:
            raise SummarizationError(
                "Could not reach Ollama - is it running? (start the Ollama app)"
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise SummarizationError(
                    f"Model '{self.model}' is not installed - run: ollama pull {self.model}"
                ) from exc
            raise SummarizationError(f"Ollama error ({exc.status_code}): {exc.error}") from exc
        finally:
            self._unload_model()

        status = SummaryStatus.NO_RELEVANT_CONTENT if _is_empty_verdict(notes) else SummaryStatus.OK
        return SummaryResult(to_markdown(notes), status)

    # ---- internals ---------------------------------------------------------

    def _summarize(self, transcript: str, progress: Callable[[str], None]) -> str:
        chunks = split_into_chunks(transcript, self.chunk_words)
        if len(chunks) == 1:
            progress("Generating notes...")
            return self._chat(SYSTEM_PROMPT, chunks[0])

        # Map: condense each chunk to short notes. If the combined notes still
        # don't fit one call (extremely long interview), condense them again.
        for _ in range(MAX_CONDENSE_PASSES):
            partial_notes = []
            for index, chunk in enumerate(chunks, start=1):
                progress(f"Summarizing part {index}/{len(chunks)}...")
                partial_notes.append(self._chat(CHUNK_PROMPT, chunk))
            partial_notes = [n for n in partial_notes if not _is_empty_verdict(n)]
            if not partial_notes:
                return NO_CONTENT_MESSAGE

            combined = "\n\n".join(partial_notes)
            chunks = split_into_chunks(combined, self.chunk_words)
            if len(chunks) == 1:
                # Reduce: merge the notes into the final structured summary.
                progress("Combining the notes...")
                return self._chat(SYSTEM_PROMPT, MERGE_PREFIX + combined)

        raise SummarizationError("The transcript is too long to summarize.")

    def _chat(self, system_prompt: str, user_content: str) -> str:
        if self._should_cancel is not None and self._should_cancel():
            raise OperationCancelled("summarization cancelled")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        # temperature=0: deterministic, format-following output. This is a
        # structured-extraction task, not creative writing - we want the same
        # transcript to reliably produce the same section layout.
        # keep_alive: stay loaded between the calls of one summary; the model is
        # unloaded explicitly afterwards (see _unload_model).
        options = {"temperature": 0, "num_ctx": self.num_ctx}
        if self._force_cpu:
            options["num_gpu"] = 0

        try:
            response = self._client.chat(
                model=self.model, messages=messages, options=options, keep_alive="5m"
            )
        except ollama.ResponseError as exc:
            # A 5xx means the model runner itself crashed while loading (seen with
            # Ollama's CUDA backend: driver mismatch, or not enough free VRAM).
            # Retry once on the CPU - slower, but it works. Client errors such as
            # "model not found" (404) are not retried.
            if exc.status_code < 500 or self._force_cpu:
                raise
            logger.warning("Ollama GPU inference failed (%s); retrying on CPU.", exc.error)
            self._force_cpu = True
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={**options, "num_gpu": 0},
                keep_alive="5m",
            )
        return response["message"]["content"].strip()

    def _unload_model(self) -> None:
        """Frees the model's (V)RAM right away instead of after Ollama's default
        5 minutes, so Whisper has the GPU to itself on the next recording."""
        try:
            self._client.generate(model=self.model, keep_alive=0)
        except Exception as exc:  # best effort: never mask the real result or error
            logger.debug("Could not unload model '%s': %s", self.model, exc)
