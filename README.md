# AI Interview Notetaker

A Windows desktop app that records a job interview (your microphone **and** the
system audio, e.g. a video call), transcribes it with Whisper, and has a local LLM
turn the transcript into structured **Hungarian HR notes** (skills, experience,
key findings). Everything runs on your own machine - there are no cloud APIs
and no API keys.

```
record (mic + system audio)  ->  Whisper (faster-whisper)  ->  Ollama (llama3.1)  ->  notes
```

## Requirements

- Windows 10/11 (system-audio capture uses WASAPI loopback)
- Python 3.10 - 3.13
- [Ollama](https://ollama.com/download) with the `llama3.1` model
- About 10 GB of free disk space (models) and 16 GB RAM recommended
- Optional: an NVIDIA GPU for fast transcription (the app falls back to the CPU)

## Setup

Run `setup.bat` once (safe to re-run). It creates `.venv`, installs the packages
(plus the CUDA libraries if an NVIDIA GPU is found), installs Ollama if missing,
pulls `llama3.1`, downloads the Whisper `medium` model, and builds the launcher.

## Run

Double-click **`AI Interview Notetaker.exe`** (built by `setup.bat`; it must stay in
the project folder), or run `.venv\Scripts\python.exe main.py`.

1. **Start Recording**, hold the interview, **Stop Recording**.
2. The app transcribes the audio, then generates the notes. The Start button is
   disabled until both steps finish.
3. **Copy notes** puts the notes on the clipboard; **Open folder** opens the
   interview's folder.

## Where things are saved

Each interview gets its own folder, `data/interviews/<date>_<time>/`:

| File             | Content                          |
|------------------|----------------------------------|
| `recording.wav`  | mic on the left, system audio on the right channel |
| `transcript.txt` | the Whisper transcript           |
| `notes.md`       | the generated notes (Markdown)   |

Nothing is deleted automatically. These files contain personal data of the
people interviewed - store and delete them according to your data-protection
rules (e.g. GDPR) and make sure participants have agreed to the recording.

## How it works

| Module | Responsibility |
|--------|----------------|
| `app/audio/recorder.py` | Records mic + WASAPI loopback, follows default-device changes (e.g. headphones), writes one stereo WAV |
| `app/transcription/transcriber.py` | faster-whisper wrapper: GPU when possible, CPU fallback, model unloaded after use to free VRAM |
| `app/summarization/summarizer.py` | Ollama wrapper: prompts, map-reduce for long transcripts, typed results and errors |
| `app/gui/main_window.py` | PyQt6 window; transcription and summarization run on background threads |
| `app/bootstrap.py` | Starts the app, logging and the last-resort error handler |

Long interviews are summarized in parts (each part is condensed, then the notes
are merged), because the model's context window is much smaller than a
one-hour transcript.

## Configuration

There is no config file yet; the settings are constants in the code:

- Whisper model size and language: `TranscriptionWorker` in `app/gui/main_window.py`
  (`"medium"`, `"hu"`)
- LLM model and context size: `InterviewSummarizer(...)` arguments in
  `app/summarization/summarizer.py` (`"llama3.1"`, `num_ctx=8192`)
- The prompt that shapes the notes: `SYSTEM_PROMPT` in `summarizer.py`

## Tests

```
.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

The tests need no microphone, GPU or running Ollama (a fake Ollama client is used).
`requirements-lock.txt` lists the exact package versions the app was tested with.

## Troubleshooting

- **"Could not reach Ollama"** - start the Ollama app (tray icon), then try again.
- **"Model 'llama3.1' is not installed"** - run `ollama pull llama3.1`.
- **Notes say "Nincs elegendő információ"** - the transcript had no interview
  content (or was too short). Read `transcript.txt`: was the right audio recorded?
- **Microphone/system audio "essentially silent" in the log** - check the default
  input/output device in Windows. Some Bluetooth headsets expose no loopback
  stream while in call mode; the app then records the microphone only.
- **Transcription is slow** - it is running on the CPU. Run `setup.bat` again
  (installs the CUDA libraries) and check that `app.log` says
  `Loading 'medium' model on CUDA`.
- **App closes immediately / does nothing** - see `app.log` in the project folder.

## Known limitations

- Windows only. The Whisper language is fixed to Hungarian (English audio would
  be transcribed badly) and the notes are always written in Hungarian.
- The whole recording is kept in RAM until you press Stop, so very long
  recordings (1 h+) need several GB of free memory.
- Closing the window while transcription or summarization is still running
  ends the app abruptly.
