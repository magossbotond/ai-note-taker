"""The app's settings in one place (previously constants scattered through the GUI code)."""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppConfig:
    # Speech-to-text
    whisper_model: str = "medium"  # best accuracy/speed tradeoff for Hungarian; "small" is faster
    language: str = "hu"  # ISO 639-1 code of the spoken language

    # Notes (Ollama)
    llm_model: str = "llama3.1"  # must be pulled: `ollama pull llama3.1`
    llm_context: int = 8192  # context window in tokens; more = slower and more (V)RAM

    # Recording
    samplerate: int = 44100

    # Every interview gets its own timestamped folder here
    interviews_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "interviews")
