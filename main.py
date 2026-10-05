"""Application entry point. Keep this file free of business logic."""
import sys

from app.bootstrap import run


if __name__ == "__main__":
    sys.exit(run())
