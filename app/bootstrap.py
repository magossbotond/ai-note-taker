"""Wires together the QApplication and the main window."""
import logging
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from app.gui.main_window import MainWindow
from app.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def _install_excepthook() -> None:
    """Last line of defence: PyQt6 aborts the process when a Python exception
    escapes a slot. Log it and tell the user instead of silently vanishing."""

    def hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
        QMessageBox.critical(
            None,
            "AI Interview Notetaker",
            f"An unexpected error occurred:\n\n{exc_value}\n\nSee the log for details.",
        )

    sys.excepthook = hook


def run() -> int:
    setup_logging()
    _install_excepthook()

    app = QApplication(sys.argv)
    app.setApplicationName("AI Interview Notetaker")

    window = MainWindow()
    window.show()

    return app.exec()
