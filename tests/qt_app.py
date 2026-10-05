"""Shared Qt test helpers. Importing this module selects Qt's headless "offscreen"
platform, so no display is needed (it must be imported before any widget is created)."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

# Must stay referenced for the whole test run: when Python garbage-collects the
# QApplication wrapper, Qt tears the application down underneath the tests.
_app = None


def get_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


def wait_until(predicate, timeout: float = 10.0) -> bool:
    """Runs the Qt event loop until predicate() is true; False on timeout."""
    app = get_app()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())
