"""Exceptions shared across the app's layers."""


class OperationCancelled(Exception):
    """A long-running step (transcription, summarization) was cancelled on purpose,
    e.g. because the user closed the window. Not an error - callers should stop quietly."""
