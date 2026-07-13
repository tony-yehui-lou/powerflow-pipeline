"""Errors that distinguish a rejected scan from a failed pipeline run."""

from __future__ import annotations


class PipelineError(RuntimeError):
    """Base error for failures that must stop a pipeline run."""


class ScanRejected(PipelineError):
    """Expected validation failure for one scan in an otherwise valid batch."""


class PublishError(PipelineError):
    """Failure while atomically publishing staged pipeline output."""
