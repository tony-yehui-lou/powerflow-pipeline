"""Consistent source-to-output messages for Prefect tasks."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from prefect import get_run_logger
from prefect.exceptions import MissingContextError


def log_task_paths(source: Path, outputs: Path | Iterable[Path] | None) -> None:
    """Log the input and output locations for a task run.

    Task functions are also called through ``.fn`` in unit tests, where Prefect has no active
    run context. Use the module logger in that case while preserving Prefect task logs in runs.
    """

    if outputs is None:
        output_text = "<in-memory>"
    elif isinstance(outputs, Path):
        output_text = str(outputs)
    else:
        output_text = ", ".join(str(path) for path in outputs)

    logger: Any
    try:
        logger = get_run_logger()
    except MissingContextError:
        logger = logging.getLogger(__name__)
    logger.info("Processing source=%s output=%s", source, output_text)
