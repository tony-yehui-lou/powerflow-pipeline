"""Run Prefect against a disposable local database instead of a real API."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True, scope="session")
def prefect_backend() -> Iterator[None]:
    with prefect_test_harness():
        yield
