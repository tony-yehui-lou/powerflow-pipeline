"""The run options every pipeline's config shares: which captures, and whether to write."""

from __future__ import annotations

from fnmatch import fnmatch

from pydantic import BaseModel


class RunOptions(BaseModel):
    """Capture selection and the write mode, common to preprocess, detect and analyze."""

    # Restrict the run to capture ids matching this glob; unset, every capture is processed.
    only: str | None = None
    overwrite: bool = False
    dry_run: bool = False

    def selects(self, capture_id: str) -> bool:
        """Whether `--only` admits this capture. Unset admits everything."""

        return self.only is None or fnmatch(capture_id, self.only)
