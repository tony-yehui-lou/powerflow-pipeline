"""The published body model documents must not drift from the constants they come from.

`skeleton.human-v2.json` is generated, never hand-edited, so the only thing keeping it honest is
this comparison. `powerflow-ui` is a separate repository, so these skip when it is not checked
out beside this one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from powerflow_pipeline.data.common.models import PoseDocument, skeleton_document

EXAMPLES = (
    Path(__file__).resolve().parents[2].parent
    / "powerflow-ui"
    / "docs"
    / "schema-normalized"
    / "examples"
)

REGENERATE = "uv run python scripts/generate_body_model_documents.py"

requires_ui = pytest.mark.skipif(
    not EXAMPLES.is_dir(),
    reason="powerflow-ui is not checked out beside this repository",
)


@requires_ui
def test_published_skeleton_document_still_matches_the_human_skeleton_constant() -> None:
    published = json.loads((EXAMPLES / "skeleton.human-v2.json").read_text(encoding="utf-8"))
    assert published == skeleton_document(), f"stale document -- regenerate with: {REGENERATE}"


@requires_ui
def test_published_pose_example_satisfies_every_stored_invariant() -> None:
    raw = (EXAMPLES / "pose.11-july-30kg-set1.side.json").read_text(encoding="utf-8")
    document = PoseDocument.from_document(json.loads(raw))
    assert document.frames.count == 3
    assert set(document.joints) == set(skeleton_document()["joints"])


@requires_ui
def test_published_pose_example_demonstrates_the_occlusion_convention() -> None:
    raw = (EXAMPLES / "pose.11-july-30kg-set1.side.json").read_text(encoding="utf-8")
    document = PoseDocument.from_document(json.loads(raw))
    occluded = document.joints["leftWrist"]
    assert occluded.position_at(1) is None
    assert occluded.confidence[1] == 0.0
    assert occluded.position_at(0) is not None
