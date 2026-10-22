"""S5's model boundary: the detector registry, the top-1 rule, devices, and checkpoint cards.

Nothing here runs a real detector. `TorchvisionBarbellDetector` is covered by the one `slow`
smoke test at the bottom, which needs a promoted checkpoint on disk and is skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from powerflow_pipeline.data.detect import barbell_model
from powerflow_pipeline.data.detect.barbell_model import (
    MODELS,
    CheckpointCard,
    build_model,
    card_path,
    promote_checkpoint,
    read_card,
    resolve_device,
    sha256_file,
    top1,
)
from powerflow_pipeline.data.detect.config import BarbellDevice


class TestRegistry:
    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_every_registered_architecture_builds_with_a_three_class_head(self, name: str) -> None:
        model = build_model(name, 64, 96, weights=None, pretrained_backbone=False)

        assert model is not None

    def test_an_unknown_architecture_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match=r"unknown model 'yolo'.*fasterrcnn_resnet50_fpn_v2"):
            build_model("yolo", min_size=64, max_size=96, weights=None)


class TestTop1:
    def test_keeps_the_highest_scoring_detection_per_class(self) -> None:
        boxes = [[0, 0, 2, 2], [1, 1, 3, 3], [5, 5, 9, 9], [6, 6, 8, 8]]
        scores = [0.6, 0.9, 0.95, 0.4]
        labels = [1, 1, 2, 2]

        found = top1(boxes, scores, labels, threshold=0.3)

        cap, collar = found["cap"], found["collar"]
        assert cap is not None and cap.box == (1.0, 1.0, 3.0, 3.0) and cap.score == 0.9
        assert collar is not None and collar.score == 0.95

    def test_below_threshold_is_not_recorded(self) -> None:
        found = top1([[0, 0, 2, 2]], [0.29], [1], threshold=0.3)

        assert found == {"cap": None, "collar": None}

    def test_background_and_unknown_labels_are_ignored(self) -> None:
        found = top1([[0, 0, 2, 2], [0, 0, 2, 2]], [0.99, 0.99], [0, 7], threshold=0.3)

        assert found == {"cap": None, "collar": None}

    def test_rounds_boxes_and_scores_as_the_script_did(self) -> None:
        found = top1([[0.04, 0.06, 2.0, 2.0]], [0.123456], [1], threshold=0.1)

        cap = found["cap"]
        assert cap is not None and cap.box == (0.0, 0.1, 2.0, 2.0) and cap.score == 0.1235


class TestDevice:
    @pytest.mark.parametrize(
        ("mps", "cuda", "expected"),
        [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")],
    )
    def test_auto_prefers_mps_then_cuda_then_cpu(
        self, mps: bool, cuda: bool, expected: str
    ) -> None:
        assert resolve_device(BarbellDevice.AUTO, mps=mps, cuda=cuda) == expected

    def test_an_explicit_device_is_honoured(self) -> None:
        assert resolve_device(BarbellDevice.CPU, mps=True, cuda=True) == "cpu"

    def test_an_unavailable_explicit_device_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cuda is not available"):
            resolve_device(BarbellDevice.CUDA, mps=True, cuda=False)


def _training_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "fasterrcnn_resnet50_fpn_v2-20261019-003709"
    run.mkdir(parents=True)
    torch.save(
        {
            "model": "fasterrcnn_resnet50_fpn_v2",
            "state_dict": {"w": torch.zeros(2)},
            "epoch": 6,
            "metrics": {"recall": 0.97},
        },
        run / "best.pt",
    )
    (run / "history.json").write_text(json.dumps({"args": {"val_fraction": 0.2, "seed": 7}}))
    return run / "best.pt"


class TestCheckpointCard:
    def test_promotion_copies_the_weights_and_writes_a_card(self, tmp_path: Path) -> None:
        source = _training_run(tmp_path)
        destination = tmp_path / "models" / "s5_barbell" / "frcnn-v2-2026-10-19.pt"

        card = promote_checkpoint(source, destination)

        assert destination.read_bytes() == source.read_bytes()
        assert card == read_card(destination)
        assert card.architecture == "fasterrcnn_resnet50_fpn_v2"
        assert card.classes == ("cap", "collar")
        assert card.training_epoch == 6
        assert card.training_run == "fasterrcnn_resnet50_fpn_v2-20261019-003709"
        assert card.val_metrics == {"recall": 0.97}
        assert card.training_args == {"val_fraction": 0.2, "seed": 7}
        assert card.sha256 == sha256_file(destination)
        assert json.loads(card_path(destination).read_text())["trainingEpoch"] == 6

    def test_promotion_never_overwrites_a_model(self, tmp_path: Path) -> None:
        source = _training_run(tmp_path)
        destination = tmp_path / "models" / "m.pt"
        promote_checkpoint(source, destination)

        with pytest.raises(FileExistsError, match="never rewritten"):
            promote_checkpoint(source, destination)

    def test_a_checkpoint_without_its_card_is_refused(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "m.pt"
        checkpoint.write_bytes(b"weights")

        with pytest.raises(FileNotFoundError, match=r"m\.json"):
            read_card(checkpoint)

    def test_a_missing_checkpoint_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"no checkpoint at .*m\.pt"):
            read_card(tmp_path / "m.pt")

    def test_a_card_naming_an_unknown_architecture_is_refused(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "m.pt"
        checkpoint.write_bytes(b"weights")
        card = CheckpointCard(
            architecture="yolo",
            classes=("cap", "collar"),
            training_epoch=1,
            training_run="r",
            source_checkpoint="r/best.pt",
            sha256=sha256_file(checkpoint),
        )
        card_path(checkpoint).write_text(json.dumps(card.to_document()))

        with pytest.raises(ValueError, match="unknown model 'yolo'"):
            read_card(checkpoint)

    def test_a_card_whose_hash_disagrees_with_the_weights_is_refused(self, tmp_path: Path) -> None:
        source = _training_run(tmp_path)
        destination = tmp_path / "models" / "m.pt"
        promote_checkpoint(source, destination)
        destination.write_bytes(b"swapped")

        with pytest.raises(ValueError, match="sha256"):
            read_card(destination)


# --- the real detector ------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
PROMOTED = DATA_ROOT / "models" / "s5_barbell" / "frcnn-v2-2026-10-19.pt"
CLIP = DATA_ROOT / "s4_crop_output" / "10 September" / "50kg3" / "rgb.mp4"


@pytest.mark.slow
@pytest.mark.skipif(not (PROMOTED.is_file() and CLIP.is_file()), reason="no promoted checkpoint")
def test_the_real_detector_finds_the_collar_on_a_real_clip() -> None:
    detector = barbell_model.TorchvisionBarbellDetector(
        PROMOTED, device=BarbellDevice.AUTO, record_threshold=0.3, batch_size=4
    )

    found = detector.detect(CLIP, n_frames=8)  # decodes only the first 8 frames

    assert len(found["cap"]) == len(found["collar"]) == 8
    assert sum(hit is not None for hit in found["collar"]) >= 6
    assert detector.provenance.checkpoint_sha256 == sha256_file(PROMOTED)
