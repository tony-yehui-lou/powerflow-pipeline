"""The barbell-landmark model boundary for S5 Barbell (S5-barbell-detection-stage-draft3.md §3).

`BarbellDetector.detect` is the whole seam: the task calls it and never imports torch, so every
test of the stage's plumbing runs against a stand-in. `TorchvisionBarbellDetector` is the real
implementation, and `MODELS` / `build_model` are the one place an architecture is built --
`scripts/train_landmark_detector.py` imports them, so training and inference cannot drift.

A checkpoint is only ever loaded through its **card**, the `.json` beside the `.pt` that
`promote_checkpoint` writes: it names the architecture and pins the weights by sha256. There is
no "latest training run" lookup, which is what made the old script unreproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import av
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from powerflow_pipeline.data.common.models import (
    BARBELL_CLASSES,
    BarbellClass,
    BarbellDetection,
    BarbellProvenance,
)
from powerflow_pipeline.data.preprocess.config import BarbellDevice, PreprocessConfig

# background, cap, collar
NUM_CLASSES: Final = 3
CLASS_NAMES: Final[dict[int, BarbellClass]] = {1: "cap", 2: "collar"}
CARD_SCHEMA_VERSION: Final = "1.0.0"


@dataclass(frozen=True)
class ModelSpec:
    """How to build one architecture and repoint its classifier at our classes."""

    factory: str  # torchvision.models.detection.<factory>
    family: str  # fasterrcnn | retinanet | fcos
    takes_size: bool = True


MODELS: Final[dict[str, ModelSpec]] = {
    "fasterrcnn_resnet50_fpn": ModelSpec("fasterrcnn_resnet50_fpn", "fasterrcnn"),
    "fasterrcnn_resnet50_fpn_v2": ModelSpec("fasterrcnn_resnet50_fpn_v2", "fasterrcnn"),
    "fasterrcnn_mobilenet_v3_large_fpn": ModelSpec(
        "fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn"
    ),
    "fasterrcnn_mobilenet_v3_large_320_fpn": ModelSpec(
        "fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn"
    ),
    "retinanet_resnet50_fpn": ModelSpec("retinanet_resnet50_fpn", "retinanet"),
    "retinanet_resnet50_fpn_v2": ModelSpec("retinanet_resnet50_fpn_v2", "retinanet"),
    "fcos_resnet50_fpn": ModelSpec("fcos_resnet50_fpn", "fcos"),
}


def _known(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {', '.join(sorted(MODELS))}")
    return MODELS[name]


def build_model(
    name: str,
    min_size: int,
    max_size: int,
    weights: str | None = "DEFAULT",
    *,
    pretrained_backbone: bool = True,
) -> Any:
    """COCO-pretrained detector with its classifier repointed at cap/collar.

    `weights=None` skips the COCO download, and `pretrained_backbone=False` the ImageNet
    backbone too: inference loads a state dict over both anyway, and a registry check only needs
    the architecture to build. Training keeps the backbone even in a `weights=None` ablation.
    """

    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.fcos import FCOSClassificationHead
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    spec = _known(name)
    factory = getattr(torchvision.models.detection, spec.factory)
    kwargs: dict[str, Any] = {"weights": weights}
    if not pretrained_backbone:
        kwargs["weights_backbone"] = None
    if spec.takes_size:
        kwargs |= {"min_size": min_size, "max_size": max_size}
    model = factory(**kwargs)

    if spec.family == "fasterrcnn":
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    elif spec.family == "retinanet":
        head = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=_head_in_channels(head),
            num_anchors=model.anchor_generator.num_anchors_per_location()[0],
            num_classes=NUM_CLASSES,
            norm_layer=None,
        )
    elif spec.family == "fcos":
        head = model.head.classification_head
        model.head.classification_head = FCOSClassificationHead(
            in_channels=_head_in_channels(head),
            num_anchors=model.anchor_generator.num_anchors_per_location()[0],
            num_classes=NUM_CLASSES,
        )
    else:  # pragma: no cover - the registry and this branch move together
        raise ValueError(f"no head-replacement rule for family {spec.family!r}")
    return model


def _head_in_channels(head: Any) -> int:
    first = head.conv[0]
    channels: int = first[0].in_channels if hasattr(first, "__getitem__") else first.in_channels
    return channels


def top1(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    threshold: float,
) -> dict[BarbellClass, BarbellDetection | None]:
    """The highest-scoring detection per class at or above `threshold`, or None.

    The scene holds at most one cap and one collar that matter, so a spurious box can only
    survive by outscoring the real one. Rounded as the old script rounded, so its output and
    this stage's compare to the digit.
    """

    found: dict[BarbellClass, BarbellDetection | None] = dict.fromkeys(BARBELL_CLASSES)
    for box, score, label in zip(boxes, scores, labels, strict=True):
        name = CLASS_NAMES.get(int(label))
        if name is None or score < threshold:
            continue
        best = found[name]
        if best is None or score > best.score:
            x0, y0, x1, y1 = (round(float(v), 1) for v in box)
            found[name] = BarbellDetection(box=(x0, y0, x1, y1), score=round(float(score), 4))
    return found


def resolve_device(requested: BarbellDevice, *, mps: bool, cuda: bool) -> str:
    """The torch device for `requested`, given what this machine has."""

    available = {"mps": mps, "cuda": cuda, "cpu": True}
    if requested is BarbellDevice.AUTO:
        return next(name for name in ("mps", "cuda", "cpu") if available[name])
    if not available[requested.value]:
        raise ValueError(f"{requested.value} is not available on this machine")
    return requested.value


# --- checkpoint cards ---------------------------------------------------------------------


class CheckpointCard(BaseModel):
    """What a promoted checkpoint is: the `.json` written once beside its `.pt`."""

    model_config = ConfigDict(frozen=True, alias_generator=to_camel, populate_by_name=True)

    schema_version: str = CARD_SCHEMA_VERSION
    architecture: str
    classes: tuple[BarbellClass, ...]
    training_epoch: int | None
    training_run: str
    source_checkpoint: str
    sha256: str
    val_metrics: dict[str, Any] = Field(default_factory=dict)
    training_args: dict[str, Any] = Field(default_factory=dict)

    def to_document(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")


def card_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_card(checkpoint: Path) -> CheckpointCard:
    """The card for `checkpoint`, refusing anything that would make a run unreproducible."""

    if not checkpoint.is_file():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    path = card_path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"{checkpoint} has no card at {path}; promote it with scripts/promote_barbell_model.py"
        )
    card = CheckpointCard.model_validate(json.loads(path.read_text(encoding="utf-8")))
    _known(card.architecture)
    actual = sha256_file(checkpoint)
    if actual != card.sha256:
        raise ValueError(f"{checkpoint}: sha256 {actual} disagrees with its card's {card.sha256}")
    return card


def promote_checkpoint(source: Path, destination: Path) -> CheckpointCard:
    """Copy a training run's checkpoint to a model path, once, and write its card beside it."""

    import torch

    if destination.exists() or card_path(destination).exists():
        raise FileExistsError(f"{destination} already exists; a promoted model is never rewritten")
    blob = torch.load(source, map_location="cpu", weights_only=False)
    history_path = source.parent / "history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() else {}

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    card = CheckpointCard(
        architecture=blob.get("model", "fasterrcnn_resnet50_fpn_v2"),
        classes=BARBELL_CLASSES,
        training_epoch=blob.get("epoch"),
        training_run=source.parent.name,
        source_checkpoint=str(source),
        sha256=sha256_file(destination),
        val_metrics=blob.get("metrics", {}),
        training_args=history.get("args", {}),
    )
    card_path(destination).write_text(
        json.dumps(card.to_document(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return card


# --- the detector -------------------------------------------------------------------------


@runtime_checkable
class BarbellDetector(Protocol):
    """Something that finds the cap and collar, per frame of one video."""

    @property
    def provenance(self) -> BarbellProvenance: ...

    def detect(
        self, rgb_path: Path, n_frames: int
    ) -> dict[BarbellClass, list[BarbellDetection | None]]:
        """One list per class, each `n_frames` long; None where nothing passed the threshold."""
        ...


class TorchvisionBarbellDetector:  # pragma: no cover - exercised by the `slow` smoke test
    """The trained torchvision detector, loaded once and run at each video's native size."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: BarbellDevice,
        record_threshold: float,
        batch_size: int,
    ) -> None:
        # One torchvision op has no MPS kernel; without this it raises instead of using the CPU.
        mps_fallback = "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        import torchvision

        self._card = read_card(checkpoint)
        self._state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        self._device = resolve_device(
            device, mps=torch.backends.mps.is_available(), cuda=torch.cuda.is_available()
        )
        self._threshold = record_threshold
        self._batch_size = batch_size
        self._models: dict[tuple[int, int], Any] = {}
        self._provenance = BarbellProvenance(
            architecture=self._card.architecture,
            checkpoint=str(checkpoint),
            checkpoint_sha256=self._card.sha256,
            training_epoch=self._card.training_epoch,
            record_threshold=record_threshold,
            device=self._device,
            mps_fallback=mps_fallback and self._device == "mps",
            versions={"torch": torch.__version__, "torchvision": torchvision.__version__},
        )

    @property
    def provenance(self) -> BarbellProvenance:
        return self._provenance

    def _model_for(self, width: int, height: int) -> Any:
        # Trained at native resolution: a resize shifts every box, so build at the frame's size.
        key = (width, height)
        if key not in self._models:
            model = build_model(
                self._card.architecture,
                min_size=min(width, height),
                max_size=max(width, height),
                weights=None,
                pretrained_backbone=False,
            )
            model.load_state_dict(self._state)
            self._models[key] = model.to(self._device).eval()
        return self._models[key]

    def detect(
        self, rgb_path: Path, n_frames: int
    ) -> dict[BarbellClass, list[BarbellDetection | None]]:
        import torch

        found: dict[BarbellClass, list[BarbellDetection | None]] = {
            name: [] for name in BARBELL_CLASSES
        }
        with torch.inference_mode():
            for batch in _batches(rgb_path, n_frames, self._batch_size):
                height, width = batch[0].shape[:2]
                model = self._model_for(width, height)
                tensors = [
                    torch.from_numpy(frame).permute(2, 0, 1).float().div(255).to(self._device)
                    for frame in batch
                ]
                for output in model(tensors):
                    hits = top1(
                        output["boxes"].cpu().tolist(),
                        output["scores"].cpu().tolist(),
                        output["labels"].cpu().tolist(),
                        threshold=self._threshold,
                    )
                    for name in BARBELL_CLASSES:
                        found[name].append(hits[name])
        for name in BARBELL_CLASSES:
            found[name].extend([None] * (n_frames - len(found[name])))
        return found


def _batches(  # pragma: no cover - exercised by the `slow` smoke test
    rgb_path: Path, n_frames: int, size: int
) -> Iterator[list[np.ndarray]]:
    """Up to `n_frames` RGB frames, `size` at a time, decoded as they're needed."""

    batch: list[np.ndarray] = []
    with av.open(str(rgb_path)) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index >= n_frames:
                break
            batch.append(frame.to_ndarray(format="rgb24"))
            if len(batch) == size:
                yield batch
                batch = []
    if batch:
        yield batch


def build_barbell_detector(config: PreprocessConfig) -> BarbellDetector:  # pragma: no cover
    """The detector `config.barbell_checkpoint` names. Loads torch, so only called to run S5."""

    assert config.barbell_checkpoint is not None
    return TorchvisionBarbellDetector(
        config.barbell_checkpoint,
        device=config.barbell_device,
        record_threshold=config.barbell_record_threshold,
        batch_size=config.barbell_batch_size,
    )
