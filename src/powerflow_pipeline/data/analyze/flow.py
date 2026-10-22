"""The analyze flow: S8 kinematics, S9 phase.

Ordered orchestration only.

Runs per camera, on what preprocess and detect published. No video is read and no detector is
loaded, so a rule change re-runs in seconds: `stages=(PHASE,)` re-runs S9 against the S8 tree
already in `kinematics_root`.
"""

from __future__ import annotations

from pathlib import Path

from prefect import flow

from powerflow_pipeline.data.analyze.config import AnalyzeConfig
from powerflow_pipeline.data.analyze.tasks.kinematics import solve_kinematics_camera
from powerflow_pipeline.data.analyze.tasks.phase import detect_phases_camera
from powerflow_pipeline.data.common.camera_records import discover_published_cameras
from powerflow_pipeline.data.common.context import OutputMode, RunContext
from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.manifest import RunManifest, emit_manifest, scan_outcome
from powerflow_pipeline.data.common.models import RejectedScan, StepResult

PIPELINE = "analyze"


@flow(name=PIPELINE)
def analyze(config: AnalyzeConfig) -> RunManifest:
    """Solve the joints in 3D and split each lift into its phases. Or reject, with a reason.

    S8 runs when it is selected and `config.kinematics_root` is set; S9 likewise with
    `config.phase_root`. S9 reads S8's published `joint_hva.json`, whether this run wrote it or
    an earlier one did.
    """

    if config.runs_phase and config.kinematics_root is None:
        raise ValueError(
            "config.phase_root is set but config.kinematics_root is unset: no kinematics to read"
        )
    if config.runs_phase and config.barbell_root is None:
        raise ValueError(
            "config.phase_root is set but config.barbell_root is unset: no bar to segment"
        )
    if not (config.runs_kinematics or config.runs_phase):
        raise ValueError(
            "nothing to run: select S8 with a kinematics root, or S9 with a phase root"
        )

    context = RunContext.create(
        pipeline=PIPELINE,
        input_root=config.pose_root,
        output_root=config.phase_root if config.runs_phase else config.kinematics_root,
        output_mode=OutputMode.PUBLISH,
        dry_run=config.dry_run,
    )
    manifest = RunManifest(
        pipeline=context.pipeline,
        input_root=context.input_root,
        output_root=context.output_root,
        output_mode=context.output_mode,
        dry_run=context.dry_run,
    )

    found = discover_published_cameras(
        config.record_root, config.cut_root, config.crop_root, only=config.only
    )
    manifest.rejected_scans.extend(found.rejected)

    for record in found.records:
        ran: list[tuple[str, StepResult]] = []
        try:
            if config.runs_kinematics:
                ran.append(("kinematics", solve_kinematics_camera(record, config)[1]))
            if config.runs_phase:
                ran.append(("phase", detect_phases_camera(record, config)[1]))
        except ScanRejected as rejection:
            manifest.rejected_scans.append(
                RejectedScan(scan_id=record.capture_id, source=record.source, reason=str(rejection))
            )
            continue
        manifest.scans.append(
            scan_outcome(record.capture_id, record.source, ran, dry_run=config.dry_run)
        )

    manifest_path = None if config.dry_run else _manifest_path(context.output_root)
    return emit_manifest(manifest, manifest_path)


def _manifest_path(root: Path | None) -> Path:
    """The run's manifest sits in the tree of the last stage it ran: S9, else S8."""

    assert root is not None  # publish mode always has an output root
    root.mkdir(parents=True, exist_ok=True)
    return root / "manifest.json"
