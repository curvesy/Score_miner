from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML

from score_miner_core.validator_sim.pgt_loader import (
    _ensure_turbovision_importable,
    load_pseudo_ground_truth,
)
from score_miner_core.validator_sim.replay_loader import (
    load_json_object,
    load_predictions_from_response,
    load_replay_response,
    write_json,
)


class ValidatorSimReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replay_dir: str
    manifest_path: str
    element_id: str
    pgt_path: str
    pgt_frames: int
    prediction_frames: int
    score_available: bool
    breakdown: dict[str, Any]
    mean_weighted: float | None = None
    errors: list[str]


def run_validator_sim(
    *,
    replay_dir: Path,
    pgt_path: Path,
    manifest_path: Path,
    turbovision_path: Path,
    element_id: str,
    output_path: Path | None = None,
) -> ValidatorSimReport:
    _ensure_turbovision_importable(turbovision_path)

    try:
        from scorevision.utils.data_models import SVRunOutput
        from scorevision.utils.evaluate import get_element_scores
        from scorevision.utils.manifest import Manifest
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise ModuleNotFoundError(
            "TurboVision scorer dependencies are not installed in this Python environment. "
            "Run validator_sim with the TurboVision uv environment, for example: "
            "`cd ../turbovision && PYTHONPATH=../score_miner_project/score_miner_dev/src "
            "uv run python -m score_miner_core.validator_sim.score_runner ...`. "
            f"Missing module: {missing}"
        ) from exc

    response = load_replay_response(replay_dir)
    predictions = load_predictions_from_response(response)
    if predictions is None:
        report = ValidatorSimReport(
            replay_dir=str(replay_dir),
            manifest_path=str(manifest_path),
            element_id=element_id,
            pgt_path=str(pgt_path),
            pgt_frames=0,
            prediction_frames=0,
            score_available=False,
            breakdown={},
            mean_weighted=None,
            errors=[str(response.get("error") or "Replay response has no predictions")],
        )
        _write_report_if_requested(report, output_path)
        return report

    manifest = _load_manifest_for_element(
        manifest_path=manifest_path,
        element_id=element_id,
        manifest_cls=Manifest,
    )
    pgt = load_pseudo_ground_truth(pgt_path, turbovision_path=turbovision_path)
    miner_run = SVRunOutput(
        success=bool(response.get("success")),
        latency_ms=0.0,
        predictions=predictions,
        error=response.get("error"),
    )

    errors: list[str] = []
    try:
        breakdown = get_element_scores(
            manifest=manifest,
            pseudo_gt_annotations=pgt,
            miner_run=miner_run,
            frame_store={},
            challenge_type_id=None,
            element_id=element_id,
        )
    except Exception as exc:
        breakdown = {}
        errors.append(str(exc))

    mean_weighted = breakdown.get("mean_weighted") if isinstance(breakdown, dict) else None
    report = ValidatorSimReport(
        replay_dir=str(replay_dir),
        manifest_path=str(manifest_path),
        element_id=element_id,
        pgt_path=str(pgt_path),
        pgt_frames=len(pgt),
        prediction_frames=len(predictions.get("frames") or []),
        score_available=not errors,
        breakdown=breakdown,
        mean_weighted=float(mean_weighted) if mean_weighted is not None else None,
        errors=errors,
    )
    _write_report_if_requested(report, output_path)
    return report


def _load_manifest_for_element(
    *,
    manifest_path: Path,
    element_id: str,
    manifest_cls: type,
):
    yaml = YAML(typ="safe")
    data = yaml.load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest YAML must decode to an object: {manifest_path}")
    elements = data.get("elements")
    if not isinstance(elements, list):
        raise ValueError(f"Manifest YAML must contain elements: []: {manifest_path}")
    selected = [element for element in elements if isinstance(element, dict) and element.get("id") == element_id]
    if not selected:
        raise ValueError(f"Element {element_id!r} not found in manifest {manifest_path}")
    data = dict(data)
    data["elements"] = selected
    return manifest_cls(**data)


def _write_report_if_requested(report: ValidatorSimReport, output_path: Path | None) -> None:
    if output_path is None:
        return
    write_json(output_path, report.model_dump(mode="json"))
    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Validator Sim Report",
        "",
        f"- replay_dir: `{report.replay_dir}`",
        f"- element_id: `{report.element_id}`",
        f"- score_available: `{report.score_available}`",
        f"- mean_weighted: `{report.mean_weighted}`",
        f"- pgt_frames: `{report.pgt_frames}`",
        f"- prediction_frames: `{report.prediction_frames}`",
        "",
        "## Errors",
        "",
        "```json",
        __import__("json").dumps(report.errors, indent=2, sort_keys=True),
        "```",
        "",
        "## Breakdown",
        "",
        "```json",
        __import__("json").dumps(report.breakdown, indent=2, sort_keys=True),
        "```",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a replay with TurboVision scorer and PGT JSON.")
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--pgt", type=Path, required=True, help="Pseudo/ground-truth JSON file.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("../turbovision/tests/test_data/manifests/example_manifest.yml"),
    )
    parser.add_argument("--turbovision-path", type=Path, default=Path("../turbovision"))
    parser.add_argument("--element-id", default="PlayerDetect_v1@1.0")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.replay_dir / "score_report.json"
    report = run_validator_sim(
        replay_dir=args.replay_dir,
        pgt_path=args.pgt,
        manifest_path=args.manifest,
        turbovision_path=args.turbovision_path,
        element_id=args.element_id,
        output_path=output,
    )
    print(load_json_object(output) if output else report.model_dump(mode="json"))


if __name__ == "__main__":
    main()
