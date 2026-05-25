#!/usr/bin/env python3
"""Run an Ultralytics baseline training config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from public_detect.elements import load_element_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PROJECT_PREFIX = "score_miner_project/public_detect/"


TRAIN_KEYS = {
    "data",
    "epochs",
    "imgsz",
    "batch",
    "patience",
    "device",
    "workers",
    "seed",
    "optimizer",
    "cos_lr",
    "close_mosaic",
    "cache",
    "amp",
    "plots",
    "val",
    "project",
    "name",
}


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"invalid training config: {path}")
    return data


def project_path(value: str | Path) -> Path:
    """Resolve config paths from this package root, including old repo-root paths."""
    path_text = str(value)
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path_text.startswith(LEGACY_PROJECT_PREFIX):
        path = Path(path_text.removeprefix(LEGACY_PROJECT_PREFIX))
    return PROJECT_ROOT / path


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    element_config = config.get("element_config")
    if not element_config:
        raise ValueError("training config missing element_config")
    element_config_path = project_path(element_config)
    spec = load_element_spec(element_config_path)

    data_yaml = project_path(config["data"])
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml does not exist: {data_yaml}")
    dataset = yaml.safe_load(data_yaml.read_text())
    names = dataset.get("names")
    if isinstance(names, dict):
        dataset_names = tuple(str(names[idx]) for idx in sorted(names))
    elif isinstance(names, list):
        dataset_names = tuple(str(item) for item in names)
    else:
        raise ValueError(f"dataset names must be list or dict in {data_yaml}")
    if dataset_names != spec.objects:
        raise ValueError(
            f"dataset class order mismatch for {spec.element_id}: "
            f"expected {spec.objects}, got {dataset_names}"
        )

    model_name = str(config["model"])
    YOLO(model_name)
    return {
        "element_id": spec.element_id,
        "objects": spec.objects,
        "model": model_name,
        "data": str(data_yaml),
    }


def train_args(config: dict[str, Any], overrides: argparse.Namespace) -> dict[str, Any]:
    args = {key: config[key] for key in TRAIN_KEYS if key in config}
    if "project" in args:
        args["project"] = str(project_path(args["project"]).resolve())
    if "data" in args:
        args["data"] = str(project_path(args["data"]).resolve())
    if overrides.epochs is not None:
        args["epochs"] = overrides.epochs
    if overrides.imgsz is not None:
        args["imgsz"] = overrides.imgsz
    if overrides.batch is not None:
        args["batch"] = overrides.batch
    if overrides.device is not None:
        args["device"] = overrides.device
    if overrides.name_suffix:
        args["name"] = f"{args.get('name', Path(config['model']).stem)}_{overrides.name_suffix}"
    return args


def expected_run_dir(args: dict[str, Any]) -> str:
    project = Path(str(args["project"]))
    name = str(args.get("name") or "train")
    return str(project / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--name-suffix")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = validate_config(config)
    fit_args = train_args(config, args)
    print(
        json.dumps(
            {
                "summary": summary,
                "train_args": fit_args,
                "expected_run_dir": expected_run_dir(fit_args),
                "expected_best_checkpoint": str(Path(expected_run_dir(fit_args)) / "weights" / "best.pt"),
                "expected_last_checkpoint": str(Path(expected_run_dir(fit_args)) / "weights" / "last.pt"),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    model = YOLO(str(config["model"]))
    model.train(**fit_args)


if __name__ == "__main__":
    main()
