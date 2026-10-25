"""Node-impact workflow with settings declared in config.py."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

from mdacn.config import impact_settings
from mdacn.impact import validate_design
from mdacn.impact_pipeline import (
    prepare_impact_study,
    save_impact_reports,
    simulate_impact_study,
)


def _positive_integer(value: object, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _reset_study(output: Path, inputs: tuple[Path, ...]) -> Path:
    """Clear generated study files only, after checking resolved paths."""
    root = output.resolve()
    study = output / "study"
    target = study.resolve()
    config_file = output / "config.json"
    if Path.cwd().resolve().is_relative_to(root) or target.parent != root:
        raise ValueError("Results directory must be separate from the project and linked folders")
    if config_file.resolve() != root / "config.json":
        raise ValueError("config.json must not link outside the results directory")
    if any(
        path.resolve().is_relative_to(target) or path.resolve() == config_file.resolve()
        for path in inputs
    ):
        raise ValueError("Input files must not be stored among generated study files")
    if study.is_dir():
        shutil.rmtree(study)
    elif study.exists():
        study.unlink()
    return study


def run_node_impact(
    settings: dict | None = None,
    *,
    save_reports: bool = True,
) -> Path:
    """Run paired interventions and save outcomes, with optional statistical reports."""
    config = deepcopy(impact_settings() if settings is None else settings)
    expected = {
        "": {
            "graph",
            "results_dir",
            "features",
            "feature_metadata",
            "simulation",
            "analysis",
        },
        "simulation": {"fractions", "models", "realizations", "seed", "max_steps"},
        "analysis": {"bootstrap", "seed"},
    }
    for section, keys in expected.items():
        values = config if not section else config.get(section)
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError(f"Invalid {section or 'root'} config; expected keys: {sorted(keys)}")
    simulation, analysis = config["simulation"], config["analysis"]
    for name, value in (
        ("simulation.realizations", simulation["realizations"]),
        ("simulation.max_steps", simulation["max_steps"]),
    ):
        _positive_integer(value, name)
    _positive_integer(analysis["bootstrap"], "analysis.bootstrap", minimum=2)
    for section in (simulation, analysis):
        _positive_integer(section["seed"], "seed", minimum=0)
    validate_design(
        tuple(simulation["fractions"]),
        tuple(simulation["models"]),
        simulation["realizations"],
        simulation["max_steps"],
    )
    for name in ("graph", "results_dir", "features", "feature_metadata"):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError(f"{name} must be a nonempty path")
    graph_path, output = Path(config["graph"]), Path(config["results_dir"])
    if output.exists() and not output.is_dir():
        raise ValueError("results_dir must be a directory")
    features_path = Path(config["features"])
    feature_metadata_path = Path(config["feature_metadata"])
    for path, stage in (
        (graph_path, "characterise"),
        (features_path, "features"),
        (feature_metadata_path, "features"),
    ):
        if not path.is_file():
            raise ValueError(f"Missing input: {path}; run {stage} first")
    study = _reset_study(output, (graph_path, features_path, feature_metadata_path))
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    prepare_impact_study(
        graph_path,
        results_dir=study,
        features_path=features_path,
        feature_metadata_path=feature_metadata_path,
        fractions=tuple(simulation["fractions"]),
        models=tuple(simulation["models"]),
        realizations=simulation["realizations"],
        seed=simulation["seed"],
        max_steps=simulation["max_steps"],
    )
    simulate_impact_study(study)
    if save_reports:
        save_impact_reports(study, bootstrap=analysis["bootstrap"], seed=analysis["seed"])
    print(f"Node-impact workflow complete: {output}")
    return output
