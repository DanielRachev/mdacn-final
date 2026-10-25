"""Project stages connected through saved artifacts and shared config."""

from pathlib import Path

from mdacn import config
from mdacn.artifacts import checksum, write_json
from mdacn.centrality_pipeline import load_centrality_scores, save_centrality_scores
from mdacn.data import download_dataset, load_graph
from mdacn.features import validate_features
from mdacn.ico_pipeline import run_ico_experiment, save_ico_reports
from mdacn.impact_pipeline import _read_features, save_impact_reports
from mdacn.impact_workflow import run_node_impact
from mdacn.nco_pipeline import run_nco_experiment, run_sign_shuffle_experiment, save_nco_reports
from mdacn.pipeline import characterise

STAGES = (
    "setup",
    "characterise",
    "centrality",
    "features",
    "nco",
    "sign-shuffle",
    "ico",
    "node-impact",
    "reports",
)


def _require(paths: list[Path], stage: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Missing inputs: {', '.join(missing)}. Run {stage} first.")


def run_features() -> None:
    """Merge and validate contributor exports; never calculate node scores."""
    paths = config.stage_paths()
    _require([paths["core"]], "characterise")
    load_centrality_scores(paths["core"])
    source_paths = (
        paths["centrality"],
        paths["signed_leverage"],
        paths["incident_signs"],
        paths["temporal_activity"],
    )
    _require(list(source_paths), "the contributor's export stage")
    supplied, sources = _read_features(source_paths)
    features = validate_features(supplied, load_graph(paths["core"]))
    if paths["features"].resolve() in {path.resolve() for path in source_paths}:
        raise ValueError("Feature output must not also be a supplied input table")
    features.to_csv(paths["features"], index=False)
    write_json(
        paths["feature_metadata"],
        {
            "graph_sha256": checksum(paths["core"]),
            "features_sha256": checksum(paths["features"]),
            "columns": features.columns.tolist(),
            "sources": sources,
        },
    )
    print(f"Saved node features: {paths['features']}")


def save_reports() -> None:
    """Save tables and plots from available stored experiment results."""
    paths = config.stage_paths()
    study = paths["node-impact"] / "study"
    available = []
    for stage, filename in (
        ("nco", "fraction_sweep.csv"),
        ("sign-shuffle", "sign_shuffle_sweep.csv"),
        ("ico", "intervention_sweep.csv"),
    ):
        if (paths[stage] / filename).is_file():
            _require([paths[stage] / "metadata.json"], stage)
            available.append(stage)
    if (study / "runs.csv").is_file():
        _require([study / "simulation_metadata.json", study / "plan.json"], "node-impact")
        available.append("node-impact")
    if not available:
        raise ValueError(
            "No saved experiment results; run nco, sign-shuffle, ico, or node-impact first"
        )
    for stage in available:
        if stage == "node-impact":
            save_impact_reports(study, **config.NODE_IMPACT["analysis"])
        elif stage == "ico":
            save_ico_reports(paths[stage])
        else:
            save_nco_reports(paths[stage], sign_shuffle=stage == "sign-shuffle")


def run_stage(stage: str, **overrides) -> None:
    paths = config.stage_paths()
    if stage == "setup":
        dataset = overrides.get("dataset", "body")
        default = paths["dataset"] if dataset == "body" else Path(config.PATHS["title_dataset"])
        destination = overrides.get("output") or default
        download_dataset(destination, dataset=dataset)
        print(f"Dataset ready: {destination}")
    elif stage == "characterise":
        options = {
            "input_path": paths["dataset"],
            "processed_dir": paths["processed"],
            "results_dir": paths["topology"],
            **config.CHARACTERISATION,
            **overrides,
        }
        _require([options["input_path"]], "setup")
        characterise(**options)
    elif stage == "centrality":
        _require([paths["core"], paths["nodes"]], "characterise")
        save_centrality_scores(paths["core"])
    elif stage == "features":
        run_features()
    elif stage in ("nco", "sign-shuffle", "ico"):
        options = {
            "graph_path": paths["core"],
            "results_dir": paths[stage],
            **config.SIMULATION,
            **overrides,
        }
        options["fractions"] = tuple(options["fractions"])
        _require([options["graph_path"]], "characterise")
        if stage == "ico":
            options.setdefault("budgets", tuple(config.ICO["budgets"]))
        experiment = {
            "nco": run_nco_experiment,
            "sign-shuffle": run_sign_shuffle_experiment,
            "ico": run_ico_experiment,
        }[stage]
        experiment(**options, save_reports=False)
    elif stage == "node-impact":
        _require([paths["core"]], "characterise")
        _require([paths["features"], paths["feature_metadata"]], "features")
        run_node_impact(save_reports=False)
    elif stage == "reports":
        save_reports()
    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_project() -> None:
    for stage in STAGES:
        print(f"Running stage: {stage}", flush=True)
        run_stage(stage)
