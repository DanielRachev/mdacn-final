from pathlib import Path

import pytest

from mdacn import __main__ as cli
from mdacn import workflow


@pytest.mark.parametrize(
    "command,target",
    [
        ("nco", "run_nco_experiment"),
        ("sign-shuffle", "run_sign_shuffle_experiment"),
    ],
)
@pytest.mark.parametrize("override", [False, True])
def test_simulation_commands_use_project_defaults_and_allow_overrides(
    monkeypatch, command, target, override
):
    monkeypatch.setitem(cli.config.PATHS, "processed_dir", "custom/processed")
    monkeypatch.setitem(cli.config.PATHS, "results_dir", "custom/results")
    for key, value in {
        "fractions": [0.2, 0.6],
        "realizations": 3,
        "seed": 17,
        "max_steps": 21,
    }.items():
        monkeypatch.setitem(cli.config.SIMULATION, key, value)
    calls = []
    monkeypatch.setattr(workflow, "_require", lambda *args: None)
    monkeypatch.setattr(
        workflow, target, lambda graph_path, **kwargs: calls.append((graph_path, kwargs))
    )
    args = ["mdacn", command]
    if override:
        args += [
            "--graph",
            "another.graphml",
            "--results-dir",
            "another/results",
            "--fractions",
            "0.5",
            "--realizations",
            "7",
            "--seed",
            "19",
            "--max-steps",
            "40",
        ]
    monkeypatch.setattr("sys.argv", args)
    cli.main()
    graph, options = calls[0]
    if override:
        assert graph == Path("another.graphml")
        assert options == {
            "results_dir": Path("another/results"),
            "fractions": (0.5,),
            "realizations": 7,
            "seed": 19,
            "max_steps": 40,
            "save_reports": False,
        }
    else:
        assert graph == Path("custom/processed/core.graphml")
        suffix = "nco" if command == "nco" else "sign_shuffle"
        assert options == {
            "results_dir": Path("custom/results") / suffix,
            "fractions": (0.2, 0.6),
            "realizations": 3,
            "seed": 17,
            "max_steps": 21,
            "save_reports": False,
        }


@pytest.mark.parametrize("override", [False, True])
def test_characterise_uses_project_paths_and_extraction_settings(monkeypatch, override):
    for key, value in {
        "dataset": "custom/body.tsv",
        "processed_dir": "custom/core",
        "results_dir": "custom/results",
    }.items():
        monkeypatch.setitem(cli.config.PATHS, key, value)
    for key, value in {
        "core_method": "activity_lcc",
        "k": 8,
        "min_interactions": 55,
        "seed": 11,
    }.items():
        monkeypatch.setitem(cli.config.CHARACTERISATION, key, value)
    calls = []
    monkeypatch.setattr(workflow, "_require", lambda *args: None)
    monkeypatch.setattr(
        workflow, "characterise", lambda input_path, **kwargs: calls.append((input_path, kwargs))
    )
    args = ["mdacn", "characterise"]
    if override:
        args += ["--core-method", "k_core", "--k", "12", "--min-interactions", "70", "--seed", "22"]
    monkeypatch.setattr("sys.argv", args)
    cli.main()
    path, options = calls[0]
    assert path == Path("custom/body.tsv")
    assert options["processed_dir"] == Path("custom/core")
    assert options["results_dir"] == Path("custom/results")
    assert (options["core_method"], options["k"], options["min_interactions"], options["seed"]) == (
        ("k_core", 12, 70, 22) if override else ("activity_lcc", 8, 55, 11)
    )


def test_setup_uses_shared_dataset_path(tmp_path, monkeypatch):
    dataset = tmp_path / "body.tsv"
    monkeypatch.setitem(cli.config.PATHS, "dataset", str(dataset))
    calls = []
    monkeypatch.setattr(workflow, "download_dataset", calls.append)
    monkeypatch.setattr("sys.argv", ["mdacn", "setup"])
    cli.main()
    assert calls == [dataset]
    assert not dataset.exists()
