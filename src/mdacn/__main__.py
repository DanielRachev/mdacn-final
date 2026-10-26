"""Command-line entry points for preprocessing, dynamics, and impact analysis."""

import argparse
from pathlib import Path

import matplotlib

from mdacn import config
from mdacn.extraction_comparison import compare_extractions
from mdacn.link_comparison import compare_link_sources
from mdacn.workflow import run_project, run_stage


def main() -> None:
    matplotlib.use("Agg")  # Commands save figures to files; no GUI is needed.
    parser = argparse.ArgumentParser(
        prog="python -m mdacn", description="MDACN Project Reddit Graph Echo Chamber"
    )

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="Run all implemented project stages using config.py")
    setup = commands.add_parser("setup", help="Download a SNAP link TSV; keep an existing file")
    setup.add_argument("--dataset", choices=("body", "title"), default="body")
    setup.add_argument("--output", type=Path)

    characterisation = commands.add_parser(
        "characterise", help="Save aggregate/core graphs, metrics, and plots"
    )
    characterisation.add_argument("--input", type=Path, default=Path(config.PATHS["dataset"]))
    characterisation.add_argument(
        "--core-method",
        choices=("k_core", "activity_lcc"),
        default=config.CHARACTERISATION["core_method"],
    )
    characterisation.add_argument(
        "--k",
        type=int,
        default=config.CHARACTERISATION["k"],
        help="Minimum degree within the k-core",
    )
    characterisation.add_argument(
        "--min-interactions",
        type=int,
        default=config.CHARACTERISATION["min_interactions"],
        help="Original activity threshold",
    )
    characterisation.add_argument("--seed", type=int, default=config.CHARACTERISATION["seed"])
    characterisation.add_argument(
        "--processed-dir",
        "--processed_dir",
        type=Path,
        default=Path(config.PATHS["processed_dir"]),
    )
    characterisation.add_argument(
        "--results-dir", type=Path, default=config.stage_paths()["topology"]
    )

    extraction = commands.add_parser(
        "compare-extractions", help="Compare k-core and activity-LCC topology; export each core"
    )
    extraction.add_argument("--input", type=Path, default=Path(config.PATHS["dataset"]))
    extraction.add_argument("--ks", type=int, nargs="+", default=(8, 10, 12))
    extraction.add_argument("--min-interactions", type=int, default=100)
    extraction.add_argument("--seed", type=int, default=0)
    extraction.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(config.PATHS["processed_dir"]) / "extraction_comparison",
    )
    extraction.add_argument(
        "--results-dir",
        type=Path,
        default=Path(config.PATHS["results_dir"]) / "extraction_comparison",
    )

    links = commands.add_parser(
        "compare-link-sources", help="Audit body/title coverage and overlap without simulations"
    )
    links.add_argument("--body-input", type=Path, default=Path(config.PATHS["dataset"]))
    links.add_argument("--title-input", type=Path, default=Path(config.PATHS["title_dataset"]))
    links.add_argument("--k", type=int, default=10)
    links.add_argument(
        "--results-dir", type=Path, default=Path(config.PATHS["results_dir"]) / "link_sources"
    )

    simulation_commands = {
        "nco": "Compare unsigned and signed NCO on the saved core",
        "sign-shuffle": "Compare empirical signed NCO against a sign-shuffled null model",
        "ico": "Run existing random, degree, and leverage ICO interventions",
    }
    paths = config.stage_paths()
    for name, help_text in simulation_commands.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--graph", type=Path, default=paths["core"])
        command.add_argument("--results-dir", type=Path, default=paths[name])
        command.add_argument(
            "--fractions", type=float, nargs="+", default=config.SIMULATION["fractions"]
        )
        command.add_argument("--realizations", type=int, default=config.SIMULATION["realizations"])
        command.add_argument("--seed", type=int, default=config.SIMULATION["seed"])
        command.add_argument("--max-steps", type=int, default=config.SIMULATION["max_steps"])
        if name == "ico":
            command.add_argument("--budgets", type=float, nargs="+", default=config.ICO["budgets"])

    commands.add_parser("centrality", help="Calculate and save the existing core centrality scores")
    commands.add_parser("features", help="Save node scores for subsequent experiments")
    commands.add_parser("node-impact", help="Run paired interventions using saved node scores")
    commands.add_parser(
        "reports", help="Save summaries, impact statistics, and plots from stored results"
    )

    args = parser.parse_args()
    try:
        if args.command == "run":
            run_project()
        elif args.command == "setup":
            run_stage("setup", output=args.output, dataset=args.dataset)
        elif args.command == "characterise":
            run_stage(
                "characterise",
                input_path=args.input,
                core_method=args.core_method,
                k=args.k,
                min_interactions=args.min_interactions,
                seed=args.seed,
                processed_dir=args.processed_dir,
                results_dir=args.results_dir,
            )
        elif args.command == "compare-extractions":
            compare_extractions(
                args.input,
                ks=tuple(args.ks),
                min_interactions=args.min_interactions,
                seed=args.seed,
                processed_dir=args.processed_dir,
                results_dir=args.results_dir,
            )
        elif args.command == "compare-link-sources":
            compare_link_sources(
                args.body_input,
                args.title_input,
                k=args.k,
                results_dir=args.results_dir,
            )
        elif args.command in simulation_commands:
            run_stage(
                args.command,
                graph_path=args.graph,
                results_dir=args.results_dir,
                fractions=tuple(args.fractions),
                realizations=args.realizations,
                seed=args.seed,
                max_steps=args.max_steps,
                **({"budgets": tuple(args.budgets)} if args.command == "ico" else {}),
            )
        else:
            run_stage(args.command)
    except Exception as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
