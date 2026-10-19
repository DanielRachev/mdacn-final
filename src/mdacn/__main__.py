"""Commands: python -m mdacn setup / characterise / nco."""

import argparse
from pathlib import Path

from mdacn.data import download_dataset
from mdacn.experiments import DEFAULT_FRACTIONS
from mdacn.nco_pipeline import run_nco_experiment
from mdacn.pipeline import characterise

DEFAULT_DATASET_PATH = Path("data/raw/soc-RedditHyperlinks-body.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mdacn", description="MDACN Project Reddit Graph Echo Chamber"
    )

    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Download the SNAP body TSV; keep an existing file")
    setup.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)

    analysis = commands.add_parser(
        "characterise", help="Save aggregate/core graphs, metrics, and plots"
    )
    analysis.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH)
    analysis.add_argument("--core-method", choices=("k_core", "activity_lcc"), default="k_core")
    analysis.add_argument(
        "--k", type=int, default=10, help="Minimum degree within the k-core (default: 10)"
    )
    analysis.add_argument(
        "--min-interactions",
        type=int,
        default=100,
        help="Original activity threshold (default: 100)",
    )
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument(
        "--processed-dir", "--processed_dir", type=Path, default=Path("data/processed")
    )
    analysis.add_argument("--results-dir", type=Path, default=Path("results/topology"))

    nco = commands.add_parser("nco", help="Compare unsigned and signed NCO on the saved core")
    nco.add_argument("--graph", type=Path, default=Path("data/processed/core.graphml"))
    nco.add_argument("--results-dir", type=Path, default=Path("results/nco"))
    nco.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=DEFAULT_FRACTIONS,
        help="Initial negative fractions; default adds 0.02-spaced points between 0.4 and 0.6",
    )
    nco.add_argument("--realizations", type=int, default=20)
    nco.add_argument("--seed", type=int, default=0)
    nco.add_argument("--max-steps", type=int, default=1000)

    args = parser.parse_args()
    try:
        if args.command == "setup":
            exists = args.output.is_file()
            print(f"{'Using existing' if exists else 'Downloading'}")
            download_dataset(args.output)
            print(f"Dataset ready: {args.output}")
        elif args.command == "characterise":
            characterise(
                args.input,
                core_method=args.core_method,
                k=args.k,
                min_interactions=args.min_interactions,
                seed=args.seed,
                processed_dir=args.processed_dir,
                results_dir=args.results_dir,
            )
        elif args.command == "nco":
            run_nco_experiment(
                args.graph,
                results_dir=args.results_dir,
                fractions=tuple(args.fractions),
                realizations=args.realizations,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        else:
            print(f"Unknown command supplied: {args.command}")
    except Exception as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
