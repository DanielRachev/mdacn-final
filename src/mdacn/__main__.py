"""Commands: python -m mdacn setup / characterise"""

import argparse
from pathlib import Path

from mdacn.data import download_dataset
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
        help="Original activity threshold (default: 100)"
    )
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    analysis.add_argument("--results-dir", type=Path, default=Path("results"))

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
                results_dir=args.results_dir
            )
        else:
            print(f"Unknown command supplied: {args.command}")
    except Exception as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
