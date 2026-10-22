"""Dataset setup, graph characterisation, structural comparisons, and experiments."""

import argparse
from pathlib import Path

from mdacn.data import download_dataset
from mdacn.experiments import DEFAULT_FRACTIONS
from mdacn.extraction_comparison import compare_extractions
from mdacn.link_comparison import compare_link_sources
from mdacn.nco_pipeline import run_nco_experiment, run_sign_shuffle_experiment
from mdacn.pipeline import characterise

DEFAULT_DATASET_PATH = Path("data/raw/soc-RedditHyperlinks-body.tsv")
DEFAULT_TITLE_PATH = Path("data/raw/soc-RedditHyperlinks-title.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mdacn", description="MDACN Project Reddit Graph Echo Chamber"
    )

    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Download a SNAP link TSV; keep an existing file")
    setup.add_argument("--dataset", choices=("body", "title"), default="body")
    setup.add_argument("--output", type=Path)

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

    extraction = commands.add_parser(
        "compare-extractions", help="Compare k-core and activity-LCC topology; export each core"
    )
    extraction.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH)
    extraction.add_argument("--ks", type=int, nargs="+", default=(8, 10, 12))
    extraction.add_argument("--min-interactions", type=int, default=100)
    extraction.add_argument("--seed", type=int, default=0)
    extraction.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/extraction_comparison")
    )
    extraction.add_argument(
        "--results-dir", type=Path, default=Path("results/extraction_comparison")
    )

    links = commands.add_parser(
        "compare-link-sources", help="Audit body/title coverage and overlap without simulations"
    )
    links.add_argument("--body-input", type=Path, default=DEFAULT_DATASET_PATH)
    links.add_argument("--title-input", type=Path, default=DEFAULT_TITLE_PATH)
    links.add_argument("--k", type=int, default=10)
    links.add_argument("--results-dir", type=Path, default=Path("results/link_sources"))

    nco = commands.add_parser("nco", help="Compare unsigned and signed NCO on the saved core")
    nco.add_argument("--graph", type=Path, default=Path("data/processed/core.graphml"))
    nco.add_argument("--results-dir", type=Path, default=Path("results/nco"))
    nco.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=DEFAULT_FRACTIONS,
        help="Initial negative fractions; default is 0.02 to 1.00 in steps of 0.02",
    )
    nco.add_argument("--realizations", type=int, default=20)
    nco.add_argument("--seed", type=int, default=0)
    nco.add_argument("--max-steps", type=int, default=1000)

    sign_shuffle = commands.add_parser(
        "sign-shuffle", help="Compare empirical signed NCO against sign-shuffled null model"
    )
    sign_shuffle.add_argument("--graph", type=Path, default=Path("data/processed/core.graphml"))
    sign_shuffle.add_argument("--results-dir", type=Path, default=Path("results/sign_shuffle"))
    sign_shuffle.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=DEFAULT_FRACTIONS,
        help="Initial negative fractions; default is 0.02 to 1.00 in steps of 0.02",
    )
    sign_shuffle.add_argument("--realizations", type=int, default=20)
    sign_shuffle.add_argument("--seed", type=int, default=0)
    sign_shuffle.add_argument("--max-steps", type=int, default=1000)

    args = parser.parse_args()
    try:
        if args.command == "setup":
            output = args.output or (
                DEFAULT_DATASET_PATH if args.dataset == "body" else DEFAULT_TITLE_PATH
            )
            exists = output.is_file()
            print(f"{'Using existing' if exists else 'Downloading'}")
            download_dataset(output, dataset=args.dataset)
            print(f"Dataset ready: {output}")
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
        elif args.command == "nco":
            run_nco_experiment(
                args.graph,
                results_dir=args.results_dir,
                fractions=tuple(args.fractions),
                realizations=args.realizations,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        elif args.command == "sign-shuffle":
            run_sign_shuffle_experiment(
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
