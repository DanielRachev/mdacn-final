"""Commands: python -m mdacn setup / characterise / nco / sign-shuffle / rewire / rewire-ico."""

import argparse
from pathlib import Path

from mdacn.data import download_dataset
from mdacn.experiments import DEFAULT_FRACTIONS
from mdacn.nco_pipeline import run_nco_experiment, run_sign_shuffle_experiment
from mdacn.pipeline import characterise
from mdacn.rewiring_pipeline import (
    DEFAULT_BETAS,
    DEFAULT_BUDGETS,
    default_specs,
    run_rewired_ico_experiment,
    run_rewiring_experiment,
)

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

    rewiring = commands.add_parser(
        "rewire", help="Export degree-preserving rewired graphs (reference, insular, integrated)"
    )
    rewiring.add_argument("--graph", type=Path, default=Path("data/processed/core.graphml"))
    rewiring.add_argument("--nodes", type=Path, default=Path("data/processed/nodes.json"))
    rewiring.add_argument(
        "--centrality",
        type=Path,
        default=Path("data/processed/centrality.csv"),
        help="Saved empirical scores; computed once and saved if missing",
    )
    rewiring.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/topology/metrics.json"),
        help="Saved core metrics for the empirical assortativity",
    )
    rewiring.add_argument("--results-dir", type=Path, default=Path("results/rewiring"))
    rewiring.add_argument("--replicates", type=int, default=5)
    rewiring.add_argument(
        "--attempts-per-edge", type=int, default=100, help="Maximum attempts per edge"
    )
    rewiring.add_argument(
        "--window-per-edge", type=float, default=2.0, help="Convergence window; 0 disables"
    )
    rewiring.add_argument(
        "--tolerance", type=float, default=0.002, help="Allowed score change per window"
    )
    for target in ("insular", "integrated"):
        rewiring.add_argument(
            f"--{target}-betas",
            type=float,
            nargs="+",
            default=DEFAULT_BETAS[target],
            help=f"Metropolis betas for the {target} target",
        )
    rewiring.add_argument("--checkpoints", type=int, default=20)
    rewiring.add_argument("--seed", type=int, default=0)

    rewired_ico = commands.add_parser(
        "rewire-ico", help="Compare signed ICO Phi_c on the empirical and rewired graphs"
    )
    rewired_ico.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Saved empirical unsigned.npz, signed.npz, and nodes.json",
    )
    rewired_ico.add_argument("--rewired-dir", type=Path, default=Path("results/rewiring"))
    rewired_ico.add_argument("--results-dir", type=Path, default=Path("results/rewiring_ico"))
    rewired_ico.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    rewired_ico.add_argument("--f", type=float, default=0.5, help="Initial negative fraction")
    rewired_ico.add_argument("--initializations", type=int, default=20)
    rewired_ico.add_argument(
        "--strategies",
        nargs="+",
        choices=("degree", "random", "leverage"),
        default=("degree", "random"),
    )
    rewired_ico.add_argument("--seed", type=int, default=0)
    rewired_ico.add_argument("--max-steps", type=int, default=1000)

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
        elif args.command == "sign-shuffle":
            run_sign_shuffle_experiment(
                args.graph,
                results_dir=args.results_dir,
                fractions=tuple(args.fractions),
                realizations=args.realizations,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        elif args.command == "rewire":
            run_rewiring_experiment(
                args.graph,
                nodes_path=args.nodes,
                centrality_path=args.centrality,
                metrics_path=args.metrics,
                results_dir=args.results_dir,
                specs=default_specs(
                    {
                        "insular": tuple(args.insular_betas),
                        "integrated": tuple(args.integrated_betas),
                    }
                ),
                replicates=args.replicates,
                attempts_per_edge=args.attempts_per_edge,
                window_per_edge=args.window_per_edge,
                tolerance=args.tolerance,
                seed=args.seed,
                checkpoints=args.checkpoints,
                show_progress=True,
            )
        elif args.command == "rewire-ico":
            run_rewired_ico_experiment(
                args.processed_dir,
                rewired_dir=args.rewired_dir,
                results_dir=args.results_dir,
                budgets=tuple(args.budgets),
                negative_fraction=args.f,
                initializations=args.initializations,
                strategies=tuple(args.strategies),
                seed=args.seed,
                max_steps=args.max_steps,
                show_progress=True,
            )
        else:
            print(f"Unknown command supplied: {args.command}")
    except Exception as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
