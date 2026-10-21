"""Central entry point for all temporal analyses; existing runners remain reusable."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNERS = {
    "analysis": "run_temporal_analysis.py",
    "robustness": "run_temporal_robustness.py",
    "final": "run_temporal_final.py",
    "diagnostics": "run_temporal_diagnostics.py",
    "compare": "run_static_temporal_comparison.py",
    "ico": "run_temporal_ico.py",
    "audit": "audit_temporal_baselines.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected or all temporal experiments.")
    parser.add_argument("task", choices=("all", *RUNNERS))
    parser.add_argument("--output", type=Path, default=Path("reports/temporal_runs"))
    parser.add_argument(
        "--signed-leverage", type=Path, help="Existing teammate scores CSV (node,signed_leverage)"
    )
    parser.add_argument(
        "--previous",
        type=Path,
        help="fraction_sweep.csv from an earlier robustness run (needed for standalone final)",
    )
    parser.add_argument("--frequency", default="W-SUN")
    parser.add_argument("--initializations", type=int, default=20)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--sign-shuffles", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = list(RUNNERS) if args.task == "all" else [args.task]
    if "final" in tasks and "robustness" not in tasks and args.previous is None:
        parser.error("The final task requires --previous unless running 'all'")

    output = (ROOT / args.output).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    previous = args.previous.resolve() if args.previous is not None else None

    for task in tasks:
        destination = output / task
        command = [
            sys.executable,
            str(ROOT / "scripts" / RUNNERS[task]),
            "--output",
            str(destination),
            "--seed",
            str(args.seed),
        ]

        if task == "analysis":
            command += [
                "--frequency",
                args.frequency,
                "--repetitions",
                str(args.initializations),
                "--shuffles",
                str(args.shuffles),
            ]
        elif task == "robustness":
            command += [
                "--frequency",
                args.frequency,
                "--initializations",
                str(args.initializations),
                "--shuffles",
                str(args.shuffles),
                "--sign-shuffles",
                str(args.sign_shuffles),
            ]
        elif task == "final":
            if previous is None:
                candidates = sorted((output / "robustness").rglob("fraction_sweep.csv"))
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Expected one robustness fraction_sweep.csv, found {len(candidates)}"
                    )
                previous = candidates[0]
            command += [
                "--frequency",
                args.frequency,
                "--previous",
                str(previous),
                "--initializations",
                str(args.initializations),
                "--shuffles",
                str(args.shuffles),
                "--sign-shuffles",
                str(args.sign_shuffles),
                "--bootstrap",
                str(args.bootstrap),
            ]
        elif task == "diagnostics":
            command += [
                "--frequency",
                args.frequency,
                "--shuffles",
                str(args.shuffles),
                "--sign-shuffles",
                str(args.sign_shuffles),
                "--max-pairs",
                str(args.max_pairs),
            ]
        elif task == "audit":
            command += ["--initializations", str(args.initializations)]
        elif task == "ico":
            command += [
                "--frequency",
                args.frequency,
                "--initializations",
                str(args.initializations),
                "--shuffles",
                str(args.shuffles),
                "--bootstrap",
                str(args.bootstrap),
            ]
            if args.signed_leverage is not None:
                command += ["--signed-leverage", str(args.signed_leverage.resolve())]
        elif task == "compare":
            command += [
                "--frequency",
                args.frequency,
                "--initializations",
                str(args.initializations),
                "--shuffles",
                str(args.shuffles),
                "--bootstrap",
                str(args.bootstrap),
            ]

        print(f"\n=== Running {task} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    print(f"\nAll requested runs saved under {output}")


if __name__ == "__main__":
    main()
