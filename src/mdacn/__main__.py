"""Commands: python -m mdacn setup / characterise"""

import argparse
from pathlib import Path

from mdacn.data import download_dataset

DEFAULT_DATASET_PATH = Path("data/raw/soc-RedditHyperlinks-body.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mdacn", description="MDACN Project Reddit Graph Echo Chamber"
    )

    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Download the SNAP body TSV; keep an existing file")
    setup.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)

    args = parser.parse_args()
    try:
        if args.command == "setup":
            exists = args.output.is_file()
            print(f"{'Using existing' if exists else 'Downloading'}")
            download_dataset(args.output)
            print(f"Dataset ready: {args.output}")
        else:
            print(f"Unknown command supplied: {args.command}")
    except Exception as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
