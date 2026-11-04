"""Run reproducible recommendation research without requiring a notebook."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from timerec.experiment import run
from timerec.data import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download official MovieLens data for local research")
    parser.add_argument("--benchmark", action="store_true", help="Train, select on validation and evaluate test once")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "u.data")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    arguments = parser.parse_args()
    if arguments.download:
        print(download(ROOT / "data" / "raw"))
    if arguments.benchmark:
        if not arguments.data.exists():
            parser.error("Data not found. Read DATA_CARD.md, then use --download or pass --data.")
        run(arguments.data, arguments.output)
    if not arguments.download and not arguments.benchmark:
        parser.print_help()


if __name__ == "__main__":
    main()
