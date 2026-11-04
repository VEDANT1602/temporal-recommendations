"""Regenerate narrative and charts from saved metrics, without retraining."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from timerec.experiment import chart, coverage_chart, narrative

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reports", type=Path, default=ROOT / "reports")
arguments = parser.parse_args()
result = json.loads((arguments.reports / "metrics.json").read_text(encoding="utf-8"))
chart(result, arguments.reports / "ranking.svg")
coverage_chart(result, arguments.reports / "coverage.svg")
(arguments.reports / "case-study.md").write_text(narrative(result), encoding="utf-8")
print("Report regenerated from saved metrics; no model fitting or new numerical claims.")
