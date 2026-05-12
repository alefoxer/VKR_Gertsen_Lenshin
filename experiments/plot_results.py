from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_summary(summary_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["goal_rate", "max_final_score", "mean_gain"]:
        fig = plt.figure(figsize=(10, 5))
        pivot = df.pivot_table(index="attack_method", columns="input_mode", values=metric, aggfunc="mean")
        pivot.plot(kind="bar", ax=plt.gca())
        plt.title(metric)
        plt.tight_layout()
        fig.savefig(output_dir / f"{metric}.png")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()
    plot_summary(args.summary_csv, args.output_dir)
    print(f"Saved plots to: {args.output_dir}")
