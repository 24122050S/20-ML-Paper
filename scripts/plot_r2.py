"""Vẽ ablation R2 và paired gamma effects; chỉ nhận summary đủ3 seeds."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures/r2_cifar100_k5_ablation.png")
    args = parser.parse_args()
    summary = json.loads((args.reports_dir / "r2_cifar100_k5_summary.json").read_bytes())
    assert summary["status"] == "complete" and summary["seeds"] == [42, 43, 44] and summary["ddof"] == 1
    assert summary["n_new_cases"] == 9 and summary["n_reused_r1_cases"] == 3
    rows = summary["rows"]; assert len(rows) == 4
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
    values = [row["metrics"]["average_incremental_accuracy_pct"] for row in rows]
    axes[0].bar(range(4), [value["mean"] for value in values], yerr=[value["sample_sd"] for value in values],
                color=["#97a4ae", "#2274a5", "#3487b8", "#65a7c7"], capsize=4, width=0.62,
                error_kw={"linewidth": 1.2}, label="Ours: mean ± sample SD")
    axes[0].scatter(range(4), [row["paper_average"] for row in rows], marker="D", facecolors="white",
                    edgecolors="#20252b", s=30, zorder=5, label="Paper Table 2")
    axes[0].set_xticks(range(4), ["No FE\ng=0.1", "FE8000\ng=0.1", "FE8000\ng=0.01", "FE8000\ng=0.001"])
    axes[0].set(title="Feature expansion and gamma", ylabel="Average incremental accuracy (%)", ylim=(0, 76))
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    deltas = [rows[i]["paired_delta_average_vs_r1_pp"] for i in (2, 3)]
    axes[1].errorbar(range(2), [value["mean"] for value in deltas], yerr=[value["sample_sd"] for value in deltas],
                    color="#2274a5", marker="o", markersize=5, linewidth=1.8, capsize=5, linestyle="none")
    axes[1].axhline(0, color="#5b6268", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(2), ["g=0.01", "g=0.001"])
    axes[1].set(title="Paired gamma effect", ylabel="Change from g=0.1 (pp)", xlim=(-0.5, 1.5))
    for axis in axes:axis.grid(axis="y", color="#e0e5e8", linewidth=0.6);axis.set_axisbelow(True)
    fig.suptitle("R2 · CIFAR-100 / K=5 · same 160-epoch backbone per seed\nn=3 seeds; error bars are sample SD, not confidence intervals", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180);plt.close(fig)
    print("Saved R2 ablation and paired-delta figure:", args.output)


if __name__ == "__main__":
    main()
