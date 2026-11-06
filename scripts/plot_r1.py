"""Vẽ kết quả R1 đã đủ3 seeds; cài requirements-plots.txt trong môi trường riêng."""

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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    summary = json.loads((args.reports_dir / "r1_cifar100_k5_summary.json").read_bytes())
    assert summary["status"] == "complete" and summary["seeds"] == [42, 43, 44]
    assert summary["n_completed_seeds"] == 3 and summary["ddof"] == 1
    reports = [json.loads((args.reports_dir / f"benchmark_seed{seed}_kaggle_results.json").read_bytes()) for seed in summary["seeds"]]
    colors = ("#2274a5", "#e07a27", "#3a8f5c")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    classes = [row["n_classes"] for row in summary["phase_metrics"]]
    for seed, report, color in zip(summary["seeds"], reports, colors):
        ax.plot(classes, [row["accuracy_pct"] for row in report["results"]], color=color,
                linewidth=1.3, marker="o", markersize=3, alpha=0.65, label=f"Seed {seed}")
    ax.errorbar(classes, [row["accuracy_pct"]["mean"] for row in summary["phase_metrics"]],
                yerr=[row["accuracy_pct"]["sample_sd"] for row in summary["phase_metrics"]],
                color="#20252b", linewidth=2.2, marker="o", markersize=4, capsize=4,
                label="Mean ± sample SD (n=3)", zorder=5)
    metric = summary["metrics"]["average_incremental_accuracy_pct"]
    ax.set_title("ACIL reproduction · CIFAR-100 / K=5\n160 base epochs · d=8000 · gamma=0.1", loc="left", pad=12)
    ax.text(0.02, 0.04, f"Average incremental accuracy: {metric['mean']:.2f} ± {metric['sample_sd']:.2f}%",
            transform=ax.transAxes, fontsize=10, bbox={"facecolor": "white", "edgecolor": "#d5dadd", "pad": 5})
    ax.set(xlabel="Number of seen classes", ylabel="All-seen test accuracy (%)", xticks=classes)
    ax.grid(axis="y", color="#e1e5e8", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.savefig(args.output_dir / "r1_cifar100_k5_accuracy.png", dpi=args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", gridspec_kw={"width_ratios": [2.2, 1]})
    for seed, report, color in zip(summary["seeds"], reports, colors):
        history = report["training_history"]
        assert len(history) == 160
        axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history],
                     color=color, linewidth=1.25, label=f"Seed {seed}")
    history = reports[0]["training_history"]
    assert all([row["learning_rate"] for row in report["training_history"]] == [row["learning_rate"] for row in history] for report in reports)
    axes[1].step([row["epoch"] for row in history], [row["learning_rate"] for row in history],
                 where="post", color="#20252b", linewidth=1.7)
    axes[1].set_yscale("log")
    axes[0].set(title="Base training · 25,000 images/epoch", xlabel="Epoch", ylabel="Training cross entropy", xlim=(1, 160))
    axes[1].set(title="Shared learning-rate schedule", xlabel="Epoch", ylabel="Learning rate", xlim=(1, 160))
    for ax in axes:
        ax.grid(axis="y", color="#e1e5e8", linewidth=0.7)
        for epoch in (80, 120):ax.axvline(epoch, color="#9ba6af", linestyle="--", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(args.output_dir / "r1_cifar100_k5_training.png", dpi=args.dpi)
    plt.close(fig)
    print("Saved accuracy mean±sample_SD and training/LR PNG figures:", args.output_dir)


if __name__ == "__main__":
    main()
