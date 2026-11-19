"""Tự vẽ biểu đồ thực nghiệm phụ cuối kỳ E3 (MSE Class-Imbalance & FD3-WRLS vs E2b Woodbury Drift).

Script đọc trực tiếp tệp tổng hợp ``reports/e3_mse_imbalance_summary.json`` và
``reports/e2_memory_precision_summary.json`` đã khóa SHA-256, xuất biểu đồ 2 panels
vào ``reports/figures/e3_mse_imbalance_vs_woodbury_drift.png`` phục vụ trình bày
hạn chế thứ 3 (P3) trong ``README.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def configure_plot_style() -> None:
    """Thiết lập cấu hình hiển thị phông chữ và khung trục chuẩn học thuật cho Matplotlib.

    Tham số:
        Không nhận tham số.

    Trả về:
        None.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_e3_vs_e2b_figure(
    e3_summary_path: Path,
    e2_summary_path: Path,
    output_path: Path,
) -> Path:
    """Vẽ biểu đồ 2 panels tách bạch MSE Class-Imbalance Collapse (E3/FD3) với float32 Woodbury Drift (E2b).

    Tham số:
        e3_summary_path: Đường dẫn tới ``reports/e3_mse_imbalance_summary.json``.
        e2_summary_path: Đường dẫn tới ``reports/e2_memory_precision_summary.json``.
        output_path: Đường dẫn lưu ảnh PNG đích.

    Trả về:
        Đối tượng ``Path`` của tệp ảnh PNG đã tạo.
    """
    configure_plot_style()
    e3 = json.loads(Path(e3_summary_path).read_text(encoding="utf-8"))
    e2 = json.loads(Path(e2_summary_path).read_text(encoding="utf-8"))
    if e3.get("status") != "complete" or e2.get("status") != "complete":
        raise ValueError("Summary E3 hoặc E2 chưa ở trạng thái complete")

    curves = e3["curves_summary"]
    std_rows = sorted(
        [v for v in curves.values() if v["variant"] == "ACIL_standard"],
        key=lambda r: r["imbalance_ratio"],
    )
    fd3_rows = sorted(
        [v for v in curves.values() if v["variant"] == "FD3_weighted_rls"],
        key=lambda r: r["imbalance_ratio"],
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.8, 5.0), layout="constrained", gridspec_kw={"width_ratios": [1.18, 1.0]}
    )

    # Panel A: E3 Class-Imbalance Collapse vs FD3-WRLS Recovery (3 seeds, d=1024)
    ratios = [r["imbalance_ratio"] for r in std_rows]
    std_inc_m = [r["metrics"]["final_inc_accuracy_pct"]["mean"] for r in std_rows]
    std_inc_sd = [r["metrics"]["final_inc_accuracy_pct"]["sample_sd"] for r in std_rows]
    fd3_inc_m = [r["metrics"]["final_inc_accuracy_pct"]["mean"] for r in fd3_rows]
    fd3_inc_sd = [r["metrics"]["final_inc_accuracy_pct"]["sample_sd"] for r in fd3_rows]

    std_all_m = [r["metrics"]["final_accuracy_pct"]["mean"] for r in std_rows]
    fd3_all_m = [r["metrics"]["final_accuracy_pct"]["mean"] for r in fd3_rows]
    std_w_ratio = [r["metrics"]["w_norm_inc_over_base_ratio"]["mean"] * 100.0 for r in std_rows]

    ax1.errorbar(
        ratios,
        std_inc_m,
        yerr=std_inc_sd,
        color="#c1121f",
        marker="o",
        linewidth=2.2,
        capsize=3.5,
        label=r"Standard ACIL $A_{\mathrm{inc}}$ (50 new classes)",
        zorder=5,
    )
    ax1.errorbar(
        ratios,
        fd3_inc_m,
        yerr=fd3_inc_sd,
        color="#2b9348",
        marker="s",
        linewidth=2.2,
        capsize=3.5,
        label=r"FD3 Weighted RLS $A_{\mathrm{inc}}$ ($\omega_c = N_{\mathrm{base}}/N_c$)",
        zorder=5,
    )
    ax1.plot(
        ratios,
        std_all_m,
        color="#c1121f",
        linestyle="--",
        marker="^",
        markersize=4.5,
        alpha=0.75,
        label=r"Standard ACIL $A_{\mathrm{final}}$ (all 100 classes)",
    )
    ax1.plot(
        ratios,
        fd3_all_m,
        color="#2b9348",
        linestyle="--",
        marker="v",
        markersize=4.5,
        alpha=0.80,
        label=r"FD3 Weighted RLS $A_{\mathrm{final}}$ (all 100 classes)",
    )
    ax1.plot(
        ratios,
        std_w_ratio,
        color="#6c757d",
        linestyle=":",
        marker="d",
        markersize=4.5,
        label=r"Weight-norm ratio $100 \times \|\mathbf{w}_{\mathrm{inc}}\|_2 / \|\mathbf{w}_{\mathrm{base}}\|_2$",
    )

    ax1.set_xscale("log")
    ax1.set_xticks(ratios)
    ax1.set_xticklabels([f"{int(r)}x\n(N={int(500//r)})" for r in ratios], fontsize=8.5)
    ax1.set_ylim(-2.5, 95.0)
    ax1.set_xlabel("Base-to-Incremental Class Imbalance Ratio $\\rho = N_{\\mathrm{base}} / N_{\\mathrm{inc}}$")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("(A) E3: MSE One-Hot Collapse under Class Imbalance & FD3 Recovery\n(CIFAR-100, d=1024, float64, mean ± SD across seeds 42/43/44)", fontsize=10.2, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="upper right", fontsize=8.1, framealpha=0.92)

    if len(std_rows) >= 5:
        r10_std = [r for r in std_rows if int(r["imbalance_ratio"]) == 10][0]
        r10_fd3 = [r for r in fd3_rows if int(r["imbalance_ratio"]) == 10][0]
        y_lo = r10_std["metrics"]["final_inc_accuracy_pct"]["mean"]
        y_hi = r10_fd3["metrics"]["final_inc_accuracy_pct"]["mean"]
        ax1.annotate(
            "",
            xy=(10.0, y_hi - 0.8),
            xytext=(10.0, y_lo + 0.8),
            arrowprops={"arrowstyle": "<->", "color": "#1b4332", "lw": 1.6},
            zorder=6,
        )
        ax1.text(
            13.2,
            13.8,
            f"+{y_hi - y_lo:.1f} pp\nFD3 recovery",
            fontsize=8.2,
            fontweight="bold",
            color="#1b4332",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#2b9348", "alpha": 0.92},
            zorder=7,
        )

    # Panel B: Mechanism Decoupling at Controlled d=1024 (E2b float32 drift vs E3 MSE imbalance)
    e2b_200 = e2["e2b_summary"]["float32__K5b128__g0.1"]
    e2b_1575 = e2["e2b_summary"]["float32__K25b16__g0.001"]
    std_500 = curves["ACIL_standard__N500"]
    std_100 = curves["ACIL_standard__N100"]
    std_50 = curves["ACIL_standard__N50"]
    fd3_50 = curves["FD3_weighted_rls__N50"]

    categories = [
        "E2b: fp32 Woodbury\n(200 steps, N=500)",
        "E2b: fp32 Woodbury\n(1575 steps, N=500)",
        "E3: fp64 MSE 5x\n(N_inc=100)",
        "E3: fp64 MSE 10x\n(N_inc=50)",
        "FD3: fp64 WRLS 10x\n(N_inc=50)",
    ]
    acc_drops = [
        abs(e2b_200["final_accuracy_drop_pp_mean"]),
        abs(e2b_1575["final_accuracy_drop_pp_mean"]),
        std_500["metrics"]["final_inc_accuracy_pct"]["mean"] - std_100["metrics"]["final_inc_accuracy_pct"]["mean"],
        std_500["metrics"]["final_inc_accuracy_pct"]["mean"] - std_50["metrics"]["final_inc_accuracy_pct"]["mean"],
        std_500["metrics"]["final_inc_accuracy_pct"]["mean"] - fd3_50["metrics"]["final_inc_accuracy_pct"]["mean"],
    ]
    rel_w_vals = [
        e2b_200["max_relative_w_error_mean"],
        e2b_1575["max_relative_w_error_mean"],
        std_100["joint_oracle_comparison"]["mean_relative_w_error"],
        std_50["joint_oracle_comparison"]["mean_relative_w_error"],
        fd3_50["joint_oracle_comparison"]["mean_relative_w_error"],
    ]

    x_pos = np.arange(len(categories))
    bar_colors = ["#457b9d", "#1d3557", "#e63946", "#9b2226", "#2b9348"]
    bars = ax2.bar(x_pos, acc_drops, width=0.56, color=bar_colors, alpha=0.88, zorder=3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontsize=8.0, rotation=14, ha="right")
    ax2.set_ylabel("Accuracy Degradation (pp)\n[E2b: $|\\Delta A_{\\mathrm{final}}|$, E3/FD3: $\\Delta A_{\\mathrm{inc}}$ vs N=500]")
    ax2.set_ylim(0.0, max(acc_drops) * 1.28)
    ax2.set_title("(B) Decoupling Numerical Drift (E2b) vs Objective Bias (E3)\n(Same R1 Backbones & W_fe at d=1024, 3 Seeds)", fontsize=10.2, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.55)

    for bar, drop_val, rw_val in zip(bars, acc_drops, rel_w_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.65,
            f"-{drop_val:.2f} pp\n(rel_W={rw_val:.1e})",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Phân tích đối số dòng lệnh cho script vẽ hình E3 đối chiếu E2b.

    Tham số:
        argv: Danh sách đối số tùy chọn.

    Trả về:
        Đối tượng ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ E3 (MSE Class-Imbalance & FD3) đối chiếu E2b.")
    parser.add_argument("--e3-summary", type=Path, default=ROOT / "reports/e3_mse_imbalance_summary.json")
    parser.add_argument("--e2-summary", type=Path, default=ROOT / "reports/e2_memory_precision_summary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/figures/e3_mse_imbalance_vs_woodbury_drift.png",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Hàm chính tạo tệp ảnh biểu đồ thực nghiệm phụ E3.

    Tham số:
        argv: Danh sách đối số dòng lệnh tùy chọn.

    Trả về:
        Mã thoát ``0`` khi xuất hình thành công.
    """
    args = parse_args(argv)
    out = plot_e3_vs_e2b_figure(args.e3_summary, args.e2_summary, args.output)
    print(f"Saved E3 figure: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
