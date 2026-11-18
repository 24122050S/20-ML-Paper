"""Tự vẽ bộ biểu đồ thực nghiệm cuối kỳ E1 và E2 phục vụ Poster A2 dọc và báo cáo phản biện.

Script hỗ trợ chạy độc lập cho từng nhánh (``--mode e2``, ``--mode e1`` hoặc ``--mode all``)
từ các tệp JSON tổng hợp đã kiểm chứng trong ``reports/``, xuất ảnh PNG độ phân giải cao
vào ``reports/figures/``.
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


def _configure_style() -> None:
    """Thiết lập phông chữ, màu nền và khung trục nhất quán cho biểu đồ học thuật.

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


def plot_e2_figure(summary_path: Path, output_path: Path) -> Path:
    """Vẽ biểu đồ 2 panels cho Thực nghiệm E2 (Ngân sách Byte E2a/FD2 & Độ trôi Woodbury E2b).

    Tham số:
        summary_path: Đường dẫn tới tệp ``reports/e2_memory_precision_summary.json``.
        output_path: Đường dẫn xuất ảnh PNG (mặc định ``reports/figures/e2_memory_and_woodbury_precision.png``).

    Trả về:
        Đối tượng ``Path`` của tệp ảnh PNG vừa lưu.
    """
    _configure_style()
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("Summary E2 chưa ở trạng thái complete")

    curves = summary["e2a_and_fd2_summary"]
    dense_rows = sorted(
        [v for v in curves.values() if v["branch"] == "E2a"],
        key=lambda item: item["dimension"],
    )
    fd2_rows = sorted(
        [v for v in curves.values() if v["branch"] == "FD2"],
        key=lambda item: item["state_memory"]["mb_fp64"],
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 5.1), layout="constrained", gridspec_kw={"width_ratios": [1.22, 1.0]}
    )

    # --- Panel A: E2a + FD2 Pareto Trade-off ---
    mb_x = [row["state_memory"]["mb_fp64"] for row in dense_rows]
    abar_m = [row["metrics"]["average_incremental_accuracy_pct"]["mean"] for row in dense_rows]
    abar_sd = [row["metrics"]["average_incremental_accuracy_pct"]["sample_sd"] for row in dense_rows]
    afin_m = [row["metrics"]["final_accuracy_pct"]["mean"] for row in dense_rows]
    afin_sd = [row["metrics"]["final_accuracy_pct"]["sample_sd"] for row in dense_rows]

    ax1.axvspan(0.05, 6.144, color="#e8f5e9", alpha=0.45, zorder=0, label="Fits 2,000-image byte budget (<= 6.14 MB)")
    ax1.axvline(6.144, color="#c1121f", linestyle="-.", linewidth=1.5, zorder=2)

    ax1.errorbar(
        mb_x, abar_m, yerr=abar_sd, color="#1d6996", marker="o", markersize=5.2,
        linewidth=2.0, capsize=3.5, label=r"ACIL dense $\bar{A}$ (mean ± SD, n=3)", zorder=4,
    )
    ax1.errorbar(
        mb_x, afin_m, yerr=afin_sd, color="#2b9348", marker="s", markersize=4.8,
        linewidth=1.8, linestyle="--", capsize=3.5, label=r"ACIL dense $A_{\mathrm{final}}$ (100 classes)", zorder=4,
    )

    if fd2_rows:
        fd2_x = [row["state_memory"]["mb_fp64"] for row in fd2_rows]
        fd2_abar = [row["metrics"]["average_incremental_accuracy_pct"]["mean"] for row in fd2_rows]
        fd2_sd = [row["metrics"]["average_incremental_accuracy_pct"]["sample_sd"] for row in fd2_rows]
        fd2_afin = [row["metrics"]["final_accuracy_pct"]["mean"] for row in fd2_rows]
        ax1.errorbar(
            fd2_x, fd2_abar, yerr=fd2_sd, color="#7b2cbf", marker="D", markersize=6.5,
            linestyle="none", capsize=4, label=r"FD2 Block-Diag $\bar{A}$ (d=8000, B=16/80)", zorder=6,
        )
        ax1.scatter(
            fd2_x, fd2_afin, color="#9d4edd", marker="d", s=46,
            label=r"FD2 Block-Diag $A_{\mathrm{final}}$", zorder=6,
        )
        for row in fd2_rows:
            blk = row["num_blocks"]
            x_val = row["state_memory"]["mb_fp64"]
            y_val = row["metrics"]["average_incremental_accuracy_pct"]["mean"]
            ax1.annotate(
                f"B={blk}\n({x_val:.1f} MB)",
                xy=(x_val, y_val),
                xytext=(0, -24 if blk == 16 else 10),
                textcoords="offset points",
                ha="center",
                fontsize=7.8,
                color="#5a189a",
                fontweight="bold",
            )

    by_dim = {row["dimension"]: row for row in dense_rows}
    if 798 in by_dim and 8000 in by_dim:
        r798 = by_dim[798]
        r8000 = by_dim[8000]
        drop_abar = (
            r798["metrics"]["average_incremental_accuracy_pct"]["mean"]
            - r8000["metrics"]["average_incremental_accuracy_pct"]["mean"]
        )
        drop_fin = (
            r798["metrics"]["final_accuracy_pct"]["mean"]
            - r8000["metrics"]["final_accuracy_pct"]["mean"]
        )
        ax1.annotate(
            f"Iso-Byte fp64 (d=798)\n"
            f"A-bar: {r798['metrics']['average_incremental_accuracy_pct']['mean']:.2f}% ({drop_abar:+.2f} pp)\n"
            f"Final: {r798['metrics']['final_accuracy_pct']['mean']:.2f}% ({drop_fin:+.2f} pp)",
            xy=(r798["state_memory"]["mb_fp64"], r798["metrics"]["average_incremental_accuracy_pct"]["mean"]),
            xytext=(-18, -42),
            textcoords="offset points",
            ha="right",
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.28", "fc": "#fff8f8", "ec": "#c1121f", "lw": 0.9},
            arrowprops={"arrowstyle": "->", "color": "#c1121f", "lw": 1.0},
            zorder=7,
        )
        ax1.annotate(
            f"Paper d=8000\n522.5 MB (85x)\n{r8000['metrics']['average_incremental_accuracy_pct']['mean']:.2f}%",
            xy=(r8000["state_memory"]["mb_fp64"], r8000["metrics"]["average_incremental_accuracy_pct"]["mean"]),
            xytext=(-12, -34),
            textcoords="offset points",
            ha="right",
            fontsize=8.0,
            color="#1d3557",
            fontweight="bold",
        )
    if 64 in by_dim:
        r64 = by_dim[64]
        ax1.annotate(
            f"No-FE (d=64)\n{r64['metrics']['average_incremental_accuracy_pct']['mean']:.2f}%",
            xy=(r64["state_memory"]["mb_fp64"], r64["metrics"]["average_incremental_accuracy_pct"]["mean"]),
            xytext=(8, -16),
            textcoords="offset points",
            fontsize=7.8,
            color="#495057",
        )

    ax1.set_xscale("log")
    ax1.set_xlim(0.055, 880)
    ax1.set_ylim(33.5, 69.8)
    ax1.text(
        6.144 * 1.12,
        67.6,
        "Strict Budget: 6.14 MB\n(2,000 raw CIFAR-100 imgs)",
        fontsize=7.8,
        color="#c1121f",
        fontweight="bold",
        va="top",
    )
    ax1.set_xlabel("Persistent Analytic State Memory (MB, float64; log scale)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(A) E2a & FD2: Accuracy vs. Real Byte Memory Budget", fontweight="bold")
    ax1.grid(True, which="major", color="#dfe4ea", linewidth=0.65)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower right", fontsize=7.6, frameon=True, facecolor="white", framealpha=0.95)

    # --- Panel B: E2b Woodbury Precision Drift (float32 vs float64) ---
    e2b_sum = summary["e2b_summary"]
    gammas = sorted({item["gamma"] for item in e2b_sum.values()}, reverse=True)
    series_specs = (
        ("float64", 5, 128, "#1d6996", "o", "-", "float64 (K=5, b=128; 200 steps)"),
        ("float64", 25, 16, "#4ea8de", "s", "--", "float64 (K=25, b=16; 1,575 steps)"),
        ("float32", 5, 128, "#e07a5f", "D", "-", "float32 (K=5, b=128; 200 steps)"),
        ("float32", 25, 16, "#c1121f", "^", "-", "float32 (K=25, b=16; 1,575 steps)"),
    )
    for dtype_name, k_val, b_val, color, marker, lstyle, label in series_specs:
        xs, ys, dis_list = [], [], []
        for g_val in gammas:
            key = f"{dtype_name}__K{k_val}b{b_val}__g{g_val:g}"
            if key in e2b_sum and e2b_sum[key]["max_relative_w_error_mean"] is not None:
                xs.append(g_val)
                ys.append(e2b_sum[key]["max_relative_w_error_mean"])
                dis_list.append(e2b_sum[key]["final_prediction_disagreements_mean"])
        if xs:
            ax2.plot(xs, ys, color=color, marker=marker, linestyle=lstyle, linewidth=2.0, markersize=5.8, label=label)
            if dtype_name == "float32" and k_val == 25:
                for gx, gy, dis in zip(xs, ys, dis_list):
                    if dis and dis > 0:
                        ax2.annotate(
                            f"{dis:.1f} flips",
                            xy=(gx, gy),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7.8,
                            color="#c1121f",
                            fontweight="bold",
                        )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1.5e-11, 3.5e-2)
    ax2.invert_xaxis()
    ax2.set_xlabel(r"Ridge Regularization $\gamma$ (log scale, conditioning worsens $\rightarrow$)")
    ax2.set_ylabel(r"Max Relative Weight Error vs. fp64 Joint Oracle")
    ax2.set_title("(B) E2b: Woodbury Numerical Drift (float32 vs. float64)", fontweight="bold")
    ax2.grid(True, which="major", color="#dfe4ea", linewidth=0.65)
    ax2.set_axisbelow(True)
    ax2.legend(loc="center left", fontsize=8.0, frameon=True, facecolor="white", framealpha=0.95)

    fig.suptitle(
        "Empirical Demonstration E2 (Issue P2) · Real Byte Memory Scaling & Woodbury Floating-Point Stability\n"
        "CIFAR-100, frozen R1 ResNet-32 checkpoints (seeds 42, 43, 44; mean ± sample SD, ddof=1)",
        fontsize=11.5,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_e1_figure(
    e1_summary_path: Path,
    r1_summary_path: Path,
    e2_summary_path: Path | None,
    output_path: Path,
) -> Path:
    """Vẽ biểu đồ 2 panels cho Thực nghiệm E1 (Tỷ lệ lớp Base E1a & Dịch chuyển Superclass E1b).

    Hợp nhất kết quả E1a/E1b mới trên GPU với chuẩn đối chứng ``c_0 = 50`` đồng phân phối từ R1.

    Tham số:
        e1_summary_path: Đường dẫn tới tệp ``reports/e1_base_assumption_summary.json``.
        r1_summary_path: Đường dẫn tới tệp ``reports/r1_cifar100_k5_summary.json`` (đối chứng c0=50).
        e2_summary_path: Đường dẫn tới ``reports/e2_memory_precision_summary.json`` để đọc ``A_inc``
            của R1 d=8000 (hoặc suy ra đại số chính xác từ ``2 * A_final - B_5``).
        output_path: Đường dẫn xuất ảnh PNG (mặc định ``reports/figures/e1_base_representation_bottleneck.png``).

    Trả về:
        Đối tượng ``Path`` của tệp ảnh PNG vừa lưu.
    """
    _configure_style()
    e1_sum = json.loads(Path(e1_summary_path).read_text(encoding="utf-8"))
    r1_sum = json.loads(Path(r1_summary_path).read_text(encoding="utf-8"))
    groups = e1_sum["groups"]

    # Trích đối chứng c_0 = 50 (uniform) từ R1 chuẩn đã nghiệm thu
    r1_fin_by_seed = r1_sum["metrics"]["final_accuracy_pct"]["values_by_seed"]
    r1_abar_by_seed = r1_sum["metrics"]["average_incremental_accuracy_pct"]["values_by_seed"]
    r1_b5_by_seed = r1_sum["phase_metrics"][-1]["base_accuracy_pct"]["values_by_seed"]
    r1_n5_by_seed = r1_sum["phase_metrics"][-1]["new_class_accuracy_pct"]["values_by_seed"]
    r1_inc_vals = np.asarray(
        [2.0 * r1_fin_by_seed[s] - r1_b5_by_seed[s] for s in ("42", "43", "44")], dtype=np.float64
    )
    r1_b5_vals = np.asarray([r1_b5_by_seed[s] for s in ("42", "43", "44")], dtype=np.float64)
    r1_fin_vals = np.asarray([r1_fin_by_seed[s] for s in ("42", "43", "44")], dtype=np.float64)
    r1_abar_vals = np.asarray([r1_abar_by_seed[s] for s in ("42", "43", "44")], dtype=np.float64)
    r1_n5_vals = np.asarray([r1_n5_by_seed[s] for s in ("42", "43", "44")], dtype=np.float64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.1), layout="constrained")

    # --- Panel A: E1a Base-Class Ratio Curve ---
    c0_grid = []
    fin_m, fin_sd = [], []
    base_m, base_sd = [], []
    inc_m, inc_sd = [], []
    for c0 in (10, 20, 30):
        g_key = f"E1a__uniform__c0_{c0}"
        if g_key in groups:
            m = groups[g_key]["metrics"]
            c0_grid.append(c0)
            fin_m.append(m["final_accuracy_pct"]["mean"])
            fin_sd.append(m["final_accuracy_pct"]["sample_sd"])
            base_m.append(m["final_base_accuracy_pct"]["mean"])
            base_sd.append(m["final_base_accuracy_pct"]["sample_sd"])
            inc_m.append(m["final_inc_accuracy_pct"]["mean"])
            inc_sd.append(m["final_inc_accuracy_pct"]["sample_sd"])
    c0_grid.append(50)
    fin_m.append(float(np.mean(r1_fin_vals)))
    fin_sd.append(float(np.std(r1_fin_vals, ddof=1)))
    base_m.append(float(np.mean(r1_b5_vals)))
    base_sd.append(float(np.std(r1_b5_vals, ddof=1)))
    inc_m.append(float(np.mean(r1_inc_vals)))
    inc_sd.append(float(np.std(r1_inc_vals, ddof=1)))

    ax1.errorbar(
        c0_grid, base_m, yerr=base_sd, color="#457b9d", marker="^", linestyle=":",
        linewidth=1.8, markersize=6, capsize=4, label=r"Base classes $B_{\mathrm{final}}$ ($< c_0$)",
    )
    ax1.errorbar(
        c0_grid, fin_m, yerr=fin_sd, color="#1d3557", marker="o", linestyle="-",
        linewidth=2.2, markersize=6, capsize=4, label=r"All 100 classes $A_{\mathrm{final}}$ (primary)",
    )
    ax1.errorbar(
        c0_grid, inc_m, yerr=inc_sd, color="#c1121f", marker="s", linestyle="-",
        linewidth=2.1, markersize=6, capsize=4, label=r"Incremental classes $A_{\mathrm{inc}}$ ($\geq c_0$)",
    )
    for cx, fy, iy in zip(c0_grid, fin_m, inc_m):
        ax1.annotate(f"{fy:.1f}%", (cx, fy), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.2, fontweight="bold", color="#1d3557")
        ax1.annotate(f"{iy:.1f}%", (cx, iy), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8.0, color="#c1121f")

    ax1.set_xticks(c0_grid, [f"c0 = {c}\n(+{(100-c)//10}x10)" for c in c0_grid])
    ax1.set_xlim(5, 55)
    ax1.set_ylim(max(0.0, min(inc_m) - 8.0), min(100.0, max(base_m) + 9.0))
    ax1.set_xlabel("Number of Frozen Phase-0 Base Classes (all end at 100 classes)")
    ax1.set_ylabel("Final-Phase Test Accuracy (%)")
    ax1.set_title("(A) E1a: Representation Collapse Under Smaller Base", fontweight="bold")
    ax1.grid(True, axis="y", color="#dfe4ea", linewidth=0.65)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower right", fontsize=8.2, frameon=True, facecolor="white")

    # --- Panel B: E1b Semantic Superclass Shift ---
    disjoint_key = "E1b__superclass_disjoint__c0_50"
    if disjoint_key in groups:
        dm = groups[disjoint_key]["metrics"]
        labels = [
            r"$\bar{A}$" + "\n(6 phases)",
            r"$A_{\mathrm{final}}$" + "\n(100 cls)",
            r"$B_5$" + "\n(50 Base)",
            r"$A_{\mathrm{inc}}$" + "\n(50 Inc)",
            r"$N_5$" + "\n(Last 10)",
        ]
        u_means = [
            float(np.mean(r1_abar_vals)),
            float(np.mean(r1_fin_vals)),
            float(np.mean(r1_b5_vals)),
            float(np.mean(r1_inc_vals)),
            float(np.mean(r1_n5_vals)),
        ]
        u_sds = [
            float(np.std(r1_abar_vals, ddof=1)),
            float(np.std(r1_fin_vals, ddof=1)),
            float(np.std(r1_b5_vals, ddof=1)),
            float(np.std(r1_inc_vals, ddof=1)),
            float(np.std(r1_n5_vals, ddof=1)),
        ]
        d_means = [
            dm["average_incremental_accuracy_pct"]["mean"],
            dm["final_accuracy_pct"]["mean"],
            dm["final_base_accuracy_pct"]["mean"],
            dm["final_inc_accuracy_pct"]["mean"],
            dm["final_new_phase_accuracy_pct"]["mean"],
        ]
        d_sds = [
            dm["average_incremental_accuracy_pct"]["sample_sd"],
            dm["final_accuracy_pct"]["sample_sd"],
            dm["final_base_accuracy_pct"]["sample_sd"],
            dm["final_inc_accuracy_pct"]["sample_sd"],
            dm["final_new_phase_accuracy_pct"]["sample_sd"],
        ]
        x_pos = np.arange(len(labels))
        width = 0.36
        bars_u = ax2.bar(
            x_pos - width / 2, u_means, width, yerr=u_sds, capsize=3.5,
            color="#2274a5", label="Uniform Base 50 (R1 control, 25k imgs)",
        )
        bars_d = ax2.bar(
            x_pos + width / 2, d_means, width, yerr=d_sds, capsize=3.5,
            color="#e07a5f", label="Superclass-Disjoint Base 50 (Natural -> Artifacts, 25k imgs)",
        )
        for rect, sd_val in zip(bars_u, u_sds):
            h = rect.get_height()
            ax2.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h + sd_val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.8, color="#1d3557")
        for rect, sd_val in zip(bars_d, d_sds):
            h = rect.get_height()
            ax2.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h + sd_val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.8, color="#9b2226", fontweight="bold")

        manmade = dm.get("final_inc_manmade_accuracy_pct")
        bio = dm.get("final_inc_biological_accuracy_pct")
        if manmade and bio:
            ax2.text(
                0.99, 0.89,
                f"Inside Disjoint Incremental 50:\n"
                f"• Man-Made/Vehicles (30 cls): {manmade['mean']:.2f} ± {manmade['sample_sd']:.2f}%\n"
                f"• Unseen Bio/People (20 cls): {bio['mean']:.2f} ± {bio['sample_sd']:.2f}%",
                transform=ax2.transAxes,
                ha="right",
                va="top",
                fontsize=7.6,
                bbox={"boxstyle": "round,pad=0.28", "fc": "#fff5f2", "ec": "#e07a5f", "lw": 0.9},
            )

        ax2.set_xticks(x_pos, labels)
        ax2.set_ylim(0, 91.0)
        ax2.set_ylabel("Test Accuracy (%)")
        ax2.set_title("(B) E1b: Semantic Superclass Shift (Same 25k Base Budget)", fontweight="bold")
        ax2.grid(True, axis="y", color="#dfe4ea", linewidth=0.65)
        ax2.set_axisbelow(True)
        ax2.legend(loc="upper left", fontsize=7.9, frameon=True, facecolor="white", framealpha=0.95)

    fig.suptitle(
        "Empirical Demonstration E1 (Issue P1) · Frozen Phase-0 Representation Bottleneck\n"
        "CIFAR-100, ResNet-32 (160 epochs), FE d=8000, gamma=0.1 (n=3 seeds: 42, 43, 44; mean ± sample SD)",
        fontsize=11.5,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    """Điểm vào CLI tạo biểu đồ E2, E1 hoặc cả hai vào ``reports/figures/``.

    Tham số:
        Không nhận tham số hàm; đọc ``--mode`` và đường dẫn các tệp summary từ CLI.

    Trả về:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["e2", "e1", "all"], default="all")
    parser.add_argument("--e2-summary", type=Path, default=ROOT / "reports/e2_memory_precision_summary.json")
    parser.add_argument("--e1-summary", type=Path, default=ROOT / "reports/e1_base_assumption_summary.json")
    parser.add_argument("--r1-summary", type=Path, default=ROOT / "reports/r1_cifar100_k5_summary.json")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()

    if args.mode in ("e2", "all"):
        out_e2 = plot_e2_figure(
            args.e2_summary, args.figures_dir / "e2_memory_and_woodbury_precision.png"
        )
        print("Saved E2 figure:", out_e2)
    if args.mode in ("e1", "all"):
        out_e1 = plot_e1_figure(
            args.e1_summary,
            args.r1_summary,
            args.e2_summary if args.e2_summary.is_file() else None,
            args.figures_dir / "e1_base_representation_bottleneck.png",
        )
        print("Saved E1 figure:", out_e1)


if __name__ == "__main__":
    main()
