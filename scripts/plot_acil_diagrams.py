"""Vẽ sơ đồ pipeline end-to-end của ACIL và hình minh họa bài toán/tiền nhiệm.

Hình do nhóm tự dựng bằng Matplotlib phục vụ slide đọc hiểu giữa kỳ và README;
không sao chép hình từ bài báo gốc. Chạy với môi trường có matplotlib (vd:
requirements-plots.txt).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]


def _draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    facecolor: str,
    edgecolor: str,
    title_color: str = "#16202a",
    body_color: str = "#2f3b47",
    title_size: float = 9.8,
    body_size: float = 8.1,
    linewidth: float = 1.5,
    linestyle: str = "-",
) -> None:
    """Vẽ một khối chữ nhật bo góc có tiêu đề và nội dung căn chỉnh gọn.

    Tham số:
        ax: Axes đích để vẽ patch và text.
        x: Tọa độ x góc dưới-trái theo hệ tọa độ trục [0, 100].
        y: Tọa độ y góc dưới-trái theo hệ tọa độ trục [0, 100].
        w: Chiều rộng khối.
        h: Chiều cao khối.
        title: Dòng tiêu đề in đậm ở phần trên khối.
        body: Nội dung nhiều dòng mô tả bước hoặc công thức.
        facecolor: Màu nền khối.
        edgecolor: Màu viền khối.
        title_color: Màu chữ tiêu đề.
        body_color: Màu chữ nội dung.
        title_size: Cỡ chữ tiêu đề (pt).
        body_size: Cỡ chữ nội dung (pt).
        linewidth: Độ dày viền khối.
        linestyle: Kiểu nét viền ('-' hoặc '--').

    Trả về:
        None. Hàm vẽ trực tiếp lên ``ax``.
    """
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    title_y = y + h - 2.3
    ax.text(
        x + w / 2,
        title_y,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    body_y = y + (h - 4.4) / 2
    ax.text(
        x + w / 2,
        body_y,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=body_color,
        linespacing=1.35,
        zorder=3,
    )


def _draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#42505e",
    label: str = "",
    label_offset: tuple[float, float] = (0.0, 1.5),
    connectionstyle: str = "arc3,rad=0.0",
    linestyle: str = "-",
    linewidth: float = 1.6,
    label_size: float = 7.7,
) -> None:
    """Vẽ mũi tên nối hai điểm kèm nhãn tùy chọn.

    Tham số:
        ax: Axes đích.
        start: Tọa độ (x, y) điểm bắt đầu.
        end: Tọa độ (x, y) điểm kết thúc.
        color: Màu mũi tên và nhãn.
        label: Nhãn mô tả luồng dữ liệu hoặc trạng thái.
        label_offset: Độ lệch (dx, dy) của nhãn so với trung điểm.
        connectionstyle: Kiểu đường cong mũi tên của Matplotlib.
        linestyle: Kiểu nét ('-' hoặc '--').
        linewidth: Độ dày mũi tên.
        label_size: Cỡ chữ nhãn.

    Trả về:
        None. Hàm thêm arrow patch trực tiếp vào ``ax``.
    """
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>,head_length=5.5,head_width=3.2",
        connectionstyle=connectionstyle,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=4,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=label_size,
            color=color,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
            zorder=5,
        )


def draw_pipeline_diagram(output_path: Path, dpi: int = 200) -> Path:
    """Vẽ sơ đồ pipeline end-to-end của ACIL từ Phase 0 đến cập nhật tăng dần và suy luận.

    Tham số:
        output_path: Đường dẫn file PNG đầu ra.
        dpi: Độ phân giải ảnh khi lưu.

    Trả về:
        Đường dẫn file PNG đã lưu.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(15.5, 9.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Tiêu đề tổng
    ax.text(
        50,
        98.2,
        "ACIL End-to-End Pipeline: Phase 0 Base Learning, Incremental RLS & All-Seen Inference",
        ha="center",
        va="top",
        fontsize=14.2,
        fontweight="bold",
        color="#14212e",
    )
    ax.text(
        50,
        94.8,
        "Backpropagation runs ONLY in Phase 0; later phases freeze the CNN + W_fe and update (R_k, W_k) recursively without old samples",
        ha="center",
        va="top",
        fontsize=9.6,
        color="#4a5968",
    )

    # Khung nền 3 giai đoạn
    bands = [
        (2.0, 64.0, 96.0, 27.5, "#f4f8fb", "#b7cde0",
         "STAGE 1 · Phase 0: Base BP & Init (0..49)"),
        (2.0, 28.0, 96.0, 30.5, "#fdf7f2", "#e5c3a6",
         "STAGE 2 · Phases k=1..K (No Replay)"),
        (2.0, 4.2, 96.0, 19.5, "#f3faf6", "#b2d9c2",
         "STAGE 3 · All-Seen Inference (No Task ID)"),
    ]
    for bx, by, bw, bh, fc, ec, banner in bands:
        band = FancyBboxPatch(
            (bx, by),
            bw,
            bh,
            boxstyle="round,pad=0.3,rounding_size=1.4",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.3,
            zorder=1,
        )
        ax.add_patch(band)
        ax.text(
            bx + 1.4,
            by + bh - 1.2,
            banner,
            ha="left",
            va="top",
            fontsize=9.6,
            fontweight="bold",
            color="#1c2a38",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": ec,
                "linewidth": 0.9,
            },
            zorder=2,
        )

    # Stage 1 boxes
    y1, h1 = 66.2, 20.0
    _draw_box(
        ax, 3.3, y1, 17.0, h1,
        "1. Base Data (X_0, Y_0)",
        "25,000 imgs (32x32x3)\nGlobal IDs: 0..49\nAugment (BP only):\npad4, crop32, flip\nBase-only mean/std",
        facecolor="#ffffff", edgecolor="#2274a5", title_size=9.2,
    )
    _draw_box(
        ax, 22.8, y1, 17.6, h1,
        "2. Base BP Training",
        "ResNet-32 (3 stages)\n+ FC50 (Cross-Entropy)\n160 epochs, SGD\nLR: 0.1 -> 0.01 -> 0.001\n(milestones 80, 120)",
        facecolor="#e9f2f9", edgecolor="#2274a5", title_size=9.2,
    )
    _draw_box(
        ax, 42.9, y1, 16.8, h1,
        "3. Freeze & Clean Pass",
        "Reload checkpoint\nDiscard FC50 head\neval() + freeze params\n(BN stats fixed)\nClean pass -> F_0 (n, 64)",
        facecolor="#ffffff", edgecolor="#2274a5", title_size=9.2,
    )
    _draw_box(
        ax, 62.2, y1, 16.8, h1,
        "4. Feature Expansion",
        "FP32 -> FP64 features\nFixed W_fe in R^(64 x d)\n~ N(0, 1), seed+1000\nH_0 = ReLU(F_0 W_fe)\nShape: (n, d), d=8000",
        facecolor="#e9f2f9", edgecolor="#2274a5", title_size=9.2,
    )
    _draw_box(
        ax, 81.5, y1, 15.2, h1,
        "5. Analytic Init",
        "Streaming Gram/Cross\nCholesky (+ gamma*I)\nR_0 = (H_0^T H_0 + gI)^-1\nW_0 = R_0 H_0^T Y_0\nR_0:(d,d), W_0:(d,50)",
        facecolor="#d9ebf7", edgecolor="#165780", linewidth=1.8, title_size=9.2,
    )

    _draw_arrow(ax, (20.6, y1 + h1 / 2), (22.5, y1 + h1 / 2), color="#2274a5")
    _draw_arrow(ax, (40.7, y1 + h1 / 2), (42.6, y1 + h1 / 2), color="#2274a5")
    _draw_arrow(ax, (60.0, y1 + h1 / 2), (61.9, y1 + h1 / 2), color="#2274a5", label="F_0", label_offset=(0, 2.2))
    _draw_arrow(ax, (79.3, y1 + h1 / 2), (81.2, y1 + h1 / 2), color="#2274a5", label="H_0", label_offset=(0, 2.2))

    # Stage 2 boxes
    y2, h2 = 33.4, 19.6
    _draw_box(
        ax, 3.3, y2, 16.8, h2,
        "6. Phase k Data (X_k, Y_k)",
        "5,000 imgs / phase\nDisjoint IDs: 50..99\nMinibatch size b=128\nNo crop/flip augment\nZero old samples kept",
        facecolor="#ffffff", edgecolor="#d46a18", title_size=9.0,
    )
    _draw_box(
        ax, 22.6, y2, 18.2, h2,
        "7. Frozen Extractor + FE",
        "Same frozen ResNet-32\n+ fixed W_fe + ReLU\nNo gradient / No BP\nF_k (b, 64) -> FP64\nH_k = ReLU(F_k W_fe)",
        facecolor="#fdf0e5", edgecolor="#d46a18", title_size=9.0,
    )
    _draw_box(
        ax, 43.3, y2, 19.2, h2,
        "8. Global Column Padding",
        "Expand W columns:\nW_pad = [W_{k-1}, 0]\nShape: (d, C_{k-1}) -> (d, C_k)\nOne-hot Y_k in R^(b x C_k)\nOld classes keep columns",
        facecolor="#ffffff", edgecolor="#d46a18", title_size=9.0,
    )
    _draw_box(
        ax, 65.0, y2, 31.7, h2,
        "9. Recursive Analytic Update (Eqs. 9-10)",
        "Woodbury inverse Gram update (no old H):\nS_b = I + H_k R_{k-1} H_k^T\nR_k = R_{k-1} - R_{k-1} H_k^T (S_b)^-1 H_k R_{k-1}\nResidual classifier update (gamma added once):\nW_k = W_pad + R_k H_k^T (Y_k - H_k W_pad)",
        facecolor="#fbe3cf", edgecolor="#b55209", linewidth=1.8, title_size=9.2, body_size=7.8,
    )

    _draw_arrow(ax, (20.4, y2 + h2 / 2), (22.3, y2 + h2 / 2), color="#d46a18")
    _draw_arrow(ax, (41.1, y2 + h2 / 2), (43.0, y2 + h2 / 2), color="#d46a18", label="H_k", label_offset=(0, 2.2))
    _draw_arrow(ax, (62.8, y2 + h2 / 2), (64.7, y2 + h2 / 2), color="#d46a18")

    # Truyền state từ Stage 1 xuống Stage 2 và vòng lặp giữa các minibatch/phase
    _draw_arrow(
        ax,
        (89.1, y1 - 0.4),
        (81.5, y2 + h2 + 0.5),
        color="#165780",
        label="Initial state (R_0, W_0)",
        label_offset=(4.8, 0.4),
        linewidth=1.8,
    )
    _draw_arrow(
        ax,
        (51.3, y1 - 0.4),
        (36.0, y2 + h2 + 0.5),
        color="#556473",
        linestyle="--",
        label="Reuse frozen backbone & W_fe",
        label_offset=(-1.0, 0.4),
        linewidth=1.4,
    )
    _draw_arrow(
        ax,
        (76.0, y2 - 0.3),
        (53.0, y2 - 0.3),
        color="#b55209",
        linestyle="--",
        connectionstyle="arc3,rad=-0.16",
        label="Next batch / phase (R_{k-1}, W_{k-1})",
        label_offset=(0.0, -3.5),
        linewidth=1.4,
    )

    # Stage 3 boxes
    y3, h3 = 6.0, 12.6
    _draw_box(
        ax, 3.3, y3, 21.2, h3,
        "10. Test Input (All Seen)",
        "Test imgs of classes 0..C_k-1\n(5,000 -> 10,000 imgs)\nSame base-only norm",
        facecolor="#ffffff", edgecolor="#2d7a4f", title_size=9.2,
    )
    _draw_box(
        ax, 27.6, y3, 22.5, h3,
        "11. Frozen Forward + FE",
        "ResNet-32 (frozen) -> F_test\nH_test = ReLU(F_test W_fe)\nShape: (n_test, d) float64",
        facecolor="#e8f6ee", edgecolor="#2d7a4f", title_size=9.2,
    )
    _draw_box(
        ax, 53.2, y3, 21.2, h3,
        "12. Linear Score Matrix",
        "S = H_test W_k\nShape: (n_test, C_k)\nAll seen classes compete",
        facecolor="#e8f6ee", edgecolor="#2d7a4f", title_size=9.2,
    )
    _draw_box(
        ax, 77.5, y3, 19.2, h3,
        "13. Unified Prediction",
        "y_hat = argmax_c S_{:,c}\nNo Task ID provided\nMap col -> original class",
        facecolor="#d6efe0", edgecolor="#21613d", linewidth=1.8, title_size=9.2,
    )

    _draw_arrow(ax, (24.8, y3 + h3 / 2), (27.3, y3 + h3 / 2), color="#2d7a4f")
    _draw_arrow(ax, (50.4, y3 + h3 / 2), (52.9, y3 + h3 / 2), color="#2d7a4f", label="H_test", label_offset=(0, 2.0))
    _draw_arrow(ax, (74.7, y3 + h3 / 2), (77.2, y3 + h3 / 2), color="#2d7a4f", label="S", label_offset=(0, 2.0))
    _draw_arrow(
        ax,
        (85.0, y2 - 0.4),
        (65.0, y3 + h3 + 0.5),
        color="#21613d",
        label="Final classifier W_k in R^(d x C_k)",
        label_offset=(4.8, 0.3),
        linewidth=1.8,
    )

    ax.text(
        50,
        1.7,
        "Self-drawn diagram of ACIL (Zhuang et al., NeurIPS 2022) matching repo modules: src/resnet.py, src/base_training.py, src/analytic.py, src/streaming.py.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#5b6977",
        style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return output_path


def draw_problem_and_predecessors_diagram(output_path: Path, dpi: int = 200) -> Path:
    """Vẽ hình minh họa bài toán Class-Incremental Learning và so sánh các hướng tiền nhiệm với ACIL.

    Tham số:
        output_path: Đường dẫn file PNG đầu ra.
        dpi: Độ phân giải ảnh khi lưu.

    Trả về:
        Đường dẫn file PNG đã lưu.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(15.0, 9.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        50,
        98.0,
        "Class-Incremental Learning (CIL): Problem Setting & Predecessor Paradigms vs. ACIL",
        ha="center",
        va="top",
        fontsize=14.2,
        fontweight="bold",
        color="#14212e",
    )
    ax.text(
        50,
        94.6,
        "Why continual learning is hard without old data, where predecessor methods struggle, and how ACIL reformulates the update",
        ha="center",
        va="top",
        fontsize=9.7,
        color="#4a5968",
    )

    # Panel A: Problem Setting
    panel_a = FancyBboxPatch(
        (2.0, 58.0),
        96.0,
        33.5,
        boxstyle="round,pad=0.3,rounding_size=1.4",
        facecolor="#f7f9fb",
        edgecolor="#c4cfd9",
        linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(panel_a)
    ax.text(
        3.5,
        89.8,
        "PANEL A · The Class-Incremental Learning (CIL) Problem & Real-World Constraints",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#1c2a38",
    )

    # Chuỗi Phase 0 -> Phase 1 -> ... -> Phase K -> Test
    _draw_box(
        ax, 3.8, 71.2, 16.5, 14.2,
        "Phase 0 (Base)",
        "Train classes 0..49\n(e.g., 50 base labels)\nLarge initial dataset\nAvailable at t = 0 only",
        facecolor="#e9f2f9", edgecolor="#2274a5", body_size=8.0,
    )
    _draw_box(
        ax, 23.2, 71.2, 16.5, 14.2,
        "Phase 1 (+10 cls)",
        "New classes 50..59\nOld images X_0 gone\n(privacy / retention\npolicy or storage cap)",
        facecolor="#fdf0e5", edgecolor="#d46a18", body_size=8.0,
    )
    _draw_box(
        ax, 42.6, 71.2, 16.5, 14.2,
        "Phases 2..K",
        "Sequential tasks\n+10 classes/phase\n(total 100 classes)\nOnly current batch seen",
        facecolor="#fdf0e5", edgecolor="#d46a18", body_size=8.0,
    )
    _draw_box(
        ax, 63.0, 71.2, 33.2, 14.2,
        "All-Seen Evaluation (No Task ID)",
        "Test set contains ALL seen classes 0..C_k-1\nNaive fine-tuning -> Catastrophic Forgetting\n(new gradients overwrite old class boundaries)\nGoal: High all-seen A_bar & low forgetting F",
        facecolor="#eef7f1", edgecolor="#2d7a4f", linewidth=1.7, body_size=8.1,
    )

    _draw_arrow(ax, (20.6, 78.3), (22.9, 78.3), color="#4a5968")
    _draw_arrow(ax, (40.0, 78.3), (42.3, 78.3), color="#4a5968")
    _draw_arrow(ax, (59.4, 78.3), (62.7, 78.3), color="#2d7a4f")

    # Ba hộp thách thức dưới Panel A
    _draw_box(
        ax, 3.8, 59.8, 29.5, 8.8,
        "Constraint 1 · No Raw Old Data",
        "Strict privacy (medical/edge) forbids storing past images.",
        facecolor="#ffffff", edgecolor="#93a3b3", title_size=9.0, body_size=7.9,
    )
    _draw_box(
        ax, 35.3, 59.8, 29.5, 8.8,
        "Constraint 2 · Stability vs. Plasticity",
        "Must learn new classes without erasing old representations.",
        facecolor="#ffffff", edgecolor="#93a3b3", title_size=9.0, body_size=7.9,
    )
    _draw_box(
        ax, 66.8, 59.8, 29.4, 8.8,
        "Constraint 3 · Argmax Competition",
        "Even with fixed features, adding classes shrinks margins.",
        facecolor="#ffffff", edgecolor="#93a3b3", title_size=9.0, body_size=7.9,
    )

    # Panel B: Predecessor Paradigms vs ACIL
    panel_b = FancyBboxPatch(
        (2.0, 4.5),
        96.0,
        50.5,
        boxstyle="round,pad=0.3,rounding_size=1.4",
        facecolor="#f9fafb",
        edgecolor="#c4cfd9",
        linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(panel_b)
    ax.text(
        3.5,
        53.4,
        "PANEL B · Predecessor Approaches (Core Idea, Remaining Limitation, Practical Consequence) vs. ACIL",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#1c2a38",
    )

    cards = [
        {
            "x": 3.6,
            "w": 29.6,
            "fc": "#fff9f8",
            "ec": "#c0564b",
            "header_fc": "#f8e2df",
            "title": "Predecessor 1 · Regularization\n(e.g., LwF, EWC)",
            "sections": [
                ("Core Idea", "Retain old knowledge via knowledge\ndistillation (LwF) or Fisher parameter\npenalties (EWC) during iterative BP."),
                ("Stored State", "Previous teacher model or Fisher diag\n(exemplar-free; no raw images saved)."),
                ("Remaining Limitation", "Gradient interference & approximate\npenalties drift over long sequences."),
                ("Practical Consequence", "Severe forgetting at large K; weak\naccuracy on old classes vs. new ones."),
            ],
        },
        {
            "x": 35.2,
            "w": 29.6,
            "fc": "#fffaf3",
            "ec": "#c97a2b",
            "header_fc": "#f8ead7",
            "title": "Predecessor 2 · Exemplar Replay\n(e.g., iCaRL, PODNet, AANets)",
            "sections": [
                ("Core Idea", "Keep a raw exemplar buffer of old\nclasses; retrain via BP + distillation\non buffer + current-phase images."),
                ("Stored State", "Raw exemplar images (e.g., 20/class)\nplus previous teacher snapshot."),
                ("Remaining Limitation", "Depends on raw data retention;\nbuffer imbalance as C_k grows."),
                ("Practical Consequence", "Violates strict privacy / zero-replay\nrules; degrades if buffer is forbidden."),
            ],
        },
        {
            "x": 66.8,
            "w": 29.6,
            "fc": "#f4f9fd",
            "ec": "#2274a5",
            "header_fc": "#dcecf8",
            "title": "ACIL (Zhuang et al., NeurIPS 2022)\nAnalytic Class-Incremental Learning",
            "sections": [
                ("Core Idea", "Train CNN in Phase 0 only, freeze it,\nexpand via fixed random FE + ReLU,\nand solve ridge recursively (Woodbury)."),
                ("Stored State", "Inverse correlation R_k in R^(d x d)\nand W_k in R^(d x C_k); NO raw images."),
                ("Key Strength (Verified)", "Exact equivalence to joint ridge on\nfixed features (R3 max err ~ 1e-11)."),
                ("Theoretical Trade-off", "Frozen representation cannot adapt;\nR_k costs O(d^2) memory (~512 MB d=8000)."),
            ],
        },
    ]

    for card in cards:
        cx, cw = card["x"], card["w"]
        cy, ch = 6.5, 44.2
        outer = FancyBboxPatch(
            (cx, cy),
            cw,
            ch,
            boxstyle="round,pad=0.25,rounding_size=1.2",
            facecolor=card["fc"],
            edgecolor=card["ec"],
            linewidth=1.6,
            zorder=2,
        )
        ax.add_patch(outer)
        header = FancyBboxPatch(
            (cx, cy + ch - 7.8),
            cw,
            7.8,
            boxstyle="round,pad=0.25,rounding_size=1.1",
            facecolor=card["header_fc"],
            edgecolor=card["ec"],
            linewidth=1.3,
            zorder=3,
        )
        ax.add_patch(header)
        ax.text(
            cx + cw / 2,
            cy + ch - 3.9,
            card["title"],
            ha="center",
            va="center",
            fontsize=9.3,
            fontweight="bold",
            color="#15212d",
            zorder=4,
        )
        sy = cy + ch - 10.2
        for heading, text in card["sections"]:
            ax.text(
                cx + 1.3,
                sy,
                f"[{heading}]",
                ha="left",
                va="top",
                fontsize=8.1,
                fontweight="bold",
                color=card["ec"],
                zorder=4,
            )
            ax.text(
                cx + 1.3,
                sy - 2.3,
                text,
                ha="left",
                va="top",
                fontsize=7.9,
                color="#2c3742",
                linespacing=1.28,
                zorder=4,
            )
            sy -= 8.8

    ax.text(
        50,
        1.8,
        "Self-drawn conceptual diagram comparing CIL problem constraints, LwF/EWC, iCaRL/PODNet/AANets, and ACIL (NeurIPS 2022).",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#5b6977",
        style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return output_path


def main() -> int:
    """Sinh hai sơ đồ đọc hiểu ACIL vào thư mục hình của báo cáo.

    Tham số:
        Không nhận tham số hàm; đọc ``--output-dir`` và ``--dpi`` từ CLI.

    Trả về:
        Mã thoát 0 sau khi ghi xong hai file PNG.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    p1 = draw_pipeline_diagram(args.output_dir / "acil_pipeline_end_to_end.png", dpi=args.dpi)
    p2 = draw_problem_and_predecessors_diagram(
        args.output_dir / "acil_problem_and_predecessors.png", dpi=args.dpi
    )
    print("Saved self-drawn ACIL diagrams:")
    print(" -", p1)
    print(" -", p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
