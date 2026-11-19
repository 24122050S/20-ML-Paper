"""Điều phối thực nghiệm phụ cuối kỳ E3 (kiểm chứng hạn chế thứ 3 P3 & nguyên mẫu FD3).

Module này trình bày bằng chứng bổ sung trong README.md (ngoài 2 hạn chế chính P1, P2 trên Poster A2),
tách bạch hai cơ chế suy giảm thường bị gộp chung trong ACIL:
1. Sai số xấp xỉ số học của cập nhật Woodbury ở float32 (đã đo tại E2b, d=1024): làm mất tương đương
   đại số chính xác (max_rel_W ~ 1.5e-3..4.0e-3, phát sinh mẫu đổi nhãn argmax) nhưng làm lệch độ chính xác
   tổng thể chưa tới 0.01 pp.
2. Sai lệch bản chất của hàm mục tiêu tuyến tính MSE One-Hot tách cột khi mất cân bằng số mẫu lớp (E3, d=1024):
   dù thỏa mãn Định lý 3.1 chính xác tới ~1e-11 ở float64 (0 mẫu lệch so với Joint Ridge), khi các lớp mới
   chỉ có N_inc in {250, 100, 50, 25, 10} mẫu/lớp so với N_base = 500 mẫu/lớp ở Phase 0, chuẩn cột trọng số
   ||w_c||_2 co nhỏ có hệ thống và kéo độ chính xác 50 lớp mới A_inc sụt về 0.00%.
3. Hướng khắc phục nghiệm đóng FD3 (Importance-Weighted Recursive Least Squares - WRLS): cân bằng khối lượng
   mẫu hiệu dụng omega_c = N_base / N_c ngay trong công thức đệ quy Woodbury một lượt duyệt (không lưu mẫu cũ,
   không tăng bộ nhớ ma trận R_k), bảo toàn 100% tương đương học gộp có trọng số và khôi phục mạnh A_inc.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2_memory_precision_runner import prepare_or_load_seed_fixture
from src.analytic import incremental_update, initialize_batches, joint_ridge
from src.cifar_data import sha256_file
from src.metrics import summarize
from src.reporting import collect_provenance, save_json
from src.run_attempt import RunAttempt


DEFAULT_E3_N_INC_LIST: tuple[int, ...] = (500, 250, 100, 50, 25, 10)
DEFAULT_E3_DIMENSION: int = 1024
DEFAULT_E3_GAMMA: float = 0.1


def weighted_joint_ridge(
    h: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    n_classes: int,
    gamma: float,
) -> np.ndarray:
    """Giải nghiệm chuẩn Joint Ridge có trọng số mẫu dùng làm mốc đối chiếu đại số cho FD3-WRLS.

    Giải bài toán ``min_W sum_i omega_i ||e_{y_i} - h_i^T W||_2^2 + gamma ||W||_F^2`` ở ``float64``.

    Tham số:
        h: Ma trận đặc trưng mở rộng kích thước ``(N, d)``.
        labels: Vector nhãn lớp nguyên toàn cục kích thước ``(N,)``.
        sample_weights: Vector trọng số mẫu dương ``omega_i`` kích thước ``(N,)``.
        n_classes: Tổng số cột lớp ``C`` của ma trận trọng số.
        gamma: Hệ số điều chuẩn Ridge dương.

    Trả về:
        Ma trận trọng số ``W`` kích thước ``(d, n_classes)`` kiểu ``float64``.
    """
    h64 = np.asarray(h, dtype=np.float64)
    y64 = np.asarray(labels, dtype=np.int64)
    w64 = np.asarray(sample_weights, dtype=np.float64)
    if h64.ndim != 2 or y64.ndim != 1 or w64.ndim != 1:
        raise ValueError("Kích thước đầu vào của weighted_joint_ridge không hợp lệ")
    if not (len(h64) == len(y64) == len(w64)) or len(h64) == 0:
        raise ValueError("Số phần tử h, labels, sample_weights phải bằng nhau và khác rỗng")
    if np.any(w64 <= 0.0) or not np.isfinite(w64).all():
        raise ValueError("sample_weights phải dương và hữu hạn")
    if gamma <= 0.0 or not np.isfinite(gamma):
        raise ValueError("gamma phải dương và hữu hạn")

    sqrt_w = np.sqrt(w64)
    h_scaled = h64 * sqrt_w[:, None]
    y_scaled = np.zeros((len(y64), n_classes), dtype=np.float64)
    y_scaled[np.arange(len(y64)), y64] = sqrt_w
    gram = h_scaled.T @ h_scaled + float(gamma) * np.eye(h64.shape[1], dtype=np.float64)
    return np.linalg.solve(gram, h_scaled.T @ y_scaled)


def weighted_incremental_update(
    r_old: np.ndarray,
    w_old: np.ndarray,
    h: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    n_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cập nhật đệ quy Woodbury một lượt có trọng số mẫu cho nguyên mẫu FD3-WRLS (không lưu mẫu cũ).

    Nhân mỗi hàng đặc trưng và vector đích one-hot với ``sqrt(omega_i)`` để cân bằng đồng thời
    ma trận hiệp phương sai ``H^T Omega H`` và vế phải ``H^T Omega Y`` qua đúng bổ đề Woodbury.

    Tham số:
        r_old: Ma trận nghịch đảo Gram có trọng số từ bước trước kích thước ``(d, d)``.
        w_old: Ma trận trọng số phân loại từ bước trước kích thước ``(d, C_old)``.
        h: Ma trận đặc trưng của minibatch hiện tại kích thước ``(b, d)``.
        labels: Vector nhãn nguyên toàn cục của minibatch kích thước ``(b,)``.
        sample_weights: Vector trọng số dương ``omega_i`` kích thước ``(b,)``.
        n_classes: Tổng số lớp đích ``C`` sau bước cập nhật (tùy chọn).

    Trả về:
        Tuple ``(r_new, w_new)`` kiểu ``float64`` khớp đại số với ``weighted_joint_ridge``.
    """
    h64 = np.asarray(h, dtype=np.float64)
    y64 = np.asarray(labels, dtype=np.int64)
    w64 = np.asarray(sample_weights, dtype=np.float64)
    r_prev = np.asarray(r_old, dtype=np.float64)
    w_prev = np.asarray(w_old, dtype=np.float64)

    if h64.ndim != 2 or len(h64) == 0 or not np.isfinite(h64).all():
        raise ValueError("H phải là ma trận 2 chiều hữu hạn, khác rỗng")
    if y64.ndim != 1 or w64.ndim != 1 or not (len(h64) == len(y64) == len(w64)):
        raise ValueError("labels và sample_weights phải có cùng độ dài với H")
    if np.any(w64 <= 0.0) or not np.isfinite(w64).all():
        raise ValueError("sample_weights phải dương và hữu hạn")
    d = h64.shape[1]
    if r_prev.shape != (d, d) or w_prev.ndim != 2 or w_prev.shape[0] != d:
        raise ValueError("Kích thước r_old hoặc w_old không khớp chiều đặc trưng")

    required_classes = max(w_prev.shape[1], int(y64.max()) + 1)
    count = required_classes if n_classes is None else int(n_classes)
    if count < required_classes:
        raise ValueError("n_classes nhỏ hơn số lớp hiện có")

    w_padded = np.zeros((d, count), dtype=np.float64)
    w_padded[:, : w_prev.shape[1]] = w_prev

    sqrt_w = np.sqrt(w64)
    h_tilde = h64 * sqrt_w[:, None]
    y_tilde = np.zeros((len(y64), count), dtype=np.float64)
    y_tilde[np.arange(len(y64)), y64] = sqrt_w

    left = r_prev @ h_tilde.T
    middle = np.eye(len(h_tilde), dtype=np.float64) + h_tilde @ left
    r_new = r_prev - left @ np.linalg.solve(middle, h_tilde @ r_prev)
    w_new = w_padded + r_new @ h_tilde.T @ (y_tilde - h_tilde @ w_padded)
    return r_new, w_new


def sample_imbalanced_incremental_indices(
    train_labels: np.ndarray,
    *,
    n_inc_per_class: int,
    seed: int,
    base_classes: int = 50,
    n_incremental_phases: int = 5,
    classes_per_phase: int = 10,
) -> list[np.ndarray]:
    """Lấy mẫu con tất định không hoàn lại cho từng pha tăng dần với đúng ``n_inc_per_class`` mẫu/lớp.

    Tham số:
        train_labels: Vector nhãn huấn luyện đã ánh xạ ``0..99`` kích thước ``(50000,)``.
        n_inc_per_class: Số lượng mẫu huấn luyện giữ lại cho mỗi lớp tăng dần (``1..500``).
        seed: Hạt giống cơ sở của lượt chạy để tạo bộ sinh ngẫu nhiên độc lập.
        base_classes: Số lớp ở Phase 0 (mặc định 50).
        n_incremental_phases: Số pha tăng dần (mặc định 5).
        classes_per_phase: Số lớp mới mỗi pha (mặc định 10).

    Trả về:
        Danh sách gồm ``n_incremental_phases`` mảng chỉ số mẫu đã xáo trộn tất định theo pha.
    """
    if n_inc_per_class < 1:
        raise ValueError("n_inc_per_class phải >= 1")
    labels = np.asarray(train_labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed) * 10000 + int(n_inc_per_class))
    phase_indices: list[np.ndarray] = []

    for phase_id in range(1, n_incremental_phases + 1):
        lo = base_classes + (phase_id - 1) * classes_per_phase
        hi = base_classes + phase_id * classes_per_phase
        selected = []
        for cls_id in range(lo, hi):
            cls_idx = np.flatnonzero(labels == cls_id)
            if len(cls_idx) < n_inc_per_class:
                raise ValueError(f"Lớp {cls_id} chỉ có {len(cls_idx)} mẫu < {n_inc_per_class}")
            chosen = (
                cls_idx.copy()
                if len(cls_idx) == n_inc_per_class
                else rng.choice(cls_idx, size=n_inc_per_class, replace=False)
            )
            selected.append(np.sort(chosen))
        merged = np.concatenate(selected)
        rng.shuffle(merged)
        phase_indices.append(merged)
    return phase_indices


def run_e3_imbalance_case(
    *,
    seed: int,
    variant: str,
    n_inc_per_class: int,
    n_base_per_class: int,
    base_r0: np.ndarray,
    base_w0: np.ndarray,
    h_train: np.ndarray,
    train_labels: np.ndarray,
    h_test: np.ndarray,
    test_labels: np.ndarray,
    gamma: float = DEFAULT_E3_GAMMA,
    batch_size: int = 128,
    base_classes: int = 50,
    n_incremental_phases: int = 5,
    classes_per_phase: int = 10,
    threads: int = 4,
) -> dict:
    """Chạy một cấu hình tỷ lệ mất cân bằng lớp cho ``ACIL_standard`` hoặc ``FD3_weighted_rls``.

    Tham số:
        seed: Hạt giống chuẩn (``42``, ``43``, ``44``).
        variant: Tên biến thể (``"ACIL_standard"`` hoặc ``"FD3_weighted_rls"``).
        n_inc_per_class: Số mẫu train mỗi lớp tăng dần ở Phases 1..K.
        n_base_per_class: Số mẫu train mỗi lớp Base ở Phase 0 (mặc định 500).
        base_r0: Ma trận ``R_0`` khởi tạo từ Phase 0.
        base_w0: Ma trận ``W_0`` khởi tạo từ Phase 0.
        h_train: Ma trận đặc trưng mở rộng ``ReLU(F_train @ W_fe)`` kích thước ``(N_train, d)``.
        train_labels: Vector nhãn train ``0..99``.
        h_test: Ma trận đặc trưng mở rộng của tập test kích thước ``(N_test, d)``.
        test_labels: Vector nhãn test ``0..99``.
        gamma: Hệ số điều chuẩn Ridge dương.
        batch_size: Kích thước minibatch cập nhật Woodbury (mặc định 128).
        base_classes: Số lớp ở Phase 0 (mặc định 50).
        n_incremental_phases: Số pha tăng dần (mặc định 5).
        classes_per_phase: Số lớp mới mỗi pha (mặc định 10).
        threads: Giới hạn số luồng BLAS/CPU.

    Trả về:
        Dict kết quả gồm chuỗi phase, metric tổng hợp, thống kê chuẩn cột ``||w_c||_2`` và kiểm chứng Joint Oracle.
    """
    if variant not in ("ACIL_standard", "FD3_weighted_rls"):
        raise ValueError(f"Biến thể E3 không hợp lệ: {variant}")

    imbalance_ratio = float(n_base_per_class) / float(n_inc_per_class)
    phase_idx_list = sample_imbalanced_incremental_indices(
        train_labels,
        n_inc_per_class=n_inc_per_class,
        seed=seed,
        base_classes=base_classes,
        n_incremental_phases=n_incremental_phases,
        classes_per_phase=classes_per_phase,
    )

    base_mask = train_labels < base_classes
    seen_h = [h_train[base_mask]]
    seen_y = [train_labels[base_mask]]
    seen_w = [np.ones(int(np.sum(base_mask)), dtype=np.float64)]

    r_curr = np.array(base_r0, dtype=np.float64, copy=True)
    w_curr = np.array(base_w0, dtype=np.float64, copy=True)

    phase_rows = []
    max_rel_w_err = 0.0
    total_disagreements = 0

    t_start = time.perf_counter()
    with threadpool_limits(limits=threads):
        # Đánh giá mốc Phase 0
        test_m0 = test_labels < base_classes
        scores0 = h_test[test_m0] @ w_curr
        preds0 = np.argmax(scores0, axis=1)
        acc0 = float(np.mean(preds0 == test_labels[test_m0]) * 100.0)

        w_joint0 = (
            joint_ridge(seen_h[0], seen_y[0], n_classes=base_classes, gamma=gamma)
            if variant == "ACIL_standard"
            else weighted_joint_ridge(
                seen_h[0], seen_y[0], seen_w[0], n_classes=base_classes, gamma=gamma
            )
        )
        rel_w0 = float(np.linalg.norm(w_curr - w_joint0) / np.linalg.norm(w_joint0))
        dis0 = int(np.sum(preds0 != np.argmax(h_test[test_m0] @ w_joint0, axis=1)))
        max_rel_w_err = max(max_rel_w_err, rel_w0)
        total_disagreements += dis0
        phase_rows.append({
            "phase": 0,
            "n_classes": base_classes,
            "test_samples_all_seen": int(np.sum(test_m0)),
            "accuracy_pct": acc0,
            "base_accuracy_pct": acc0,
            "inc_accuracy_pct": None,
            "relative_w_error": rel_w0,
            "prediction_disagreements": dis0,
            "checks_passed": bool(rel_w0 < 1e-8 and dis0 == 0),
        })

        # Cập nhật đệ quy qua các pha tăng dần 1..K
        for phase_id, idx_phase in enumerate(phase_idx_list, start=1):
            seen_classes = base_classes + phase_id * classes_per_phase
            hk = h_train[idx_phase]
            yk = train_labels[idx_phase]
            wk = (
                np.ones(len(yk), dtype=np.float64)
                if variant == "ACIL_standard"
                else np.full(len(yk), imbalance_ratio, dtype=np.float64)
            )

            for start_idx in range(0, len(yk), batch_size):
                hb = hk[start_idx : start_idx + batch_size]
                yb = yk[start_idx : start_idx + batch_size]
                if variant == "ACIL_standard":
                    r_curr, w_curr = incremental_update(
                        r_curr, w_curr, hb, yb, n_classes=seen_classes
                    )
                else:
                    wb = wk[start_idx : start_idx + batch_size]
                    r_curr, w_curr = weighted_incremental_update(
                        r_curr, w_curr, hb, yb, wb, n_classes=seen_classes
                    )

            seen_h.append(hk)
            seen_y.append(yk)
            seen_w.append(wk)

            h_cat = np.concatenate(seen_h, axis=0)
            y_cat = np.concatenate(seen_y, axis=0)
            w_cat = np.concatenate(seen_w, axis=0)
            w_joint = (
                joint_ridge(h_cat, y_cat, n_classes=seen_classes, gamma=gamma)
                if variant == "ACIL_standard"
                else weighted_joint_ridge(
                    h_cat, y_cat, w_cat, n_classes=seen_classes, gamma=gamma
                )
            )

            test_mk = test_labels < seen_classes
            yk_test = test_labels[test_mk]
            scores_k = h_test[test_mk] @ w_curr
            preds_k = np.argmax(scores_k, axis=1)
            preds_joint_k = np.argmax(h_test[test_mk] @ w_joint, axis=1)

            rel_wk = float(np.linalg.norm(w_curr - w_joint) / np.linalg.norm(w_joint))
            dis_k = int(np.sum(preds_k != preds_joint_k))
            max_rel_w_err = max(max_rel_w_err, rel_wk)
            total_disagreements += dis_k

            base_sub = yk_test < base_classes
            inc_sub = yk_test >= base_classes
            acc_k = float(np.mean(preds_k == yk_test) * 100.0)
            base_acc_k = float(np.mean(preds_k[base_sub] == yk_test[base_sub]) * 100.0)
            inc_acc_k = float(np.mean(preds_k[inc_sub] == yk_test[inc_sub]) * 100.0)

            phase_rows.append({
                "phase": phase_id,
                "n_classes": seen_classes,
                "test_samples_all_seen": int(np.sum(test_mk)),
                "accuracy_pct": acc_k,
                "base_accuracy_pct": base_acc_k,
                "inc_accuracy_pct": inc_acc_k,
                "relative_w_error": rel_wk,
                "prediction_disagreements": dis_k,
                "checks_passed": bool(rel_wk < 1e-8 and dis_k == 0),
            })

    elapsed = time.perf_counter() - t_start
    if (
        base_classes == 50
        and classes_per_phase == 10
        and phase_rows[0]["test_samples_all_seen"] == 5000
    ):
        std_summary = summarize(
            phase_rows,
            {
                "mode": "benchmark",
                "seed": seed,
                "incremental_phases": n_incremental_phases,
                "run_incremental_phases": n_incremental_phases,
            },
        )
        avg_inc_acc = float(std_summary["average_incremental_accuracy_pct"])
        base_forgetting = float(std_summary["base_forgetting_pp"])
    else:
        avg_inc_acc = float(np.mean([r["accuracy_pct"] for r in phase_rows]))
        base_forgetting = float(phase_rows[0]["base_accuracy_pct"] - phase_rows[-1]["base_accuracy_pct"])

    final_row = phase_rows[-1]
    col_norms = np.linalg.norm(w_curr, axis=0)
    norm_base = float(np.mean(col_norms[:base_classes]))
    norm_inc = float(np.mean(col_norms[base_classes:]))

    final_scores = h_test @ w_curr
    mean_max_score_base = float(np.mean(np.max(final_scores[:, :base_classes], axis=1)))
    mean_max_score_inc = float(np.mean(np.max(final_scores[:, base_classes:], axis=1)))

    metric_summary = {
        "average_incremental_accuracy_pct": avg_inc_acc,
        "final_accuracy_pct": float(final_row["accuracy_pct"]),
        "final_base_accuracy_pct": float(final_row["base_accuracy_pct"]),
        "final_inc_accuracy_pct": float(final_row["inc_accuracy_pct"]),
        "final_base_minus_inc_gap_pp": float(
            final_row["base_accuracy_pct"] - final_row["inc_accuracy_pct"]
        ),
        "base_forgetting_pp": base_forgetting,
        "mean_w_norm_base": norm_base,
        "mean_w_norm_inc": norm_inc,
        "w_norm_inc_over_base_ratio": float(norm_inc / norm_base),
        "mean_max_logit_base": mean_max_score_base,
        "mean_max_logit_inc": mean_max_score_inc,
    }

    verification_passed = bool(max_rel_w_err < 1e-8 and total_disagreements == 0)
    return {
        "branch": "E3" if variant == "ACIL_standard" else "FD3",
        "variant": variant,
        "seed": seed,
        "dimension": int(h_train.shape[1]),
        "gamma": float(gamma),
        "n_base_per_class": int(n_base_per_class),
        "n_inc_per_class": int(n_inc_per_class),
        "imbalance_ratio": imbalance_ratio,
        "wall_seconds": float(elapsed),
        "joint_oracle_comparison": {
            "max_relative_w_error": max_rel_w_err,
            "total_prediction_disagreements": total_disagreements,
        },
        "metric_summary": metric_summary,
        "phases": phase_rows,
        "verification_passed": verification_passed,
    }


def log_e3_suite_wandb(
    *,
    seed: int,
    cases: list[dict],
    dimension: int,
    gamma: float,
    wandb_mode: str,
    entity: str,
    project: str,
    wandb_dir: Path,
    provenance: dict,
    wandb_client=None,
) -> dict:
    """Ghi nhận 1 run W&B tổng hợp cho toàn bộ lưới mất cân bằng lớp E3 & FD3 trên một seed.

    Tham số:
        seed: Hạt giống của lượt chạy (``42``, ``43``, ``44``).
        cases: Danh sách các dict kết quả từ ``run_e3_imbalance_case``.
        dimension: Chiều mở rộng đặc trưng ``d`` (mặc định 1024).
        gamma: Hệ số điều chuẩn Ridge.
        wandb_mode: ``"disabled"``, ``"offline"`` hoặc ``"online"``.
        entity: Tên entity W&B.
        project: Tên project W&B.
        wandb_dir: Thư mục lưu tệp log W&B cục bộ.
        provenance: Dict thông tin mã nguồn và SHA-256.
        wandb_client: Module/mock ``wandb`` tùy chọn dùng trong unit test.

    Trả về:
        Dict ghi trạng thái tracking, ``wandb_run_id``, ``wandb_run_url`` và danh sách ``tags``.
    """
    if wandb_mode == "disabled":
        return {
            "wandb_status": "not_logged",
            "wandb_run_id": None,
            "wandb_run_url": None,
            "tags": [],
        }
    if wandb_mode not in ("offline", "online"):
        raise ValueError("wandb_mode phải là 'disabled', 'offline' hoặc 'online'")

    client = wandb_client
    if client is None:
        import wandb as client

    tags = [
        "final-empirics",
        "P3-mse-imbalance",
        "FD3-weighted-rls",
        "E3-class-imbalance",
        f"seed{seed}",
    ]
    with client.init(
        entity=entity,
        project=project,
        name=f"E3-mse-imbalance-seed{seed}",
        job_type="final-empirics-e3",
        mode=wandb_mode,
        dir=str(wandb_dir),
        tags=tags,
        config={
            "branch": "E3_and_FD3",
            "issue_tested": "P3 — MSE One-Hot Class-Imbalance Collapse vs float32 Woodbury Drift",
            "seed": seed,
            "dimension": dimension,
            "gamma": gamma,
            "n_cases": len(cases),
            "source_commit": provenance.get("git_commit"),
            "runner_script": "scripts/e3_mse_imbalance_runner.py",
        },
    ) as run:
        for step_idx, item in enumerate(cases):
            ms = item["metric_summary"]
            jo = item["joint_oracle_comparison"]
            run.log(
                {
                    "is_fd3_weighted_rls": item["variant"] == "FD3_weighted_rls",
                    "n_inc_per_class": item["n_inc_per_class"],
                    "imbalance_ratio": item["imbalance_ratio"],
                    "average_incremental_accuracy_pct": ms["average_incremental_accuracy_pct"],
                    "final_accuracy_pct": ms["final_accuracy_pct"],
                    "final_base_accuracy_pct": ms["final_base_accuracy_pct"],
                    "final_inc_accuracy_pct": ms["final_inc_accuracy_pct"],
                    "final_base_minus_inc_gap_pp": ms["final_base_minus_inc_gap_pp"],
                    "w_norm_inc_over_base_ratio": ms["w_norm_inc_over_base_ratio"],
                    "max_relative_w_error": jo["max_relative_w_error"],
                    "total_prediction_disagreements": jo["total_prediction_disagreements"],
                },
                step=step_idx,
            )
            prefix = f"{item['variant']}_N{item['n_inc_per_class']}"
            run.summary[f"{prefix}_final_acc"] = ms["final_accuracy_pct"]
            run.summary[f"{prefix}_inc_acc"] = ms["final_inc_accuracy_pct"]
            run.summary[f"{prefix}_w_ratio"] = ms["w_norm_inc_over_base_ratio"]
            run.summary[f"{prefix}_max_rel_w"] = jo["max_relative_w_error"]

        run.summary["verification_passed"] = all(c["verification_passed"] for c in cases)
        run_id = getattr(run, "id", None)
        run_url = getattr(run, "url", None)

    if wandb_mode == "online" and (not run_id or not run_url):
        raise RuntimeError("W&B online E3 không trả về run ID hoặc URL hợp lệ")

    return {
        "wandb_status": "logged" if wandb_mode == "online" else "offline_pending_sync",
        "wandb_run_id": run_id,
        "wandb_run_url": run_url,
        "tags": tags,
    }


def run_e3_seed_suite(
    *,
    seed: int = 42,
    output_dir: str | Path,
    fixture_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    data_dir: str | Path = ROOT / "data",
    project_root: Path = ROOT,
    dimension: int = DEFAULT_E3_DIMENSION,
    gamma: float = DEFAULT_E3_GAMMA,
    n_inc_list: tuple[int, ...] = DEFAULT_E3_N_INC_LIST,
    batch_size: int = 128,
    device: str = "auto",
    threads: int = 4,
    wandb_mode: str = "disabled",
    entity: str = "hsangg-2006",
    project: str = "acil-reproduction",
    wandb_client=None,
) -> dict:
    """Thực thi trọn bộ thực nghiệm E3 (ACIL chuẩn & FD3-WRLS) cho một seed trong ``RunAttempt``.

    Tham số:
        seed: Hạt giống chuẩn của R1 (``42``, ``43`` hoặc ``44``).
        output_dir: Thư mục gốc lưu ``attempts/<id>/results.json``.
        fixture_dir: Thư mục chứa đặc trưng ``F64`` của R1 đã kiểm SHA-256 (tùy chọn).
        checkpoint_path: Đường dẫn tới ``base_checkpoint.pt`` nếu cần trích xuất mới (tùy chọn).
        data_dir: Thư mục dữ liệu CIFAR-100.
        project_root: Thư mục gốc repository.
        dimension: Chiều mở rộng ngẫu nhiên ``d`` (mặc định 1024 để khớp trực tiếp với E2b).
        gamma: Hệ số điều chuẩn Ridge dương (mặc định 0.1).
        n_inc_list: Danh sách số mẫu/lớp ở các pha tăng dần (mặc định ``500, 250, 100, 50, 25, 10``).
        batch_size: Cỡ lô cập nhật Woodbury (mặc định 128).
        device: Thiết bị trích xuất đặc trưng nếu chưa có fixture.
        threads: Giới hạn số luồng CPU/BLAS.
        wandb_mode: Chế độ ghi nhận W&B (``"disabled"``, ``"offline"``, ``"online"``).
        entity: Tên entity W&B.
        project: Tên project W&B.
        wandb_client: Client W&B tùy chọn cho kiểm thử đơn vị.

    Trả về:
        Dict báo cáo kết quả đã hoàn tất và vượt qua toàn bộ kiểm tra đại số.
    """
    out_root = Path(output_dir)
    resolved_fixture = (
        Path(fixture_dir) if fixture_dir is not None else (out_root / f"fixture_seed{seed}")
    )

    with RunAttempt(out_root, seed=seed) as attempt:
        attempt.progress("prepare_fixture", seed=seed)
        provenance = collect_provenance(project_root)
        runner_path = Path(__file__).resolve()
        provenance["runner_script"] = runner_path.relative_to(project_root).as_posix()
        provenance["runner_sha256"] = sha256_file(runner_path)

        meta, train_x, train_y, test_x, test_y = prepare_or_load_seed_fixture(
            seed,
            resolved_fixture,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            project_root=project_root,
            device=device,
            threads=threads,
        )

        with threadpool_limits(limits=threads):
            rng_fe = np.random.default_rng(int(seed) + 1000)
            w_fe = rng_fe.standard_normal((train_x.shape[1], dimension)).astype(np.float64)
            h_train = np.maximum(0.0, train_x.astype(np.float64) @ w_fe)
            h_test = np.maximum(0.0, test_x.astype(np.float64) @ w_fe)

            base_mask = train_y < 50
            h_base = h_train[base_mask]
            y_base = train_y[base_mask].astype(np.int64)
            base_batches = [
                (h_base[s : s + batch_size], y_base[s : s + batch_size])
                for s in range(0, len(y_base), batch_size)
            ]
            base_r0, base_w0 = initialize_batches(
                base_batches, n_features=dimension, n_classes=50, gamma=gamma
            )

        n_base_per_class = int(meta.get("train_per_class", 500))
        cases = []
        for n_inc in n_inc_list:
            for variant in ("ACIL_standard", "FD3_weighted_rls"):
                attempt.progress("e3_imbalance_case", variant=variant, n_inc_per_class=n_inc)
                case_out = run_e3_imbalance_case(
                    seed=seed,
                    variant=variant,
                    n_inc_per_class=n_inc,
                    n_base_per_class=n_base_per_class,
                    base_r0=base_r0,
                    base_w0=base_w0,
                    h_train=h_train,
                    train_labels=train_y.astype(np.int64),
                    h_test=h_test,
                    test_labels=test_y.astype(np.int64),
                    gamma=gamma,
                    batch_size=batch_size,
                    threads=threads,
                )
                cases.append(case_out)

        suite_tracking = log_e3_suite_wandb(
            seed=seed,
            cases=cases,
            dimension=dimension,
            gamma=gamma,
            wandb_mode=wandb_mode,
            entity=entity,
            project=project,
            wandb_dir=attempt.directory,
            provenance=provenance,
            wandb_client=wandb_client,
        )

        # Kiểm chéo: tại N_inc = N_base (tỷ số 1x), FD3_weighted_rls phải trùng khít ACIL_standard
        balanced_std = [
            c for c in cases if c["variant"] == "ACIL_standard" and c["n_inc_per_class"] == n_base_per_class
        ]
        balanced_fd3 = [
            c for c in cases if c["variant"] == "FD3_weighted_rls" and c["n_inc_per_class"] == n_base_per_class
        ]
        balanced_identity_ok = True
        if balanced_std and balanced_fd3:
            diff_acc = abs(
                balanced_std[0]["metric_summary"]["final_accuracy_pct"]
                - balanced_fd3[0]["metric_summary"]["final_accuracy_pct"]
            )
            balanced_identity_ok = diff_acc < 1e-12

        report = {
            "experiment": "ACIL-E3-MSE-Class-Imbalance",
            "issue_tested": "P3 — MSE One-Hot Class-Imbalance Collapse vs float32 Woodbury Drift",
            "seed": seed,
            "dimension": dimension,
            "gamma": gamma,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "threads": threads,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("scipy", "threadpoolctl")
                },
            },
            "provenance": provenance,
            "fixture_metadata": meta,
            "suite_tracking": suite_tracking,
            "cases": cases,
            "checks": {
                "training_never_executed": meta["training_executed_this_run"] is False,
                "all_cases_match_joint_oracle": all(c["verification_passed"] for c in cases),
                "balanced_fd3_matches_standard": balanced_identity_ok,
            },
        }
        report["verification_passed"] = all(report["checks"].values())
        save_json(report, attempt.directory / "results.json")
        return attempt.complete(report)


def aggregate_e3_reports(
    report_paths: list[str | Path],
    output_path: str | Path | None = None,
    e2b_summary_path: str | Path | None = ROOT / "reports/e2_memory_precision_summary.json",
) -> dict:
    """Tổng hợp mean ± sample SD (ddof=1) trên 3 seeds cho E3 & FD3, kèm bảng đối chiếu trực tiếp E2b.

    Tham số:
        report_paths: Danh sách đường dẫn tới các tệp ``e3_seed*_results.json`` hoàn tất.
        output_path: Đường dẫn ghi tệp ``e3_mse_imbalance_summary.json`` (tùy chọn).
        e2b_summary_path: Đường dẫn tới ``e2_memory_precision_summary.json`` để trích bảng đối chiếu
            cùng chiều ``d=1024`` giữa độ trôi ``float32`` Woodbury và mất cân bằng lớp MSE (tùy chọn).

    Trả về:
        Dict tổng hợp đầy đủ các đường cong ``ACIL_standard``, ``FD3_weighted_rls`` và ``contrast_vs_e2b``.
    """
    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in report_paths]
    for path, item in zip(report_paths, payloads):
        if item.get("status") != "complete" or not item.get("verification_passed"):
            raise ValueError(f"Báo cáo E3 chưa hoàn tất hợp lệ: {path}")

    grouped: dict[str, list[dict]] = {}
    for item in payloads:
        for case in item["cases"]:
            key = f"{case['variant']}__N{case['n_inc_per_class']}"
            grouped.setdefault(key, []).append(case)

    metric_keys = (
        "average_incremental_accuracy_pct",
        "final_accuracy_pct",
        "final_base_accuracy_pct",
        "final_inc_accuracy_pct",
        "final_base_minus_inc_gap_pp",
        "base_forgetting_pp",
        "mean_w_norm_base",
        "mean_w_norm_inc",
        "w_norm_inc_over_base_ratio",
    )

    curves_summary = {}
    for key, items in sorted(
        grouped.items(),
        key=lambda kv: (kv[1][0]["variant"], -kv[1][0]["n_inc_per_class"]),
    ):
        items_sorted = sorted(items, key=lambda x: x["seed"])
        seeds = [x["seed"] for x in items_sorted]
        metrics_agg = {}
        for mk in metric_keys:
            vals = np.asarray([x["metric_summary"][mk] for x in items_sorted], dtype=np.float64)
            metrics_agg[mk] = {
                "mean": float(np.mean(vals)),
                "sample_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
                "ddof": 1,
                "values_by_seed": {str(x["seed"]): float(x["metric_summary"][mk]) for x in items_sorted},
            }
        rel_errs = [x["joint_oracle_comparison"]["max_relative_w_error"] for x in items_sorted]
        disagrees = [x["joint_oracle_comparison"]["total_prediction_disagreements"] for x in items_sorted]
        curves_summary[key] = {
            "variant": items_sorted[0]["variant"],
            "n_base_per_class": items_sorted[0]["n_base_per_class"],
            "n_inc_per_class": items_sorted[0]["n_inc_per_class"],
            "imbalance_ratio": items_sorted[0]["imbalance_ratio"],
            "dimension": items_sorted[0]["dimension"],
            "gamma": items_sorted[0]["gamma"],
            "seeds": seeds,
            "joint_oracle_comparison": {
                "max_relative_w_error": float(max(rel_errs)),
                "mean_relative_w_error": float(np.mean(rel_errs)),
                "total_prediction_disagreements": int(sum(disagrees)),
            },
            "metrics": metrics_agg,
        }

    contrast = {}
    if e2b_summary_path is not None and Path(e2b_summary_path).is_file():
        e2_data = json.loads(Path(e2b_summary_path).read_text(encoding="utf-8"))
        e2b_200 = e2_data["e2b_summary"]["float32__K5b128__g0.1"]
        e2b_1575 = e2_data["e2b_summary"]["float32__K25b16__g0.001"]
        std_500 = curves_summary.get("ACIL_standard__N500")
        std_50 = curves_summary.get("ACIL_standard__N50")
        fd3_50 = curves_summary.get("FD3_weighted_rls__N50")
        if std_500 and std_50 and fd3_50:
            contrast = {
                "controlled_dimension": 1024,
                "seeds": std_500["seeds"],
                "float32_woodbury_200steps_g0p1": {
                    "satisfies_theorem_3_1_exact": False,
                    "max_relative_w_error_mean": e2b_200["max_relative_w_error_mean"],
                    "final_prediction_disagreements_total": e2b_200["final_prediction_disagreements_total"],
                    "final_accuracy_drop_pp_mean": e2b_200["final_accuracy_drop_pp_mean"],
                },
                "float32_woodbury_1575steps_g0p001": {
                    "satisfies_theorem_3_1_exact": False,
                    "max_relative_w_error_mean": e2b_1575["max_relative_w_error_mean"],
                    "final_prediction_disagreements_total": e2b_1575["final_prediction_disagreements_total"],
                    "final_accuracy_drop_pp_mean": e2b_1575["final_accuracy_drop_pp_mean"],
                },
                "mse_imbalance_10x_N50_fp64": {
                    "satisfies_theorem_3_1_exact": True,
                    "max_relative_w_error": std_50["joint_oracle_comparison"]["max_relative_w_error"],
                    "total_prediction_disagreements": std_50["joint_oracle_comparison"]["total_prediction_disagreements"],
                    "final_accuracy_drop_pp_mean": float(
                        std_500["metrics"]["final_accuracy_pct"]["mean"]
                        - std_50["metrics"]["final_accuracy_pct"]["mean"]
                    ),
                    "inc_accuracy_drop_pp_mean": float(
                        std_500["metrics"]["final_inc_accuracy_pct"]["mean"]
                        - std_50["metrics"]["final_inc_accuracy_pct"]["mean"]
                    ),
                    "fd3_inc_accuracy_recovery_pp_mean": float(
                        fd3_50["metrics"]["final_inc_accuracy_pct"]["mean"]
                        - std_50["metrics"]["final_inc_accuracy_pct"]["mean"]
                    ),
                },
            }

    out = {
        "status": "complete",
        "experiment_family": "ACIL-Final-Empirics-E3",
        "issue_tested": "P3 — MSE One-Hot Class-Imbalance Collapse vs float32 Woodbury Drift",
        "curves_summary": curves_summary,
        "contrast_vs_e2b": contrast,
    }
    if output_path is not None:
        save_json(out, Path(output_path))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Phân tích tham số dòng lệnh cho script chạy thực nghiệm phụ E3 & FD3.

    Tham số:
        argv: Danh sách đối số dòng lệnh tùy chọn; mặc định đọc từ ``sys.argv[1:]``.

    Trả về:
        Đối tượng ``argparse.Namespace`` chứa cấu hình chạy.
    """
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm phụ cuối kỳ E3 (MSE Class-Imbalance & FD3-WRLS).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports/local/final_empirics_e3_20261118")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=ROOT / "reports/local/final_empirics_e2_20261118",
        help="Thư mục gốc chứa fixture_seed{42,43,44} từ R1.",
    )
    parser.add_argument("--dimension", type=int, default=DEFAULT_E3_DIMENSION)
    parser.add_argument("--gamma", type=float, default=DEFAULT_E3_GAMMA)
    parser.add_argument("--n-inc-list", type=int, nargs="+", default=list(DEFAULT_E3_N_INC_LIST))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--publish-reports-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI thực thi E3 trên danh sách seeds và xuất tệp tổng hợp nếu được yêu cầu.

    Tham số:
        argv: Danh sách đối số dòng lệnh tùy chọn.

    Trả về:
        Mã thoát tiến trình (``0`` nếu toàn bộ các seed đạt kiểm chứng).
    """
    args = parse_args(argv)
    report_paths: list[Path] = []

    for seed in args.seeds:
        seed_out = args.output_root / f"seed{seed}"
        fixture_dir = args.fixture_root / f"fixture_seed{seed}"
        result = run_e3_seed_suite(
            seed=seed,
            output_dir=seed_out,
            fixture_dir=fixture_dir,
            dimension=args.dimension,
            gamma=args.gamma,
            n_inc_list=tuple(args.n_inc_list),
            threads=args.threads,
            wandb_mode=args.wandb_mode,
        )
        seed_json = Path(result["output_dir"]) / "results.json"
        if args.publish_reports_dir is not None:
            dst = args.publish_reports_dir / f"e3_seed{seed}_results.json"
            save_json(result, dst)
            report_paths.append(dst)
        else:
            report_paths.append(seed_json)

    if args.publish_reports_dir is not None and len(report_paths) > 1:
        aggregate_e3_reports(
            report_paths,
            output_path=args.publish_reports_dir / "e3_mse_imbalance_summary.json",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
