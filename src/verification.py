"""Sinh dữ liệu tổng hợp và đối chiếu joint/phase/minibatch một cách độc lập."""

from __future__ import annotations

import platform

import numpy as np

from .analytic import initialize, incremental_update, joint_ridge


def make_synthetic_phases(config):
    """Sinh bộ dữ liệu tổng hợp theo nhiều phase với ma trận mở rộng đặc trưng cố định.

    Tham số:
        config: Dict cấu hình gồm ``seed``, ``d_input``, ``d_feature``, ``n_phases``,
            ``classes_per_phase``, ``n_train_per_class``, ``n_test_per_class`` và ``minibatch_size``.

    Trả về:
        Danh sách các tuple ``(H_train, y_train, H_test, y_test)`` cho từng phase.
    """
    for key in ("d_input", "d_feature", "n_phases", "classes_per_phase", "n_train_per_class", "n_test_per_class", "minibatch_size"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} phải là số nguyên dương")
    rng = np.random.default_rng(config["seed"])
    w_fe = rng.normal(size=(config["d_input"], config["d_feature"]))
    phases = []
    for k in range(config["n_phases"]):
        train_x, train_y, test_x, test_y = [], [], [], []
        for local_label in range(config["classes_per_phase"]):
            label = k * config["classes_per_phase"] + local_label
            center = rng.normal(0, 2, size=config["d_input"])
            train_x.append(center + rng.normal(size=(config["n_train_per_class"], config["d_input"])))
            test_x.append(center + rng.normal(size=(config["n_test_per_class"], config["d_input"])))
            train_y.extend([label] * config["n_train_per_class"])
            test_y.extend([label] * config["n_test_per_class"])
        phases.append((
            np.maximum(np.vstack(train_x) @ w_fe, 0.0), np.array(train_y, dtype=np.int64),
            np.maximum(np.vstack(test_x) @ w_fe, 0.0), np.array(test_y, dtype=np.int64),
        ))
    return phases


def run_verification(config):
    """Chạy kiểm chứng tương đương số học giữa joint ridge, cập nhật theo phase và minibatch.

    Tham số:
        config: Dict cấu hình bài toán kiểm chứng gồm kích thước chiều, gamma và ngưỡng sai số ``rtol``/``atol``.

    Trả về:
        Dict báo cáo kết quả kiểm chứng từng phase (chưa log W&B); raise nếu sai lệch vượt ngưỡng.
    """
    config = dict(config)
    phases = make_synthetic_phases(config)
    r = w = r_batch = w_batch = None
    rows = []
    history_h, history_y, history_test_h, history_test_y = [], [], [], []
    for k, (h, labels, h_test_new, labels_test_new) in enumerate(phases):
        count = (k + 1) * config["classes_per_phase"]
        if r is None:
            r, w = initialize(h, labels, gamma=config["gamma"], n_classes=count)
        else:
            r, w = incremental_update(r, w, h, labels, n_classes=count)
        for offset in range(0, len(labels), config["minibatch_size"]):
            end = offset + config["minibatch_size"]
            if r_batch is None:
                r_batch, w_batch = initialize(h[offset:end], labels[offset:end], gamma=config["gamma"], n_classes=count)
            else:
                r_batch, w_batch = incremental_update(r_batch, w_batch, h[offset:end], labels[offset:end], n_classes=count)
            if w_batch.shape != (config["d_feature"], count):
                raise AssertionError("Số cột W không khớp số lớp của phase khi chia batch")

        # Lịch sử chỉ nằm trong bộ kiểm tra để xây nghiệm joint độc lập.
        history_h.append(h)
        history_y.append(labels)
        history_test_h.append(h_test_new)
        history_test_y.append(labels_test_new)
        pooled_h, pooled_y = np.vstack(history_h), np.concatenate(history_y)
        reference_w = joint_ridge(pooled_h, pooled_y, n_classes=count, gamma=config["gamma"])
        identity = np.eye(config["d_feature"])
        reference_r = np.linalg.solve(pooled_h.T @ pooled_h + config["gamma"] * identity, identity)
        h_test, y_test = np.vstack(history_test_h), np.concatenate(history_test_y)
        scores, scores_batch, scores_joint = h_test @ w, h_test @ w_batch, h_test @ reference_w
        for actual, expected in (
            (w, reference_w), (r, reference_r), (scores, scores_joint),
            (w_batch, reference_w), (r_batch, reference_r), (scores_batch, scores_joint),
        ):
            np.testing.assert_allclose(actual, expected, rtol=config["rtol"], atol=config["atol"])
        np.testing.assert_array_equal(scores.argmax(1), scores_joint.argmax(1))
        np.testing.assert_array_equal(scores_batch.argmax(1), scores_joint.argmax(1))
        if r.shape != (config["d_feature"], config["d_feature"]) or w.shape != (config["d_feature"], count):
            raise AssertionError("Kích thước trạng thái không hợp lệ")
        denominator = max(float(np.linalg.norm(reference_w)), np.finfo(np.float64).eps)
        rows.append({
            "phase": k, "n_classes": count, "weights_shape": list(w.shape),
            "inverse_gram_shape": list(r.shape),
            "max_abs_w_error": float(np.abs(w - reference_w).max()),
            "relative_w_error": float(np.linalg.norm(w - reference_w) / denominator),
            "max_abs_r_error": float(np.abs(r - reference_r).max()),
            "max_abs_score_error": float(np.abs(scores - scores_joint).max()),
            "prediction_disagreements": int(np.count_nonzero(scores.argmax(1) != scores_joint.argmax(1))),
            "minibatch_max_abs_w_error": float(np.abs(w_batch - reference_w).max()),
            "minibatch_relative_w_error": float(np.linalg.norm(w_batch - reference_w) / denominator),
            "minibatch_max_abs_r_error": float(np.abs(r_batch - reference_r).max()),
            "minibatch_max_abs_score_error": float(np.abs(scores_batch - scores_joint).max()),
            "minibatch_prediction_disagreements": int(np.count_nonzero(scores_batch.argmax(1) != scores_joint.argmax(1))),
            "test_accuracy": float(np.mean(scores.argmax(1) == y_test)), "checks_passed": True,
        })
    return {
        "experiment": config["experiment"],
        "description": "Joint, whole-phase and minibatch ridge on synthetic fixed features",
        "config": {**config, "dtype": "float64"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "results": rows,
        "verification_passed": bool(rows) and all(row["checks_passed"] for row in rows),
        "wandb_status": "not_logged", "wandb_run_id": None, "wandb_run_url": None,
    }
