"""Xuất metric v1.0 từ các hàng phase; không lấy final thay A-bar."""

import numpy as np


def summarize(rows, config):
    """Tổng hợp metric đánh giá ACIL theo chuẩn v1.0 từ kết quả từng phase.

    Tính accuracy trung bình tăng dần (gồm phase base), final accuracy và
    base forgetting; đặt ``average_incremental_accuracy_pct=None`` nếu chưa đủ K phase.

    Tham số:
        rows: Danh sách dict kết quả theo thứ tự phase ``0..requested``.
        config: Dict cấu hình thí nghiệm chứa ``seed``, ``incremental_phases``,
            ``run_incremental_phases`` và ``mode``.

    Trả về:
        Dict chứa bộ metric chuẩn hóa ``ACIL-metrics-v1.0``.
    """
    if not rows or [r["phase"] for r in rows] != list(range(len(rows))):
        raise ValueError("Cần các phase liên tiếp bắt đầu từ base0")
    k = config["incremental_phases"]
    requested = config["run_incremental_phases"]
    if len(rows) != requested + 1 or not 1 <= requested <= k or 50 % k:
        raise ValueError("Số phase không khớp config")
    for phase, row in enumerate(rows):
        classes = 50 + (50 // k) * phase
        if row["n_classes"] != classes or row["test_samples_all_seen"] != 100 * classes:
            raise ValueError("Sai số lớp hoặc số ảnh test seen")
        if not all(np.isfinite(row[key]) and 0 <= row[key] <= 100
                   for key in ("accuracy_pct", "base_accuracy_pct")):
            raise ValueError("Accuracy phải hữu hạn trong[0,100]")
        if row.get("checks_passed") is not True:
            raise ValueError("Không tổng hợp phase chưa đạt checks")
    average = float(np.mean([row["accuracy_pct"] for row in rows]))
    delta = float(rows[-1]["base_accuracy_pct"] - rows[0]["base_accuracy_pct"])
    complete = requested == k
    return {"protocol_version": "ACIL-C100-v1.0", "metric_version": "ACIL-metrics-v1.0",
            "mode": config.get("mode", "smoke"), "seed": config["seed"],
            "n_incremental_phases": k, "n_evaluations": len(rows),
            "complete_protocol": complete,
            "average_incremental_accuracy_pct": average if complete else None,
            "partial_average_accuracy_pct": None if complete else average,
            "final_accuracy_pct": rows[-1]["accuracy_pct"], "final_n_classes": rows[-1]["n_classes"],
            "base_forgetting_pp": -delta, "base_accuracy_delta_pp": delta,
            "n_completed_seeds": 1, "between_seed_sd": None}
