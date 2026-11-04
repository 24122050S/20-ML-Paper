"""Analytic theo batch cho benchmark: không giữ feature lịch sử hoặc joint oracle."""

from __future__ import annotations

import time
import numpy as np

from .analytic import initialize_batches, incremental_update


class StreamingHead:
    """Nhận config, giữ FE/R/W và bộ đếm; đầu vào của mỗi phương thức là iterator F64/y."""

    def __init__(self, config):
        self.expansion_enabled = config.get("feature_expansion", True)
        self.dimension = config["expansion_dim"] if self.expansion_enabled else 64
        self.expansion = (np.random.default_rng(config["seed"] + 1000).normal(size=(64, self.dimension))
                          if self.expansion_enabled else None)
        self.gamma = config["gamma"]
        self.r = self.w = None
        self.timings = {}
        self.maximum_feature_batch_rows = 0

    def transform(self, raw):
        """Nhận F(n,64), trả H float64; no-FE dùng F trực tiếp, không random projection64."""
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 64 or not np.isfinite(raw).all():
            raise ValueError("F phải hữu hạn và có shape(n,64)")
        self.maximum_feature_batch_rows = max(self.maximum_feature_batch_rows, len(raw))
        return np.maximum(raw @ self.expansion, 0.0) if self.expansion_enabled else raw

    def fit_base(self, batches, n_classes=50):
        """Đọc base đúng một lượt, tích lũy Gram/cross rồi khởi tạo R/W; trả số mẫu."""
        transformed = ((self.transform(raw), labels) for raw, labels in batches)
        self.r, self.w = initialize_batches(transformed, self.dimension, n_classes,
                                             self.gamma, timings=self.timings)
        return self.timings["base_samples"]

    def update_phase(self, batches, n_classes):
        """Đọc duy nhất mẫu phase hiện tại, trả (số mẫu,số batch); không lưu H/y."""
        if self.r is None:
            raise ValueError("Cần fit_base trước incremental")
        count = batch_count = 0
        start = time.perf_counter()
        for raw, labels in batches:
            h = self.transform(raw)
            self.r, self.w = incremental_update(self.r, self.w, h, labels, n_classes=n_classes)
            count += len(h)
            batch_count += 1
        if not count:
            raise ValueError("Phase không được rỗng")
        self.timings["last_incremental_seconds"] = time.perf_counter() - start
        return count, batch_count

    def evaluate(self, batches, n_classes, new_classes, *, expected_per_class=None):
        """Chấm từng batch trên toàn bộ cột seen; trả accuracy/base/new và số ảnh."""
        if self.w is None or self.w.shape != (self.dimension, n_classes):
            raise ValueError("W không khớp số lớp cần đánh giá")
        counts = np.zeros(n_classes, dtype=np.int64)
        correct = total = base_correct = base_total = new_correct = new_total = 0
        for raw, labels in batches:
            labels = np.asarray(labels)
            if labels.ndim != 1 or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= n_classes):
                raise ValueError("Nhãn test phải thuộc lớp đã thấy")
            scores = self.transform(raw) @ self.w
            if len(scores) != len(labels) or not np.isfinite(scores).all():
                raise FloatingPointError("Scores test sai kích thước hoặc không hữu hạn")
            matched = scores.argmax(1) == labels
            base = labels < 50
            new = np.isin(labels, new_classes)
            counts += np.bincount(labels, minlength=n_classes)
            total += len(labels)
            correct += int(matched.sum())
            base_correct += int(matched[base].sum())
            base_total += int(base.sum())
            new_correct += int(matched[new].sum())
            new_total += int(new.sum())
        if np.any(counts == 0) or (expected_per_class is not None and np.any(counts != expected_per_class)):
            raise AssertionError("Evaluation phải phủ đúng tất cả lớp đã thấy")
        return {"test_samples_all_seen": total, "accuracy_pct": 100*correct/total,
                "base_accuracy_pct": 100*base_correct/base_total,
                "new_class_accuracy_pct": None if n_classes == 50 else 100*new_correct/new_total}

    def state_bytes(self):
        """Trả bytes R/W/FE thực có; không phải peak RSS hoặc bộ nhớ CNN."""
        return sum(a.nbytes for a in (self.r, self.w, self.expansion) if a is not None)


def run_streaming_phases(config, protocol, train_batches, test_batches, *, progress):
    """Nhận factories theo phase, trả head và rows; factories không được trả lịch sử train."""
    head = StreamingHead(config)
    rows = []
    for phase in range(config["run_incremental_phases"] + 1):
        progress("analytic_streaming", phase=phase)
        classes = protocol["phase_global_classes"][phase]
        n_classes = max(classes) + 1
        if phase == 0:
            count = head.fit_base(train_batches(phase), n_classes)
            batches = None  # Gram accumulation không phải RLS incremental minibatch.
        else:
            count, batches = head.update_phase(train_batches(phase), n_classes)
        expected_train = len(classes) * 500
        if count != expected_train:
            raise AssertionError("Sai số ảnh train phase")
        evaluation = head.evaluate(test_batches(n_classes), n_classes, classes, expected_per_class=100)
        rows.append({"phase": phase, "n_classes": n_classes, "weights_shape": list(head.w.shape),
                     "inverse_gram_shape": list(head.r.shape), "train_samples_current_phase": count,
                     "analytic_batches": batches, "relative_w_error": None, "max_abs_r_error": None,
                     "max_abs_score_error": None, "prediction_disagreements": None,
                     "joint_reference_status": "not_run", "checks_passed": True, **evaluation})
        print(f"Benchmark phase {phase}: W={head.w.shape}, test={evaluation['test_samples_all_seen']}, accuracy={evaluation['accuracy_pct']:.2f}%", flush=True)
    return head, rows
