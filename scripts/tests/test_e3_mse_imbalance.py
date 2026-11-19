"""Bộ kiểm thử đơn vị cho thực nghiệm phụ cuối kỳ E3 (MSE Class-Imbalance & FD3-WRLS)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.e3_mse_imbalance_runner import (
    aggregate_e3_reports,
    run_e3_imbalance_case,
    run_e3_seed_suite,
    sample_imbalanced_incremental_indices,
    weighted_incremental_update,
    weighted_joint_ridge,
)
from src.analytic import incremental_update, initialize, initialize_batches, joint_ridge


class _DummyWandbRun:
    """Đối tượng giả lập context run của W&B để kiểm tra cấu hình, tags và summary E3."""

    def __init__(self, run_id: str, url: str) -> None:
        """Khởi tạo run giả lập với ID và URL xác định.

        Tham số:
            run_id: Mã định danh run giả lập.
            url: Đường dẫn URL giả lập của run.

        Trả về:
            None.
        """
        self.id = run_id
        self.url = url
        self.logged: list[tuple[int | None, dict]] = []
        self.summary: dict = {}

    def __enter__(self) -> "_DummyWandbRun":
        """Mở context manager cho run W&B giả lập.

        Tham số:
            Không nhận tham số.

        Trả về:
            Chính đối tượng ``_DummyWandbRun`` hiện tại.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Đóng context manager của run W&B giả lập.

        Tham số:
            exc_type: Kiểu ngoại lệ nếu phát sinh.
            exc: Giá trị ngoại lệ nếu phát sinh.
            tb: Traceback nếu phát sinh.

        Trả về:
            ``False`` để không nuốt ngoại lệ.
        """
        return False

    def log(self, data: dict, step: int | None = None) -> None:
        """Ghi lại bản ghi metric theo từng bước để kiểm tra trong unit test.

        Tham số:
            data: Dict metric của bước hiện tại.
            step: Chỉ số bước tùy chọn.

        Trả về:
            None.
        """
        self.logged.append((step, dict(data)))


class _DummyWandbClient:
    """Client giả lập SDK W&B dùng kiểm thử hàm ``log_e3_suite_wandb`` không cần mạng."""

    def __init__(self) -> None:
        """Khởi tạo danh sách lưu các lần gọi ``init``.

        Tham số:
            Không nhận tham số.

        Trả về:
            None.
        """
        self.inits: list[tuple[dict, _DummyWandbRun]] = []

    def init(self, **kwargs) -> _DummyWandbRun:
        """Tạo một ``_DummyWandbRun`` mới và lưu toàn bộ đối số cấu hình.

        Tham số:
            **kwargs: Các đối số truyền vào ``wandb.init``.

        Trả về:
            Đối tượng ``_DummyWandbRun`` vừa khởi tạo.
        """
        idx = len(self.inits) + 1
        run = _DummyWandbRun(f"e3mock{idx:02d}", f"https://wandb.example/runs/e3mock{idx:02d}")
        self.inits.append((kwargs, run))
        return run


class E3MseImbalanceTests(unittest.TestCase):
    """Kiểm thử đại số, lấy mẫu tất định và luồng tổng hợp của thực nghiệm phụ E3 & FD3."""

    def test_weighted_rls_matches_weighted_joint_and_reduces_to_standard_at_unit_weights(self) -> None:
        """Xác minh FD3-WRLS trùng khít ACIL chuẩn khi trọng số bằng 1 và khớp Weighted Joint ở trọng số bất kỳ.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        rng = np.random.default_rng(2026)
        h0 = rng.normal(size=(40, 18))
        y0 = rng.integers(0, 4, size=40, dtype=np.int64)
        r_std, w_std = initialize(h0, y0, gamma=0.25, n_classes=4)
        r_wrls, w_wrls = r_std.copy(), w_std.copy()

        h1 = rng.normal(size=(26, 18))
        y1 = rng.integers(4, 8, size=26, dtype=np.int64)
        unit_w = np.ones(len(y1), dtype=np.float64)

        r_u, w_u = weighted_incremental_update(r_wrls, w_wrls, h1, y1, unit_w, n_classes=8)
        r_s, w_s = incremental_update(r_std, w_std, h1, y1, n_classes=8)
        np.testing.assert_allclose(r_u, r_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(w_u, w_s, rtol=1e-12, atol=1e-12)

        # Kiểm tra nhiều minibatch với trọng số lớp không đồng nhất
        class_weights = {4: 10.0, 5: 5.0, 6: 2.5, 7: 20.0}
        sample_w1 = np.array([class_weights[int(c)] for c in y1], dtype=np.float64)
        for start in range(0, len(y1), 7):
            r_wrls, w_wrls = weighted_incremental_update(
                r_wrls,
                w_wrls,
                h1[start : start + 7],
                y1[start : start + 7],
                sample_w1[start : start + 7],
                n_classes=8,
            )

        h_all = np.concatenate([h0, h1], axis=0)
        y_all = np.concatenate([y0, y1], axis=0)
        w_all = np.concatenate([np.ones(len(y0), dtype=np.float64), sample_w1], axis=0)
        w_oracle = weighted_joint_ridge(h_all, y_all, w_all, n_classes=8, gamma=0.25)
        np.testing.assert_allclose(w_wrls, w_oracle, rtol=1e-11, atol=1e-11)

    def test_imbalance_collapses_standard_mse_and_fd3_recovers_on_synthetic_gaussian_classes(self) -> None:
        """Kiểm chứng cơ chế co chuẩn cột ||w_c||_2 khi mất cân bằng 25x và khả năng khôi phục của FD3-WRLS.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        rng = np.random.default_rng(99)
        d_cnn = 8
        d = 64
        base_classes = 6
        inc_classes = 4
        total_classes = base_classes + inc_classes
        prototypes = rng.normal(scale=0.85, size=(total_classes, d_cnn))
        w_fe = rng.normal(size=(d_cnn, d))

        n_base = 200
        n_inc = 4  # Tỷ lệ mất cân bằng 50x trên cùng không gian đặc trưng mở rộng
        train_h, train_y = [], []
        test_h, test_y = [], []
        for cls_id in range(total_classes):
            count_tr = n_base if cls_id < base_classes else n_inc
            f_tr = prototypes[cls_id] + 0.80 * rng.normal(size=(count_tr, d_cnn))
            f_te = prototypes[cls_id] + 0.80 * rng.normal(size=(40, d_cnn))
            train_h.append(np.maximum(0.0, f_tr @ w_fe))
            train_y.append(np.full(count_tr, cls_id, dtype=np.int64))
            test_h.append(np.maximum(0.0, f_te @ w_fe))
            test_y.append(np.full(40, cls_id, dtype=np.int64))

        h_tr = np.concatenate(train_h, axis=0)
        y_tr = np.concatenate(train_y, axis=0)
        h_te = np.concatenate(test_h, axis=0)
        y_te = np.concatenate(test_y, axis=0)

        m0 = y_tr < base_classes
        r0, w0 = initialize_batches([(h_tr[m0], y_tr[m0])], n_features=d, n_classes=base_classes, gamma=0.5)

        std_case = run_e3_imbalance_case(
            seed=42,
            variant="ACIL_standard",
            n_inc_per_class=n_inc,
            n_base_per_class=n_base,
            base_r0=r0,
            base_w0=w0,
            h_train=h_tr,
            train_labels=y_tr,
            h_test=h_te,
            test_labels=y_te,
            gamma=0.5,
            batch_size=8,
            base_classes=base_classes,
            n_incremental_phases=2,
            classes_per_phase=2,
            threads=1,
        )
        fd3_case = run_e3_imbalance_case(
            seed=42,
            variant="FD3_weighted_rls",
            n_inc_per_class=n_inc,
            n_base_per_class=n_base,
            base_r0=r0,
            base_w0=w0,
            h_train=h_tr,
            train_labels=y_tr,
            h_test=h_te,
            test_labels=y_te,
            gamma=0.5,
            batch_size=8,
            base_classes=base_classes,
            n_incremental_phases=2,
            classes_per_phase=2,
            threads=1,
        )

        self.assertTrue(std_case["verification_passed"])
        self.assertTrue(fd3_case["verification_passed"])
        self.assertLess(std_case["metric_summary"]["w_norm_inc_over_base_ratio"], 0.65)
        self.assertLess(
            std_case["metric_summary"]["w_norm_inc_over_base_ratio"],
            fd3_case["metric_summary"]["w_norm_inc_over_base_ratio"],
        )
        self.assertGreater(
            fd3_case["metric_summary"]["final_inc_accuracy_pct"],
            std_case["metric_summary"]["final_inc_accuracy_pct"] + 25.0,
        )

    def test_suite_runner_and_aggregation_preserve_wandb_tags_and_checks(self) -> None:
        """Kiểm tra trọn luồng ``run_e3_seed_suite``, lấy mẫu tất định và ``aggregate_e3_reports``.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        rng = np.random.default_rng(7)
        labels_full = np.repeat(np.arange(100, dtype=np.int64), 20)
        idx_a = sample_imbalanced_incremental_indices(labels_full, n_inc_per_class=5, seed=42)
        idx_b = sample_imbalanced_incremental_indices(labels_full, n_inc_per_class=5, seed=42)
        for arr_a, arr_b in zip(idx_a, idx_b):
            np.testing.assert_array_equal(arr_a, arr_b)
            self.assertEqual(len(arr_a), 50)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = []
            for seed in (42, 43):
                fake_meta = {
                    "seed": seed,
                    "train_per_class": 20,
                    "training_executed_this_run": False,
                }
                tr_x = rng.normal(size=(2000, 16)).astype(np.float32)
                tr_y = np.repeat(np.arange(100, dtype=np.int64), 20)
                te_x = rng.normal(size=(1000, 16)).astype(np.float32)
                te_y = np.repeat(np.arange(100, dtype=np.int64), 10)
                dummy_wb = _DummyWandbClient()

                with mock.patch(
                    "scripts.e3_mse_imbalance_runner.prepare_or_load_seed_fixture",
                    return_value=(fake_meta, tr_x, tr_y, te_x, te_y),
                ):
                    res = run_e3_seed_suite(
                        seed=seed,
                        output_dir=tmp_path / f"s{seed}",
                        dimension=32,
                        gamma=0.1,
                        n_inc_list=(20, 5),
                        batch_size=32,
                        threads=1,
                        wandb_mode="online",
                        wandb_client=dummy_wb,
                    )
                self.assertEqual(res["status"], "complete")
                self.assertTrue(res["verification_passed"])
                self.assertEqual(len(dummy_wb.inits), 1)
                init_kwargs, _ = dummy_wb.inits[0]
                self.assertIn("final-empirics", init_kwargs["tags"])
                self.assertIn("P3-mse-imbalance", init_kwargs["tags"])
                paths.append(Path(res["output_dir"]) / "results.json")

            agg_out = tmp_path / "summary.json"
            summary = aggregate_e3_reports(paths, output_path=agg_out, e2b_summary_path=None)
            self.assertEqual(summary["status"], "complete")
            self.assertIn("ACIL_standard__N5", summary["curves_summary"])
            self.assertIn("FD3_weighted_rls__N5", summary["curves_summary"])
            self.assertTrue(agg_out.is_file())


if __name__ == "__main__":
    unittest.main()
