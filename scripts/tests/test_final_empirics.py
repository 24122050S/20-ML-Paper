"""Kiểm thử đơn vị cho hai runner thực nghiệm cuối kỳ E1 và E2 trước khi chạy đầy đủ."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import e1_base_assumption_runner as e1
from scripts import e2_memory_precision_runner as e2
from src.analytic import joint_ridge
from src.cifar_data import CifarArrays, make_protocol, sha256_file
from src.reporting import save_json
from src.streaming import StreamingHead


class _FakeWandbRun:
    """Đối tượng W&B Run giả lập phục vụ kiểm tra nhãn (tags), config và summary."""

    def __init__(self, record: dict):
        """Khởi tạo run giả lập gắn với bản ghi lưu trữ trong bộ nhớ.

        Tham số:
            record: Dict lưu thông tin khởi tạo, lịch sử ``logged_steps`` và ``summary``.

        Trả về:
            None.
        """
        self.record = record
        self.id = f"fake-{len(record['all_runs']):03d}"
        self.url = f"https://wandb.ai/hsangg-2006/acil-reproduction/runs/{self.id}"
        self.summary: dict = {}

    def __enter__(self) -> "_FakeWandbRun":
        """Đi vào context manager của run giả lập.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            Chính đối tượng ``_FakeWandbRun``.
        """
        return self

    def __exit__(self, *_args) -> bool:
        """Đóng run giả lập và lưu bản chụp ``summary`` vào bản ghi kiểm thử.

        Tham số:
            *_args: Thông tin ngoại lệ tiêu chuẩn của context manager.

        Trả về:
            ``False`` để không chặn ngoại lệ nếu có.
        """
        self.record["summary"] = dict(self.summary)
        self.record["finished"] = True
        return False

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Ghi nhận một bước log số liệu theo phase.

        Tham số:
            metrics: Dict các chỉ số số học tại bước hiện tại.
            step: Chỉ số bước/phase.

        Trả về:
            None.
        """
        self.record["logged_steps"].append((step, dict(metrics)))


class _FakeWandbClient:
    """SDK W&B giả lập thu thập toàn bộ các lượt ``init`` để kiểm tra tách biệt tag cuối kỳ."""

    def __init__(self):
        """Khởi tạo danh sách rỗng chứa các run đã gọi ``init``.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        self.runs: list[dict] = []

    def init(self, **kwargs) -> _FakeWandbRun:
        """Tạo một ``_FakeWandbRun`` mới và lưu toàn bộ đối số ``kwargs`` để kiểm tra.

        Tham số:
            **kwargs: Các đối số cấu hình W&B như ``entity``, ``project``, ``tags``, ``mode``...

        Trả về:
            Đối tượng ``_FakeWandbRun`` dùng trong khối ``with``.
        """
        record = {
            "kwargs": kwargs,
            "logged_steps": [],
            "summary": {},
            "finished": False,
            "all_runs": self.runs,
        }
        self.runs.append(record)
        return _FakeWandbRun(record)


def _make_synthetic_cifar(seed: int = 7, *, train_per_class: int = 2, test_per_class: int = 2) -> CifarArrays:
    """Tạo đối tượng ``CifarArrays`` giả lập kích thước nhỏ, cân bằng đủ 100 lớp cho unit test.

    Tham số:
        seed: Hạt giống sinh ảnh uint8 ngẫu nhiên.
        train_per_class: Số mẫu train mỗi lớp.
        test_per_class: Số mẫu test mỗi lớp.

    Trả về:
        Đối tượng ``CifarArrays`` hợp lệ cho ``audit_e1_data`` và ``run_e1_case``.
    """
    rng = np.random.default_rng(seed)
    n_train = 100 * train_per_class
    n_test = 100 * test_per_class
    train_images = rng.integers(0, 256, size=(n_train, 32, 32, 3), dtype=np.uint8)
    test_images = rng.integers(0, 256, size=(n_test, 32, 32, 3), dtype=np.uint8)
    train_labels = np.repeat(np.arange(100, dtype=np.int64), train_per_class)
    test_labels = np.repeat(np.arange(100, dtype=np.int64), test_per_class)
    names = tuple(f"class_{idx:02d}" for idx in range(100))
    source = {"dataset_id": "synthetic/cifar100-unit-test", "revision": "unit-test-rev", "files": {}}
    return CifarArrays(train_images, train_labels, test_images, test_labels, names, source)


class FinalEmpiricsTests(unittest.TestCase):
    """Bộ kiểm thử toàn diện cho hai runner thực nghiệm cuối kỳ E1 và E2."""

    def setUp(self) -> None:
        """Tạo thư mục tạm thời riêng cho từng ca kiểm thử.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_coarse_mapping_bijection_and_parquet_verification(self) -> None:
        """Kiểm tra bảng ánh xạ 100 fine -> 20 coarse và đối chiếu tệp Parquet nếu có sẵn.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        self.assertEqual(len(e1.CANONICAL_FINE_TO_COARSE), 100)
        counts = np.bincount(np.asarray(e1.CANONICAL_FINE_TO_COARSE), minlength=20)
        self.assertTrue(np.all(counts == 5))
        self.assertEqual(set(e1.NATURAL_SUPERCLASSES) & set(e1.SHIFTED_SUPERCLASSES), set())
        self.assertEqual(set(e1.NATURAL_SUPERCLASSES) | set(e1.SHIFTED_SUPERCLASSES), set(range(20)))
        workspace_data = e1.ROOT.parent / "data"
        verified = e1.verify_or_load_coarse_map(workspace_data if workspace_data.is_dir() else None)
        self.assertEqual(verified, e1.CANONICAL_FINE_TO_COARSE)

    def test_e1a_protocol_invariants_and_exact_r1_identity(self) -> None:
        """Xác nhận E1a giữ cố định class_order trên cùng seed và trùng khít R1 tại c_0 = 50.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        r1_proto = make_protocol(seed=42, n_incremental_phases=5)
        e1_c50 = e1.make_e1_protocol(seed=42, base_classes=50, phase_size=10, split_mode="uniform")
        for field in (
            "class_order",
            "original_to_global",
            "base_classes",
            "incremental_phases",
            "phase_global_classes",
            "phase_original_classes",
        ):
            self.assertEqual(e1_c50[field], r1_proto[field])

        for base_c, expected_k in ((10, 9), (20, 8), (30, 7), (50, 5)):
            proto = e1.make_e1_protocol(seed=42, base_classes=base_c, phase_size=10, split_mode="uniform")
            self.assertEqual(proto["class_order"], r1_proto["class_order"])
            self.assertEqual(proto["incremental_phases"], expected_k)
            self.assertEqual(len(proto["phase_global_classes"]), expected_k + 1)
            self.assertEqual(proto["phase_global_classes"][0], list(range(base_c)))
            merged = [cls for group in proto["phase_global_classes"] for cls in group]
            self.assertEqual(merged, list(range(100)))

    def test_e1_base_only_normalization_blocks_future_and_test_leakage(self) -> None:
        """Đảm bảo mean/std tiền xử lý ở E1a chỉ tính từ mẫu train của c_0 lớp Base, không rò rỉ.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        arrays_a = _make_synthetic_cifar(seed=11, train_per_class=3, test_per_class=2)
        proto = e1.make_e1_protocol(seed=42, base_classes=10, phase_size=10, split_mode="uniform")
        mapping = np.asarray(proto["original_to_global"], dtype=np.int64)

        train_b = arrays_a.train_images.copy()
        future_mask = mapping[arrays.train_labels] >= 10 if (arrays := arrays_a) else None
        train_b[future_mask] = 255 - train_b[future_mask]
        test_b = 255 - arrays_a.test_images.copy()
        arrays_b = CifarArrays(
            train_b, arrays_a.train_labels.copy(), test_b, arrays_a.test_labels.copy(),
            arrays_a.class_names, arrays_a.source,
        )

        audit_a = e1.audit_e1_data(arrays_a, proto, train_per_class=3, test_per_class=2)
        audit_b = e1.audit_e1_data(arrays_b, proto, train_per_class=3, test_per_class=2)
        self.assertEqual(audit_a["normalization"]["samples"], 30)
        self.assertEqual(audit_a["normalization"], audit_b["normalization"])

    def test_e1b_superclass_disjoint_vs_stratified_partitions(self) -> None:
        """Kiểm tra tính rời nhau hoàn toàn của superclass_disjoint và độ phủ 20/20 của stratified.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        disjoint = e1.make_e1_protocol(seed=42, base_classes=50, phase_size=10, split_mode="superclass_disjoint")
        reverse = e1.make_e1_protocol(
            seed=42, base_classes=50, phase_size=10, split_mode="superclass_disjoint_reverse"
        )
        stratified = e1.make_e1_protocol(seed=42, base_classes=50, phase_size=10, split_mode="stratified")

        self.assertEqual(disjoint["shared_superclass_count"], 0)
        self.assertEqual(disjoint["base_superclasses"], list(e1.NATURAL_SUPERCLASSES))
        self.assertEqual(disjoint["incremental_superclasses"], list(e1.SHIFTED_SUPERCLASSES))

        self.assertEqual(reverse["shared_superclass_count"], 0)
        self.assertEqual(reverse["base_superclasses"], list(e1.SHIFTED_SUPERCLASSES))
        self.assertEqual(reverse["incremental_superclasses"], list(e1.NATURAL_SUPERCLASSES))

        self.assertEqual(stratified["shared_superclass_count"], 20)
        self.assertEqual(len(stratified["base_superclasses"]), 20)
        self.assertEqual(len(stratified["incremental_superclasses"]), 20)

    def test_e1_end_to_end_smoke_theorem_3_1_and_wandb_tags(self) -> None:
        """Chạy trọn vòng pipeline E1 thu nhỏ, xác minh khớp Joint Oracle và tag W&B cuối kỳ.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        synth = _make_synthetic_cifar(seed=21, train_per_class=2, test_per_class=2)
        fake_wandb = _FakeWandbClient()
        report_paths = []

        for seed in (42, 43):
            cfg = e1.default_e1_config(
                branch="E1a", seed=seed, base_classes=20, phase_size=40, split_mode="uniform"
            )
            cfg.update(
                epochs=1,
                batch_size=16,
                eval_batch_size=32,
                analytic_batch_size=13,
                base_statistics_batch_size=16,
                expansion_dim=24,
                torch_threads=1,
                device="cpu",
                train_per_class=2,
                test_per_class=2,
            )
            out_dir = self.root / f"e1a_seed{seed}"
            rep = e1.run_e1_case(
                cfg,
                data_dir=self.root / "unused_data",
                output_dir=out_dir,
                arrays_override=synth,
                verify_joint_oracle=True,
                wandb_mode="online",
                wandb_client=fake_wandb,
            )
            self.assertEqual(rep["status"], "complete")
            self.assertTrue(rep["verification_passed"])
            self.assertTrue(rep["checks"]["theorem_3_1_joint_oracle_matched"])
            self.assertEqual(rep["backbone_sha256_before"], rep["backbone_sha256_after"])
            self.assertEqual(rep["metric_summary"]["final_n_classes"], 100)
            self.assertIsNotNone(rep["metric_summary"]["final_inc_accuracy_pct"])
            self.assertIsNotNone(rep["metric_summary"]["final_base_minus_inc_gap_pp"])
            report_paths.append(out_dir / "results.json")

        self.assertEqual(len(fake_wandb.runs), 2)
        tags = fake_wandb.runs[0]["kwargs"]["tags"]
        self.assertIn("final-empirics", tags)
        self.assertIn("P1-base-assumption", tags)
        self.assertIn("E1a-base-ratio", tags)

        agg = e1.aggregate_e1_reports(report_paths, self.root / "e1_summary.json")
        self.assertIn("E1a__uniform__c0_20", agg["groups"])
        self.assertEqual(agg["groups"]["E1a__uniform__c0_20"]["seeds"], [42, 43])

    def test_e2a_iso_byte_ceilings_and_exact_nbytes_accounting(self) -> None:
        """Kiểm tra nghiệm nguyên d_max (798 ở fp64, 1160 ở fp32) và công thức state_bytes.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        d_fp64 = e2.max_iso_byte_dimension(bytes_per_elem=8)
        d_fp32 = e2.max_iso_byte_dimension(bytes_per_elem=4)
        self.assertEqual(d_fp64, 798)
        self.assertEqual(d_fp32, 1160)

        self.assertLessEqual(e2.compute_state_memory_budget(798)["bytes_fp64"], 6_144_000)
        self.assertGreater(e2.compute_state_memory_budget(799)["bytes_fp64"], 6_144_000)
        self.assertLessEqual(e2.compute_state_memory_budget(1160)["bytes_fp32"], 6_144_000)
        self.assertGreater(e2.compute_state_memory_budget(1161)["bytes_fp32"], 6_144_000)
        self.assertEqual(e2.compute_state_memory_budget(8000)["bytes_fp64"], 522_496_000)
        self.assertEqual(
            e2.compute_state_memory_budget(64, feature_expansion=False)["bytes_fp64"], 83_968
        )

        rng = np.random.default_rng(12)
        train_x = rng.normal(size=(200, 64))
        train_y = np.repeat(np.arange(100, dtype=np.int64), 2)
        test_x = rng.normal(size=(200, 64))
        test_y = np.repeat(np.arange(100, dtype=np.int64), 2)

        original_eval = e2.evaluate_seen_classes

        def checked_eval(*args, **kwargs):
            """Bọc hàm đánh giá để kiểm tra LearningWindow.active luôn tắt khi tính test.

            Tham số:
                *args: Đối số vị trí chuyển tiếp cho ``evaluate_seen_classes``.
                **kwargs: Đối số từ khóa chuyển tiếp cho ``evaluate_seen_classes``.

            Trả về:
                Dict kết quả đánh giá từ ``evaluate_seen_classes``.
            """
            self.assertFalse(e2.LearningWindow.active)
            return original_eval(*args, **kwargs)

        with mock.patch.object(e2, "evaluate_seen_classes", side_effect=checked_eval):
            for dim, fe_flag in ((64, False), (128, True)):
                res = e2.run_e2a_case(
                    seed=42,
                    dimension=dim,
                    train_features=train_x,
                    train_labels=train_y,
                    test_features=test_x,
                    test_labels=test_y,
                    feature_expansion=fe_flag,
                    batch_size=17,
                    threads=1,
                    train_per_class=2,
                    test_per_class=2,
                )
                self.assertTrue(res["verification_passed"])
                self.assertTrue(res["state_memory"]["formula_exact_match"])

    def test_fd2_block_diagonal_matches_dense_at_b1_and_reduces_r_memory_by_b(self) -> None:
        """Xác nhận FD2 tại B=1 trùng khít ACIL chuẩn và tại B>1 giảm bộ nhớ R đúng B lần.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        rng = np.random.default_rng(99)
        train_x = rng.normal(size=(200, 64))
        train_y = np.repeat(np.arange(100, dtype=np.int64), 2)
        test_x = rng.normal(size=(50, 64))
        phases = e2.slice_phase_arrays(train_x, train_y, base_classes=50, n_incremental_phases=5)

        dense = StreamingHead({"seed": 42, "expansion_dim": 32, "feature_expansion": True, "gamma": 0.1})
        block1 = e2.BlockDiagonalStreamingHead(seed=42, expansion_dim=32, num_blocks=1, gamma=0.1)
        block4 = e2.BlockDiagonalStreamingHead(seed=42, expansion_dim=32, num_blocks=4, gamma=0.1)

        for p_idx, (px, py, classes) in enumerate(phases):
            n_cls = max(classes) + 1
            if p_idx == 0:
                dense.fit_base(e2.iter_array_batches(px, py, 19), n_classes=n_cls)
                block1.fit_base(e2.iter_array_batches(px, py, 19), n_classes=n_cls)
                block4.fit_base(e2.iter_array_batches(px, py, 19), n_classes=n_cls)
            else:
                dense.update_phase(e2.iter_array_batches(px, py, 19), n_classes=n_cls)
                block1.update_phase(e2.iter_array_batches(px, py, 19), n_classes=n_cls)
                block4.update_phase(e2.iter_array_batches(px, py, 19), n_classes=n_cls)

        np.testing.assert_allclose(block1.r_blocks[0], dense.r, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(block1.w, dense.w, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(block1.score(test_x), dense.transform(test_x) @ dense.w, rtol=1e-12, atol=1e-12)
        self.assertEqual(block4.r_blocks.nbytes * 4, dense.r.nbytes)

        full_h = block4.transform(train_x)
        for blk_idx in range(4):
            h_sub = full_h[:, blk_idx * 8 : (blk_idx + 1) * 8]
            w_sub_ref = joint_ridge(h_sub, train_y, n_classes=100, gamma=0.1)
            w_sub_got = block4.w[blk_idx * 8 : (blk_idx + 1) * 8, :]
            np.testing.assert_allclose(w_sub_got, w_sub_ref, rtol=1e-11, atol=1e-11)

    def test_e2_checkpoint_sha_guard_refuses_missing_or_corrupt_checkpoint(self) -> None:
        """Đảm bảo E2 kiểm tra chặt SHA-256 của R1, khớp backbone thật nếu có local và chặn file giả.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        import torch
        from src.cifar_smoke import fingerprint
        from src.resnet import ResNet32, freeze_backbone

        ref42 = e2.load_r1_seed_reference(42)
        self.assertEqual(len(ref42["checkpoint_sha256"]), 64)
        bogus = self.root / "fake_checkpoint.pt"
        bogus.write_bytes(b"not-a-valid-r1-checkpoint")
        with self.assertRaises(ValueError):
            e2.find_verified_r1_checkpoint(42, ref42["checkpoint_sha256"], checkpoint_path=bogus)

        local_dir = e2.ROOT / "reports/local"
        if local_dir.is_dir():
            try:
                real_ckpt = e2.find_verified_r1_checkpoint(42, ref42["checkpoint_sha256"])
            except ValueError:
                real_ckpt = None
            if real_ckpt is not None:
                saved = torch.load(real_ckpt, map_location="cpu", weights_only=True)
                model = ResNet32(n_classes=50)
                model.load_state_dict(saved["model_state_dict"])
                freeze_backbone(model)
                self.assertEqual(fingerprint(model), ref42["backbone_sha256"])

    def test_e2b_strict_float32_vs_float64_woodbury_drift_and_no_silent_upcast(self) -> None:
        """Kiểm tra E2b giữ đúng kiểu float32/float64 và phát hiện độ trôi số học ở gamma nhỏ.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        rng = np.random.default_rng(123)
        basis = rng.normal(size=(300, 8))
        proj = rng.normal(size=(8, 64))
        train_x = basis @ proj + 1e-4 * rng.normal(size=(300, 64))
        train_y = np.repeat(np.arange(100, dtype=np.int64), 3)
        test_x = rng.normal(size=(200, 64))
        test_y = np.repeat(np.arange(100, dtype=np.int64), 2)

        res64 = e2.run_typed_woodbury_case(
            seed=42,
            dimension=48,
            gamma=1e-4,
            dtype_name="float64",
            n_incremental_phases=25,
            batch_size=3,
            train_features=train_x,
            train_labels=train_y,
            test_features=test_x,
            test_labels=test_y,
            threads=1,
            compare_all_phases=True,
        )
        res32 = e2.run_typed_woodbury_case(
            seed=42,
            dimension=48,
            gamma=1e-4,
            dtype_name="float32",
            n_incremental_phases=25,
            batch_size=3,
            train_features=train_x,
            train_labels=train_y,
            test_features=test_x,
            test_labels=test_y,
            threads=1,
            compare_all_phases=True,
        )
        self.assertTrue(all(item["dtype_verified"] == "float64" for item in res64["comparisons"]))
        self.assertFalse(res64["summary"]["numerical_breakdown"])
        self.assertLess(res64["summary"]["max_relative_w_error"], 1e-8)

        if not res32["summary"]["numerical_breakdown"]:
            self.assertTrue(all(item["dtype_verified"] == "float32" for item in res32["comparisons"]))
            self.assertGreater(
                res32["summary"]["max_relative_w_error"],
                res64["summary"]["max_relative_w_error"] * 1e3,
            )

    def test_e2_seed_suite_with_prebuilt_fixture_and_aggregation(self) -> None:
        """Kiểm tra điều phối trọn gói run_e2_seed_suite, nhãn W&B P2 và gộp kết quả E2.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        ref42 = e2.load_r1_seed_reference(42)
        fixture = self.root / "fixture_seed42"
        fixture.mkdir(parents=True)
        rng = np.random.default_rng(77)
        train_x = rng.normal(size=(50000, 64))
        train_y = np.repeat(np.arange(100, dtype=np.int64), 500)
        test_x = rng.normal(size=(10000, 64))
        test_y = np.repeat(np.arange(100, dtype=np.int64), 100)

        files = {}
        for fname, arr in (
            ("train_features.npy", train_x),
            ("train_labels.npy", train_y),
            ("test_features.npy", test_x),
            ("test_labels.npy", test_y),
        ):
            np.save(fixture / fname, arr, allow_pickle=False)
            files[fname] = sha256_file(fixture / fname)
        save_json(
            {
                "scope": "synthetic_e2_unit_test",
                "seed": 42,
                "checkpoint_sha256": ref42["checkpoint_sha256"],
                "backbone_sha256": ref42["backbone_sha256"],
                "baseline_wandb_run_id": ref42["wandb_run_id"],
                "baseline_metrics": ref42["baseline_metrics"],
                "train_samples": 50000,
                "test_samples": 10000,
                "train_per_class": 500,
                "test_per_class": 100,
                "feature_extraction_seconds_excluded": 0.0,
                "training_executed_this_run": False,
                "files": files,
            },
            fixture / "fixture.json",
        )

        fake_wandb = _FakeWandbClient()
        out_dir = self.root / "e2_suite_out"
        suite_report = e2.run_e2_seed_suite(
            seed=42,
            output_dir=out_dir,
            fixture_dir=fixture,
            e2a_dimensions=(64, 96),
            include_fd2=False,
            e2b_dimension=32,
            e2b_gammas=(0.1,),
            e2b_regimes=((5, 512),),
            threads=1,
            wandb_mode="offline",
            wandb_client=fake_wandb,
        )
        self.assertEqual(suite_report["status"], "complete")
        self.assertTrue(suite_report["verification_passed"])
        self.assertEqual(len(fake_wandb.runs), 2)
        for rec in fake_wandb.runs:
            self.assertIn("final-empirics", rec["kwargs"]["tags"])
            self.assertIn("P2-memory-precision", rec["kwargs"]["tags"])
            self.assertIn("P2-memory-scaling", rec["kwargs"]["tags"])

        agg = e2.aggregate_e2_reports([out_dir / "results.json"], self.root / "e2_summary.json")
        self.assertEqual(agg["iso_byte_anchors"]["d_max_fp64"], 798)
        self.assertEqual(agg["iso_byte_anchors"]["d_max_fp32"], 1160)


if __name__ == "__main__":
    unittest.main()
