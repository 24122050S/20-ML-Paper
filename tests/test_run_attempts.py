"""Hồi quy lỗi artifact cũ còn báo pass khi một lượt CIFAR mới thất bại."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HAS_CIFAR = all(importlib.util.find_spec(name) is not None for name in
                ("torch", "huggingface_hub", "pyarrow", "PIL", "threadpoolctl", "filelock"))
if HAS_CIFAR:
    import torch
    from src.cifar_smoke import run
    from src.run_attempt import RunAttempt


@unittest.skipUnless(HAS_CIFAR, "Cài requirements-cifar.txt để kiểm tra attempt CIFAR")
class RunAttemptTests(unittest.TestCase):
    def config(self, seed=42):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/cifar100_smoke.json").read_text(encoding="utf-8"))
        return {**config, "seed": seed, "device": "cpu", "torch_threads": 1}

    def report(self):
        return {"experiment": "test", "config": self.config(), "environment": {"numpy": "2.1.3"},
                "results": [{"phase": 0, "n_classes": 50}], "verification_passed": True,
                "wandb_status": "logged", "wandb_run_id": "old", "wandb_run_url": "https://old.invalid/run"}

    def read(self, root, name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    def assert_latest(self, root, status):
        latest, progress, result = (self.read(root, name) for name in
                                     ("latest.json", "progress.json", "results.json"))
        self.assertEqual({x["attempt_id"] for x in (latest, progress, result)}, {latest["attempt_id"]})
        self.assertTrue(all(x["status"] == status for x in (latest, progress, result)))
        self.assertEqual(result["verification_passed"], status == "complete")
        own = self.read(root, latest["results_path"])
        self.assertEqual(own, result)
        return result

    def fake_data(self):
        arrays = SimpleNamespace(source={"revision": "test-revision"})
        audit = {"passed": True, "normalization": {"mean": [0.5]*3, "std": [0.25]*3}}
        return mock.patch("src.cifar_smoke.load_cifar100", return_value=arrays), mock.patch("src.cifar_smoke.audit_data", return_value=audit)

    def test_old_success_then_missing_checkpoint_records_failed_seed43(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_bytes = (json.dumps(self.report(), indent=2) + "\r\n").encode()
            old_progress = b'{"stage":"complete","checks_passed":true}\r\n'
            (root / "results.json").write_bytes(old_bytes)
            (root / "progress.json").write_bytes(old_progress)
            (root / "base_checkpoint.pt").write_bytes(b"old checkpoint must stay unchanged")
            data, audit = self.fake_data()
            with data, audit, self.assertRaises(FileNotFoundError):
                run(self.config(43), data_dir=root / "data", output_dir=root,
                    reuse_checkpoint=root / "missing.pt")
            failed = self.assert_latest(root, "failed")
            self.assertEqual(failed["config"]["seed"], 43)
            self.assertEqual(failed["error_type"], "FileNotFoundError")
            self.assertEqual(failed["failed_stage"], "checkpoint_load")
            self.assertIsNone(failed["wandb_run_id"])
            self.assertIsNone(failed["wandb_run_url"])
            self.assertEqual(failed["results"], [])
            archive = list((root / "attempts").glob("legacy-*/results.json"))
            self.assertEqual(len(archive), 1)
            self.assertEqual(archive[0].read_bytes(), old_bytes)
            self.assertEqual((archive[0].parent / "progress.json").read_bytes(), old_progress)
            self.assertEqual((root / "base_checkpoint.pt").read_bytes(), b"old checkpoint must stay unchanged")

    def test_invalid_config_is_failed_before_any_data_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("src.cifar_smoke.load_cifar100") as load, self.assertRaises(ValueError):
                run({**self.config(), "gamma": 0}, data_dir=root / "data", output_dir=root)
            load.assert_not_called()
            failed = self.assert_latest(root, "failed")
            self.assertEqual(failed["failed_stage"], "validating")

    def test_data_failure_is_not_left_running_or_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("src.cifar_smoke.load_cifar100", side_effect=ConnectionError), self.assertRaises(ConnectionError):
                run(self.config(), data_dir=root / "data", output_dir=root)
            failed = self.assert_latest(root, "failed")
            self.assertEqual(failed["failed_stage"], "download_and_audit")

    def test_incompatible_checkpoint_is_failed_without_replacing_old_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with RunAttempt(root, seed=42) as previous:
                previous.complete(self.report())
            previous_bytes = (previous.directory / "results.json").read_bytes()
            checkpoint = root / "wrong-seed.pt"
            torch.save({"config": self.config(42)}, checkpoint)
            data, audit = self.fake_data()
            with data, audit, self.assertRaisesRegex(ValueError, "seed"):
                run(self.config(43), data_dir=root / "data", output_dir=root, reuse_checkpoint=checkpoint)
            self.assert_latest(root, "failed")
            self.assertEqual((previous.directory / "results.json").read_bytes(), previous_bytes)

    def test_successful_repeated_runs_get_distinct_artifacts_and_matching_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("src.cifar_smoke._compute_report", side_effect=lambda *a, **k: self.report()):
                first = run(self.config(), data_dir=root / "data", output_dir=root)
                first_path = Path(first["output_dir"]) / "results.json"
                first_bytes = first_path.read_bytes()
                second = run(self.config(), data_dir=root / "data", output_dir=root)
            self.assertNotEqual(first["attempt_id"], second["attempt_id"])
            self.assertNotEqual(first["output_dir"], second["output_dir"])
            self.assertEqual(first_path.read_bytes(), first_bytes)
            self.assertEqual(self.assert_latest(root, "complete"), second)
            self.assertEqual((root / "results.json").read_bytes(), (Path(second["output_dir"]) / "results.json").read_bytes())
            self.assertNotIn(b"\r\n", (root / "results.json").read_bytes())
            self.assertEqual(second["wandb_status"], "not_logged")
            self.assertIsNone(second["wandb_run_url"])

    def test_running_marker_is_visible_before_pipeline_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def computing(*args, **kwargs):
                latest, result = self.read(root, "latest.json"), self.read(root, "results.json")
                self.assertEqual(latest["status"], "running")
                self.assertEqual(result["status"], "running")
                self.assertFalse(result["verification_passed"])
                self.assertEqual(result["results"], [])
                self.assertEqual(result["attempt_id"], latest["attempt_id"])
                self.assertFalse((kwargs["output"] / "results.json").exists())
                return self.report()
            with mock.patch("src.cifar_smoke._compute_report", side_effect=computing):
                run(self.config(), data_dir=root / "data", output_dir=root)
            self.assert_latest(root, "complete")

    def test_mid_training_failure_keeps_stage_and_reraises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def computing(*args, **kwargs):
                kwargs["progress"]("base_training", epoch=1, batch=2)
                raise FloatingPointError("loss")
            with mock.patch("src.cifar_smoke._compute_report", side_effect=computing), self.assertRaises(FloatingPointError):
                run(self.config(), data_dir=root / "data", output_dir=root)
            self.assertEqual(self.assert_latest(root, "failed")["failed_stage"], "base_training")

    def test_wandb_finish_failure_cannot_publish_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class FinishFailure:
                id, url = "new-run", "https://wandb.ai/entity/project/runs/new-run"
                summary = {}
                def __enter__(self):
                    self.path = root / json.loads((root / "latest.json").read_text())["results_path"]
                    if self.path.exists():
                        raise AssertionError("Không ghi kết quả trước logging")
                    return self
                def log(self, row, step):
                    if self.path.exists():
                        raise AssertionError("Kết quả được ghi trước finish")
                def __exit__(self, *args):
                    raise RuntimeError("simulated finish failure")
            client = SimpleNamespace(__version__="test", init=lambda **kwargs: FinishFailure())
            with mock.patch.dict(sys.modules, {"wandb": client}), mock.patch("src.cifar_smoke._compute_report", side_effect=lambda *a, **k: self.report()), self.assertRaises(RuntimeError):
                run(self.config(), data_dir=root / "data", output_dir=root, log_wandb=True)
            failed = self.assert_latest(root, "failed")
            self.assertEqual(failed["failed_stage"], "tracking")
            self.assertEqual(failed["wandb_status"], "failed")
            self.assertTrue(failed["numerical_verification_passed"])
            self.assertIsNone(failed["wandb_run_url"])

    def test_keyboard_interrupt_records_failed_instead_of_previous_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("src.cifar_smoke._compute_report", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                run(self.config(), data_dir=root / "data", output_dir=root)
            self.assertEqual(self.assert_latest(root, "failed")["error_type"], "KeyboardInterrupt")

    def test_missing_or_invalid_cli_config_is_recorded_as_failed(self):
        for invalid_json in (False, True):
            with self.subTest(invalid_json=invalid_json), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / "config-input.json"
                if invalid_json:
                    config_path.write_text("{broken", encoding="utf-8")
                error = json.JSONDecodeError if invalid_json else FileNotFoundError
                with self.assertRaises(error):
                    run(None, config_path=config_path, data_dir=root / "data", output_dir=root)
                self.assertEqual(self.assert_latest(root, "failed")["failed_stage"], "config_load")

    def test_tracking_setup_failure_is_inside_attempt_before_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setup = mock.Mock(side_effect=RuntimeError("login failed"))
            with mock.patch("src.cifar_smoke._compute_report") as compute, self.assertRaises(RuntimeError):
                run(self.config(), data_dir=root / "data", output_dir=root,
                    log_wandb=True, prepare_tracking=setup)
            compute.assert_not_called()
            setup.assert_called_once()
            self.assertEqual(self.assert_latest(root, "failed")["failed_stage"], "tracking_setup")

    def test_config_file_overrides_are_applied_inside_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config-input.json"
            original = self.config(42)
            config_path.write_text(json.dumps(original), encoding="utf-8")
            def computing(config, **kwargs):
                self.assertEqual(config["seed"], 43)
                result = self.report()
                result["config"] = config
                return result
            with mock.patch("src.cifar_smoke._compute_report", side_effect=computing):
                result = run(None, config_path=config_path, config_overrides={"seed": 43},
                             data_dir=root / "data", output_dir=root)
            self.assertEqual(self.assert_latest(root, "complete")["config"]["seed"], 43)
            self.assertEqual(result["requested_seed"], 43)
            self.assertEqual(json.loads(config_path.read_text())["seed"], 42)

    def test_older_completion_cannot_overwrite_newer_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with RunAttempt(root, seed=42) as older:
                with self.assertRaises(ValueError):
                    with RunAttempt(root, seed=43) as newer:
                        raise ValueError("newer failed")
                older.progress("analytic", phase=1)
                older.complete(self.report())
            latest = self.assert_latest(root, "failed")
            self.assertEqual(latest["attempt_id"], newer.id)
            self.assertEqual(self.read(older.directory, "progress.json")["status"], "complete")

    def test_older_failure_cannot_overwrite_newer_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                with RunAttempt(root, seed=42) as older:
                    with RunAttempt(root, seed=43) as newer:
                        newer.complete(self.report())
                    raise RuntimeError("older failed late")
            self.assertEqual(self.assert_latest(root, "complete")["attempt_id"], newer.id)
            self.assertEqual(self.read(older.directory, "progress.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
