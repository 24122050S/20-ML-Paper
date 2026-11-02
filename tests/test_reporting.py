"""Kiểm tra JSON chỉ ghi tracking thành công sau khi run finish, dùng SDK giả."""

import json
from pathlib import Path
import tempfile
import unittest

from src.reporting import log_and_save_report


class FakeRun:
    def __init__(self,path,fail_at=None):
        self.path=path; self.fail_at=fail_at
        self.id="actual-new-run"; self.url="https://wandb.ai/entity/project/runs/actual-new-run"
        self.summary={}; self.finished=False

    def __enter__(self):
        if self.path.exists():raise AssertionError("Không được xuất report thành công trước logging")
        return self

    def log(self,row,step):
        if self.path.exists():raise AssertionError("Report bị ghi khi run chưa kết thúc")
        if self.fail_at=="log":raise RuntimeError("simulated log failure")

    def __exit__(self,*args):
        if self.fail_at=="finish":raise RuntimeError("simulated finish failure")
        self.finished=True


class FakeClient:
    __version__="test"
    def __init__(self,run):self.run=run
    def init(self,**kwargs):return self.run


class ReportingTests(unittest.TestCase):
    def report(self):
        return {"experiment":"VERIFY-02","config":{"seed":42},"environment":{"numpy":"2.1.3"},
                "results":[{"phase":0,"relative_w_error":1e-12}],"verification_passed":True,
                "wandb_status":"synced","wandb_run_url":"https://old.invalid/run"}

    def test_offline_local_export_has_no_fake_wandb_url(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"nested/result.json"
            result=log_and_save_report(self.report(),path)
            self.assertEqual(result["wandb_status"],"not_logged")
            self.assertIsNone(result["wandb_run_url"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),result)

    def test_success_uses_actual_sdk_id_and_writes_after_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"result.json"
            run=FakeRun(path)
            original=self.report()
            result=log_and_save_report(original,path,log_wandb=True,entity="entity",project="project",wandb_client=FakeClient(run))
            self.assertTrue(run.finished)
            self.assertEqual(result["wandb_status"],"logged")
            self.assertEqual(result["wandb_run_id"],run.id)
            self.assertEqual(result["wandb_run_url"],run.url)
            self.assertEqual(original["wandb_run_url"],"https://old.invalid/run")

    def test_log_or_finish_failure_never_reports_logged(self):
        for stage in ("log","finish"):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"result.json"
                with self.assertRaises(RuntimeError):
                    log_and_save_report(self.report(),path,log_wandb=True,entity="entity",project="project",wandb_client=FakeClient(FakeRun(path,stage)))
                saved=json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["wandb_status"],"failed")
                self.assertIsNone(saved["wandb_run_url"])
                self.assertIsNone(saved["wandb_run_id"])

    def test_init_failure_is_saved_as_failed_without_old_url(self):
        class InitFailure:
            def init(self,**kwargs):raise RuntimeError("simulated init failure")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"result.json"
            with self.assertRaises(RuntimeError):
                log_and_save_report(self.report(),path,log_wandb=True,entity="entity",project="project",wandb_client=InitFailure())
            saved=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["wandb_status"],"failed")
            self.assertIsNone(saved["wandb_run_url"])

    def test_missing_sdk_url_cannot_be_claimed_as_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"result.json"
            run=FakeRun(path)
            run.url=None
            with self.assertRaises(RuntimeError):
                log_and_save_report(self.report(),path,log_wandb=True,entity="entity",project="project",wandb_client=FakeClient(run))
            saved=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["wandb_status"],"failed")
            self.assertIsNone(saved["wandb_run_url"])


if __name__ == "__main__":unittest.main()
