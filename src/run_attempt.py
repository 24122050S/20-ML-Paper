"""Quản lý từng lượt CIFAR: artifact riêng, trạng thái mới nhất và ghi JSON atomic.

FileLock chỉ bảo vệ cập nhật metadata; hai lượt không dùng chung checkpoint/state.
Không xóa hoặc sửa dữ liệu của các lượt trước.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from filelock import FileLock


def _now():
    """Trả timestamp UTC rõ múi giờ để nhận diện lượt chạy."""
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    """Đọc metadata nếu hợp lệ; trả None cho file thiếu hoặc JSON hỏng."""
    try:
        value = Path(path).read_text(encoding="utf-8-sig")
        result = json.loads(value)
        return result if isinstance(result, dict) else None
    except (OSError, ValueError):
        return None


def _atomic_json(path, payload):
    """Ghi payload UTF-8/LF qua file tạm cùng thư mục rồi replace atomic."""
    path = Path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".attempt-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RunAttempt:
    """Một lượt có thư mục riêng; root/results và root/progress chỉ theo lượt mới nhất.

    output_root là thư mục đích của CLI/notebook, seed chỉ là metadata yêu cầu.
    Context ghi running khi enter, failed khi exception, complete qua complete().
    Không dùng exception message trong JSON để tránh lưu credential từ thư viện.
    """

    def __init__(self, output_root, *, seed=None):
        self.root = Path(output_root).resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.id = f"{timestamp}-{uuid4().hex[:12]}"
        self.directory = self.root / "attempts" / self.id
        self.relative_dir = self.directory.relative_to(self.root).as_posix()
        self.seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
        self.started_at = _now()
        self.lock = FileLock(str(self.root / ".attempts.lock"), timeout=30)
        self.record = None
        self.finished = False

    def _identity(self):
        return {"attempt_id": self.id, "artifacts_dir": self.relative_dir,
                "output_dir": str(self.directory), "requested_seed": self.seed,
                "started_at_utc": self.started_at}

    def _current(self):
        latest = _read_json(self.root / "latest.json")
        return latest is not None and latest.get("attempt_id") == self.id

    def _archive_legacy(self):
        """Giữ nguyên JSON kiểu cũ trước khi thay summary; không dời checkpoint cũ."""
        names = ("results.json", "progress.json", "config.json", "protocol.json", "data_audit.json")
        if (self.root / "latest.json").exists() or not any((self.root / n).is_file() for n in names):
            return
        legacy = self.root / "attempts" / f"legacy-{self.id}"
        legacy.mkdir()
        for name in names:
            source = self.root / name
            if source.is_file():
                shutil.copy2(source, legacy / name)
        _atomic_json(legacy / "legacy.json", {
            "note": "JSON của pipeline cũ được giữ nguyên. Checkpoint/state cũ vẫn ở output root; lượt mới chỉ ghi vào thư mục attempt riêng.",
            "artifacts_at_output_root": [n for n in ("base_checkpoint.pt", "analytic_state.npz")
                                         if (self.root / n).is_file()]})

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.directory.mkdir(parents=True, exist_ok=False)
        with self.lock:
            self._archive_legacy()
            self.record = {**self._identity(), "status": "running", "stage": "starting",
                           "checks_passed": False, "updated_at_utc": _now(),
                           "results_path": f"{self.relative_dir}/results.json"}
            _atomic_json(self.directory / "progress.json", self.record)
            # Invalidate summary cũ ngay cả khi lượt mới lỗi trước bước tải dữ liệu.
            _atomic_json(self.root / "results.json", self._pending_report())
            _atomic_json(self.root / "progress.json", self.record)
            _atomic_json(self.root / "latest.json", self.record)
        print(f"Attempt: {self.id}\nArtifacts: {self.directory}", flush=True)
        return self

    def _pending_report(self):
        return {**self._identity(), "status": "running", "verification_passed": False,
                "results": [], "wandb_status": "not_started", "wandb_run_id": None,
                "wandb_run_url": None}

    def progress(self, stage, **details):
        """Ghi stage của lượt này; cập nhật root chỉ khi nó còn là lượt mới nhất."""
        if self.finished:
            raise RuntimeError("Attempt đã kết thúc")
        self.record = {**self._identity(), "status": "running", "stage": stage,
                       "checks_passed": False, "updated_at_utc": _now(),
                       "results_path": f"{self.relative_dir}/results.json", **details}
        _atomic_json(self.directory / "progress.json", self.record)
        with self.lock:
            if self._current():
                _atomic_json(self.root / "results.json", self._pending_report())
                _atomic_json(self.root / "progress.json", self.record)
                _atomic_json(self.root / "latest.json", self.record)

    def complete(self, report):
        """Lưu report đã log/kiểm tra xong, trả report có attempt ID và status complete."""
        if self.finished or not report.get("verification_passed"):
            raise ValueError("Chỉ complete một attempt đang chạy với kiểm chứng đã đạt")
        result = {**copy.deepcopy(report), **self._identity(), "status": "complete"}
        self._finish(result, status="complete", checks_passed=True)
        return result

    def fail(self, error):
        """Đánh dấu failed, giữ phần chẩn đoán của chính attempt này nếu đã có."""
        result = _read_json(self.directory / "results.json")
        if result is None or result.get("attempt_id") != self.id:
            result = self._pending_report()
        else:
            result["numerical_verification_passed"] = bool(result.get("verification_passed"))
        config = _read_json(self.directory / "config.json")
        if config is not None:
            result["config"] = config
        result.update(self._identity(), status="failed", verification_passed=False,
                      error_type=type(error).__name__, failed_stage=self.record["stage"])
        self._finish(result, status="failed", checks_passed=False)

    def _finish(self, result, *, status, checks_passed):
        self.record = {**self._identity(), "status": status, "stage": status,
                       "checks_passed": checks_passed, "updated_at_utc": _now(),
                       "results_path": f"{self.relative_dir}/results.json"}
        if status == "failed":
            self.record.update(error_type=result["error_type"], failed_stage=result["failed_stage"])
        _atomic_json(self.directory / "results.json", result)
        _atomic_json(self.directory / "progress.json", self.record)
        with self.lock:
            if self._current():
                _atomic_json(self.root / "results.json", result)
                _atomic_json(self.root / "progress.json", self.record)
                _atomic_json(self.root / "latest.json", self.record)
        self.finished = True

    def __exit__(self, error_type, error, traceback):
        if error is not None:
            try:
                self.fail(error)
            except Exception as status_error:
                # Giữ exception gốc; disk/lock lỗi có thể khiến status dừng ở running.
                if hasattr(error, "add_note"):
                    error.add_note(f"Không ghi được failed status: {type(status_error).__name__}")
            return False
        if not self.finished:
            error = RuntimeError("Attempt kết thúc mà chưa complete")
            self.fail(error)
            raise error
        return False
