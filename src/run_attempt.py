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
    """Tạo dấu thời gian UTC chuẩn ISO-8601 để nhận diện và sắp xếp lượt chạy.

    Tham số:
        Không nhận tham số.

    Trả về:
        Chuỗi thời gian UTC có thông tin múi giờ.
    """
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    """Đọc file JSON dạng dict một cách an toàn, bỏ qua file thiếu hoặc bị hỏng.

    Tham số:
        path: Đường dẫn file JSON cần đọc.

    Trả về:
        Dict đã giải mã nếu hợp lệ, ngược lại trả ``None``.
    """
    try:
        value = Path(path).read_text(encoding="utf-8-sig")
        result = json.loads(value)
        return result if isinstance(result, dict) else None
    except (OSError, ValueError):
        return None


def _atomic_json(path, payload):
    """Ghi dict ra file JSON theo chuẩn UTF-8/LF qua file tạm cùng thư mục rồi replace atomic.

    Tham số:
        path: Đường dẫn file JSON đích.
        payload: Đối tượng có thể tuần tự hóa thành JSON (không chứa NaN/Inf).

    Trả về:
        None.
    """
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
        """Khởi tạo ngữ cảnh quản lý thư mục attempt riêng, attempt ID và khóa file.

        Tham số:
            output_root: Thư mục gốc chứa ``latest.json``, ``results.json`` và ``attempts/``.
            seed: Seed yêu cầu tùy chọn để ghi vào metadata nhận diện lượt chạy.

        Trả về:
            None.
        """
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
        """Tạo dict thông tin nhận diện cố định của attempt hiện tại.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            Dict gồm ``attempt_id``, ``artifacts_dir``, ``output_dir``, ``requested_seed`` và ``started_at_utc``.
        """
        return {"attempt_id": self.id, "artifacts_dir": self.relative_dir,
                "output_dir": str(self.directory), "requested_seed": self.seed,
                "started_at_utc": self.started_at}

    def _current(self):
        """Kiểm tra attempt này có còn là lượt mới nhất trong ``latest.json`` hay không.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            ``True`` nếu ``latest.json`` trỏ đúng vào ``self.id``, ngược lại ``False``.
        """
        latest = _read_json(self.root / "latest.json")
        return latest is not None and latest.get("attempt_id") == self.id

    def _archive_legacy(self):
        """Lưu giữ các file JSON kiểu cũ vào thư mục archive trước khi cập nhật root summary.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
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
        """Khởi tạo thư mục attempt, đánh dấu trạng thái ``running`` và làm mới summary root.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            Chính đối tượng ``RunAttempt``.
        """
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
        """Tạo cấu trúc báo cáo tạm thời khi lượt chạy đang ở trạng thái ``running``.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            Dict báo cáo placeholder với ``verification_passed=False`` và ``status="running"``.
        """
        return {**self._identity(), "status": "running", "verification_passed": False,
                "results": [], "wandb_status": "not_started", "wandb_run_id": None,
                "wandb_run_url": None}

    def progress(self, stage, **details):
        """Cập nhật giai đoạn đang thực thi của attempt hiện tại và đồng bộ root nếu vẫn mới nhất.

        Tham số:
            stage: Tên giai đoạn hiện tại (ví dụ ``"base_training"``, ``"tracking"``).
            **details: Các trường metadata bổ sung cần lưu vào ``progress.json``.

        Trả về:
            None.
        """
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
        """Đánh dấu attempt hoàn tất hợp lệ và ghi báo cáo cuối cùng ra thư mục attempt và root.

        Tham số:
            report: Dict báo cáo đã có ``verification_passed=True``.

        Trả về:
            Dict báo cáo hoàn chỉnh đã gắn thông tin nhận diện attempt và ``status="complete"``.
        """
        if self.finished or not report.get("verification_passed"):
            raise ValueError("Chỉ complete một attempt đang chạy với kiểm chứng đã đạt")
        result = {**copy.deepcopy(report), **self._identity(), "status": "complete"}
        self._finish(result, status="complete", checks_passed=True)
        return result

    def fail(self, error):
        """Đánh dấu attempt thất bại an toàn mà không ghi đè dữ liệu chi tiết bằng message nhạy cảm.

        Tham số:
            error: Ngoại lệ phát sinh khiến lượt chạy dừng lại.

        Trả về:
            None.
        """
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
        """Ghi nhận trạng thái kết thúc (``complete`` hoặc ``failed``) vào thư mục attempt và root.

        Tham số:
            result: Dict báo cáo kết quả cuối cùng của attempt.
            status: Chuỗi trạng thái kết thúc (``"complete"`` hoặc ``"failed"``).
            checks_passed: Giá trị bool cho biết các kiểm tra có đạt hay không.

        Trả về:
            None.
        """
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
        """Xử lý thoát context manager: ghi ``failed`` nếu có lỗi hoặc chưa gọi ``complete()``.

        Tham số:
            error_type: Lớp ngoại lệ nếu khối ``with`` phát sinh lỗi.
            error: Thể hiện ngoại lệ phát sinh (hoặc ``None``).
            traceback: Đối tượng traceback tương ứng (hoặc ``None``).

        Trả về:
            ``False`` để không nuốt ngoại lệ gốc.
        """
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
