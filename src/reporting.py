"""Ghi provenance và kết quả; chỉ đánh dấu logged sau khi W&B context kết thúc."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess


def collect_provenance(root):
    """Nhận root repo; trả SHA-256 source/config, Git commit và trạng thái dirty.

    Không ghi giá trị biến môi trường, credential hoặc thông tin tài khoản bí mật.
    Commit có thể null nếu chạy từ source không có Git; source hashes vẫn có.
    """
    root = Path(root).resolve()
    files = [*sorted((root / "src").glob("*.py")), root / "configs/verify_analytic_update.json"]
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    commit, dirty = None, None
    try:
        current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, timeout=5, check=True)
        commit, dirty = current.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"repository": "https://github.com/24122050S/20-ML-Paper", "git_commit": commit, "source_dirty": dirty, "source_sha256": hashes}


def save_json(report, output_path):
    """Ghi report JSON UTF-8 vào output_path; tự tạo thư mục cha và trả Path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_path


def log_and_save_report(report, output_path, *, log_wandb=False, entity=None, project=None, wandb_client=None):
    """Log run khi được bật rồi xuất JSON với ID/URL lấy từ chính SDK.

    report chứa config, environment và các result rows đã kiểm tra. output_path
    là file đích; entity/project là đích W&B. wandb_client tùy chọn phục vụ test.
    Trả bản sao report đã lưu. Nếu init/log/finish lỗi, ghi tracking failed với
    URL null rồi raise; không ghi thành công trước khi context kết thúc.
    """
    result = copy.deepcopy(report)
    result.update(wandb_status="not_logged", wandb_run_id=None, wandb_run_url=None)
    if log_wandb:
        if not entity or not project:
            raise ValueError("Cần entity và project khi bật W&B")
        try:
            if wandb_client is None:
                import wandb as wandb_client
            with wandb_client.init(
                entity=entity, project=project, job_type="verification", mode="online",
                tags=["verification", "synthetic", "float64", "minibatch"],
                config={**result["config"], "numpy_version": result["environment"]["numpy"],
                        "source_commit": result.get("provenance", {}).get("git_commit"),
                        "source_sha256": result.get("provenance", {}).get("source_sha256", {})},
            ) as run:
                for row in result["results"]:
                    metrics = {key: value for key, value in row.items() if isinstance(value, (int, float, bool))}
                    run.log(metrics, step=row["phase"])
                run.summary["verification_passed"] = result["verification_passed"]
                run.summary["experiment"] = result["experiment"]
                actual_id, actual_url = run.id, run.url
            # Chỉ đến đây sau khi __exit__/finish đã hoàn tất thành công.
            if not actual_id or not actual_url:
                raise RuntimeError("W&B chưa trả ID/URL cho run online")
            result.update(wandb_status="logged", wandb_run_id=actual_id, wandb_run_url=actual_url)
            result["environment"]["wandb"] = getattr(wandb_client, "__version__", None)
        except Exception as error:
            result.update(wandb_status="failed", wandb_run_id=None, wandb_run_url=None)
            result["tracking_error_type"] = type(error).__name__
            save_json(result, output_path)
            raise
    save_json(result, output_path)
    return result
