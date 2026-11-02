"""Đóng gói source hiện tại vào notebook Kaggle; không cần chờ source lên GitHub."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
import zipfile


def build(output=None):
    """Tạo notebook 02 từ module thật; chạy lại sau mỗi lần sửa source/config/test."""
    import nbformat as nb
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.glob("src/*.py"), *root.glob("configs/*.json"),
                    *root.glob("tests/*.py"), *root.glob("requirements-*.txt")])
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), (2026, 11, 2, 0, 0, 0))
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    raw = payload.getvalue()
    encoded, digest = base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()
    markdown = nb.v4.new_markdown_cell
    code = nb.v4.new_code_cell
    setup = '''import os, sys, json, base64, hashlib, io, tempfile, zipfile
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
LOG_WANDB = False  # Bật True sau khi enable secret WANDB_API_KEY trên Kaggle.
candidates = (Path.cwd(), Path.cwd().parent)
PROJECT_ROOT = next((p for p in candidates if (p / "src/cifar_smoke.py").is_file()), None)
BUNDLE_SHA256 = __DIGEST__
if PROJECT_ROOT is None:
    data = base64.b64decode(__PAYLOAD__)
    assert hashlib.sha256(data).hexdigest() == BUNDLE_SHA256
    parent = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(tempfile.gettempdir())
    PROJECT_ROOT = Path(tempfile.mkdtemp(prefix="acil-cifar-source-", dir=parent))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            assert (PROJECT_ROOT / name).resolve().is_relative_to(PROJECT_ROOT.resolve())
        archive.extractall(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print("Source:", PROJECT_ROOT)
print("Embedded bundle SHA-256:", BUNDLE_SHA256)
'''.replace("__DIGEST__", repr(digest)).replace("__PAYLOAD__", repr(encoded))
    cells = [markdown('''# 02 — CIFAR-100 smoke: tự chạy 50 base + 10 lớp mới

**Mục tiêu:** tải dữ liệu tự động, huấn luyện ResNet-32 trên 50 lớp base trong 2 epochs, đóng băng backbone/BatchNorm và expansion, cập nhật analytic theo minibatch rồi đánh giá toàn bộ 60 lớp.

Trên Kaggle: import notebook này, bật Internet, chọn GPU nếu có, **Save & Run All**. Source được đóng gói kèm nên không cần chờ thành viên khác hoặc clone một nhánh chưa có trên GitHub. Lần chạy mới mặc định train base từ đầu; không phụ thuộc checkpoint cũ trong máy Sang.

Đây là smoke test với expansion 256; accuracy chưa dùng so benchmark paper. W&B mặc định tắt; bật `LOG_WANDB` ở cell đầu và enable secret nếu muốn log. Không nhập API key trực tiếp vào source.'''),
        code(setup), markdown('''## Cài dependency và kiểm tra môi trường

Giữ PyTorch có sẵn của Kaggle để dùng GPU phù hợp môi trường; phiên bản thực tế được lưu trong JSON. Bản local đã kiểm tra dùng PyTorch 2.11.0. NumPy cho analytic là 2.1.3.'''),
        code('''import subprocess, importlib.util, importlib.metadata
pins = {"numpy": "2.1.3", "huggingface_hub": "1.10.1", "pyarrow": "23.0.1",
        "Pillow": "11.1.0", "threadpoolctl": "3.6.0", "wandb": "0.28.1"}
missing = []
for package, version in pins.items():
    try:
        current = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        current = None
    if current != version:
        missing.append(f"{package}=={version}")
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
if importlib.util.find_spec("torch") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.11.0"], check=True)
import torch, numpy as np
if np.__version__ != "2.1.3":
    raise RuntimeError("NumPy đã được kernel import trước khi cài; restart kernel rồi Run All.")
print("NumPy:", np.__version__, "| Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
'''), markdown('''## Kiểm tra trước khi train

Test xác nhận mapping dùng chung, phase rời nhau, checkpoint tương thích, backbone/BatchNorm cố định và analytic minibatch khớp joint.'''),
        code('''checked = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v"],
                         cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
print(checked.stdout)
print(checked.stderr)
checked.check_returncode()
'''), markdown('''## Chạy smoke 50 → 60 lớp

H có shape `(n,256)`, nhãn base `0..49`, phase mới `50..59`. API analytic trả `(R,W)`; mapping thuộc pipeline. Bộ oracle joint giữ feature lịch sử chỉ để chẩn đoán; hàm incremental không nhận dữ liệu train cũ.'''),
        code('''from src.cifar_smoke import run
config_path = PROJECT_ROOT / "configs/cifar100_smoke.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config["device"] = "auto"
config["torch_threads"] = min(8, os.cpu_count() or 2)
if LOG_WANDB:
    import wandb
    if Path("/kaggle/working").is_dir():
        from kaggle_secrets import UserSecretsClient
        wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
    else:
        wandb.login()
working = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else PROJECT_ROOT / "reports/local"
report = run(config, data_dir=working / "cifar-data", output_dir=working / "cifar100_smoke",
             log_wandb=LOG_WANDB, config_path=config_path)
'''), markdown('''## Bằng chứng cần giữ

Lấy `cifar100_smoke/results.json`, `protocol.json`, `data_audit.json`, `base_checkpoint.pt` và `analytic_state.npz` ở Output. JSON ghi rõ checkpoint mới/reuse, source hashes, môi trường, kiểm tra và W&B ID/URL của chính run nếu bật logging. Source bundle trên Kaggle không có Git: commit có thể null; dùng source hashes để nhận diện, không gắn một commit sạch giả.

Điều kiện đạt: W 50 → 60 cột, R giữ 256×256, test 5.000 → 6.000 ảnh, dự đoán khớp joint và mọi check đạt.'''),
        code('''assert report["verification_passed"]
assert report["results"][-1]["weights_shape"] == [256, 60]
for row in report["results"]:
    print({key: row[key] for key in ("phase", "n_classes", "weights_shape", "test_samples_all_seen", "accuracy_pct", "relative_w_error", "prediction_disagreements")})
print("Tracking:", report["wandb_status"], "| W&B:", report["wandb_run_url"])
print("CIFAR-SMOKE-01 COMPLETE")
''')]
    cells[1]["metadata"]["jupyter"] = {"source_hidden": True}
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "acil_source_bundle_sha256": digest})
    destination = Path(output) if output is not None else root / "notebooks/02_cifar100_smoke.ipynb"
    nb.write(notebook, destination)
    print(f"Notebook: {destination}; bundle files={len(files)}, sha256={digest}")
    return destination


if __name__ == "__main__":
    build()
