"""Đóng gói notebook04 cho resume/profile/benchmark; giữ snapshot02/03 bất biến."""

from __future__ import annotations
import base64
import hashlib
import io
from pathlib import Path
import zipfile


def build(output=None):
    """Đóng gói toàn bộ source/configs/tests/requirements vào notebook 04 và ghi ra đĩa.

    Chặn việc ghi đè lên các snapshot notebook 02 và 03 đã công bố.

    Tham số:
        output: Đường dẫn file ``.ipynb`` đích (tùy chọn); mặc định tạo
            ``notebooks/04_resume_benchmark_profile.ipynb``.

    Trả về:
        Đối tượng ``Path`` trỏ tới file notebook vừa được tạo.
    """
    import nbformat as nb
    root=Path(__file__).resolve().parents[1]
    destination=Path(output) if output is not None else root/'notebooks/04_resume_benchmark_profile.ipynb'
    if destination.resolve() in {(root/'notebooks'/name).resolve() for name in
                                  ('02_cifar100_smoke.ipynb','03_cifar100_safe_attempts.ipynb')}:
        raise ValueError('Notebook02/03 là snapshot đã công bố; chọn notebook mới')
    files=sorted([*root.glob('src/*.py'),*root.glob('configs/*.json'),*root.glob('tests/*.py'),*root.glob('requirements-*.txt')])
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info=zipfile.ZipInfo(path.relative_to(root).as_posix(),(2026,11,4,0,0,0))
            info.create_system=3
            archive.writestr(info,path.read_bytes().replace(b'\r\n',b'\n'),compress_type=zipfile.ZIP_DEFLATED)
    raw=payload.getvalue()
    digest=hashlib.sha256(raw).hexdigest()
    setup='''import os, sys, json, base64, hashlib, io, tempfile, zipfile
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ACTION = "profile_analytic"  # profile_analytic / profile_base / benchmark
CHECKPOINT = None  # Profile: attach Output notebook03 làm Input; tự tìm nếu có đúng1 checkpoint.
RESUME_CHECKPOINT = None  # Chỉ checkpoint v2 đang dở; khác reuse base hoàn tất.
STOP_AFTER_EPOCH = None  # Ví dụ1 để kiểm chạy ngắt; giữ None khi chạy tới target.
LOG_WANDB = True  # Enable secret WANDB_API_KEY trên Kaggle.
PROFILE_DIMENSION = 8000
RUN_INCREMENTAL_PHASES = 5
BUNDLE_SHA256 = __DIGEST__
candidates = (Path.cwd(), Path.cwd().parent)
PROJECT_ROOT = next((p for p in candidates if (p / "src/profile_analytic.py").is_file()), None)
if PROJECT_ROOT is None:
    data = base64.b64decode(__PAYLOAD__)
    assert hashlib.sha256(data).hexdigest() == BUNDLE_SHA256
    parent = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(tempfile.gettempdir())
    PROJECT_ROOT = Path(tempfile.mkdtemp(prefix="acil-benchmark-source-", dir=parent))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            assert (PROJECT_ROOT / name).resolve().is_relative_to(PROJECT_ROOT.resolve())
        archive.extractall(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print("Source:", PROJECT_ROOT)
print("Embedded bundle SHA-256:", BUNDLE_SHA256)
'''.replace('__DIGEST__',repr(digest)).replace('__PAYLOAD__',repr(base64.b64encode(raw).decode()))
    dependencies='''import subprocess, importlib.util, importlib.metadata
pins = {"numpy":"2.1.3", "huggingface_hub":"1.10.1", "pyarrow":"23.0.1",
        "Pillow":"11.1.0", "threadpoolctl":"3.6.0", "wandb":"0.28.1",
        "filelock":"4.2.6", "scipy":"1.17.0", "psutil":"7.2.2"}
missing = []
for package, version in pins.items():
    try: current = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: current = None
    if current != version: missing.append(f"{package}=={version}")
if missing: subprocess.run([sys.executable,"-m","pip","install","-q",*missing],check=True)
if importlib.util.find_spec("torch") is None:
    subprocess.run([sys.executable,"-m","pip","install","torch==2.11.0"],check=True)
import torch, numpy as np
if np.__version__ != "2.1.3": raise RuntimeError("Restart kernel sau khi đổi NumPy rồi Run All")
print("NumPy:",np.__version__,"Torch:",torch.__version__,"CUDA:",torch.cuda.is_available())
'''
    checks='''checked = subprocess.run([sys.executable,"-B","-X","utf8","-m","unittest","discover","-s","tests","-v"],
                         cwd=PROJECT_ROOT,capture_output=True,text=True,encoding="utf-8")
print(checked.stdout)
print(checked.stderr)
checked.check_returncode()
'''
    execute='''from src.cifar_smoke import run
from src.profile_analytic import profile
def prepare_tracking():
    import wandb
    if Path("/kaggle/working").is_dir():
        from kaggle_secrets import UserSecretsClient
        wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
    else: wandb.login()
working = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else PROJECT_ROOT / "reports/local"
if ACTION == "profile_analytic":
    profile_checkpoint = CHECKPOINT
    if profile_checkpoint is None and Path("/kaggle/input").is_dir():
        candidates = list(Path("/kaggle/input").rglob("base_checkpoint.pt"))
        if len(candidates) == 1: profile_checkpoint = candidates[0]
    report = profile(profile_checkpoint,data_dir=working/"cifar-data",output_dir=working/"profile_d8000",
                     dimension=PROFILE_DIMENSION,threads=min(4,os.cpu_count() or 2),device="auto",
                     log_wandb=LOG_WANDB,prepare_tracking=prepare_tracking)
elif ACTION in ("profile_base","benchmark"):
    overrides = {"device":"auto","torch_threads":min(4,os.cpu_count() or 2),
                 "run_incremental_phases":RUN_INCREMENTAL_PHASES}
    if ACTION == "profile_base":
        overrides.update(experiment="ACIL-BASE-PROFILE",purpose="profile_base",epochs=5,
                         expansion_dim=256,run_incremental_phases=1)
    report = run(None,config_path=PROJECT_ROOT/"configs/cifar100_benchmark_k5.json",
                 config_overrides=overrides,data_dir=working/"cifar-data",output_dir=working/ACTION,
                 reuse_checkpoint=CHECKPOINT,resume_checkpoint=RESUME_CHECKPOINT,
                 stop_after_epoch=STOP_AFTER_EPOCH,log_wandb=LOG_WANDB,prepare_tracking=prepare_tracking)
else: raise ValueError("ACTION phải là profile_analytic, profile_base hoặc benchmark")
'''
    final='''assert report["status"] == "complete" and report["verification_passed"]
if "profile_scope" in report:
    assert report["profile_scope"]["full_protocol_evaluated"] is False
    assert report["state_shapes"]["W"] == [report["config"]["expansion_dim"],100]
    print("PROFILE subset:",report["profile_scope"])
    print("Timings:",report["timings"])
    print("Memory:",report["memory"])
else:
    config = report["config"]
    expected_classes = 50 + (50 // config["incremental_phases"]) * config["run_incremental_phases"]
    dimension = config["expansion_dim"] if config.get("feature_expansion",True) else 64
    assert len(report["results"]) == config["run_incremental_phases"] + 1
    assert report["results"][-1]["weights_shape"] == [dimension,expected_classes]
    assert report["results"][-1]["test_samples_all_seen"] == expected_classes*100
    assert report.get("joint_reference_status","not_run") == "not_run"
    print("Metric summary:",report.get("metric_summary"))
    for row in report["results"]: print(row)
print("Tracking:",report["wandb_status"],"W&B:",report["wandb_run_url"])
print("Attempt:",report["attempt_id"],"Artifacts:",report["output_dir"])
print("RESUME-BENCHMARK-PROFILE COMPLETE")
'''
    md,code=nb.v4.new_markdown_cell,nb.v4.new_code_cell
    cells=[md("""# 04 — Resume, benchmark streaming và profile d=8000

Mặc định ACTION=profile_analytic: đo subset CIFAR thật, không mở BP160 epochs. Attach Output notebook03 làm Input để lấy checkpoint base; nếu có nhiều checkpoint thì điền CHECKPOINT trong cell đầu. Bật Internet/GPU và secret WANDB_API_KEY khi LOG_WANDB=True.

ACTION=profile_base chạy BP5 epochs và một phase analytic d256 để đo BP; ACTION=benchmark dùng config160 epochs/d8000. Chỉ chọn benchmark sau khi nghiệm thu resume và profile. Resume dùng checkpoint v2 của phiên đang dở và đúng target/config/runtime/source; checkpoint cũ hoàn tất chỉ dùng reuse/profile.

Mỗi lượt có attempts/<id>/ và failed status khi lỗi. Không ghi đè snapshot notebook02/03. Mở cell code đầu đang thu gọn để sửa các biến."""),code(setup),
           md("## Dependency — giữ PyTorch hiện có của Kaggle"),code(dependencies),
           md("## Tests — có kiểm continuous vs resume, RNG/shuffle, streaming và metric; CUDA test chỉ chạy khi có GPU"),code(checks),
           md("## Chạy action đã chọn — checkpoint mỗi epoch nằm trong output_dir của attempt; cần giữ Output qua phiên Kaggle"),code(execute),
           md("## Kiểm kết quả — profile là subset/resource probe; benchmark không chạy joint oracle và không gán error=0"),code(final)]
    cells[1]['metadata']['jupyter']={'source_hidden':True}
    notebook=nb.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'acil_source_bundle_sha256':digest})
    nb.write(notebook,destination)
    print(f'Notebook: {destination}; bundle files={len(files)}, sha256={digest}')
    return destination


if __name__=='__main__':build()
