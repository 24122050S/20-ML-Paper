"""Tạo notebook dùng cùng module với CLI/tests; không nhúng URL run cố định."""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def build():
    """Ghi notebook 01 chưa có output; trả đường dẫn notebook để execute/kiểm tra."""
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    md(r"""
    # 01 — Verify ACIL analytic update, phase và minibatch

    **VERIFY-02** bổ sung hỗ trợ minibatch với nhãn toàn cục. Dữ liệu tổng hợp
    vẫn có 3 phase, mỗi phase thêm 2 lớp, feature 24 chiều, gamma=1, float64.
    Đây là kiểm chứng số học, chưa phải tái lập benchmark ảnh.

    Joint tính trực tiếp bằng `solve`; incremental chỉ nhận R/W cũ và batch hiện tại.
    Code dùng chung với `src/analytic.py`, `src/verification.py` và bộ test trong repo.
    Nguồn: ACIL, NeurIPS 2022, mục 3.1–3.2, phương trình (4), (9), (10).

    Trên Kaggle: Accelerator **None** là đủ; bật Internet để lấy source và log W&B.
    Bật secret `WANDB_API_KEY` cho notebook nếu dùng tracking. Cell thiết lập ghi rõ
    commit/source hashes để không suy diễn rằng mọi lần chạy đều dùng cùng source.
    """)
    code('''
import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

REPOSITORY = "https://github.com/24122050S/20-ML-Paper.git"
# Có thể đặt ACIL_SOURCE_REF thành commit cụ thể để chạy lại đúng source cũ.
SOURCE_REF = os.environ.get("ACIL_SOURCE_REF", "main")
LOG_WANDB = os.environ.get("ACIL_LOG_WANDB", "1") == "1"
WANDB_ENTITY = "hsangg-2006"
WANDB_PROJECT = "acil-reproduction"

candidates = (Path.cwd(), Path.cwd().parent)
PROJECT_ROOT = next((p for p in candidates if (p / "src/analytic.py").is_file()
                     and (p / "configs/verify_analytic_update.json").is_file()), None)
if PROJECT_ROOT is None:
    # Import độc lập trên Kaggle: checkout một snapshot, không sửa notebook/project cũ.
    source_parent = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(tempfile.gettempdir())
    PROJECT_ROOT = Path(tempfile.mkdtemp(prefix="acil-source-", dir=source_parent))
    subprocess.run(["git", "init", "--quiet", str(PROJECT_ROOT)], check=True)
    subprocess.run(["git", "remote", "add", "origin", REPOSITORY], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["git", "fetch", "--quiet", "--depth=1", "origin", SOURCE_REF], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=PROJECT_ROOT, check=True)
sys.path.insert(0, str(PROJECT_ROOT))
print("Source:", PROJECT_ROOT)
print("W&B enabled:", LOG_WANDB)
''')
    md(r"""
    ## Đặc tả nhãn và cập nhật

    Nhãn là ID toàn cục: base 0–1, phase sau 2–3 rồi 4–5. Trong một phase,
    lớp có thể xuất hiện lại ở nhiều minibatch; số cột W chỉ tăng khi có lớp mới.

    Với W cũ được đệm cột 0 thành $W_{pad}$ và one-hot toàn cục $Y$:

    $$R_{new}=R_{old}-R_{old}H^T(I+HR_{old}H^T)^{-1}HR_{old},$$
    $$W_{new}=W_{pad}+R_{new}H^T(Y-HW_{pad}).$$

    Khi batch chỉ có lớp mới, đây chính là hai khối trong công thức (9).
    Gamma chỉ cộng lúc khởi tạo. Cùng feature map/objective được giữ suốt run.
    """)
    code('''
from src.verification import run_verification
from src.reporting import collect_provenance, log_and_save_report

CONFIG = json.loads((PROJECT_ROOT / "configs/verify_analytic_update.json").read_text(encoding="utf-8"))
report = run_verification(CONFIG)
report["provenance"] = collect_provenance(PROJECT_ROOT)
print("Environment:", report["environment"])
print("Source commit:", report["provenance"]["git_commit"])
print("Source dirty:", report["provenance"]["source_dirty"])
for row in report["results"]:
    print(f"phase={row['phase']} W={row['weights_shape']} R={row['inverse_gram_shape']} "
          f"relative_W={row['relative_w_error']:.3e} "
          f"minibatch_relative_W={row['minibatch_relative_w_error']:.3e} "
          f"label_disagreements={row['minibatch_prediction_disagreements']}")
assert report["verification_passed"]
print("PASS: joint, whole-phase và minibatch đạt kiểm tra W/R/score/argmax.")
''')
    md("""
    ## Test dùng chung

    Bộ test còn kiểm tra lớp lặp, rank-deficient features, trạng thái cũ không bị sửa
    và lỗi tracking. Test W&B dùng SDK giả để xác nhận report chỉ ghi thành công
    sau khi run finish, và không giữ URL cũ khi logging lỗi.
    """)
    code('''
subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    cwd=PROJECT_ROOT, check=True,
)
''')
    md("""
    ## W&B và xuất JSON sau khi logging hoàn tất

    Mặc định Kaggle log vào project của nhóm. SDK được khóa version 0.28.1.
    Khi chạy local không có tài khoản, đặt `ACIL_LOG_WANDB=0` trước khi mở kernel
    hoặc đổi biến ở cell đầu. Trạng thái khi đó là `not_logged`, URL null.

    Không điền run ID bằng tay. Hàm lấy ID/URL từ SDK; chỉ ghi `logged` sau khi
    context W&B kết thúc. Nếu init/log/finish lỗi, JSON ghi `failed` và không gắn
    URL thành công cũ. Authentication lỗi trước logging sẽ dừng notebook.
    """)
    code('''
wandb_client = None
if LOG_WANDB:
    import importlib.metadata
    try:
        installed = importlib.metadata.version("wandb")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != "0.28.1":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb==0.28.1"], check=True)
    import wandb as wandb_client
    if Path("/kaggle/working").is_dir():
        from kaggle_secrets import UserSecretsClient
        wandb_client.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
    else:
        # Local dùng đăng nhập/env có sẵn; không ghi credential vào notebook.
        wandb_client.login()

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else PROJECT_ROOT / "reports/local"
OUTPUT_PATH = OUTPUT_DIR / "verify_analytic_update_v2_results.json"
report = log_and_save_report(
    report, OUTPUT_PATH, log_wandb=LOG_WANDB,
    entity=WANDB_ENTITY, project=WANDB_PROJECT, wandb_client=wandb_client,
)
print("Report:", OUTPUT_PATH)
print("Tracking:", report["wandb_status"])
print("W&B run ID:", report["wandb_run_id"])
print("W&B run:", report["wandb_run_url"])
print("VERIFY-02 COMPLETE")
''')
    md("""
    ## Bàn giao đúng một lượt chạy

    Sau Save & Run All thành công, dùng chính JSON ở Output của version đó.
    README phải ghép đúng notebook version, source commit/hash và W&B URL trong JSON.
    Nếu chạy lại, run URL có thể thay đổi và JSON sẽ tự cập nhật; không sửa URL cứng
    trong source. Giữ VERIFY-01 làm lịch sử, VERIFY-02 là kiểm tra API minibatch mới.

    Accuracy tổng hợp chỉ mô tả ví dụ này. Các kết quả không chứng minh lại định lý
    tổng quát, không bảo đảm accuracy lớp cũ bất biến và không phải benchmark ảnh.
    """)
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook_cell_{index}", "exec")
    nbf.validate(notebook)
    path = ROOT / "notebooks/01_verify_analytic_update.ipynb"
    nbf.write(notebook, path)
    return path


if __name__ == "__main__":
    print(build())
