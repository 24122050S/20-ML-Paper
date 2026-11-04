"""Tạo notebook BP5 cấu hình sẵn từ snapshot04; không đổi source training/bundle."""

from pathlib import Path
import json
import nbformat as nb


def build():
    root = Path(__file__).resolve().parents[1]
    notebook = nb.read(root / 'notebooks/04_resume_benchmark_profile.ipynb', as_version=4)
    notebook.cells[0].source = '''# 05 — Sang chạy thử BP5 trước benchmark

**Không cần sửa biến hoặc thêm Input.** Bật GPU T4, Internet và quyền dùng secret WANDB_API_KEY, rồi Save & Run All.

Notebook đã đặt ACTION=profile_base, CHECKPOINT=None, RESUME_CHECKPOINT=None, STOP_AFTER_EPOCH=None. Chạy test resume CUDA, train base5 epochs từ đầu và một phase analytic d256. Đây chưa phải benchmark160 epochs.

Cuối lượt thành công, tải **profile_base_ready.json** ở Output và gửi cùng link version. File này gộp kết quả BP5, test CUDA và resource profile; trợ lý sẽ kiểm ngân sách và chuẩn bị lượt160 epochs. Không cần tự tính hoặc chọn checkpoint. Checkpoint2/5 epochs không dùng làm base160 epochs.
'''
    old = 'ACTION = "profile_analytic"  # profile_analytic / profile_base / benchmark'
    assert notebook.cells[1].source.count(old) == 1
    notebook.cells[1].source = notebook.cells[1].source.replace(old, 'ACTION = "profile_base"  # Đã cấu hình5 epochs; không đổi sang benchmark ở lượt này.')

    # Chạy suite trong process riêng như04, bổ sung proof machine-readable.
    test_program = '''import hashlib, json, platform, sys, unittest
from pathlib import Path
import numpy as np
import torch

class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.outcomes={}
    def addSuccess(self,test):
        super().addSuccess(test);self.outcomes[test.id()]="passed"
    def addSkip(self,test,reason):
        super().addSkip(test,reason);self.outcomes[test.id()]="skipped: "+reason
    def addFailure(self,test,error):
        super().addFailure(test,error);self.outcomes[test.id()]="failed"
    def addError(self,test,error):
        super().addError(test,error);self.outcomes[test.id()]="error"

suite=unittest.defaultTestLoader.discover("tests")
result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(suite)
cuda_id="test_resume.ResumeTests.test_cuda_restart_matches_continuous_exactly"
payload={"status":"complete" if result.wasSuccessful() else "failed",
 "verification_passed":result.wasSuccessful(),"tests_run":result.testsRun,"skipped":len(result.skipped),
 "failures":len(result.failures),"errors":len(result.errors),"outcomes":result.outcomes,
 "cuda_available":torch.cuda.is_available(),"cuda_resume_test_id":cuda_id,
 "cuda_resume_passed":result.outcomes.get(cuda_id)=="passed",
 "environment":{"python":platform.python_version(),"numpy":np.__version__,"torch":str(torch.__version__),
 "cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
 "bundle_sha256":sys.argv[2]}
Path(sys.argv[1]).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8",newline="\\n")
sys.exit(0 if result.wasSuccessful() else 1)
'''
    notebook.cells[4].source = '## Tests — xuất bằng chứng resume CUDA vào JSON'
    notebook.cells[5].source = '''from datetime import datetime, timezone
import uuid
working = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else PROJECT_ROOT / "reports/local"
proof_dir = working / "profile_base_checks" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])
proof_dir.mkdir(parents=True)
proof_path = proof_dir / "tests.json"
checked = subprocess.run([sys.executable,"-B","-X","utf8","-c",__PROGRAM__,str(proof_path),BUNDLE_SHA256],
                         cwd=PROJECT_ROOT,capture_output=True,text=True,encoding="utf-8")
(proof_dir/"stdout.log").write_text(checked.stdout,encoding="utf-8",newline="\\n")
(proof_dir/"stderr.log").write_text(checked.stderr,encoding="utf-8",newline="\\n")
print(checked.stdout)
print(checked.stderr)
checked.check_returncode()
TEST_PROOF = json.loads(proof_path.read_text(encoding="utf-8"))
print("CUDA resume passed:",TEST_PROOF["cuda_resume_passed"],"Proof:",proof_path)
'''.replace('__PROGRAM__', repr(test_program))

    analytic = json.loads((root / 'reports/profile_d8000_kaggle_results.json').read_text(encoding='utf-8'))
    reference = {'wandb_run_url': analytic['wandb_run_url'], 'attempt_id': analytic['attempt_id'],
                 'device': analytic['environment']['cuda_device'], 'threads': analytic['config']['torch_threads'],
                 'rough_extrapolation': analytic['rough_extrapolation']}
    notebook.cells[6].source = '## BP5 đã cấu hình sẵn — yêu cầu test CUDA đạt; chưa chạy160 epochs'
    notebook.cells[7].source = '''from src.cifar_smoke import run
from src.profile_analytic import MemoryMonitor
from src.reporting import save_json

next_report_path = working / "profile_base_ready.json"
save_json({"status":"running","verification_passed":False,"tests":TEST_PROOF},next_report_path)
def prepare_tracking():
    import wandb
    if Path("/kaggle/working").is_dir():
        from kaggle_secrets import UserSecretsClient
        wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
    else:
        wandb.login()
try:
    if ACTION != "profile_base" or CHECKPOINT is not None or RESUME_CHECKPOINT is not None or STOP_AFTER_EPOCH is not None:
        raise ValueError("Notebook05 chỉ train mới5 epochs; giữ các tham số đã cấu hình.")
    if not TEST_PROOF["cuda_resume_passed"] or not torch.cuda.is_available():
        raise RuntimeError("Bật GPU T4 rồi Save & Run All; cần test resume CUDA đạt trước BP5.")
    with MemoryMonitor("cuda") as resources:
        report = run(None,config_path=PROJECT_ROOT/"configs/cifar100_benchmark_k5.json",
                     config_overrides={"experiment":"ACIL-BASE-PROFILE","purpose":"profile_base", "device":"cuda",
                         "torch_threads":min(4,os.cpu_count() or 2),"epochs":5,"expansion_dim":256,"run_incremental_phases":1},
                     data_dir=working/"cifar-data",output_dir=working/"profile_base",
                     reuse_checkpoint=None,resume_checkpoint=None,stop_after_epoch=None,
                     log_wandb=LOG_WANDB,prepare_tracking=prepare_tracking)
    assert report["status"]=="complete" and report["verification_passed"]
    assert len(report["training_history"])==5 and report["config"]["epochs"]==5
    checkpoint = torch.load(Path(report["output_dir"])/"base_checkpoint.pt",map_location="cpu",weights_only=True)
    assert checkpoint["checkpoint_format_version"]==2 and checkpoint["training_complete"] is True
    assert checkpoint["completed_epoch"]==checkpoint["target_epochs"]==5
    reference = __REFERENCE__
    bp_seconds = sum(row["seconds"] for row in report["training_history"])
    readiness = {"status":"complete","verification_passed":True,"scope":"BP5 and CUDA resume proof; not160-epoch benchmark",
        "attempt_id":report["attempt_id"],"base_profile":report,"tests":TEST_PROOF,
        "memory":{"scope":"whole BP5 pipeline including analytic d256 and tracking; excludes notebook tests",**resources.result},
        "analytic_profile_reference":reference,
        "time_budget":{"bp5_seconds":bp_seconds,"bp160_linear_seconds":bp_seconds*32,
            "analytic_full_linear_seconds":reference["rough_extrapolation"]["base_full25000_seconds"]+reference["rough_extrapolation"]["incremental_full25000_seconds"],
            "other_bp5_pipeline_seconds":max(0,report["seconds_this_run"]-bp_seconds),
            "note":"Component estimates only; full evaluation/feature extraction/artifact I/O and30% remaining-window reserve still need budgeting."},
        "long_benchmark_ready":False,"gpu_quota_reported_hours":26.0,"gpu_quota_reported_date":"2026-11-04",
        "quota_note":"Sang reported26 GPU hours beforeBP5; compare measured runtime with quota and the session window before160 epochs.",
        "next_step":"Send this JSON and saved Kaggle version; assistant checks quota/budget and prepares benchmark seed42."}
    save_json(readiness,Path(report["output_dir"])/"profile_base_ready.json")
    save_json(readiness,next_report_path)
except BaseException as error:
    save_json({"status":"failed","verification_passed":False,"error_type":type(error).__name__,
               "tests":TEST_PROOF},next_report_path)
    raise
'''.replace('__REFERENCE__', repr(reference))
    notebook.cells[8].source = '## Hoàn tất — tải profile_base_ready.json và gửi link version'
    notebook.cells[9].source += '''
print("TẢI FILE NÀY:",next_report_path)
print("BP5 seconds:",readiness["time_budget"]["bp5_seconds"])
print("Peak memory:",readiness["memory"])
print("CUDA resume:",TEST_PROOF["cuda_resume_passed"])
print("BP5 COMPLETE — gửi profile_base_ready.json; trợ lý sẽ tính ngân sách và chuẩn bị benchmark.")
'''
    notebook.metadata['acil_preset'] = 'profile_base_5_epochs'
    target = root / 'notebooks/05_profile_base_ready.ipynb'
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, 'notebook-cell', 'exec')
    nb.validate(notebook)
    # nbformat mặc định newline Windows; chuẩn hóa bản được công bố.
    target.write_text(nb.writes(notebook), encoding='utf-8', newline='\n')
    print('Created:',target,'bundle:',notebook.metadata['acil_source_bundle_sha256'])
    return target


if __name__ == '__main__':
    build()
