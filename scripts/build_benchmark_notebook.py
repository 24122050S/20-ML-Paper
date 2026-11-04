"""Tạo notebook R1 seed42 từ bundle đã nghiệm thu; không khởi chạy GPU."""

import json
from pathlib import Path
import nbformat as nb


def build():
    root=Path(__file__).resolve().parents[1]
    notebook=nb.read(root/'notebooks/archive/05_profile_base_api_v3.ipynb',as_version=4)
    budget=json.loads((root/'reports/benchmark_seed42_budget.json').read_text(encoding='utf-8'))
    budget.pop('long_benchmark_started',None)  # Trạng thái lúc lập kế hoạch không phải trạng thái run tương lai.
    notebook.cells[0].source='''# 06 — R1 CIFAR100/K5, seed42,160 epochs, d8000

Cấu hình chính đã chốt, không reuse checkpoint5 epochs. Test resume CUDA và hai profile đã được kiểm. Dự trù khoảng59 phút, chuẩn bị phiên2 giờ để còn hơn30% dự phòng; đây là ước lượng, không phải cam kết runtime.

Bật GPU T4/Internet. Source không cần Input. API runner dùng timeout7200 giây. W&B được ghi offline trong Output, sau đó trợ lý sync trên máy đã đăng nhập; không cần đưa key vào notebook hoặc bật secret cho lượt API.

Chạy mới giữ CHECKPOINT=None, RESUME_CHECKPOINT=None, STOP_AFTER_EPOCH=None. Nếu training bị ngắt, chỉ dùng checkpoint v2 **cùng target160/config/runtime/source** để resume. Checkpoint5 epochs bị từ chối.

Cuối lượt: benchmark_seed42_ready.json, benchmark_seed42/attempts/<id>/base_checkpoint.pt, analytic_state.npz và wandb/offline-run... phải được giữ cùng version. Không coi lỗi/số liệu cũ là kết quả mới. Tạo file notebook này không tự khởi chạy benchmark.
'''
    setup=notebook.cells[1].source
    setup=setup.replace('ACTION = "profile_base"','ACTION = "benchmark"')
    setup=setup.replace('  # Đã cấu hình5 epochs; không đổi sang benchmark ở lượt này.','  # Preset R1:160 epochs, d8000, K5, seed42.')
    setup=setup.replace('CHECKPOINT = None  # Profile: attach Output notebook03 làm Input; tự tìm nếu có đúng1 checkpoint.',
                        'CHECKPOINT = None  # Chỉ reuse base160 epochs đã hoàn tất, cùng config.')
    setup=setup.replace('RESUME_CHECKPOINT = None  # Chỉ checkpoint v2 đang dở; khác reuse base hoàn tất.',
                        'RESUME_CHECKPOINT = None  # Chỉ checkpoint v2 đang dở với target160 epochs.')
    notebook.cells[1].source=setup
    notebook.cells[5].source=notebook.cells[5].source.replace('"profile_base_checks"','"benchmark_checks"')
    notebook.cells[6].source='## Chạy R1 — yêu cầu CUDA resume đạt, giữ đúng protocol160/d8000'
    notebook.cells[7].source='''import time
from src.cifar_smoke import run
from src.profile_analytic import MemoryMonitor
from src.reporting import save_json
import wandb

ready_path=working/"benchmark_seed42_ready.json"
save_json({"status":"running","verification_passed":False,"tests":TEST_PROOF},ready_path)
try:
    if ACTION!="benchmark" or not TEST_PROOF["cuda_resume_passed"] or not torch.cuda.is_available():
        raise RuntimeError("R1 cần GPU và test resume CUDA đạt.")
    if CHECKPOINT is not None and RESUME_CHECKPOINT is not None:
        raise ValueError("Chọn reuse hoặc resume, không chọn đồng thời.")
    with MemoryMonitor("cuda") as resources:
        report=run(None,config_path=PROJECT_ROOT/"configs/cifar100_benchmark_k5.json",
            config_overrides={"experiment":"ACIL-R1-C100-K5","purpose":"benchmark","seed":42,"device":"cuda",
                              "torch_threads":min(4,os.cpu_count() or 2),"run_incremental_phases":5},
            data_dir=working/"cifar-data",output_dir=working/"benchmark_seed42",
            reuse_checkpoint=CHECKPOINT,resume_checkpoint=RESUME_CHECKPOINT,stop_after_epoch=STOP_AFTER_EPOCH,
            log_wandb=False)
    assert report["status"]=="complete" and report["verification_passed"]
    assert report["config"]["epochs"]==160 and report["config"]["expansion_dim"]==8000
    assert len(report["training_history"])==160 and len(report["results"])==6
    assert report["results"][-1]["weights_shape"]==[8000,100]
    assert report["results"][-1]["test_samples_all_seen"]==10000
    assert report["joint_reference_status"]=="not_run" and report["streaming"]["stored_training_samples"]==0
    assert report["metric_summary"]["complete_protocol"] is True
    assert report["metric_summary"]["average_incremental_accuracy_pct"] is not None
    checkpoint=torch.load(Path(report["output_dir"])/"base_checkpoint.pt",map_location="cpu",weights_only=True)
    assert checkpoint["checkpoint_format_version"]==2 and checkpoint["training_complete"]
    assert checkpoint["completed_epoch"]==checkpoint["target_epochs"]==160
    bp_seconds=sum(row["seconds"] for row in report["training_history"])
    with wandb.init(entity="hsangg-2006",project="acil-reproduction",mode="offline",dir=str(working),
                    job_type="benchmark",tags=["R1","seed42","K5","d8000","offline-to-sync"],
                    config={**report["config"],"source_sha256":report["provenance"]["source_sha256"],
                            "kaggle_attempt_id":report["attempt_id"],"logging_origin":"Kaggle offline; sync after run"}) as offline_run:
        for row in report["results"]:
            offline_run.log({k:v for k,v in row.items() if isinstance(v,(int,float,bool))},step=row["phase"])
        offline_run.summary.update(report["metric_summary"])
        offline_run.summary.update({"verification_passed":report["verification_passed"],
            "cuda_resume_passed":True,"bp160_seconds":bp_seconds,
            "profile/peak_cpu_rss_bytes":resources.result["peak_cpu_rss_bytes"],
            "profile/peak_cuda_allocated_bytes":resources.result["peak_cuda_allocated_bytes"]})
        offline_id=offline_run.id
        offline_directory=str(Path(offline_run.dir).parent)
    ready={"status":"complete","verification_passed":True,"scope":"R1 seed42 full protocol; single seed, not three-seed aggregate",
           "attempt_id":report["attempt_id"],"benchmark":report,"tests":TEST_PROOF,
           "memory":{"scope":"training+analytic; notebook tests and offline logging excluded",**resources.result},
           "planning_budget":__BUDGET__,
           "tracking":{"status":"offline_pending_sync","offline_run_id":offline_id,"offline_directory":offline_directory}}
    save_json(ready,Path(report["output_dir"])/"benchmark_seed42_ready.json")
    save_json(ready,ready_path)
except BaseException as error:
    save_json({"status":"failed","verification_passed":False,"error_type":type(error).__name__,"tests":TEST_PROOF},ready_path)
    raise
'''.replace('__BUDGET__',repr(budget))
    notebook.cells[8].source='## Kết quả đủ100 lớp và log offline của chính lượt này'
    notebook.cells[9].source='''assert ready["status"]=="complete" and ready["verification_passed"]
print("R1 SEED42 COMPLETE",ready_path)
print("Metrics:",report["metric_summary"])
print("Memory:",ready["memory"])
print("Offline tracking:",ready["tracking"])
print("Attempt:",report["attempt_id"],"Artifacts:",report["output_dir"])
print("Giữ Output/checkpoint và sync W&B trước khi công bố benchmark.")
'''
    notebook.metadata['acil_preset']='R1_seed42_160epochs_d8000_K5'
    for cell in notebook.cells:
        if cell.cell_type=='code':compile(cell.source,'cell','exec')
    nb.validate(notebook)
    target=root/'notebooks/06_benchmark_seed42_ready.ipynb'
    target.write_text(nb.writes(notebook),encoding='utf-8',newline='\n')
    print('Prepared:',target,'No GPU run was submitted.')
    return target


if __name__=='__main__':
    build()
