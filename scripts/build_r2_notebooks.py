"""Tạo3 notebook R2 từ cùng bundle R1; giữ nguyên source của các lượt đã công bố."""

import hashlib
import json
from pathlib import Path
import nbformat as nb


ROOT = Path(__file__).resolve().parents[1]


def build(seed):
    """Mỗi seed chạy3 case analytic, dùng Input checkpoint R1 và reference bất biến."""
    if seed not in (42, 43, 44):
        raise ValueError("R2 dùng đúng seeds42/43/44")
    number = seed - 33
    report_path = ROOT / f"reports/benchmark_seed{seed}_kaggle_results.json"
    report = json.loads(report_path.read_bytes())
    manifest = json.loads((ROOT / f"reports/R1-SEED{seed}-KAGGLE-01-provenance.json").read_bytes())
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == manifest["artifacts"][report_path.name]["sha256"]
    reference = {"seed": seed, "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "backbone_sha256": report["backbone_sha256_before"], "expansion_sha256": report["expansion_sha256"],
        "source_sha256": report["provenance"]["source_sha256"], "dataset_source": report["dataset_source"],
        "wandb_run_id": manifest["tracking_sync"]["wandb_run_id"], "wandb_run_url": manifest["tracking_sync"]["wandb_run_url"],
        "r1_results_sha256": manifest["artifacts"][report_path.name]["sha256"],
        "r1_kernel_source": manifest["kaggle"]["kernel"] + "/1", "r1_kaggle_url": manifest["kaggle"]["url"],
        "baseline_config": report["config"], "baseline_metrics": report["metric_summary"]}
    runner = (ROOT / "scripts/r2_runner.py").read_text(encoding="utf-8")
    notebook = nb.read(ROOT / "notebooks/06_benchmark_seed42_ready.ipynb", as_version=4)
    notebook.cells[0].source = f'''# {number:02d} — R2 ablation, CIFAR100/K5, seed{seed}

Ba case mới: no-FE/gamma0.1 (F64 trực tiếp), FE8000/gamma0.01 và0.001. Hàng FE8000/gamma0.1 tái dùng kết quả R1, không chạy lại để chọn score đẹp. Chỉ tái lập4/5 hàng Table2 theo protocol; không gồm gamma0.

GPU T4/Internet. Add Input là Output R1 seed{seed}, version1: `{reference['r1_kernel_source']}`. Chỉ nhận checkpoint160 có đúng SHA đã pin; thiếu/sai Input phải dừng, tuyệt đối không train fallback. Class order/backbone/normalization/FE giữ nguyên trong cùng seed. Không dùng checkpoint seed khác.

Timeout API7200s. Dự trù hai case d8000 khoảng23–30 phút tổng, no-FE ngắn; đây là ước lượng từ R1. Mỗi case có output/attempt và W&B offline riêng. Ma trận lớn giữ trong Output; kiểm NPZ/hash/shape/finite/symmetry ngay trên Kaggle sau numeric timer, không giữ lịch sử train features hay chạy oracle joint d8000.

Giữ `r2_seed{seed}_ready.json`, các results/progress/proofs và offline payload cùng version. Chỉ công bố aggregate sau khi đủ9 case mới,3 baseline R1 và sync các run thành công.
'''
    notebook.cells[1].source = notebook.cells[1].source.replace(
        'ACTION = "benchmark"  # Preset R1:160 epochs, d8000, K5, seed42.',
        'ACTION = "ablation"  # R2 tái dùng checkpoint, không train base.')
    notebook.cells[1].source = notebook.cells[1].source.replace(
        'CHECKPOINT = None  # Chỉ reuse base160 epochs đã hoàn tất, cùng config.',
        'CHECKPOINT = None  # R2 tìm checkpoint Input theo SHA; không có nghĩa là train mới.')
    notebook.cells[5].source = notebook.cells[5].source.replace('"benchmark_checks"', '"r2_checks"')
    notebook.cells[6].source = "## Chạy3 case R2, chặn training và kiểm checkpoint/FE trước khi công bố"
    notebook.cells[7].source = runner + "\n\nR1_REFERENCE = " + repr(reference) + f'''\nassert ACTION == "ablation"
R2_RESULT = run_suite(seed={seed}, checkpoint=CHECKPOINT, reference=R1_REFERENCE,
    project_root=PROJECT_ROOT, working=working, test_proof=TEST_PROOF, device="cuda")
'''
    notebook.cells[8].source = "## Kết quả R2 của seed này — chưa phải aggregate3 seeds"
    notebook.cells[9].source = '''assert R2_RESULT["status"]=="complete" and R2_RESULT["verification_passed"]
for case in R2_RESULT["cases"]:
    print(case["case_key"],case["report"]["metric_summary"],case["tracking"])
print("R2 seed complete; giữ Output, sync từng run trước khi tổng hợp.")
'''
    notebook.metadata["acil_preset"] = f"R2_seed{seed}_three_cases_reuse160_K5"
    notebook.metadata["acil_seed"] = seed
    notebook.metadata["acil_r2_runner_sha256"] = hashlib.sha256(runner.encode()).hexdigest()
    notebook.metadata["acil_r1_kernel_source"] = reference["r1_kernel_source"]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []; cell.execution_count = None
            compile(cell.source, "r2-notebook", "exec")
    nb.validate(notebook)
    target = ROOT / f"notebooks/{number:02d}_r2_ablation_seed{seed}.ipynb"
    target.write_text(nb.writes(notebook), encoding="utf-8", newline="\n")
    print("Prepared:", target.name, "Input:", reference["r1_kernel_source"], "No GPU submission")
    return target


if __name__ == "__main__":
    for seed in (42, 43, 44):
        build(seed)
