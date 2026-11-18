"""Tạo bộ 6 notebooks Kaggle GPU T4 cho thực nghiệm cuối kỳ E1 (E1a & E1b, 3 seeds 42/43/44).

Giữ nguyên gói mã nguồn cơ sở (bundle R1) để vượt qua toàn bộ kiểm tra phân phối và resume CUDA,
chỉ nạp bổ sung runner E1 độc lập và ghi nhận kết quả offline W&B theo đúng giao thức v1.0.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]

E1_NOTEBOOK_SPECS: tuple[dict, ...] = (
    {"num": 13, "branch": "E1a", "seed": 42, "slug": "13-e1a-base-ratio-seed42", "base_list": (10, 20, 30), "split_mode": "uniform"},
    {"num": 14, "branch": "E1a", "seed": 43, "slug": "14-e1a-base-ratio-seed43", "base_list": (10, 20, 30), "split_mode": "uniform"},
    {"num": 15, "branch": "E1a", "seed": 44, "slug": "15-e1a-base-ratio-seed44", "base_list": (10, 20, 30), "split_mode": "uniform"},
    {"num": 16, "branch": "E1b", "seed": 42, "slug": "16-e1b-superclass-shift-seed42", "base_list": (50,), "split_mode": "superclass_disjoint"},
    {"num": 17, "branch": "E1b", "seed": 43, "slug": "17-e1b-superclass-shift-seed43", "base_list": (50,), "split_mode": "superclass_disjoint"},
    {"num": 18, "branch": "E1b", "seed": 44, "slug": "18-e1b-superclass-shift-seed44", "base_list": (50,), "split_mode": "superclass_disjoint"},
)


def build_one_e1_notebook(spec: dict) -> Path:
    """Sinh một tệp notebook ``.ipynb`` hoàn chỉnh cho cấu hình E1a hoặc E1b trên Kaggle T4.

    Tham số:
        spec: Dict đặc tả gồm ``num``, ``branch``, ``seed``, ``slug``, ``base_list`` và ``split_mode``.

    Trả về:
        Đối tượng ``Path`` của tệp notebook vừa được ghi vào thư mục ``notebooks/``.
    """
    template = nb.read(ROOT / "notebooks/06_benchmark_seed42_ready.ipynb", as_version=4)
    num = spec["num"]
    branch = spec["branch"]
    seed = spec["seed"]
    base_list = spec["base_list"]
    split_mode = spec["split_mode"]
    suite_prefix = f"{branch.lower()}_seed{seed}"

    runner_text = (ROOT / "scripts/e1_base_assumption_runner.py").read_text(encoding="utf-8")
    e2_text = (ROOT / "scripts/e2_memory_precision_runner.py").read_text(encoding="utf-8")
    tests_text = (ROOT / "scripts/tests/test_final_empirics.py").read_text(encoding="utf-8")
    sources = {
        "scripts/__init__.py": "",
        "scripts/tests/__init__.py": "",
        "scripts/e1_base_assumption_runner.py": runner_text,
        "scripts/e2_memory_precision_runner.py": e2_text,
        "scripts/tests/test_final_empirics.py": tests_text,
    }

    cells = template.cells[:4]
    if branch == "E1a":
        cells[0].source = f"""# {num:02d} — Thực nghiệm Cuối kỳ E1a: Thu hẹp quy mô lớp Base c0 in {{10, 20, 30}}, seed {seed}

Kiểm chứng Hạn chế lý thuyết P1 (nhánh quy mô lớp Base đóng băng) trên CIFAR-100, seed {seed}.
Ba cấu hình mới chạy tuần tự trên GPU T4: `c0 = 10` (`K=9`), `c0 = 20` (`K=8`), `c0 = 30` (`K=7`), mỗi pha tăng dần thêm đúng 10 lớp cho tới đủ 100 lớp. Đối chứng `c0 = 50` (`K=5`) tái sử dụng trực tiếp từ R1 seed {seed}.

Giữ nguyên kiến trúc ResNet-32, SGD 160 epochs, `d = 8000`, `gamma = 0.1`, minibatch `b = 128`, `float64` và chuẩn hóa `base_train_only` trên đúng `500 * c0` ảnh train Phase 0. W&B ghi offline để đồng bộ lên `hsangg-2006/acil-reproduction` kèm nhãn `final-empirics`, `P1-base-assumption`, `P1-base-shift`.
"""
    else:
        cells[0].source = f"""# {num:02d} — Thực nghiệm Cuối kỳ E1b: Dịch chuyển ngữ nghĩa Superclass-Disjoint (50 + 5×10), seed {seed}

Kiểm chứng Hạn chế lý thuyết P1 (nhánh đồng phân phối ngữ nghĩa) trên CIFAR-100, seed {seed}.
Phase 0 huấn luyện 160 epochs trên đúng 50 lớp thuộc 10 coarse superclasses Tự nhiên/Thực vật/Động vật (`25,000` ảnh train); Phases 1..5 học giải tích tuần tự 50 lớp thuộc 10 coarse superclasses Đồ vật nhân tạo/Phương tiện & nhóm ngoại miền còn lại. Đối chứng đồng phân phối là R1 seed {seed}.

W&B ghi offline vào `/kaggle/working` để đồng bộ công khai lên `hsangg-2006/acil-reproduction` kèm nhãn `final-empirics`, `P1-base-assumption`, `P1-base-shift`, `E1b-superclass-shift`.
"""

    cells[1].source = cells[1].source.replace(
        'ACTION = "benchmark"  # Preset R1:160 epochs, d8000, K5, seed42.',
        f'ACTION = "final_empirics_{branch.lower()}"  # {branch} seed {seed}.',
    )

    cells.append(
        nb.v4.new_code_cell(
            "E1_SOURCES = "
            + repr(sources)
            + """
for rel_name, content in E1_SOURCES.items():
    dst = PROJECT_ROOT / rel_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8", newline="\\n")
print("Unpacked E1 sources:", {k: hashlib.sha256(v.encode("utf-8")).hexdigest()[:12] for k, v in E1_SOURCES.items() if v})
"""
        )
    )

    cells.append(nb.v4.new_markdown_cell("## Kiểm tra nhanh môi trường CUDA và các unit tests giao thức E1"))
    cells.append(
        nb.v4.new_code_cell(
            """from datetime import datetime, timezone
import json, subprocess, uuid
working = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else PROJECT_ROOT / "reports/local"
proof_dir = working / "e1_checks" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
proof_dir.mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available(), "Thực nghiệm E1 yêu cầu GPU CUDA T4 trên Kaggle"
e1_test_names = [
    "scripts.tests.test_final_empirics.FinalEmpiricsTests.test_coarse_mapping_bijection_and_parquet_verification",
    "scripts.tests.test_final_empirics.FinalEmpiricsTests.test_e1a_protocol_invariants_and_exact_r1_identity",
    "scripts.tests.test_final_empirics.FinalEmpiricsTests.test_e1_base_only_normalization_blocks_future_and_test_leakage",
    "scripts.tests.test_final_empirics.FinalEmpiricsTests.test_e1b_superclass_disjoint_vs_stratified_partitions",
    "scripts.tests.test_final_empirics.FinalEmpiricsTests.test_e1_end_to_end_smoke_theorem_3_1_and_wandb_tags",
]
proc = subprocess.run(
    [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "-v", *e1_test_names],
    cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8"
)
(proof_dir / "e1_unit_stdout.log").write_text(proc.stdout, encoding="utf-8", newline="\\n")
(proof_dir / "e1_unit_stderr.log").write_text(proc.stderr, encoding="utf-8", newline="\\n")
print(proc.stdout)
if proc.returncode != 0:
    print(proc.stderr)
proc.check_returncode()
TEST_PROOF = {
    "status": "complete",
    "e1_unit_tests_passed": True,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0),
}
(proof_dir / "e1_test_proof.json").write_text(json.dumps(TEST_PROOF, indent=2) + "\\n", encoding="utf-8", newline="\\n")
print("E1 pre-flight checks PASSED on:", TEST_PROOF["cuda_device"])
"""
        )
    )

    cells.append(nb.v4.new_markdown_cell(f"## Thực thi {branch} (seed {seed}) và lưu kết quả + W&B offline"))
    cells.append(
        nb.v4.new_code_cell(
            f"""import gc
from src.reporting import save_json
from scripts.e1_base_assumption_runner import default_e1_config, run_e1_case

ready_path = working / "{suite_prefix}_ready.json"
suite_state = {{
    "status": "running",
    "verification_passed": False,
    "branch": {branch!r},
    "seed": {seed},
    "split_mode": {split_mode!r},
    "expected_base_classes": list({base_list!r}),
    "tests": TEST_PROOF,
    "cases": [],
}}
save_json(suite_state, ready_path)

try:
    for base_c in {base_list!r}:
        case_key = f"{branch.lower()}_{split_mode}_c0_{{base_c}}_seed{seed}"
        suite_state["current_case"] = case_key
        save_json(suite_state, ready_path)
        gc.collect()
        torch.cuda.empty_cache()

        cfg = default_e1_config(
            branch={branch!r},
            seed={seed},
            base_classes=base_c,
            phase_size=10,
            split_mode={split_mode!r},
        )
        cfg.update(
            device="cuda",
            torch_threads=min(4, os.cpu_count() or 2),
        )
        case_dir = working / "{suite_prefix}" / f"c0_{{base_c}}"
        report = run_e1_case(
            cfg,
            data_dir=working / "cifar-data",
            output_dir=case_dir,
            save_large_r_npz=False,
            wandb_mode="offline",
            entity="hsangg-2006",
            project="acil-reproduction",
        )
        assert report["status"] == "complete" and report["verification_passed"]
        assert report["streaming"]["stored_training_samples"] == 0
        suite_state["cases"].append({{
            "case_key": case_key,
            "base_classes": base_c,
            "attempt_id": report["attempt_id"],
            "results_path": str(case_dir / "results.json"),
            "metric_summary": report["metric_summary"],
            "wandb_status": report["wandb_status"],
            "wandb_run_id": report["wandb_run_id"],
            "wandb_tags": report.get("wandb_tags", []),
        }})
        save_json(suite_state, ready_path)
        print("COMPLETED:", case_key, report["metric_summary"], flush=True)

    suite_state.update(status="complete", verification_passed=True, current_case=None)
    save_json(suite_state, ready_path)
except BaseException as err:
    suite_state.update(status="failed", verification_passed=False, error_type=type(err).__name__)
    save_json(suite_state, ready_path)
    raise
"""
        )
    )

    cells.append(nb.v4.new_markdown_cell("## Xác nhận hoàn tất toàn bộ cấu hình trong phiên"))
    cells.append(
        nb.v4.new_code_cell(
            f"""assert suite_state["status"] == "complete" and suite_state["verification_passed"]
assert [c["base_classes"] for c in suite_state["cases"]] == list({base_list!r})
for item in suite_state["cases"]:
    print(item["case_key"], "-> final_acc:", item["metric_summary"]["final_accuracy_pct"],
          "inc_acc:", item["metric_summary"]["final_inc_accuracy_pct"],
          "offline_id:", item["wandb_run_id"])
print("SUITE READY:", ready_path)
"""
        )
    )

    out_nb = nb.v4.new_notebook(cells=cells, metadata=dict(template.metadata))
    out_nb.metadata.update(
        acil_preset=f"{branch}_{split_mode}_seed{seed}",
        acil_seed=seed,
        acil_e1_runner_sha256=hashlib.sha256(runner_text.encode("utf-8")).hexdigest(),
    )
    for cell in out_nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            compile(cell.source, f"notebook-{num}", "exec")
    nb.validate(out_nb)
    target = ROOT / f"notebooks/{num:02d}_{branch.lower()}_{'base_ratio' if branch == 'E1a' else 'superclass_shift'}_seed{seed}.ipynb"
    target.write_text(nb.writes(out_nb), encoding="utf-8", newline="\n")
    return target


def build_all() -> list[Path]:
    """Sinh toàn bộ 6 tệp notebooks Kaggle GPU T4 cho thực nghiệm E1a và E1b.

    Tham số:
        Không nhận tham số.

    Trả về:
        Danh sách 6 đường dẫn ``Path`` của các tệp notebook đã tạo.
    """
    paths = [build_one_e1_notebook(spec) for spec in E1_NOTEBOOK_SPECS]
    for path in paths:
        print("Prepared E1 notebook:", path.relative_to(ROOT).as_posix())
    return paths


if __name__ == "__main__":
    build_all()
