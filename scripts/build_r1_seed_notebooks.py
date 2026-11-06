"""Tạo riêng seed43/44 từ snapshot seed42; không sửa notebook06 hay gửi job GPU."""

import ast
import copy
from pathlib import Path

import nbformat as nb


ROOT = Path(__file__).resolve().parents[1]


def build(seed):
    """Giữ nguyên bundle/recipe, chỉ đổi seed, tên artifact và metadata dự trù."""
    if seed not in (43, 44):
        raise ValueError("Script này chỉ tạo hai seed còn lại43/44 của R1")
    notebook = nb.read(ROOT / "notebooks/06_benchmark_seed42_ready.ipynb", as_version=4)
    number = seed - 36
    notebook.cells[0].source = f'''# {number:02d} — R1 CIFAR100/K5, seed{seed},160 epochs, d8000

Train base mới cho seed{seed}; class order, khởi tạo backbone/augmentation và FE đều theo seed này. Không dùng checkpoint seed42 hoặc seed khác. Giữ nguyên recipe của seed42 đã nghiệm thu: gamma0.1,50 base+5×10 lớp, LR80/120, d8000.

GPU T4/Internet; không cần Input. CHECKPOINT=None, RESUME_CHECKPOINT=None, STOP_AFTER_EPOCH=None. API dùng timeout7200 giây. Seed42 đo khoảng53,78 phút numeric; đây là tham chiếu dự trù cho phiên này, không bảo đảm runtime bằng nhau.

W&B ghi offline vào Output, sync đúng payload trên máy đã đăng nhập trước khi công bố. Giữ benchmark_seed{seed}_ready.json, checkpoint/state/progress/results và log offline cùng version/attempt. Source bundle giống notebook06; không sửa source bên trong bundle.

Chỉ sau khi đủ cả42/43/44 mới tổng hợp mean±sample_SD (ddof=1). Tạo notebook này không tự gửi job GPU.
'''
    notebook.cells[1].source = notebook.cells[1].source.replace(
        "# Preset R1:160 epochs, d8000, K5, seed42.",
        f"# Preset R1:160 epochs, d8000, K5, seed{seed}.",
    )
    execute = notebook.cells[7].source
    tree = ast.parse(execute)
    ready_assignment = next(node for node in tree.body if isinstance(node, ast.Try))
    ready_dict = next(node.value for node in ast.walk(ready_assignment)
                      if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == "ready")
    budget_node = next(value for key, value in zip(ready_dict.keys, ready_dict.values)
                       if isinstance(key, ast.Constant) and key.value == "planning_budget")
    old_budget_text = ast.get_source_segment(execute, budget_node)
    budget = copy.deepcopy(ast.literal_eval(budget_node))
    budget.update(scope=f"Planning estimate for R1 seed{seed}; transferred from seed42 profiles, not measured runtime of this seed",
                  profile_reference_seed=42,
                  completed_seed42_numeric_pipeline_seconds=3226.82625715,
                  estimate_transfer_note="Same160/d8000/K5 recipe and2-hour timeout; actual timing recorded per seed")
    assert execute.count(old_budget_text) == 1
    execute = execute.replace(old_budget_text, repr(budget))
    assert execute.count('"seed":42') == 1
    execute = execute.replace('"seed":42', f'"seed":{seed}')
    execute = execute.replace("benchmark_seed42", f"benchmark_seed{seed}")
    execute = execute.replace('"seed42"', f'"seed{seed}"')
    execute = execute.replace("R1 seed42 full protocol", f"R1 seed{seed} full protocol")
    notebook.cells[7].source = execute
    notebook.cells[9].source = notebook.cells[9].source.replace("SEED42", f"SEED{seed}")
    notebook.metadata["acil_preset"] = f"R1_seed{seed}_160epochs_d8000_K5"
    notebook.metadata["acil_seed"] = seed
    notebook.metadata["acil_derived_from"] = "06_benchmark_seed42_ready.ipynb; same core bundle/recipe"
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            compile(cell.source, "notebook", "exec")
    nb.validate(notebook)
    target = ROOT / f"notebooks/{number:02d}_benchmark_seed{seed}_ready.ipynb"
    target.write_text(nb.writes(notebook), encoding="utf-8", newline="\n")
    print("Prepared:", target.name, "fresh base; no GPU submission")
    return target


if __name__ == "__main__":
    for seed in (43, 44):
        build(seed)
