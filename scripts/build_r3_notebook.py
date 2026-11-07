"""Tạo notebook12 từ bundle bất biến R1, thêm source R3 riêng; không gửi GPU job."""
from pathlib import Path
import ast
import hashlib
import json
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def build():
    old=nb.read(ROOT/'notebooks/09_r2_ablation_seed42.ipynb',as_version=4)
    tree=ast.parse(old.cells[7].source)
    reference=ast.literal_eval(next(node.value for node in tree.body if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='R1_REFERENCE'))
    files=['scripts/r3_worker.py','scripts/r3_runner.py','scripts/tests/test_r3_analysis.py']
    sources={name:(ROOT/name).read_text(encoding='utf-8') for name in files}
    cells=old.cells[:4]
    cells[0].source='''# 12 — R3: batch32/128/512, seed42, d1024,3 phép đo mỗi batch

Add Input: `hsangg206/06-benchmark-seed42/1`, GPU T4 và Internet. Chỉ nhận checkpoint160 đúng SHA R1; không train lại backbone. Dùng đủ CIFAR100 50.000 train/10.000 test,50 base+5×10 lớp.

Trích F64 một lần ngoài phép đo.9 worker CPU mới chạy tuần tự, BLAS4 threads; thứ tự32→128→512,128→512→32,512→32→128. FE N(0,1)64×1024 từseed1042, gamma0.1 và thứ tự ảnh cố định. Batch thay cả base accumulation và incremental RLS. Đo tổng6 cửa sổ học (có FE), báo riêng base/incremental; CNN/evaluation/IO/oracle/log không tính.

Sau9 worker mới chạy joint oracle ở tiến trình riêng và so đủ54 phase comparisons theo rtol1e-6/atol1e-8, không có argmax khác. RSS lấy mẫu5ms trong các cửa sổ học; báo baseline/peak/increase và bytes state riêng. Ba lần là timing repetitions trên1 seed, không phải3 seed benchmark. Accuracy d1024 không thay số R1 d8000.

Mỗi suite có thư mục riêng dưới`r3/<suite_id>`, fixture/source hashes, trạng thái running/failed/complete. Giữ toàn bộ Output và9 payload W&B offline; chỉ công bố logged sau khi sync/xác minh thành công. Timeout API7200s là trần phiên, không phải thời gian thực đo. Chi tiết: `docs/R3_KE_HOACH_DO.md` trong repo.
'''
    cells[1].source=cells[1].source.replace('ACTION = "ablation"  # R2 tái dùng checkpoint, không train base.',
        'ACTION = "batch_analysis"  # R3 chỉ reuse checkpoint160 đã pin.')
    cells.append(nb.v4.new_code_cell('R3_SOURCES = '+repr(sources)+'''
for name,text in R3_SOURCES.items():
    target=PROJECT_ROOT/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(text,encoding="utf-8",newline="\\n")
print("R3 sources:",{name:hashlib.sha256(text.encode()).hexdigest() for name,text in R3_SOURCES.items()})
'''))
    cells.append(nb.v4.new_markdown_cell('## Core GPU tests và5 kiểm tra driver R3 độc lập'))
    test=old.cells[5].source.replace('"r2_checks"','"r3_checks"')
    # Chạy lại cùng recorder cho test R3 riêng, không sửa54 test/source core cũ.
    original_command=next(node.value for node in ast.walk(ast.parse(test))
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and 'class RecordedResult' in node.value)
    driver_command=original_command.replace('discover("tests")','discover("scripts/tests")')
    test+='\nr3_path=proof_path.with_name("r3_driver_tests.json")\nr3_checked=subprocess.run([sys.executable,"-B","-X","utf8","-c",'+repr(driver_command)+''',str(r3_path),BUNDLE_SHA256],
    cwd=PROJECT_ROOT,capture_output=True,text=True,encoding="utf-8")
(proof_dir/"r3_stdout.log").write_text(r3_checked.stdout,encoding="utf-8",newline="\\n")
(proof_dir/"r3_stderr.log").write_text(r3_checked.stderr,encoding="utf-8",newline="\\n")
print(r3_checked.stdout);print(r3_checked.stderr);r3_checked.check_returncode()
TEST_PROOF["r3_driver_tests"]=json.loads(r3_path.read_bytes())
assert TEST_PROOF["r3_driver_tests"]["tests_run"]==5 and TEST_PROOF["r3_driver_tests"]["skipped"]==0
'''
    cells.append(nb.v4.new_code_cell(test))
    cells.append(nb.v4.new_markdown_cell('## Chạy R3:9 phép đo, oracle và9 W&B offline'))
    cells.append(nb.v4.new_code_cell('from scripts.r3_runner import run_suite\nR1_REFERENCE = '+repr(reference)+'''
assert ACTION=="batch_analysis"
R3_RESULT=run_suite(reference=R1_REFERENCE,checkpoint=CHECKPOINT,project_root=PROJECT_ROOT,
                    working=working,test_proof=TEST_PROOF)
'''))
    cells.append(nb.v4.new_markdown_cell('## Chỉ hoàn tất khi đủ9 case/54 so sánh'))
    cells.append(nb.v4.new_code_cell('''assert R3_RESULT["status"]=="complete" and R3_RESULT["verification_passed"]
assert len(R3_RESULT["cases"])==9
for case in R3_RESULT["cases"]:
    print(case["key"],case["report"]["learning_seconds"],case["tracking"])
print("R3 complete; giữ Output và sync9 payload trước khi công bố tracking.")
'''))
    notebook=nb.v4.new_notebook(cells=cells,metadata=dict(old.metadata))
    notebook.metadata.update(acil_preset='R3_seed42_d1024_batch32_128_512_repeat3',acil_r3_source_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()})
    notebook.metadata.pop('acil_r2_runner_sha256',None)
    for cell in notebook.cells:
        if cell.cell_type=='code':compile(cell.source,'r3-notebook','exec')
    nb.validate(notebook);path=ROOT/'notebooks/12_r3_batch_analysis.ipynb'
    path.write_text(nb.writes(notebook),encoding='utf-8',newline='\n')
    print('Prepared',path.name,'No GPU submission');return path


if __name__=='__main__':build()
