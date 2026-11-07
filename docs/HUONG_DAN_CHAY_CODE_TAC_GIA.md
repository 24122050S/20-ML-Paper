# Chạy đối chiếu code tác giả ACIL

Đây là comparison với source tác giả ở commit `0afd6bf87622e808ce7f730f8556e78229b9e758`. Nó dùng checkpoint160 seed42 và full features của R3, không train backbone theo recipe mặc định300epochs của repo tác giả. Dùng [kế hoạch và ngưỡng kiểm](DOI_CHIEU_CODE_TAC_GIA_KE_HOACH.md) để diễn giải đúng phạm vi.

## 1. Môi trường riêng

Chạy từ thư mục gốc `20-ML-Paper` trong PowerShell. Không cài đè môi trường R1/R2/R3:

```powershell
py -3.12 -m venv .venv-author
.\.venv-author\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-author\Scripts\python.exe -m pip install -r requirements-author-comparison.txt
git clone --no-checkout https://github.com/ZHUANGHP/Analytic-continual-learning.git ..\external\Analytic-continual-learning
git -C ..\external\Analytic-continual-learning checkout --detach 0afd6bf87622e808ce7f730f8556e78229b9e758
```

Nếu checkout đã tồn tại, kiểm HEAD/status và dùng lại đúng commit; không clone đè hoặc xóa thư mục có dữ liệu. YAML tác giả ghi Python3.11/torch>=2.2; lượt này dùng Python3.12/torch2.5.1CPU và công bố sai khác môi trường đó.

## 2. Lấy input đúng version

Máy Sang đã có các input được kiểm hash. Khi chạy từ máy mới, dùng Kaggle CLI **2.2.5**, đăng nhập bằng credential lưu tại máy, rồi tải Output cố định `/1`; bản CLI2.1 cũ không được dùng để xác nhận pin version:

```powershell
kaggle kernels output hsangg206/12-r3-batch-analysis/1 -p reports/local/author-inputs/r3 --page-size 200 --file-pattern "^(r3_ready\.json|r3/.*\.(json|npy|npz))$"
kaggle kernels output hsangg206/06-benchmark-seed42/1 -p reports/local/author-inputs/r1 --page-size 200 --file-pattern ".*base_checkpoint\.pt$"
```

Nếu Output trả `nextPageToken`, tải tiếp các trang; không chạy khi input thiếu. Các version dùng ở đây là R3 scriptVersionId366444048 và R1seed42 scriptVersionId365844618. Full R3 Output gồm fixture ảnh/feature và state/oracle phục vụ kiểm chứng; đây là dữ liệu có sẵn từ Kaggle, không chuẩn bị dataset bằng tay.

`--r3-root` phải trỏ vào thư mục suite có `fixture/`, `cases/`, `oracle/`, ví dụ `.../r3/20261107T012047Z-0ca7e9aa10`. `r3_ready.json` nằm ở thư mục Output, hai cấp phía trên suite. `--checkpoint` trỏ đúng file `base_checkpoint.pt`, SHA-256 `1f2a747347e26db933b11a9910ee780a8f8597187b40bc2191ad42108cea550d`. Runner kiểm hash fixture, từng file, checkpoint, source nhóm và Git source tác giả trước khi chạy.

## 3. Chạy hai phép đối chiếu

Ví dụ sau khi tải theo đường dẫn ở mục2:

```powershell
$authorSuite = Get-ChildItem -LiteralPath reports/local/author-inputs/r3/r3 -Directory
$authorCheckpoint = Get-ChildItem -LiteralPath reports/local/author-inputs/r1 -Recurse -File -Filter base_checkpoint.pt
if ($authorSuite.Count -ne 1 -or $authorCheckpoint.Count -ne 1) { throw "Input thiếu hoặc không duy nhất; kiểm version/hash trước khi chạy." }
.\.venv-author\Scripts\python.exe -B -u -X utf8 scripts/compare_author_acil.py --author-root ..\external\Analytic-continual-learning --r3-root $authorSuite.FullName --checkpoint $authorCheckpoint.FullName --output-dir reports/local/author-comparison
```

- C1 gọi `ACIL.fit/forward/update` thật, FE1024 giống R3, cùng sample order/batch128. So từng phase với state nhóm và joint đã kiểm.
- C2 gọi `main.main(args)` thật, giữ native FE initializer và native shuffle của tác giả. Adapter ban đầu trả `Features` đã pin thay nguồn ảnh; observer lưu R/W và guard cấm train mới. Group reference/joint của C2 dùng chính FE native đó.
- Mapping/preprocessing/backbone được kiểm riêng; sai khác FP32 vẫn được giữ cùng kết quả kiểm, không nới tolerance để gọi là khớp.

Output nằm trong attempt riêng. Root `results.json`/`progress.json` trỏ cùng attempt; input lỗi hoặc kiểm không đạt phải ghi failed, không dùng success cũ. State lớn, cache và ảnh ở `reports/local/`, không commit Git. Core tác giả được import trực tiếp từ checkout ngoài repo; không có bản sao code lõi tác giả trong `src/` nhóm.

Lượt đã đo có C1 score gate không đạt ở2phase dù mọi phase đã thực thi; runner trả exit1 và giữ `status=failed`. Đây là kết quả cần phân tích, không sửa JSON/tolerance để đổi sang pass. Helper tracking chỉ chấp nhận trường hợp có đủ hai case/sáu phase và đúng lỗi gate đã khai báo; lỗi kỹ thuật hoặc input thiếu không được nhận làm kết quả hoàn chỉnh.

## 4. Tracking sau khi kiểm kết quả

Dùng môi trường tracking hiện có của nhóm (có `requirements-tracking.txt`), không cần cài W&B vào venv tác giả:

```powershell
python scripts/track_author_comparison.py --suite reports/local/author-comparison/results.json --output reports/local/author-comparison/ready.json
```

Helper lưu2payload offline, lấy run ID/URL thật, sync và đối chiếu config/summary/6history rows qua SDK lẫn API công khai. Nếu sync lỗi, chạy lại helper trên cùng JSON để dùng payload đã lưu; không chạy lại numeric. Raw JSON không bị sửa thành synced trước khi logging kết thúc. Không dán API key vào notebook/command/chat.

Diagnostic C1R hậu nghiệm có thể chạy riêng bằng `scripts/diagnose_author_base_precision.py`, truyền `--suite`, `--r3-root`, `--author-root` như trên và `--output` là một thư mục mới. Nó tiêm state base R3 rồi kiểm5updates; không được gọi là cold-start hoặc thay C1. Khi log, thêm `--diagnostic <thư_mục_diagnostic>/results.json` vào lệnh tracking để có run thứ3/5history rows. Finished mô tả tracking hoàn tất; summary/config vẫn giữ scientific flag C1=false.

Môi trường thực chạy được đóng băng tại [environment freeze](../reports/author_comparison_environment_freeze.txt). Sau khi cài torch/torchvision từ CPU index, có thể dùng freeze này để khôi phục chính xác các dependency còn lại. Tái tạo summary bằng `python scripts/aggregate_author_comparison.py`, vẽ hình bằng `python scripts/plot_author_comparison.py` trong môi trường `requirements-plots.txt`.
