# Sang tự chạy CIFAR-100: 50 base + 10 lớp mới

Bạn có thể thực hiện toàn bộ bước này một mình. Mục tiêu là nối pipeline ảnh với analytic head đã kiểm chứng, không chờ source của thành viên khác.

## Ba quy ước đã chốt

- ResNet-32 trả F `(n,64)`; `H=ReLU(F @ W_fe)` có shape `(n,256)`, float64. CNN/BatchNorm và expansion được giữ cố định sau base training.
- Nhãn CIFAR gốc được remap bằng cùng `class_order` cho train và test: base `0..49`, phase đầu `50..59`. Metadata lớp thuộc pipeline; API chỉ trả R/W.
- Base có W `(256,50)`. Sau mọi minibatch của phase đầu có W `(256,60)`, đánh giá đủ 6.000 ảnh của cả 60 lớp. Joint chỉ là nhánh kiểm chứng giữ lịch sử riêng.

Chi tiết và ví dụ nhãn: [đặc tả, mục 9](Dac_ta_ACIL.md#9-hợp-đồng-giữa-pipeline-cnn-và-analytic-head).

## Cách chạy trên Kaggle

1. Import [notebook 03 có trạng thái từng lượt](../notebooks/03_cifar100_safe_attempts.ipynb), bật Internet, chọn GPU nếu có. Notebook 02 giữ snapshot của run Version 2 đã công bố.
2. Source được đóng gói trong notebook, không phụ thuộc file nằm riêng ở workspace cũ hoặc một commit chưa được push.
3. Cell code đầu có `LOG_WANDB`, `RUN_INCREMENTAL_PHASES`, `REUSE_CHECKPOINT`. Giữ 2 epochs, expansion 256 và `RUN_INCREMENTAL_PHASES=1` cho 50+10; đặt 5 khi kiểm đủ 100 lớp. Save & Run All.
4. Kiểm tra `CIFAR-SMOKE-01 COMPLETE`; tải thư mục `cifar100_smoke` ở Output.
5. Nếu cần W&B, enable secret `WANDB_API_KEY` và đặt `LOG_WANDB=True`. JSON chỉ ghi `logged` sau khi run context kết thúc; URL lấy từ SDK.

Notebook giữ PyTorch có sẵn trên Kaggle để dùng GPU và ghi phiên bản thực tế. CPU cũng chạy được. Các thư viện còn lại được cài theo version đã chốt. Kết quả chỉ được gọi là chạy Kaggle sau khi có version và Output thật; notebook chưa chạy không được gắn run URL.

## Chạy trên máy từ gốc repo

Trong môi trường Python đã tạo, cài dependency. Với máy CPU có thể cài wheel CPU từ nguồn chính thức trước:

```powershell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cifar.txt
python -B -m unittest discover -s tests -v
python -B -m src.cifar_smoke
```

Mặc định tải tự động CIFAR-100 từ Hugging Face theo revision cố định, cache trong `data/`, huấn luyện base mới. Root là `reports/local/cifar100_smoke/`; artifact đầy đủ nằm trong `attempts/<attempt_id>/`. Mã lượt gồm timestamp UTC và UUID. Local chưa bật tracking có `not_logged`, ID/URL null.

Máy Sang đã có cache cũ tại `D:\ML_Paper\data`, có thể dùng lại dữ liệu để tránh tải lại:

```powershell
python -B -m src.cifar_smoke --data-dir D:\ML_Paper\data --device cpu
```

Sau một lượt có status complete, lấy đúng checkpoint trong `output_dir` của report:

```powershell
$runResult = Get-Content -Raw reports/local/cifar100_smoke/results.json | ConvertFrom-Json
$baseCheckpoint = Join-Path $runResult.output_dir "base_checkpoint.pt"
python -B -m src.cifar_smoke --reuse-checkpoint $baseCheckpoint
```

Reuse bị từ chối nếu cấu hình base, class order, normalization hoặc dataset revision khác. JSON phân biệt train mới/reuse. Nếu lượt mới nhất failed, chọn checkpoint từ thư mục attempt trước có status complete. Với Kaggle phiên mới, gắn Output cũ làm Input và copy đường dẫn checkpoint thật vào `REUSE_CHECKPOINT`.

## Đọc kết quả theo thứ tự

1. Root `latest.json`, `progress.json`, `results.json` phải cùng `attempt_id`, `status=complete`; progress có `stage=complete`.
2. `results.json`: `verification_passed=true`, các `checks` đều true. File gốc này luôn chỉ lượt mới nhất.
3. `results`: W `(256,50)` rồi `(256,60)`; test 5.000 rồi 6.000; `prediction_disagreements=0`. Đọc cả `base_accuracy_pct` và `new_class_accuracy_pct`.
4. Mở `output_dir` trong report để lấy config/protocol/data audit/checkpoint/state của chính lượt đó. Khi chạy 5 phase, W cuối là `(256,100)` và test 10.000 ảnh.
5. Status failed không được dùng như kết quả đạt; xem `error_type`, `failed_stage` và exception trên notebook/terminal. Exception vẫn được ném, lệnh vẫn báo lỗi.
6. Giữ checkpoint/state cùng report. Không commit dữ liệu, `.pt`, `.npz` hoặc credential lên GitHub.

Một lần gọi `run()` tạo thư mục mới, không ghi đè artifact của lần gọi trước. JSON flat của pipeline cũ được giữ nguyên trong `attempts/legacy-.../`; checkpoint/state cũ vẫn ở root để đường dẫn reuse đã có tiếp tục hoạt động. Với lượt mới, lấy artifact từ `output_dir`.

Ctrl+C được ghi failed. Nếu kernel/process bị tắt cứng, Python không có cơ hội ghi failed: lượt đó có thể còn running và không được coi là complete. Có thể khởi chạy lượt mới với ID riêng.

## Vì sao cấu hình này và bước tiếp theo?

Hai epochs, expansion 256 và một phase giúp kiểm tra đường đi dữ liệu, gradient, save/load, freeze và cập nhật theo batch với tài nguyên nhỏ. Đây là lựa chọn kiểm tra pipeline; chưa kết luận tái lập accuracy của paper. Dùng backbone và dữ liệu thật vẫn không biến một smoke test thành benchmark.

Mốc smoke đủ 5 phase (100 lớp) đã có [JSON](../reports/cifar100_kaggle_full5_results.json) và [W&B blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) finished, được đối chiếu ngày 04/11/2026. [Review](../reports/REVIEW_CIFAR_KAGGLE_FULL5_20261104.md) ghi cách kiểm và giới hạn metadata Kaggle. [Protocol](protocol_chot.md) và [metrics](metrics.md) đã chốt; tiếp theo cần đạt các gate resume, benchmark mode và profile d8000 trước run160 epochs. Run notebook 02 Version 2 vẫn là mốc lịch sử riêng.

Khi source/config/test đổi, chạy `python -m src.notebook_bundle` để tạo lại notebook 03 trước khi upload. Generator chặn ghi đè notebook 02 để giữ source bundle của bằng chứng cũ.

Nguồn kỹ thuật: [PyTorch reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html), [Module.eval và requires_grad](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Module.html). Seed không bảo đảm bitwise giống nhau giữa mọi thiết bị/phiên bản.
