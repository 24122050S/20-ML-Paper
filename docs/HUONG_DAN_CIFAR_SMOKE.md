# Sang tự chạy CIFAR-100: 50 base + 10 lớp mới

Bạn có thể thực hiện toàn bộ bước này một mình. Mục tiêu là nối pipeline ảnh với analytic head đã kiểm chứng, không chờ source của thành viên khác.

## Ba quy ước đã chốt

- ResNet-32 trả F `(n,64)`; `H=ReLU(F @ W_fe)` có shape `(n,256)`, float64. CNN/BatchNorm và expansion được giữ cố định sau base training.
- Nhãn CIFAR gốc được remap bằng cùng `class_order` cho train và test: base `0..49`, phase đầu `50..59`. Metadata lớp thuộc pipeline; API chỉ trả R/W.
- Base có W `(256,50)`. Sau mọi minibatch của phase đầu có W `(256,60)`, đánh giá đủ 6.000 ảnh của cả 60 lớp. Joint chỉ là nhánh kiểm chứng giữ lịch sử riêng.

Chi tiết và ví dụ nhãn: [đặc tả, mục 9](Dac_ta_ACIL.md#9-hợp-đồng-giữa-pipeline-cnn-và-analytic-head).

## Cách chạy trên Kaggle

1. Import [notebook 02](../notebooks/02_cifar100_smoke.ipynb), bật Internet, chọn GPU nếu có.
2. Source được đóng gói trong notebook, không phụ thuộc file nằm riêng ở workspace cũ hoặc một commit chưa được push.
3. Giữ 2 epochs, expansion 256 và một phase incremental cho lượt đầu. Save & Run All.
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

Mặc định tải tự động CIFAR-100 từ Hugging Face theo revision cố định, cache trong `data/`, huấn luyện base mới và lưu vào `reports/local/cifar100_smoke/`. Không cần chuẩn bị ảnh/nhãn bằng tay. Local chưa bật tracking có `not_logged`, ID/URL null.

Máy Sang đã có cache cũ tại `D:\ML_Paper\data`, có thể dùng lại dữ liệu để tránh tải lại:

```powershell
python -B -m src.cifar_smoke --data-dir D:\ML_Paper\data --device cpu
```

Sau một run đã tạo checkpoint, chạy lại analytic mà không train lại bằng:

```powershell
python -B -m src.cifar_smoke --reuse-checkpoint reports/local/cifar100_smoke/base_checkpoint.pt
```

Reuse bị từ chối nếu cấu hình base, class order, normalization hoặc dataset revision khác. JSON phân biệt rõ train mới và checkpoint tái sử dụng.

## Đọc kết quả theo thứ tự

1. `progress.json`: stage phải là `complete`.
2. `results.json`: `verification_passed=true`, các `checks` đều true.
3. `results`: W `(256,50)` rồi `(256,60)`; test 5.000 rồi 6.000; `prediction_disagreements=0`. Đọc cả `base_accuracy_pct` và `new_class_accuracy_pct`.
4. `protocol.json`: permutation và mapping nhất quán, không trùng lớp giữa phase.
5. Giữ checkpoint/state cùng report để biết chính xác run đã dùng gì. Không commit dữ liệu, `.pt`, `.npz` hoặc credential lên GitHub.

## Vì sao cấu hình này và bước tiếp theo?

Hai epochs, expansion 256 và một phase giúp kiểm tra đường đi dữ liệu, gradient, save/load, freeze và cập nhật theo batch với tài nguyên nhỏ. Đây là lựa chọn kiểm tra pipeline; chưa kết luận tái lập accuracy của paper. Dùng backbone và dữ liệu thật vẫn không biến một smoke test thành benchmark.

Sau khi lượt 50+10 đạt, chạy notebook 02 trên Kaggle để có bằng chứng môi trường GPU. Kế tiếp mở rộng đủ 5 phase (100 lớp), rồi mới chốt training dài/expansion và các ablation theo paper. Hiện một người thực hiện vẫn giữ nguyên các tiêu chí kiểm chứng này.

Khi source/config/test đổi, chạy `python -m src.notebook_bundle` để tạo lại notebook 02 trước khi upload; bundle được băm SHA-256 và giữ cùng module dùng ở CLI.

Nguồn kỹ thuật: [PyTorch reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html), [Module.eval và requires_grad](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Module.html). Seed không bảo đảm bitwise giống nhau giữa mọi thiết bị/phiên bản.
