# Thay đổi để Sang tự chạy CIFAR smoke 50+10

Ngày 02/11/2026. Mục tiêu là hoàn thành đường đi ảnh → backbone → analytic, không phụ thuộc bàn giao từ thành viên khác.

> Cập nhật 03/11/2026: pipeline và bằng chứng CIFAR đã lên main qua [PR #5](https://github.com/24122050S/20-ML-Paper/pull/5). Notebook 02 Version 2 đã chạy trên Kaggle CUDA, W&B o2r63pj2 Finished. [PR #7](https://github.com/24122050S/20-ML-Paper/pull/7) thêm notebook 03, artifact riêng cho mỗi attempt và kiểm kích thước theo config. Phần số liệu local dưới đây là lịch sử của lượt CPU ban đầu.

## Vì sao phải sửa?

Workspace cũ đã có dữ liệu và pipeline CIFAR, nhưng nó import analytic từ `acil_toy.py`; source này nằm ngoài repository chuẩn. Chỉ sao chép kết quả cũ vào repo không chứng minh module minibatch hiện tại hoạt động với đặc trưng CNN thật.

Đặc tả có hai mục số 9 và mô tả `class_ids` như trường trạng thái analytic, trong khi API thực tế chỉ trả R/W. Điều này dễ làm nhầm nhãn gốc CIFAR với chỉ số cột classifier.

## Đã thay đổi gì?

- Đưa loader/audit/protocol và ResNet-32 tự cài vào `src/cifar_data.py`, `src/resnet.py`. Tận dụng code hạ tầng đã có, thay phần nối analytic bằng module chuẩn `src/analytic.py`.
- Thêm `src/cifar_smoke.py`: tự tải/cache dữ liệu theo revision HF cố định, train 50 lớp base, kiểm checkpoint, freeze CNN/BatchNorm, mở rộng feature và cập nhật 10 lớp mới theo minibatch.
- Chốt nhãn toàn cục: cùng permutation/mapping cho train và test; `class_order` thuộc pipeline. Mở trước 60 cột cho toàn phase qua `n_classes=60`, không thêm cột theo số batch.
- Gộp mục 9 của đặc tả thành hợp đồng H/y/R/W; giữ riêng bản đặc tả trước chỉnh sửa trong `.local-archive` của workspace. Các mục công thức 1–8 được giữ.
- Thêm test mapping 60 lớp, checkpoint không tương thích và từ chối trích feature khi chưa freeze/eval. Snapshot này có 20 tests khi cài dependency CIFAR, gồm 9 tests ảnh. Sau PR #7, suite đầy đủ đã kiểm tra 37 tests trên local.
- Reporting có thể nhận tags CIFAR thay vì gắn synthetic; provenance tính cả file chưa được Git track để không gọi nhầm working tree mới là source sạch. Thứ tự log/finish rồi ghi JSON được giữ.
- Notebook 02 đóng gói source thành ZIP có SHA-256. Đây là bản phân phối của cùng module, không phải implementation analytic được viết lại trong cell. Bản 02 hiện giữ snapshot tương ứng Version 2; `python -m src.notebook_bundle` tạo bản 03 cho source mới sau PR #7.

## Kết quả thực thi local

Đã **train mới 2 epochs**, không tái sử dụng checkpoint cũ trong lượt này. Dùng cache CIFAR đã chuẩn bị theo revision `aadb3af77e9048adbea6b47c21a81e47dd092ae5`; không tải lại dữ liệu chỉ để tạo một run mới. Môi trường Python 3.12.7, NumPy 2.1.3, PyTorch 2.11.0+cpu.

| Phase | Lớp đã thấy | W | Ảnh train phase | Ảnh test toàn bộ lớp đã thấy | Accuracy | Relative W error | Dự đoán khác joint |
|---|---|---|---|---|---|---|---|
| 0 | 50 | [256, 50] | 25000 | 5000 | 32.6800% | 2.047e-11 | 0 |
| 1 | 60 | [256, 60] | 5000 | 6000 | 29.3667% | 1.771e-11 | 0 |

Phase mới xử lý 40 minibatch. R giữ 256×256; checkpoint load lại có sai lệch logits bằng 0. Fingerprint backbone/BatchNorm và expansion không đổi. Sau phase 1, accuracy riêng 50 lớp base là 29.92%, riêng 10 lớp mới là 26.60%. Mọi check trong JSON đều đạt; thời gian lượt chạy khoảng 335.9 giây.

Kết quả: [JSON nguyên bản](../reports/cifar100_smoke_results.json), [manifest](../reports/CIFAR-SMOKE-01-provenance.json). Checkpoint/state ở `reports/local/cifar100_smoke/`, không đưa file lớn lên GitHub. Hai accuracy giống lượt chạy legacy trước đây không có nghĩa tái sử dụng kết quả: JSON mới có checkpoint_origin, môi trường, checksum source và sai lệch số học của module chuẩn.

## Giới hạn và việc người học làm tiếp

Đây là smoke test 2 epochs, expansion 256, một phase mới, không phải benchmark paper. Nhánh joint giữ feature train cũ chỉ để đối chiếu; không được dùng việc giữ này để mô tả bộ nhớ của một thuật toán triển khai thật.

Lượt này chạy local với working tree chưa commit (`source_dirty=true`), W&B tắt (`not_logged`, ID/URL null). Không gắn kết quả vào commit cha như thể commit đó chứa source mới và không gán version Kaggle giả. Bằng chứng lịch sử VERIFY-01/02 được giữ nguyên.

Run Kaggle 50+10 tiếp theo đã hoàn tất và được lưu riêng: [JSON Kaggle](../reports/cifar100_kaggle_results.json), [Version 2](https://www.kaggle.com/code/hsangg206/notebook7e7946d4ee?scriptVersionId=364942764), [W&B o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2), [manifest](../reports/CIFAR-KAGGLE-01-provenance.json). Số liệu CUDA là 32,58% ở base và 29,70% ở 60 lớp. Trạng thái not_logged trong JSON CPU phía trên thuộc chính lượt CPU đó.

Cập nhật 04/11/2026: đã có [JSON smoke 100 lớp](../reports/cifar100_kaggle_full5_results.json), [manifest](../reports/CIFAR-KAGGLE-02-provenance.json) và [W&B blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) finished. Sáu phase, W cuối 256×100, test 10.000, accuracy 22,05%; source/config hashes khớp bundle 03. [Protocol](protocol_chot.md) và [metrics](metrics.md) đã chốt v1.0; việc tiếp theo là triển khai các gate trước benchmark trong protocol. Xem [review lượt mới](../reports/REVIEW_CIFAR_KAGGLE_FULL5_20261104.md).
