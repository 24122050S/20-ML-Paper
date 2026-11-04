# Kiểm tra lượt CIFAR 100 lớp — 04/11/2026

Đã nhận JSON tải xuống do Sang cung cấp và [link Kaggle scriptVersionId 365084714](https://www.kaggle.com/code/hsangg206/03-cifar100-safe-attempts/output?scriptVersionId=365084714). JSON ghi thời điểm bắt đầu 03/11/2026 02:04 UTC, status complete, verification_passed true, wandb_status logged. [W&B blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) được API công khai xác nhận finished ngày 04/11; config, source hashes và summary khớp JSON.

## Kết quả theo phase

| Phase | Lớp đã thấy | Accuracy toàn bộ lớp đã thấy | Accuracy riêng 50 lớp base | Relative W error so joint | Dự đoán khác joint |
|---|---|---|---|---|---|
| 0 | 50 | 33.4600% | 33.4600% | 3.798e-11 | 0 |
| 1 | 60 | 29.9167% | 30.6000% | 3.524e-11 | 0 |
| 2 | 70 | 27.5000% | 28.7400% | 3.252e-11 | 0 |
| 3 | 80 | 25.2250% | 27.3200% | 3.054e-11 | 0 |
| 4 | 90 | 23.3111% | 26.5800% | 2.876e-11 | 0 |
| 5 | 100 | 22.0500% | 25.5600% | 2.747e-11 | 0 |

W cuối 256×100; R 256×256; test cuối 10.000 ảnh. Mỗi phase mới nhận 5.000 ảnh train theo 40 minibatch. Mọi check trong report đạt; backbone/BatchNorm và expansion giữ nguyên, checkpoint reload có sai lệch logits bằng 0.

Train base mới 2 epochs trên CUDA, seed 42, expansion 256, gamma 0.1; loss 3,4724 → 2,9896. Môi trường Python 3.13.16, NumPy 2.1.3, PyTorch 2.11.0+cu130, W&B 0.28.1. Thời gian pipeline JSON ghi 66,823 giây; đây không phải thời gian toàn notebook đo từ giao diện Kaggle.

## Truy nguyên và giới hạn xác minh

- [JSON nguyên bản](cifar100_kaggle_full5_results.json) được copy nguyên bytes, UTF-8/LF; SHA-256 `a03d686c1e184b96d267af2b6ec4af9df08d806a8c9ce74d6e3fbaaafea9b5f3`.
- [Manifest](CIFAR-KAGGLE-02-provenance.json) ghép JSON, attempt, checkpoint hash, source bundle, config, link version và W&B.
- Toàn bộ 11 source/config hashes trong JSON khớp raw ZIP của [notebook 03 tại commit 9972be9](https://github.com/24122050S/20-ML-Paper/blob/9972be9db06b60ff1abbf0c7483325543cda8b1a/notebooks/03_cifar100_safe_attempts.ipynb). Input/effective config hashes khớp. Git commit của run vẫn null vì code chạy từ bundle; commit tham chiếu là nơi lưu source khớp.
- W&B được kiểm bằng API công khai, không dùng credential; state finished, phase 5, 100 lớp, accuracy 22,05%, verification true; toàn bộ config/source hashes trùng JSON.
- Công cụ trình duyệt không đọc được trang Kaggle. Link version do Sang cung cấp; số version hiển thị, loại GPU, thời gian notebook và quyền Private/Public chưa được xác minh độc lập. Manifest giữ các trường này null; không lấy metadata từ lượt cũ.
- Không train lại hoặc tạo W&B run mới khi nhập bằng chứng. JSON và manifest của các lượt cũ được giữ nguyên.

## Sang làm tiếp

Mốc smoke 100 lớp đã có bằng chứng. Đầu ra kế tiếp là bản protocol/metric và ngân sách benchmark: xác định scope R1/R2/R3, cấu hình base/feature expansion/gamma, số seed, cách tính metric và checkpoint chuẩn. Xử lý resume optimizer/scheduler/RNG và chi phí joint oracle trước phiên training dài nếu cần. Các mục V02/P02/S02 trong phân công là nhóm việc tiếp theo.

Giữ checkpoint/state của attempt `20261103T020415597382Z-6c34dd841810` trong Output làm artifact tái sử dụng. Accuracy 22,05% thuộc cấu hình kiểm tra 2 epochs/d256; việc đánh giá tái lập paper cần cấu hình benchmark đã chốt và thí nghiệm tương ứng.
