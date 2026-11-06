# Sang đang ở bước nào? — cập nhật 07/11/2026

**R1 và R2 đã hoàn tất trong core scope. Bước thực nghiệm còn lại là R3.**

| Phần | Đã làm | Bằng chứng |
|---|---|---|
| R1 — kết quả chính | 3 seeds42/43/44, mỗi seed train mới160 epochs/d8000/gamma0.1/K5; Ā65,1196 ± 0,6104% | [Báo cáo R1](../reports/R1_CIFAR100_K5_3SEEDS_20261106.md) |
| R2 — ablation | 9 lượt analytic mới, dùng đúng checkpoint160 từng seed;4 hàng, mỗi hàng3 seeds;9 W&B mới Finished/công khai | [Báo cáo R2](../reports/R2_CIFAR100_K5_20261107.md), [summary](../reports/r2_cifar100_k5_summary.json), [manifest](../reports/R2-C100-K5-provenance.json) |
| R3 — phân tích bổ sung | Chưa chạy sweep batch32/128/512 ở d1024 | [Protocol mục1](protocol_chot.md) |
| PPTX/poster | Chưa có sản phẩm hoàn chỉnh được nghiệm thu | Hạn PPTX21/11/2026, poster05/12/2026,23:59 |

## Kết quả R2 vừa hoàn tất

| Cấu hình | Ā: mean ± sample SD (%) |
|---|---:|
| Bỏ expansion, gamma0.1 (H64) | 53,2547 ± 0,2768 |
| FE8000, gamma0.1 (baseline R1) | 65,1196 ± 0,6104 |
| FE8000, gamma0.01 | 65,1173 ± 0,6089 |
| FE8000, gamma0.001 | 65,1163 ± 0,6088 |

Mean ± sample SD, n=3, ddof=1. FE8000/gamma0.1 tái dùng baseline R1; không train lại base. Phạm vi4/5 hàng Table2, không có gamma0. Không cần chạy lại R1/R2 để lấy bằng chứng. Đọc [phần giải thích kết quả](../reports/R2_CIFAR100_K5_20261107.md) trước khi đưa bảng vào slide.

## Việc tiếp theo

1. **R3:** dùng checkpoint160 seed42, d1024/gamma0.1, batch32/128/512. Giữ nguyên thứ tự ảnh/FE/dtype; so với joint oracle chạy ngoài vùng đo. Mỗi batch đo lặp3 lần theo protocol, báo sai số nghiệm/dự đoán, thời gian và bộ nhớ. Chưa gửi GPU job R3 trong lượt R2 này.
2. Ghép kết quả R1/R2 vào deck; dùng hình tự vẽ từ JSON và gắn nguồn paper/run. Giữ vị trí cho R3 khi có kết quả thật.
3. Hoàn thiện nội dung đọc hiểu, phân tích sai lệch và định dạng sản phẩm theo yêu cầu môn.

## Tái tạo và tìm artifact

Chạy `python scripts/aggregate_r1.py` hoặc `python scripts/aggregate_r2.py` trong root repo. Vẽ bằng môi trường riêng có `requirements-plots.txt`, với `scripts/plot_r1.py` hoặc `scripts/plot_r2.py`. Không cần GPU để tổng hợp/vẽ. Thiếu input hợp lệ sẽ ghi failed, không giữ success cũ.

Checkpoint R1 và ma trận R2 nằm trong Kaggle Output đúng version ghi ở manifest. Ma trận R2 mới đã load/kiểm trên Kaggle CPU; local chỉ tải JSON/proof/log, không tải NPZ lớn. JSON numeric giữ trạng thái capture offline; bản ready/manifest ghi actual tracking sau khi sync và xác minh. Không trộn checkpoint/mapping/normalization/FE giữa seeds.

[Protocol v1.0](protocol_chot.md) · [Metrics v1.0](metrics.md) · [Bảng notebook/config/run](../README.md).
