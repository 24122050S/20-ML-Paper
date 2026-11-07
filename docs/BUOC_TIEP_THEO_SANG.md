# Sang đang ở bước nào? — cập nhật 07/11/2026

**Ba bộ thực nghiệm R1/R2/R3 trong kế hoạch đã hoàn tất.** Còn đối chiếu code tác giả, hoàn thiện nội dung giữa kỳ/PPTX và poster.

| Phần | Đã làm | Bằng chứng |
|---|---|---|
| R1 — kết quả chính | 3 seeds42/43/44; mỗi seed train base mới160 epochs/d8000/gamma0.1/K5; Ā65,1196 ±0,6104% | [Báo cáo R1](../reports/R1_CIFAR100_K5_3SEEDS_20261106.md) |
| R2 — ablation | 4/5 hàng Table2, mỗi hàng3 seeds;9 run mới +3 baseline R1, gamma0 ngoài scope | [Báo cáo R2](../reports/R2_CIFAR100_K5_20261107.md) |
| R3 — batch size | 32/128/512 ×3 timing repeats trên seed42/d1024;54 so sánh phase đạt joint,0 argmax khác;9 W&B mới Finished/công khai | [Báo cáo R3](../reports/R3_CIFAR100_K5_BATCH_20261107.md), [summary](../reports/r3_cifar100_k5_summary.json) |
| Đối chiếu implementation tác giả | Chưa có artifact chạy source tác giả trong repo | [Protocol mục6](protocol_chot.md) |
| PPTX/poster | Chưa có sản phẩm hoàn chỉnh được nghiệm thu | Hạn PPTX21/11/2026, poster05/12/2026,23:59 |

## R3 vừa hoàn tất

| Batch | Analytic mean ± SD (s) | Peak RSS mean ± SD (MiB) | Argmax khác joint |
|---:|---:|---:|---:|
| 32 | 17,3334 ± 0,0784 | 151,3503 ± 0,9749 | 0 |
| 128 | 6,9247 ± 0,1098 | 146,1029 ± 0,9158 | 0 |
| 512 | 4,9975 ± 0,1123 | 164,5521 ± 4,3843 | 0 |

Mean ± sample SD của3 lần đo, **không phải3 seed**. Timer chỉ gồm analytic với FE, không gồm CNN/evaluation/oracle/IO/log. RSS gồm runtime/input/allocator của CPU process, lấy mẫu5ms, có thể bỏ lỡ đỉnh ngắn. R/W/FE cuối luôn9.28125MiB. Accuracy d1024 là phân tích riêng, không thay số benchmark R1 d8000. Đọc [báo cáo](../reports/R3_CIFAR100_K5_BATCH_20261107.md) trước khi diễn giải trên slide.

## Việc tiếp theo

1. **Đối chiếu code tác giả theo protocol:** dùng môi trường riêng, pin commit/config/checkpoint, chạy và lưu kết quả; đối chiếu mapping → preprocessing → feature → FE → analytic → metric. Chỉ đọc code chưa được ghi là đã chạy. Không chép implementation cốt lõi của tác giả vào module nhóm.
2. **Ghép nội dung giữa kỳ:** dùng các bảng/hình R1/R2/R3 đã xuất, giải thích công thức và điều kiện tương đương joint, trình bày sai lệch với paper và giới hạn. Giữ nguồn paper/run cho từng hình/bảng.
3. **Nghiệm thu PPTX trước21/11,23:59**, rồi hoàn thiện đề xuất/phân tích và poster trước05/12,23:59 theo hạn đã cung cấp. Nội dung/notes/nguồn/định dạng phải kiểm theo yêu cầu môn.

## Tìm và tái tạo kết quả

Notebook09/10/11 là R2, notebook12 là R3. [README](../README.md) ánh xạ từng case với source commit, JSON, Kaggle version và actual W&B run. Summary R1/R2 dùng3 seeds; summary R3 dùng3 timing repeats trên1 seed. Các script `aggregate_r1.py`, `aggregate_r2.py`, `aggregate_r3.py` trong `scripts/` tái tạo bảng từ JSON; script `plot_r*.py` dùng môi trường riêng có `requirements-plots.txt`, không cần GPU để vẽ.

Checkpoint/state lớn giữ ở Kaggle Output đúng version và local bị Git bỏ qua. R3 đã tải và kiểm đủ fixture/60 NPZ; local giải lại joint để so54 trạng thái, không dùng thời gian local làm số đo hiệu năng. Raw JSON giữ trạng thái capture offline, ready/manifest ghi trạng thái sau sync đã xác minh. R1/R2/R3 không cần thêm lượt để lấp dữ liệu còn thiếu trong kế hoạch này.

[Protocol v1.0](protocol_chot.md) · [Metrics v1.0](metrics.md) · [Kế hoạch đo R3](R3_KE_HOACH_DO.md).
