# Sang đang ở bước nào? — cập nhật 18/11/2026

**Ba bộ thực nghiệm giữa kỳ (R1/R2/R3), đối chiếu code tác giả, [Slide Deck `.pptx` giữa kỳ](../slides/ACIL_Nhom3_GiuaKy.pptx) và hai bộ thực nghiệm phản biện cuối kỳ (E1/E2) đều đã hoàn tất.** Hiện chỉ còn nộp gói giữa kỳ lên Moodle và dàn trang Poster A2 dọc nộp cuối kỳ từ số liệu/biểu đồ đã sẵn sàng.

| Phần | Đã làm | Bằng chứng |
|---|---|---|
| R1 — kết quả chính | 3 seeds 42/43/44; mỗi seed train base mới 160 epochs/d8000/gamma0.1/K5; Ā 65,1196 ± 0,6104% | [Báo cáo R1](../reports/R1_CIFAR100_K5_3SEEDS_20261106.md) |
| R2 — ablation | 4/5 hàng Table 2, mỗi hàng 3 seeds; 9 run mới + 3 baseline R1, gamma 0 ngoài scope | [Báo cáo R2](../reports/R2_CIFAR100_K5_20261107.md) |
| R3 — batch size | 32/128/512 × 3 timing repeats trên seed 42/d1024; 54 so sánh phase đạt joint, 0 argmax khác; 9 W&B mới Finished/công khai | [Báo cáo R3](../reports/R3_CIFAR100_K5_BATCH_20261107.md), [summary](../reports/r3_cifar100_k5_summary.json) |
| Đối chiếu implementation tác giả | Đã chạy 2 lượt chính + 1 diagnostic, full features CIFAR100/K5; giữ C1 strict score failures và phân tích nguyên nhân | [đối chiếu tác giả](../reports/AUTHOR_CODE_COMPARISON_20261107.md), 3 W&B Finished/public |
| Nội dung đọc hiểu & kịch bản slide | Đã soạn đầy đủ nội dung đọc hiểu, 2 sơ đồ tự vẽ và kịch bản 36 slides kèm notes | [Reading the Paper](READING_THE_PAPER_ACIL.md), [Kịch bản 36 slides](KICH_BAN_36_SLIDE_GIUA_KY.md) |
| Phản biện lý thuyết cuối kỳ (`Critical Analysis`) | Đã chốt khung 4 góc nhìn lý thuyết, 3 Strengths, cặp hạn chế `P1–P2`, giao thức `E1–E2` và `FD1–FD2` | [Khung phản biện cuối kỳ](PHAN_BIEN_CUOI_KY_ACIL.md), [Kế hoạch empirics](KE_HOACH_EMPIRICS_CUOI_KY.md) |
| Thực nghiệm cuối kỳ `E2` (kiểm chứng `P2`) | 3 seeds 42/43/44 trên checkpoints R1; quét 11 mức `d` (trần iso-byte `d=798` fp64 / `d=1160` fp32) + `FD2` + Woodbury `fp32` vs `fp64`; 6 W&B Finished/public | [Báo cáo E2](../reports/E2_MEMORY_PRECISION_20261118.md), [summary E2](../reports/e2_memory_precision_summary.json), [hình E2](../reports/figures/e2_memory_and_woodbury_precision.png) |
| Thực nghiệm cuối kỳ `E1` (kiểm chứng `P1`) | 6 notebooks Kaggle GPU T4 (`13..18`, 12 runs GPU 160 epochs trên 3 seeds): `E1a` (`c0 = 10, 20, 30` vs `50`) & `E1b` (`superclass_disjoint` `c0 = 50`); 12 W&B Finished/public | [Báo cáo E1](../reports/E1_BASE_ASSUMPTION_20261118.md), [summary E1](../reports/e1_base_assumption_summary.json), [hình E1](../reports/figures/e1_base_representation_bottleneck.png) |
| File `.pptx` giữa kỳ & Poster A2 cuối kỳ | Còn xuất `.pptx` giữa kỳ và dàn trang Poster A2 dọc từ số liệu/biểu đồ đã hoàn tất | Hạn PPTX 21/11/2026, poster 05/12/2026, 23:59 |

## R3 vừa hoàn tất

| Batch | Analytic mean ± SD (s) | Peak RSS mean ± SD (MiB) | Argmax khác joint |
|---:|---:|---:|---:|
| 32 | 17,3334 ± 0,0784 | 151,3503 ± 0,9749 | 0 |
| 128 | 6,9247 ± 0,1098 | 146,1029 ± 0,9158 | 0 |
| 512 | 4,9975 ± 0,1123 | 164,5521 ± 4,3843 | 0 |

Mean ± sample SD của 3 lần đo, **không phải 3 seed**. Timer chỉ gồm analytic với FE, không gồm CNN/evaluation/oracle/IO/log. RSS gồm runtime/input/allocator của CPU process, lấy mẫu 5ms, có thể bỏ lỡ đỉnh ngắn. R/W/FE cuối luôn 9.28125 MiB. Accuracy d1024 là phân tích riêng, không thay số benchmark R1 d8000. Đọc [báo cáo](../reports/R3_CIFAR100_K5_BATCH_20261107.md) trước khi diễn giải trên slide.

## Việc tiếp theo

1. **Dựng file `.pptx` giữa kỳ từ kịch bản đã chuẩn bị:** sử dụng [kịch bản 36 slides](KICH_BAN_36_SLIDE_GIUA_KY.md), nhúng 7 hình tự vẽ giữa kỳ trong `reports/figures/`, đặt trích dẫn IEEE ở chân từng slide và chép phần ghi chú trình bày vào Speaker Notes.
2. **Kiểm tra quyền Public & liên kết:** kiểm tra lại chế độ Public của các notebook Kaggle `01`–`18`, W&B runs và dung lượng file `.pptx` (`< 20 MB`).
3. **Nghiệm thu PPTX trước 21/11, 23:59**, sau đó dàn trang Poster A2 dọc cuối kỳ (nhúng thêm [`e1_base_representation_bottleneck.png`](../reports/figures/e1_base_representation_bottleneck.png) và [`e2_memory_and_woodbury_precision.png`](../reports/figures/e2_memory_and_woodbury_precision.png)) trước 05/12, 23:59.

## Tìm và tái tạo kết quả

Notebook 09/10/11 là R2, notebook 12 là R3. [README](../README.md) ánh xạ từng case với source commit, JSON, Kaggle version và actual W&B run. Summary R1/R2 dùng 3 seeds; summary R3 dùng 3 timing repeats trên 1 seed. Các script `aggregate_r1.py`, `aggregate_r2.py`, `aggregate_r3.py` trong `scripts/` tái tạo bảng từ JSON; script `plot_r*.py` dùng môi trường riêng có `requirements-plots.txt`, không cần GPU để vẽ.

Checkpoint/state lớn giữ ở Kaggle Output đúng version và local bị Git bỏ qua. R3 đã tải và kiểm đủ fixture/60 NPZ; local giải lại joint để so 54 trạng thái, không dùng thời gian local làm số đo hiệu năng. Raw JSON giữ trạng thái capture offline, ready/manifest ghi trạng thái sau sync đã xác minh. R1/R2/R3 không cần thêm lượt để lấp dữ liệu còn thiếu trong kế hoạch này.

[Protocol v1.0](protocol_chot.md) · [Metrics v1.0](metrics.md) · [Kế hoạch đo R3](R3_KE_HOACH_DO.md).
