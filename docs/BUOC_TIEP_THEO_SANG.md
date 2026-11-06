# Sang đang ở bước nào? — cập nhật06/11/2026

**Đã chạy xong notebook06: R1 seed42,160 epochs,d8000,50 base+5×10 lớp. Không cần chạy lại seed42.**

| Nội dung | Trạng thái/bằng chứng |
|---|---|
| R1 seed42 |[Kaggle Version1 /365844618](https://www.kaggle.com/code/hsangg206/06-benchmark-seed42/output?scriptVersionId=365844618) Complete, [W&B zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) Finished và công khai |
| Kết quả |Ā65,164550%, final55,87%, base forgetting8,32pp; n=1, SD=null |
| Kiểm cấu hình |160 hàng history,25.000 ảnh/epoch, LR80/120 đúng;6 mốc50→100 lớp; cuối W8000×100/test10.000 |
| Checkpoint/state |Checkpoint v2 hoàn tất160/160 và R/W/FE đã tải/kiểm; source/class order/normalization khớp, backbone/BN/FE cố định |
| Tài nguyên |BP41,60 phút; pipeline numeric53,78 phút, peak RSS3,55GB/GPU378,42MiB; phần setup/tests/logging nằm ngoài timer numeric |
| Tracking |Sync đúng payload offline; API công khai đối chiếu config/source/summary và6 history rows |
| Các mốc nền |Smoke100, resume CPU/CUDA, profile analytic d8000 và BP5 đều đã có bằng chứng; không cần lặp lại chỉ để lấy log |

[Báo cáo dễ đọc và bảng6 phase](../reports/R1_SEED42_KAGGLE_20261106.md) · [JSON numeric gốc](../reports/benchmark_seed42_kaggle_results.json) · [JSON tổng hợp sau sync](../reports/benchmark_seed42_kaggle_ready.json) · [manifest/checksum](../reports/R1-SEED42-KAGGLE-01-provenance.json).

## Việc tiếp theo

1. Hoàn thiện **R1 seeds43/44** với class order/backbone train mới đúng từng seed; giữ recipe160/d8000/K5. Chưa khởi chạy hai seed này trong yêu cầu chạy06.
2. Tổng hợp metric trong từng seed rồi mới mean±sample_SD, n=3; hiện65,1646% chỉ là seed42. So cùng metric Ā với66,30% của paper; không so final55,87% với Ā paper.
3. **R2:** dùng checkpoint160 đã hoàn tất của cùng seed cho no-FE và các gamma; không train lại base cho từng gamma.
4. **R3:** reuse checkpoint160/seed42, d1024 và batch32/128/512 theo [protocol](protocol_chot.md); joint oracle chạy ngoài vùng timing.

Checkpoint seed42 nằm trong Output của Version1: `benchmark_seed42/attempts/20261105T051519550968Z-47b72e2791ac/base_checkpoint.pt`. Bản tải đầy đủ ở `reports/local/kaggle_benchmark_seed42_20261105/version1_complete/`; trọng số/NPZ không đẩy GitHub. Không dùng checkpoint seed42 cho class order/backbone của seed43.

## Đọc đúng các file kết quả

JSON gốc trên Kaggle ghi `not_logged/offline_pending_sync` vì lúc tạo file đang log offline. Bản tổng hợp sau sync và manifest ghi run thực tế đã Finished; phần numeric không đổi. [Budget BP5](../reports/benchmark_seed42_budget.json) là snapshot dự trù trước benchmark; trạng thái R1 hiện tại nằm trong manifest R1. Source notebook06 đã chạy được giữ nguyên, không tạo lại notebook chỉ để sửa metadata lịch sử.

[Protocol v1.0](protocol_chot.md) · [Metrics v1.0](metrics.md) · [Chạy qua API](CHAY_KAGGLE_BANG_API.md).
