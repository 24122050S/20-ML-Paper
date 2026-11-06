# Sang đang ở bước nào? — cập nhật06/11/2026

**R1 đã hoàn tất đủ3 seeds42/43/44. Không cần chạy lại các lượt này để lấy bằng chứng.**

| Nội dung | Kết quả xác nhận |
|---|---|
| Cấu hình |Mỗi seed train base mới160 epochs, d8000/gamma0.1,50 base+5×10 lớp; có6 mốc đánh giá |
| Ā, mean ± sample SD |**65,1196 ± 0,6104%** |
| Final accuracy |**56,1300 ± 0,5575%** |
| Base forgetting |**8,3067 ± 0,6201pp** |
| Đơn vị thống kê |n=3 seeds; sample SD với ddof=1, không lấy6 phase làm6 seed |
| Tracking |[seed42](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) · [seed43](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan) · [seed44](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4); cả3 Finished/công khai, config/source/summary khớp |
| Bằng chứng |JSON gốc/progress/readiness, checkpoint160 và R/W/FE của từng seed đã tải/kiểm; summary kiểm chéo mean/SD bằng hai thư viện |

[Báo cáo tổng hợp và hình](../reports/R1_CIFAR100_K5_3SEEDS_20261106.md) · [Summary JSON](../reports/r1_cifar100_k5_summary.json) · [Manifest tổng hợp](../reports/R1-C100-K5-3SEEDS-provenance.json).

## Việc tiếp theo

1. **R2:** reuse checkpoint160 đúng cùng seed để chạy no-FE và gamma0.1/0.01/0.001 theo [protocol](protocol_chot.md). Không train lại base cho mỗi gamma và không lấy backbone seed42 thay cho43/44.
2. **R3:** checkpoint seed42, d1024/gamma0.1, batch32/128/512; chạy oracle joint ngoài vùng timing, đo lặp3 lần theo protocol.
3. Dùng báo cáo R1 làm phần kết quả chính của deck/poster; giữ phân biệt R1 đã hoàn tất với R2/R3 và sản phẩm nộp còn đang làm.

Lượt này chỉ chạy thêm43/44 và tổng hợp R1, chưa gửi GPU job R2/R3. Các checkpoint/state ở Output Kaggle đúng version trong từng manifest và `reports/local/kaggle_benchmark_seed*/version1_complete/`. Trọng số/NPZ không commit Git. Ba seeds có class order/backbone/normalization/FE riêng đã kiểm; không trộn artifact giữa chúng.

## Tái tạo bảng và hình

Trong root repo, chạy `python scripts/aggregate_r1.py` sau khi có đủ JSON/manifests. Vẽ bằng môi trường riêng với `requirements-plots.txt` rồi `python scripts/plot_r1.py`; không cần GPU. Input lỗi/thiếu seed ghi failed, không để success cũ. JSON numeric gốc giữ not_logged vì được tạo offline; bản readiness sau sync và manifest ghi run thực tế đã Finished. Bản nghiệm thu seed42 ban đầu là snapshot n=1, còn summary mới là kết quả đủ3 seed.

[Protocol v1.0](protocol_chot.md) · [Metrics v1.0](metrics.md) · [Chạy qua API](CHAY_KAGGLE_BANG_API.md).
