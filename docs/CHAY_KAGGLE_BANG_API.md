# Chạy qua Kaggle API và đồng bộ W&B

Ngày05/11/2026, công cụ điều khiển tab không đọc/click được Kaggle. Sang đã cấu hình đăng nhập Kaggle tại máy; trợ lý dùng API để chạy BP5. Không cần import lại notebook05.

## Các phiên đã có

| Version của kernel04 | Công việc | Bằng chứng |
|---|---|---|
|1 /365647614 |Profile analytic d8000 |[Output](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365647614), W&B gey0yop4 finished |
|2 |BP5 lỗi khi đọc secret W&B (HTTP400), chưa train |Output failed, không dùng làm số liệu đạt |
|3 /365703432 |BP5 hoàn tất, CUDA resume passed |[Output](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432), [manifest](../reports/BP5-KAGGLE-01-provenance.json) |

Slug04 được giữ để lưu lịch sử version. Source Version3 là variant API của notebook05, không phải chạy lại analytic Version1. Source đã lưu tại `notebooks/archive/05_profile_base_api_v3.ipynb`.

## Tại sao W&B offline?

Lượt API gặp HTTP400 khi đọc `WANDB_API_KEY` từ Kaggle Secrets. Không đưa key vào notebook. Version3 ghi log offline cùng số liệu GPU; tải đúng file `.wandb` của run `pv82dy1f` về máy rồi dùng `wandb sync` vào project nhóm sau khi đăng nhập W&B tại máy.

File đăng nhập Kaggle và W&B nằm ngoài repository. Không truyền key trong tham số lệnh hoặc commit credential. Sync chỉ chọn run của attempt này, không sync hàng loạt dữ liệu khác. Sau sync, kiểm API: finished, config/source hashes và summary đúng rồi mới cập nhật metadata logged. Manifest ghi nguyên SHA của payload offline; JSON trước sync được giữ khi tạo bản tổng hợp sau sync.

**Cập nhật05/11/2026:** đã sync đúng payload của BP5 vào [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f). API công khai xác nhận Finished, config/source hashes/accuracy/time/memory/CUDA proof khớp. [Manifest](../reports/BP5-KAGGLE-01-provenance.json) ghi checksum và thời điểm sync. [JSON gốc trước sync](../reports/archive/BP5-KAGGLE-01-original-ready.json) giữ nguyên; [bản tổng hợp](../reports/profile_base_kaggle_ready.json) chỉ thêm tracking metadata đã xác minh. Không chạy lại BP5.

## Các lỗi CLI đã xử lý

- Windows CLI cũ đọc source tiếng Việt bằng locale mặc định và báo charmap decode error. Chạy process với `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` trước khi upload. Kiểm version remote chưa đổi rồi mới thử lại.
- Kaggle CLI2.1 không áp dụng suffix/version cho output. Dùng tool environment riêng `D:\ML_Paper\tmp\kaggle_cli_225`, Kaggle2.2.5 +six1.17.0; không thay môi trường CIFAR/torch của project.
- Khi tải output, dùng version-pinned API, đối chiếu attempt/checksum trước khi công bố. Không lấy latest của một kernel có nhiều thí nghiệm để gán cho version cũ.

## Lượt06 đã hoàn tất

[Notebook06](../notebooks/06_benchmark_seed42_ready.ipynb) đã cấu hình R1 seed42,160 epochs/d8000/K5, yêu cầu CUDA test đạt và dùng checkpoint v2 đúng target/runtime. `CHECKPOINT=None`, `RESUME_CHECKPOINT=None` cho base mới; checkpoint5 epochs bị từ chối cho target160.

Dự trù trước chạy khoảng59 phút, chuẩn bị phiên2 giờ theo [budget](../reports/benchmark_seed42_budget.json). Lượt06 đã dùng timeout7200 giây; file notebook tự nó không đặt timeout cho Kaggle web UI. Đã hoàn tất [06-benchmark-seed42 Version1 /365844618](https://www.kaggle.com/code/hsangg206/06-benchmark-seed42/output?scriptVersionId=365844618) ngày05/11, nghiệm thu/công bố06/11:160 epochs,d8000,K5; Ā65,1646%, final55,87%; [W&B zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) đã sync và Finished. [Báo cáo](../reports/R1_SEED42_KAGGLE_20261106.md)/[manifest](../reports/R1-SEED42-KAGGLE-01-provenance.json) lưu kết quả thực đo53,78 phút numeric. N=1; còn seeds43/44. Source06 được giữ nguyên như lúc gửi API. Giữ Output/checkpoint để dùng lại cho R2/R3 và để resume BP khi cần; R/W incremental chưa có resume giữa phase.
