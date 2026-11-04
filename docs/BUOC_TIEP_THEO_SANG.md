# Sang đang ở bước nào? — cập nhật05/11/2026

**Đã chạy thay Sang xong BP5 trên Kaggle. Không cần import/chạy notebook05 lại.**

| Bước | Trạng thái và bằng chứng |
|---|---|
| Analytic d8000 |Đạt trên T4,34,07s/peak RSS3,20GB; [Version1](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365647614), W&B gey0yop4 finished |
| Resume CUDA |Đạt trong suite GPU53 passed/1 skip; proof nằm trong [JSON BP5](../reports/profile_base_kaggle_ready.json) |
| BP5 +analytic50+10/d256 |Đạt,76,11s cho training5 epochs, W256×60/test6.000/accuracy41,35%; [Version3](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432) |
| Checkpoint5 |Đã tải và kiểm v2, đủ optimizer/scheduler/RNG/loader state, target5 hoàn tất |
| Tracking BP5 |Run offline pv82dy1f đã lưu; trạng thái sync mới nhất trong [manifest](../reports/BP5-KAGGLE-01-provenance.json) |
| Ngân sách160 |Dự trù~59 phút, cần tối thiểu84,1 phút để còn30% dự phòng; chuẩn bị phiên2 giờ. Quota26h do Sang cung cấp |
| Notebook06 |Đã cấu hình160 epochs/d8000/K5/seed42, chưa khởi chạy |

## Phần còn lại tôi sẽ xử lý

Lượt API không đọc được secret W&B nên đã tạo offline log cùng kết quả GPU. Máy cần đăng nhập W&B để sync đúng run này; không cần train BP5 lại để lấy tracking. Không gửi key vào chat hoặc repository. Khi sync thực sự hoàn tất, metadata mới được ghi logged và API phải xác nhận finished/config/source/summary khớp.

Sau bước tracking là R1 seed42 bằng **[notebook06](../notebooks/06_benchmark_seed42_ready.ipynb)**. Các tham số đã đặt sẵn; checkpoint5 epochs không được dùng để tiếp tục thành160 epochs. Đây là lượt base mới với lịch LR80/120. Chỉ checkpoint160 đang dở, cùng config/runtime/source, mới được dùng resume.

## Những con số này chưa có nghĩa là đã tái lập paper

BP5 chạy50+10 lớp ở d256 nên mean43,215% là trung bình partial hai mốc; full A-bar=null. Không so trực tiếp accuracy41,35% của60 lớp với A-bar66,30%/K5 của paper. R1 thật cần160 epochs/d8000 và đủ5 phase, sau đó mới chạy seeds43/44 và ablation/phân tích theo protocol.

Notebook06 lưu W&B offline để trợ lý sync từ máy sau lượt GPU; không cần đưa credential vào code Kaggle. File `benchmark_seed42_ready.json`, checkpoint, state và offline payload phải cùng một attempt. Script tạo file notebook không tự khởi chạy GPU.

[Review BP5/budget](../reports/REVIEW_BP5_KAGGLE_20261105.md) · [Cách chạy API](CHAY_KAGGLE_BANG_API.md) · [Protocol](protocol_chot.md) · [Resume chi tiết](RESUME_BENCHMARK_PROFILE.md)
