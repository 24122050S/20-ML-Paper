# Sang làm gì sau profile analytic?

**Bạn đã hoàn tất bước đo analytic d8000. Bây giờ chạy notebook05 đã cấu hình sẵn để đo BP5.** Phần tính thời gian/quota và chuẩn bị benchmark160 epochs sẽ được trợ lý xử lý khi có kết quả5 epochs.

## 1. Hai file vừa gửi chứng minh điều gì?

| Nội dung | Đã kiểm |
|---|---|
| `results.json` + `progress.json` | Cùng attempt, complete, mọi check true |
| Source |16 source/config hashes khớp source bundle04 đã công bố; input checkpoint đúng của smoke Kaggle100 lớp |
| W&B |[gey0yop4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/gey0yop4) Finished, public API xác nhận config/source/memory |
| Kaggle |[Notebook04](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile) Public, Version1 theo API, Tesla T4 |
| Profile d8000 |34,07 giây, peak RSS3,20GB, GPU allocated peak45,45MB, R/W/FE522,50MB |
| BP5 và resume CUDA |Chưa có bằng chứng trong hai file này; notebook05 xuất proof riêng |

Đây là đo subset1000 ảnh base +200 ảnh mới +600 test. Accuracy9,33% trong file không dùng làm benchmark hoặc so trực tiếp với smoke22,05%. `gpu_benchmark_validated=false` nghĩa là chưa nghiệm thu benchmark dài, không phải profile vừa chạy bị lỗi.

## 2. Bạn chỉ cần làm các thao tác này

1. Lấy **[05_profile_base_ready.ipynb](../notebooks/05_profile_base_ready.ipynb)**. File trên máy ở `D:\ML_Paper\20-ML-Paper\notebooks\05_profile_base_ready.ipynb`; hoặc mở file tương ứng trên GitHub main và chọn Download raw file.
2. Trên Kaggle chọn **New Notebook → File → Import Notebook → Upload**, chọn file05. Bước này tạo một notebook mới, giữ lượt04 làm bằng chứng riêng.
3. Chọn **GPU T4**, bật **Internet**, bật quyền dùng secret `WANDB_API_KEY` đã có. Không cần thêm Output03 làm Input cho notebook05.
4. **Giữ nguyên code**, chọn **Save Version → Save & Run All**. Notebook tự kiểm resume CUDA rồi train mới5 epochs và chạy thêm10 lớp ở d256. Nếu Kaggle thiếu GPU hoặc CUDA test chưa đạt, notebook dừng trước training và ghi lý do.
5. Khi cell cuối báo **BP5 COMPLETE**, vào Output tải đúng **`profile_base_ready.json`** ở thư mục ngoài cùng. Gửi file này cùng link version đã lưu. Sang đã báo còn26 giờ GPU; chỉ cần báo lại nếu số này thay đổi đáng kể.

Notebook đã đặt sẵn `ACTION="profile_base"`, `CHECKPOINT=None`, `RESUME_CHECKPOINT=None`, `STOP_AFTER_EPOCH=None`. **Bạn không cần tìm và sửa các biến này.**

File gộp `profile_base_ready.json` chứa kết quả gốc của5 epochs, trạng thái từng test và CUDA resume, peak RAM/GPU trong toàn lượt BP5, thời gian từng epoch, run URL và source/checkpoint hashes. Output `profile_base/results.json` và checkpoint v2 ở `profile_base/attempts/<id>/base_checkpoint.pt` vẫn được giữ. Khi lượt mới lỗi, file gộp ghi failed/false, không để success cũ làm bằng chứng.

## 3. “Bước dưới” về160 epochs thực ra có nghĩa gì?

**5 epochs là lượt đo thời gian;160 epochs là lượt huấn luyện chính R1.** Không nối thêm155 epochs vào checkpoint5 epochs: target và lịch LR của lượt chính phải được chốt từ đầu. Base5 epochs hiện chỉ dùng kiểm cơ chế và đo tài nguyên.

Sau khi nhận file BP5, trợ lý sẽ:

1. Kiểm CUDA resume thật sự passed, đủ5 epochs, LR vẫn0,1, checkpoint v2 hoàn tất và đánh giá đúng60 lớp.
2. Tính thời gian BP160 từ phép đo5 epochs. Phần analytic đã có ngoại suy thô khoảng13,37 phút cho50.000 ảnh train; vẫn phải cộng trích feature/evaluation/I/O, đối chiếu RAM và GPU quota.
3. Đối chiếu khoảng thời gian còn lại để giữ ít nhất30% dự phòng. Nếu đủ, chuẩn bị notebook benchmark seed42 với160 epochs, d8000, K5 và checkpoint đầu vào=None. Bạn nhận file đã cấu hình thay vì tự chọn tham số.

Notebook05 ghi quota26 giờ do Sang cung cấp ngày04/11. `long_benchmark_ready=false` vì còn phải kiểm kết quả BP5, tổng ngân sách và giới hạn phiên hiện tại; flag này không ngăn hoàn thành BP5. Chưa có lượt160 epochs mới được khởi chạy trong lần chuẩn bị này.

## 4. Phạm vi hỗ trợ hiện tại

Công cụ điều khiển tab Kaggle gặp lỗi kết nối trong phiên hỗ trợ04/11; chưa có thao tác sửa/chạy notebook trên website. File05 đã được chuẩn bị ở repo để import; thông tin Public/Version1/source được kiểm qua API đọc công khai. API pull không chứa output test đã thực thi, nên chưa coi CUDA resume là đã nghiệm thu từ source notebook đơn thuần.

[Manifest analytic Kaggle](../reports/PROFILE-KAGGLE-01-provenance.json) · [Review bằng chứng](../reports/REVIEW_PROFILE_KAGGLE_20261104.md) · [Protocol](protocol_chot.md) · [Hướng dẫn resume chi tiết](RESUME_BENCHMARK_PROFILE.md)
