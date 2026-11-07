# R3 — kế hoạch đo batch size v1.0, chốt 07/11/2026 trước khi chạy

Đây là chi tiết phép đo của R3 trong [protocol v1.0](protocol_chot.md), không đổi seed/d/gamma/batch theo kết quả test. Dùng CIFAR-100 đầy đủ, checkpoint R1 seed 42 epoch cuối 160, 50 lớp base + 5 phase × 10 lớp. FE của R3 là `N(0,1)` shape 64×1024 từ RNG seed 1042, cố định giữa mọi case; hash riêng vì kích thước khác FE8000 của R1. Gamma = 0,1; analytic float64; CPU/BLAS 4 threads; rtol=1e-6, atol=1e-8 giữ theo protocol.

## Câu hỏi và đối chứng

1. Đổi batch 32/128/512 có thay đổi nghiệm R/W, score và dự đoán so với ridge joint ở từng phase không?
2. Thời gian analytic và peak RSS thay đổi ra sao trên cùng representation, thứ tự ảnh và máy chạy?

Trích F64 của đủ 50.000 train và 10.000 test đúng một lần từ backbone đã đóng băng. Mẫu train được chia theo phase và giữ chỉ số gốc tăng dần trong từng phase; mọi case đọc cùng cache F64/y được băm SHA. Cache là fixture của nghiên cứu và oracle, không phải dữ liệu lịch sử do learner incremental tự lưu. Không dùng kết quả này để tuyên bố benchmark toàn pipeline hoặc bảo vệ dữ liệu của hạ tầng cache.

## Vùng đo và thứ tự

- Mỗi cặp batch/repeat chạy trong **một tiến trình CPU mới**, tuần tự, không tranh CPU với case khác. Worker nạp và kiểm fixture, tạo FE, giới hạn thread, warm-up BLAS/LAPACK nhỏ trước khi đo.
- Batch được đổi ở cả tích lũy base Gram/cross và RLS incremental. Báo riêng thời gian base và tổng 5 phase incremental, rồi cộng thành thời gian analytic. Gamma chỉ cộng một lần ở base. FE transform nằm trong vùng đo.
- Không tính tải/checksum cache, checkpoint/CNN, warm-up, evaluation, joint oracle, kiểm so sánh, ghi NPZ hoặc W&B vào thời gian analytic. Mỗi phase có một cửa sổ đo `perf_counter`; tổng của 6 cửa sổ là learning_seconds.
- Ba vòng chạy theo thứ tự xoay cố định: **32→128→512; 128→512→32; 512→32→128**. Không chọn lần nhanh nhất hoặc chỉ giữ run có score đẹp. Đơn vị thống kê thời gian/bộ nhớ là 3 lần lặp trên **một seed**, không phải 3 seed độc lập.
- RSS được lấy mẫu 5ms trong từng cửa sổ học, có lấy mẫu tại biên. Báo peak RSS tuyệt đối và mức tăng so với RSS đầu worker; đây là peak quan sát được, có thể bỏ lỡ đỉnh ngắn và gồm runtime/input fixture/allocator. Báo thêm bytes R/W/FE chính xác; không coi RSS là kích thước riêng của learner. Worker không dùng GPU; CNN chạy GPU ở bước trích đặc trưng ngoài phép đo.

## Joint oracle và tiêu chí nghiệm thu

Sau khi cả 9 worker kết thúc, chạy oracle ở tiến trình riêng. Ở mỗi mốc, dùng toàn bộ H/nhãn đã thấy để giải trực tiếp `joint_ridge`; tính R bằng hệ Gram đầy đủ độc lập. Oracle có giữ lịch sử và chỉ phục vụ kiểm chứng, nằm ngoài mọi phép đo batch.

So sánh 9 case × 6 phase: shape, float64/hữu hạn, `allclose` cho R/W/score theo ngưỡng trên, relative Frobenius error của R/W, max absolute R/score error, số argmax khác nhau trên toàn bộ lớp đã thấy và accuracy. Yêu cầu không có dự đoán khác joint; nếu sai, lưu failed và điều tra, không nới ngưỡng hoặc bỏ case âm thầm. Lưu ma trận và proof tại Output cùng version. Lỗi mới phải ghi running/failed thay vì giữ complete cũ.

Số batch base/incremental dự kiến:

| Batch | Base 25.000 ảnh | Một phase 5.000 ảnh | Tổng RLS của 5 phase |
|---:|---:|---:|---:|
| 32 | 782 | 157 | 785 |
| 128 | 196 | 40 | 200 |
| 512 | 49 | 10 | 50 |

Xuất mean ± sample SD (n=3, ddof=1) cho thời gian và RSS. Accuracy/dự đoán phải ổn định giữa các lần lặp; không báo SD giữa seed từ 3 lần đo cùng seed. Các accuracy d1024 được ghi là kết quả phân tích riêng, không thay thế benchmark R1 d8000. Hiệu năng chỉ mô tả máy/phần mềm đã đo, không suy batch lớn luôn nhanh hơn hay tốt hơn trên mọi máy.

## Đầu ra

Notebook 12, source worker/coordinator, fixture/context/source hashes, 9 raw run reports và 54 comparison rows, oracle report, JSON tổng hợp, run W&B riêng cho mỗi case, hình tự vẽ và báo cáo giải thích kết quả. Trạng thái lúc viết kế hoạch: **chưa khởi chạy bộ R3 đầy đủ**.
