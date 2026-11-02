# Vì sao thay đổi từ VERIFY-01 sang VERIFY-02?

Cập nhật 02/11/2026. Đây là sửa phần kiểm chứng analytic, chưa triển khai thêm pipeline CIFAR-100 vào repo.

## Lỗi cần giải quyết

1. Hàm cũ luôn ghép thêm cột W mỗi lần gọi. Một phase có hai lớp mới chia thành hai batch làm W tăng hai lần, dù số lớp chỉ tăng một lần.
2. JSON ghi synced và một URL cố định trước khi gọi W&B. Run sau có ID mới nhưng file vẫn dẫn run cũ; nếu logging lỗi, trạng thái thành công đã bị ghi quá sớm.
3. Notebook, JSON, README và các nhánh không cùng snapshot. Tài liệu còn mô tả file chỉ có ở workspace local.

## Cách sửa và ý nghĩa

- Tách analytic head vào `src/analytic.py`. Notebook và CLI import cùng implementation để không sửa lệch hai nơi.
- Dùng ID lớp toàn cục, đệm cột W theo tổng số lớp, tạo one-hot toàn cục. Cập nhật residual `W_pad + R_new H.T (Y_global - H W_pad)` tương đương công thức khối cũ khi batch chỉ có lớp mới, đồng thời xử lý lớp lặp đúng.
- Giữ Woodbury và gamma dương. Không đổi mục tiêu ridge, không cộng lại gamma theo batch, không đọc train cũ trong hàm cập nhật.
- Kiểm chứng whole-phase và minibatch riêng với nghiệm joint trực tiếp; giữ cả R, W, score và nhãn argmax trong kiểm tra.
- Hàm reporting lấy `run.id`/`run.url` từ SDK và ghi JSON sau khi context kết thúc. Tracking tắt ghi `not_logged`; lỗi init/log/finish ghi `failed` và raise. `logged` nghĩa SDK context kết thúc thành công; quyền công khai còn được kiểm tra trên project.
- Ghi config/môi trường/source hashes và Git commit nếu có. Local run với source chưa commit phải giữ dấu dirty, không gắn nhầm là artifact của commit sạch.
- Output tự chọn Kaggle/local, tạo thư mục trước khi ghi. W&B tách thành dependency tùy chọn; NumPy là đủ chạy core/tests.

## Lịch sử được giữ thế nào?

VERIFY-01 Version 5 được archive. Bản chuẩn lịch sử Version 8 lấy numeric rows từ output đã lưu và URL từ log SDK 7aeaupq7; provenance ghi rõ đây là sửa metadata chứ không chạy lại thí nghiệm cũ.

VERIFY-02 là lần kiểm chứng API mới. Kết quả mới chỉ được ghi lên bảng README khi có output thật và đã đối chiếu version/source/run; không tái sử dụng URL VERIFY-01.

## Giới hạn còn lại

Feature map/gamma phải cố định; module hiện chỉ dùng float64 và gamma>0. Test nhiều cấu hình nhỏ không thay chứng minh tổng quát. Accuracy tổng hợp không so với benchmark ảnh. Pipeline dataset/CNN/ablation gamma=0 vẫn là phần cần phát triển tiếp.
