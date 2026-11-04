# Vì sao thay đổi từ VERIFY-01 sang VERIFY-02?

Báo cáo thay đổi VERIFY-02 ngày 02/11/2026. Phạm vi của lần sửa này là kiểm chứng analytic. Cập nhật trạng thái 03/11: pipeline CIFAR-100 đã lên main qua PR #5; notebook 03 và quản lý attempt đã lên main qua PR #7. Xem [bằng chứng hiện có](../reports/README.md).

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

## Kết quả đã kiểm tra ngày 02/11/2026

Bản sửa đã merge qua PR #3. Kaggle Version 9 (`364788888`) chạy source sạch `217684f`, cả 11 tests đạt và W sau ba phase là 24×2, 24×4, 24×6. Minibatch có sai lệch W tương đối lớn nhất khoảng 3.43e-13 so với joint; không có dự đoán khác nhau. Run W&B `wi4wau2d` đã Finished. JSON chỉ ghi `logged` sau khi SDK kết thúc.

Kaggle đồng bộ notebook có output thành commit `eddd50d`; đây là commit artifact, còn `217684f` là source thực thi. JSON kết quả được giữ nguyên và manifest riêng ghi liên kết version/commit/run cùng SHA-256. README và phân công đã phản ánh lượt chạy này. Kaggle vẫn Private, vì vậy nhóm cần tính đến quyền xem khi bàn giao; notebook output và JSON được lưu trên GitHub.

## Giới hạn còn lại

Feature map/gamma phải cố định; module hiện chỉ dùng float64 và gamma>0. Test nhiều cấu hình nhỏ không thay chứng minh tổng quát. Accuracy tổng hợp không so với benchmark ảnh. Dataset/CNN đã có smoke CIFAR đủ 100 lớp với tracking (CIFAR-KAGGLE-02); benchmark đầy đủ và ablation gamma=0 vẫn cần phát triển tiếp.

## Thu gọn thư mục sau kiểm chứng

- Gộp nội dung cần thiết của các README trong `configs/`, `src/`, `tests/`, `notebooks/` vào README gốc; bỏ `docs/README.md` chỉ có một dòng mô tả chung. Mỗi thư mục không cần thêm một trang chỉ lặp lại hướng dẫn đã có.
- Chuyển `kaggle/README.md` thành `docs/HUONG_DAN_KAGGLE.md`; hướng dẫn thuộc nhóm tài liệu, không cần một thư mục riêng chỉ chứa README.
- Bỏ `scripts/build_verification_notebook.py`, là công cụ đã dùng để tạo notebook ban đầu. Chạy notebook hiện tại không cần script này; giữ notebook có output làm artifact được review. Muốn tra công cụ tạo file vẫn có thể xem trong lịch sử Git.
- Giữ `configs/`, các module dùng chung, test reporting và ba file requirements: đây là dependency thực của quy trình kiểm chứng và tracking, không phải file thừa. Giữ JSON lịch sử và manifest để không làm mất bằng chứng đã đối chiếu.
- Giữ nguyên phần đặc tả đang được Sang chỉnh sửa. Việc dọn cấu trúc không thay đổi thuật toán, config, notebook output hoặc số liệu.
