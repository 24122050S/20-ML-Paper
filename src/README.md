# Mã nguồn dùng chung

- `analytic.py`: `joint_ridge`, `initialize`, `incremental_update`; H (n,d), nhãn nguyên toàn cục (n,). Trả R (d,d), W (d,C).
- `verification.py`: sinh dữ liệu và so sánh joint, whole-phase, minibatch.
- `reporting.py`: ghi source hashes, trạng thái tracking và JSON sau khi W&B context kết thúc.
- `verify_analytic.py`: CLI; chạy từ gốc bằng `python -m src.verify_analytic`.

Hàm cập nhật chỉ thêm cột nếu có lớp chưa có. `n_classes` tùy chọn cho phép khai báo trước số lớp của phase; không được thu hẹp cột cũ. Feature map/gamma giữ nguyên. Chỉ hỗ trợ gamma dương; precision hiện là float64.
