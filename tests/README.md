# Kiểm tra độc lập

Từ thư mục gốc:

```bash
python -m unittest discover -s tests -v
```

`test_analytic_update.py`: ba seeds, whole-phase/minibatch khớp joint, chia cùng phase không tạo cột lớp trùng, lớp cũ lặp lại, trạng thái đầu vào không bị sửa, rank-deficient features với gamma>0 và dữ liệu không hợp lệ.

`test_reporting.py`: tắt tracking phải có URL null; bật tracking dùng ID/URL thật từ SDK giả; chỉ ghi thành công sau finish; log/finish lỗi ghi failed thay vì giữ URL cũ. Test không kết nối tài khoản W&B thật.
