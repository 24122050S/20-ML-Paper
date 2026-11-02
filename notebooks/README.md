# Notebook kiểm chứng

`01_verify_analytic_update.ipynb` hiện chạy VERIFY-02 bằng module trong `src/`. Nó không chứa một implementation thứ hai của analytic head.

Chạy local trong checkout repo hoặc import độc lập trên Kaggle. Nếu không thấy module local, notebook fetch source GitHub theo SOURCE_REF và ghi commit/source hashes thực tế. Trên Kaggle bật Internet; để Accelerator None cho kiểm tra NumPy.

W&B mặc định bật khi chạy notebook; cần bật secret đúng tên. Khi chạy local không tracking, đặt `ACIL_LOG_WANDB=0` trước khi khởi động kernel hoặc đổi biến trong cell thiết lập. JSON được xuất sau cell logging, có ID/URL của run thực, không dùng URL hardcode.

Kết quả VERIFY-01 cũ được liên kết bằng commit lịch sử và file archive; không suy rằng notebook HEAD vẫn là cùng lượt chạy đó.
