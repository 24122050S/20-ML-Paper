# Bằng chứng đã công bố

- `verify_analytic_update_results.json`: VERIFY-01, đối chiếu metadata từ output/log của Kaggle Version 8, commit 23af76f và run 7aeaupq7. Trường provenance giải thích việc sửa URL hardcode cũ; numeric rows giữ nguyên từ run lưu.
- `archive/VERIFY-01-version5.json`: bản lịch sử trước tracking, giữ để truy nguyên.
- Kết quả VERIFY-02 được thêm sau khi chạy notebook/module mới và đối chiếu đúng version/source/run. Không tạo URL hay trạng thái logged bằng tay cho run chưa tồn tại.
- `local/`: output kiểm tra trên máy, bị Git bỏ qua. File local không mặc nhiên là kết quả Kaggle đã công khai.

Mỗi bản kết quả phải có config, môi trường, source commit/hashes, metrics và trạng thái tracking rõ ràng. Không dùng accuracy dữ liệu tổng hợp như benchmark ảnh của paper.
