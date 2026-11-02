# Bằng chứng đã công bố

- `verify_analytic_update_results.json`: VERIFY-01, đối chiếu metadata từ output/log của Kaggle Version 8, commit 23af76f và run 7aeaupq7. Trường provenance giải thích việc sửa URL hardcode cũ; numeric rows giữ nguyên từ run lưu.
- `archive/VERIFY-01-version5.json`: bản lịch sử trước tracking, giữ để truy nguyên.
- `verify_analytic_update_v2_results.json`: JSON nguyên bản VERIFY-02 từ Kaggle Version 9, source 217684f, run wi4wau2d. `VERIFY-02-provenance.json` liên kết version/source/notebook-output commit và checksum; 11 tests đạt. Không sửa số liệu trong JSON để khớp một run khác.
- `local/`: output kiểm tra trên máy, bị Git bỏ qua. File local không mặc nhiên là kết quả Kaggle đã công khai.

Mỗi bản kết quả phải có config, môi trường, source commit/hashes, metrics và trạng thái tracking rõ ràng. Không dùng accuracy dữ liệu tổng hợp như benchmark ảnh của paper.

- `cifar100_smoke_results.json`: lượt CPU local train mới 2 epochs, source chưa commit; 50+10 lớp, 20 unit tests và mọi check đạt. `CIFAR-SMOKE-01-provenance.json` ghi checksum và trạng thái chưa chạy Kaggle/W&B. Checkpoint/state ở `local/cifar100_smoke/` bị Git bỏ qua.
