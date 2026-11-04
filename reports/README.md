# Bằng chứng đã công bố

- `verify_analytic_update_results.json`: VERIFY-01, đối chiếu metadata từ output/log của Kaggle Version 8, commit 23af76f và run 7aeaupq7. Trường provenance giải thích việc sửa URL hardcode cũ; numeric rows giữ nguyên từ run lưu.
- `archive/VERIFY-01-version5.json`: bản lịch sử trước tracking, giữ để truy nguyên.
- `verify_analytic_update_v2_results.json`: JSON nguyên bản VERIFY-02 từ Kaggle Version 9, source 217684f, run wi4wau2d. `VERIFY-02-provenance.json` liên kết version/source/notebook-output commit và checksum; 11 tests đạt. Không sửa số liệu trong JSON để khớp một run khác.
- `local/`: output kiểm tra trên máy, bị Git bỏ qua. File local không mặc nhiên là kết quả Kaggle đã công khai.

Mỗi bản kết quả phải có config, môi trường, source commit/hashes, metrics và trạng thái tracking rõ ràng. Không dùng accuracy dữ liệu tổng hợp như benchmark ảnh của paper.

- [cifar100_smoke_results.json](cifar100_smoke_results.json): lượt CPU local train mới 2 epochs, 50+10 lớp, W&B tắt. Source dirty/chưa commit và 20 tests là metadata **tại thời điểm chạy**; pipeline sau đó đã lên main qua [PR #5](https://github.com/24122050S/20-ML-Paper/pull/5). [Manifest CPU](CIFAR-SMOKE-01-provenance.json) mô tả riêng lượt này, không phải trạng thái Kaggle/W&B của toàn dự án. Checkpoint/state ở `local/cifar100_smoke/` bị Git bỏ qua. `results_sha256` hiện khớp Git blob UTF-8/LF; `results_original_raw_sha256` giữ checksum CRLF ban đầu. [Giải thích chuẩn hóa](../docs/SUA_LOI_CHECKSUM_JSON.md); số liệu JSON không đổi, không train lại.

- [cifar100_kaggle_results.json](cifar100_kaggle_results.json): kết quả CUDA của notebook 02, [Version 2 / 364942764](https://www.kaggle.com/code/hsangg206/notebook7e7946d4ee?scriptVersionId=364942764), [W&B o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2) ghi nhận Finished. [Manifest Kaggle](CIFAR-KAGGLE-01-provenance.json) liên kết version/source bundle/checksum JSON. Base 50 lớp đạt 32,58%, mốc 60 lớp 29,70%; đây là smoke 2 epochs/d=256. Kaggle được ghi nhận Private lúc đối chiếu; JSON/manifest đã có trong repo.

- [cifar100_kaggle_full5_results.json](cifar100_kaggle_full5_results.json): smoke CUDA đủ 100 lớp, [scriptVersionId 365084714](https://www.kaggle.com/code/hsangg206/03-cifar100-safe-attempts/output?scriptVersionId=365084714), [W&B blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) được API công khai xác nhận finished. Train mới 2 epochs/d256, W cuối 256×100, test 10.000 ảnh, accuracy 22,05%; mọi check đạt. [CIFAR-KAGGLE-02-provenance.json](CIFAR-KAGGLE-02-provenance.json) đối chiếu raw JSON/source/config và ghi rõ giao diện Kaggle chưa xác minh. [Review chi tiết](REVIEW_CIFAR_KAGGLE_FULL5_20261104.md).

Cập nhật trạng thái **04/11/2026**: notebook 03 đã có JSON của chuỗi đủ 5 phase và W&B finished. [Protocol](../docs/protocol_chot.md) và [metrics](../docs/metrics.md) đã chốt v1.0; bước tiếp theo là các gate resume/benchmark mode/profile trước lượt dài; suite hiện tại đã kiểm tra 38 tests trên local. Số 11/20 tests trong các lượt cũ giữ theo snapshot. File mới được copy nguyên bytes; các JSON/manifest cũ không thay đổi.

[Review tổng quan, cập nhật 04/11](REVIEW_TONG_QUAN_20261102.md) ghi phần đã đạt và việc còn lại. `local/` tiếp tục chỉ chứa artifact trên máy, không được đưa lên GitHub.
