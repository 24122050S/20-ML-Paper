# Profile analytic Kaggle — gey0yop4

Đã kiểm hai file Sang gửi trong Downloads ngày04/11/2026. Kết quả: **lượt analytic resource profile đạt**, không phải BP5 hoặc benchmark160 epochs. [Manifest](PROFILE-KAGGLE-01-provenance.json) lưu checksum và phạm vi xác minh; [JSON kết quả](profile_d8000_kaggle_results.json), [progress](profile_d8000_kaggle_progress.json) giữ số liệu và metadata gốc.

## Bằng chứng đã đối chiếu

- Cùng attempt `20261104T140958780997Z-c2ab35794fea`; results/progress complete, progress.stage=complete, verification/checks true.
- d8000, gamma0,1, float64,4 CPU threads; Tesla T4, Torch2.11.0+cu130, Python3.13.16, NumPy2.1.3, SciPy1.17.0. Analytic dùng CPU, CUDA dùng trích feature.
- 16 source/config hashes khớp Git ở commit `d3e7405dadfb3a2e93212c343665c8bc75ffc657`. Source chạy từ bundle nên git_commit/source_dirty=null là đúng. Bundle SHA-256 `5851fa25143689342edabe9ab112635395ed2eae4f19fe95c446eaf96b6ca19c` khớp notebook04.
- Input checkpoint SHA-256 `a5bc46f46a89d5eea99e5b782925ea4ab31e809e00cf6b41442b039f5c30a0d2` khớp base2 epochs của JSON smoke Kaggle full5 đã công bố.
- Public W&B GraphQL API xác nhận [gey0yop4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/gey0yop4) finished; config/source hashes, verification và các memory summary khớp JSON.
- Public Kaggle kernels pull API xác nhận notebook Public, currentVersionNumber=1, GPU T4/Internet bật, source bundle như trên. Source của Sang thêm một cell chọn path checkpoint. Cập nhật05/11: version-pinned output API đã xác nhận scriptVersionId365647614 và raw bytes khớp analytic JSON. CUDA proof hiện có riêng ở BP5 Version3, không suy từ source V1.

## Thời gian và tài nguyên

| Stage | Giây |
|---|---:|
| Trích feature subset |0,6835|
| Tích lũy base1000 ảnh |17,3639|
| Giải R/W |12,7404|
| Cập nhật200 ảnh mới |2,8434|
| Chấm600 ảnh |0,0337|
| Cửa sổ profile |34,0693|

Peak RSS process=3.195.297.792 bytes (~3,20GB,2,98GiB); peak CUDA allocated=45.453.824 bytes (~43,35MiB). Ba ma trận R/W/FE=522.496.000 bytes. CUDA allocated của phép trích feature này không phải peak VRAM khi huấn luyện BP; notebook05 đo riêng toàn lượt BP5.

R8000×8000, W8000×100, FE64×8000; state hữu hạn và backbone freeze. Profile chỉ dùng1000 base +200 ảnh của50 lớp mới +600 test, chuyển50→100 để đo tài nguyên. Accuracy9,33% không đưa vào bảng R1; `gpu_benchmark_validated=false` giữ đúng phạm vi subset.

Ngoại suy base25.000 ảnh khoảng446,84s, incremental25.000 khoảng355,43s; tổng phần analytic thô802,26s (~13,37 phút). Chưa gồm BP160, full evaluation/trích feature và artifact I/O. Không lấy34 giây của subset làm thời gian full benchmark.

## Việc còn lại và file chuẩn bị

Đã tạo [notebook05](../notebooks/05_profile_base_ready.ipynb) từ source bundle04 không đổi module huấn luyện. Không cần Input hoặc sửa biến. Nó chạy suite, ghi proof từng test/CUDA, yêu cầu CUDA resume đạt, train BP5 từ đầu và analytic50+10/d256, đo RSS/GPU rồi xuất một file `profile_base_ready.json`. Không mở160 epochs.

Đã kiểm bootstrap05 trong kernel mới trên local CPU:54 tests,52 passed/2 skipped (CUDA và snapshot02/03), JSON proof xuất đúng; CUDA gate chặn training trên CPU và xóa trạng thái success cũ khỏi file gộp khi lỗi. Tại mốc04/11 chưa có BP5/CUDA output; kết quả mới ngày05/11 được liên kết ngay dưới.

Cập nhật05/11: đã chạy thay Sang BP5 qua API và CUDA proof đạt; [review BP5](REVIEW_BP5_KAGGLE_20261105.md), [budget](benchmark_seed42_budget.json), [notebook06](../notebooks/06_benchmark_seed42_ready.ipynb). Quota26h là thông tin Sang cung cấp; chưa có run160 epochs.
