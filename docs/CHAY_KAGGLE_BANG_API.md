# Chạy qua Kaggle API và đồng bộ W&B

Ngày05/11/2026, công cụ điều khiển tab không đọc/click được Kaggle. Sang đã cấu hình đăng nhập Kaggle tại máy; trợ lý dùng API để chạy BP5. Không cần import lại notebook05.

## Các phiên đã có

| Version của kernel04 | Công việc | Bằng chứng |
|---|---|---|
|1 /365647614 |Profile analytic d8000 |[Output](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365647614), W&B gey0yop4 finished |
|2 |BP5 lỗi khi đọc secret W&B (HTTP400), chưa train |Output failed, không dùng làm số liệu đạt |
|3 /365703432 |BP5 hoàn tất, CUDA resume passed |[Output](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432), [manifest](../reports/BP5-KAGGLE-01-provenance.json) |

Slug04 được giữ để lưu lịch sử version. Source Version3 là variant API của notebook05, không phải chạy lại analytic Version1. Source đã lưu tại `notebooks/archive/05_profile_base_api_v3.ipynb`.

## Tại sao W&B offline?

Lượt API gặp HTTP400 khi đọc `WANDB_API_KEY` từ Kaggle Secrets. Không đưa key vào notebook. Version3 ghi log offline cùng số liệu GPU; tải đúng file `.wandb` của run `pv82dy1f` về máy rồi dùng `wandb sync` vào project nhóm sau khi đăng nhập W&B tại máy.

File đăng nhập Kaggle và W&B nằm ngoài repository. Không truyền key trong tham số lệnh hoặc commit credential. Sync chỉ chọn run của attempt này, không sync hàng loạt dữ liệu khác. Sau sync, kiểm API: finished, config/source hashes và summary đúng rồi mới cập nhật metadata logged. Manifest ghi nguyên SHA của payload offline; JSON trước sync được giữ khi tạo bản tổng hợp sau sync.

**Cập nhật05/11/2026:** đã sync đúng payload của BP5 vào [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f). API công khai xác nhận Finished, config/source hashes/accuracy/time/memory/CUDA proof khớp. [Manifest](../reports/BP5-KAGGLE-01-provenance.json) ghi checksum và thời điểm sync. [JSON gốc trước sync](../reports/archive/BP5-KAGGLE-01-original-ready.json) giữ nguyên; [bản tổng hợp](../reports/profile_base_kaggle_ready.json) chỉ thêm tracking metadata đã xác minh. Không chạy lại BP5.

## Các lỗi CLI đã xử lý

- Windows CLI cũ đọc source tiếng Việt bằng locale mặc định và báo charmap decode error. Chạy process với `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` trước khi upload. Kiểm version remote chưa đổi rồi mới thử lại.
- Kaggle CLI2.1 không áp dụng suffix/version cho output. Dùng tool environment riêng `D:\ML_Paper\tmp\kaggle_cli_225`, Kaggle2.2.5 +six1.17.0; không thay môi trường CIFAR/torch của project.
- Khi tải output, dùng version-pinned API, đối chiếu attempt/checksum trước khi công bố. Không lấy latest của một kernel có nhiều thí nghiệm để gán cho version cũ.

## R1 đã hoàn tất3 seeds

[Notebook06](../notebooks/06_benchmark_seed42_ready.ipynb) đã cấu hình R1 seed42,160 epochs/d8000/K5, yêu cầu CUDA test đạt và dùng checkpoint v2 đúng target/runtime. `CHECKPOINT=None`, `RESUME_CHECKPOINT=None` cho base mới; checkpoint5 epochs bị từ chối cho target160.

R1 hiện đã hoàn tất3 kernels06/07/08, tương ứng42/43/44, mỗi lượt timeout7200s,160 epochs/d8000/K5 và fresh base. [Báo cáo tổng hợp](../reports/R1_CIFAR100_K5_3SEEDS_20261106.md)/[summary](../reports/r1_cifar100_k5_summary.json) ghi Ā65,1196 ± 0,6104%, final56,1300 ± 0,5575%, F_base8,3067 ± 0,6201pp (n=3, ddof=1), với version/source/checkpoint/W&B riêng trong [manifest tổng hợp](../reports/R1-C100-K5-3SEEDS-provenance.json). Seed43/44 chạy song song; cả3 payload đã sync và được kiểm API công khai. Giữ Output/checkpoint cho R2/R3; R/W incremental chưa có resume giữa phase. Không cần chạy lại các seeds R1 đã nghiệm thu.

## R2 đã hoàn tất — 07/11/2026

Notebook09/10/11 tương ứng seeds42/43/44, mỗi notebook có3 case analytic: no-FE/gamma0.1, FE8000/gamma0.01 và0.001. Baseline FE8000/gamma0.1 lấy từ R1 cùng seed. [Báo cáo](../reports/R2_CIFAR100_K5_20261107.md) và [manifest tổng hợp](../reports/R2-C100-K5-provenance.json) nối source commit, version Kaggle, checkpoint hash và9 actual W&B run IDs. Cả9 run đã sync và Finished/công khai; không train thêm backbone.

Input pin R1 version1 của đúng seed. `CHECKPOINT=None` ở R2 nghĩa là tìm checkpoint đã pin theo SHA trong Input; không có sẽ fail, không train mới. Hai gamma mới phải giữ FE hash giống R1. Mỗi case có attempt riêng và6 mốc đánh giá; suite complete chỉ khi đủ3 case.

Kaggle thực tế từ chối seed44 ban đầu do tài khoản đạt2 batch GPU sessions; đây là yêu cầu chưa được tạo (kernel_id/version đều0), không phải một run số liệu thất bại. Sau khi42/43 hoàn tất mới gửi44. Dùng CLI chính thức2.2.5 hỗ trợ version-pinned pull/output; không lấy latest để gắn vào version cũ. Signed output URLs và credential không lưu trong repo.

Numeric JSON giữ nguyên bytes lúc capture offline. Đồng bộ đúng3 payload của từng seed, kiểm config/source/metrics/6 history rows và API công khai không Authorization/Cookie; ready/manifest chỉ ghi synced sau khi kiểm xong. Ma trận R/W/FE mới được load và kiểm trên Kaggle CPU, giữ tại Output đúng version; local tải JSON/proof/log, không tải NPZ lớn. Không cần chạy lại R2. R3 cũng đã hoàn tất, xem mục bên dưới.

## R3 đã hoàn tất — 07/11/2026

[Notebook12](../notebooks/12_r3_batch_analysis.ipynb), [Kaggle v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048), [raw JSON](../reports/r3_cifar100_k5_results.json), [manifest](../reports/R3-KAGGLE-01-provenance.json).9 case =batch32/128/512 ×3 timing repeats, cùng checkpoint seed42/160 epochs, d1024/gamma0.1. Full50.000 train/10.000 test;9 fresh CPU subprocesses sequential với4 BLAS threads. GPU dùng trích F64 ngoài timer; oracle chạy riêng sau cả9 worker.9 W&B Finished/công khai, đủ54 history rows khớp JSON.

Output listing của lượt này có2 trang; API2.2.5 pin version1 và phân trang khi tải. Đã tải fixture,60 NPZ, JSON/proofs/offline payloads để kiểm local; prepared CIFAR cache nằm lồng trong suite cũng được nhận, tất cả ma trận/ảnh đều không commit Git. Local kiểm source/context/checksum/mapping/norm và giải lại joint cho54 so sánh. [Báo cáo](../reports/R3_CIFAR100_K5_BATCH_20261107.md) ghi rõ timer, RSS5ms và đơn vị thống kê1seed/3timing repeats. Raw capture offline giữ nguyên, actual tracking chỉ bổ sung sau sync/xác minh.

Sau sync chạy `python scripts/reconcile_r3_tracking.py --suite <raw-r3_ready.json>`: summary được chốt từ report6 phase đã kiểm, thêm alias tổng/max riêng và hash raw suite, tránh va tên với history từng phase. Chín run đã công bố đều qua reconciliation và API công khai; history/numeric bytes không đổi. Bốn tests hồi quy local cho helper tracking/JSON round-trip tách khỏi5 driver tests đã chạy trên Kaggle.
