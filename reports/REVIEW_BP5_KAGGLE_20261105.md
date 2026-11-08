# BP5 Kaggle và ngân sách benchmark — 05/11/2026

**Đã chạy thay Sang và nghiệm thu BP5 trên Tesla T4.** [Kaggle Version 3 /365703432](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432), [JSON](profile_base_kaggle_ready.json), [manifest](BP5-KAGGLE-01-provenance.json). Thời gian UTC trong artifact thuộc 04/11; ngày Việt Nam là 05/11. Đây là kiểm huấn luyện 5 epochs+d256/50+10 lớp, chưa phải R1.

## Kết quả thực đo

| Nội dung | Kết quả |
|---|---:|
| Test GPU |54 tests:53 passed,1 skipped |
| Resume CUDA |Passed, so chính xác model/optimizer/scheduler/RNG/shuffle/LR |
| Base training 5 epochs |76,1072s, trung bình 15,2214s/epoch |
| Toàn phần numeric BP5+analytic+I/O |99,5326s, chưa gồm setup notebook và offline logging |
| Train mỗi epoch |25.000 ảnh của 50 lớp base |
| LR cả 5 epochs |0,1; milestones 80/120 chưa đến |
| Peak CPU RSS |1.865.568.256 bytes (~1,87GB) |
| Peak CUDA allocated |396.805.632 bytes (~378,42MiB) |
| Analytic sau một phase |W256×60, đánh giá 6.000 ảnh, accuracy 41,35% |

Loss qua5 epochs:3,47236 →2,98960 →2,71092 →2,49433 →2,28011. Base analytic đạt 45,08%; sau thêm 10 lớp, base accuracy 42,78% và new-class accuracy 34,20%. Trung bình **partial**43,215% chỉ gồm hai mốc; full A-bar=null vì chưa đủ K5. Base drop 2,30pp là sau một phase, không phải F của toàn K5.

CUDA resume test dùng TinyCNN có BatchNorm/Dropout và dataset tiêu thụ RNG Python/NumPy/Torch; ResNet32+CIFAR thật trên CPU đã được kiểm continuous vs resume ở mốc trước. BP5 lần này huấn luyện ResNet32 trên toàn bộ 25.000 ảnh base bằng GPU. Không gọi đây là phép kiểm bitwise ResNet32/CIFAR restart trên GPU.

## Checkpoint và source

- Checkpoint v2: `completed_epoch=target_epochs=5`, training_complete=true, loader 25.000 ảnh; có optimizer, scheduler, RNG CPU/CUDA và generator của DataLoader. Đã tải về và kiểm trên CPU bằng `weights_only=True`.
- SHA-256 checkpoint: `ea34137b9cfc92024a6c8db65368d461e930c563b92320d227ce07df8aac3749`. Trọng số/NPZ ở reports/local và Output Kaggle, không commit Git.
- R/W hữu hạn, shape 256×256/256×60, không lưu train feature history và không chạy oracle trong đường benchmark. Các field sai số joint=null/not_run đúng phạm vi.
- 16 source/config hashes khớp core tại commitd3e7405; bundle 5851fa…b6ca19c không đổi. [Source API Version 3](../notebooks/archive/05_profile_base_api_v3.ipynb) lưu đúng variant đã chạy, gồm bước ghi W&B offline.
- Dùng client Kaggle 2.2.5 trong môi trường tool riêng để đối chiếu output **theo version**. Client 2.1 cũ bỏ qua suffix/version khi tải output; các lần tải cũ đã được đối chiếu lại với version-pinned bytes trước khi công bố.

## Tracking và lần lỗi trước

Version 2 dừng ở tracking_setup do HTTP400 khi đọc WANDB_API_KEY từ Kaggle Secrets, trước khi train; chưa xác nhận nguyên nhân sâu hơn của phản hồi này. Version 3 train mới 5 epochs, tắt online login và tạo offline run `pv82dy1f`. Numeric JSON gốc ghi not_logged/offline_pending_sync, không ghi logged trước khi đồng bộ.

Đã đồng bộ đúng payload `.wandb` có SHA-256 `cf4fc710f4032650454a2b12aaded0de44cde63f0573d5e871a0fa55c106084b` vào [W&B pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) ngày 05/11/2026. SDK và API công khai không đăng nhập đều xác nhận Finished; config, 16 source hashes, BP5 seconds, accuracy, peak memory và CUDA proof khớp. Không chạy lại BP5.

[Manifest](BP5-KAGGLE-01-provenance.json) ghi thời điểm sync và checksum mới. [JSON gốc trước sync](archive/BP5-KAGGLE-01-original-ready.json) giữ nguyên byte đã công bố và trạng thái tại lúc Kaggle ghi (`not_logged/offline_pending_sync`). [JSON tổng hợp sau sync](profile_base_kaggle_ready.json) bổ sung top-level `wandb_status=logged`, ID/URL thực và `tracking.status=synced`; phần `base_profile` bên trong vẫn là bằng chứng gốc. Notebook 06 được tạo lại chỉ để cập nhật trạng thái tracking BP5 trong planning budget nhúng; source bundle/config benchmark không đổi. Không lấy W&B run analytic Version 1 gán cho BP5.

## Dự trù 160 epochs, d8000, K5, seed 42

| Thành phần | Giây | Cách tính |
|---|---:|---|
| BP160 |2435,43 |BP5×32 |
| Analytic 50.000 ảnh train |802,26 |Ngoại suy từ [profile d8000 T4](profile_d8000_kaggle_results.json) |
| Trích feature/eval và overhead |54,28 |Phần non-BP của lượt 5 × 95.000/41.000 lượt ảnh |
| Setup và tests |120 |Khoản dự phòng, không phải số đo riêng |
| I/O state d8000/checkpoint |120 |Khoản dự phòng, không phải số đo riêng |
| Tổng dự trù |3531,97 (~58,87 phút) |Ước lượng, chưa phải runtime benchmark đo được |

Để còn ít nhất 30% cửa sổ thời gian, cần tối thiểu 3531,97/0,7=**84,09 phút**. Chuẩn bị timeout **2 giờ** có khoảng 50,94% thời gian dự phòng so với ước lượng. Quota 26 giờ là con số Sang cung cấp trước các lượt ngắn, không phải phép đọc tài khoản trực tiếp; job setup/download và hiệu năng host có thể biến động. Peak analytic 3,20GB và peak BP5 nêu trên là các phép đo riêng, không phải cam kết peak của full benchmark.

[Budget JSON](benchmark_seed42_budget.json) lưu công thức và khoản dự phòng. [Notebook 06](../notebooks/06_benchmark_seed42_ready.ipynb) đã chuẩn bị 160 epochs/d8000/K5/seed42, fresh base, offline tracking và resume v2. Bootstrap 06 trên CPU đạt 54 tests với 2 skip; guard chặn chạy 160 epochs trên CPU và thay success cũ bằng failed khi lỗi. **Chưa khởi chạy 160 epochs** trong lượt nghiệm thu BP5 này. Giữ checkpoint 5 epochs làm bằng chứng/profile; không resume nó thành target 160.
