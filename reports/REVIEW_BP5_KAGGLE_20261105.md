# BP5 Kaggle và ngân sách benchmark — 05/11/2026

**Đã chạy thay Sang và nghiệm thu BP5 trên Tesla T4.** [Kaggle Version3 /365703432](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432), [JSON](profile_base_kaggle_ready.json), [manifest](BP5-KAGGLE-01-provenance.json). Thời gian UTC trong artifact thuộc04/11; ngày Việt Nam là05/11. Đây là kiểm huấn luyện5 epochs+d256/50+10 lớp, chưa phải R1.

## Kết quả thực đo

| Nội dung | Kết quả |
|---|---:|
| Test GPU |54 tests:53 passed,1 skipped |
| Resume CUDA |Passed, so chính xác model/optimizer/scheduler/RNG/shuffle/LR |
| Base training5 epochs |76,1072s, trung bình15,2214s/epoch |
| Toàn phần numeric BP5+analytic+I/O |99,5326s, chưa gồm setup notebook và offline logging |
| Train mỗi epoch |25.000 ảnh của50 lớp base |
| LR cả5 epochs |0,1; milestones80/120 chưa đến |
| Peak CPU RSS |1.865.568.256 bytes (~1,87GB) |
| Peak CUDA allocated |396.805.632 bytes (~378,42MiB) |
| Analytic sau một phase |W256×60, đánh giá6.000 ảnh, accuracy41,35% |

Loss qua5 epochs:3,47236 →2,98960 →2,71092 →2,49433 →2,28011. Base analytic đạt45,08%; sau thêm10 lớp, base accuracy42,78% và new-class accuracy34,20%. Trung bình **partial**43,215% chỉ gồm hai mốc; full A-bar=null vì chưa đủ K5. Base drop2,30pp là sau một phase, không phải F của toàn K5.

CUDA resume test dùng TinyCNN có BatchNorm/Dropout và dataset tiêu thụ RNG Python/NumPy/Torch; ResNet32+CIFAR thật trên CPU đã được kiểm continuous vs resume ở mốc trước. BP5 lần này huấn luyện ResNet32 trên toàn bộ25.000 ảnh base bằng GPU. Không gọi đây là phép kiểm bitwise ResNet32/CIFAR restart trên GPU.

## Checkpoint và source

- Checkpoint v2: `completed_epoch=target_epochs=5`, training_complete=true, loader25.000 ảnh; có optimizer, scheduler, RNG CPU/CUDA và generator của DataLoader. Đã tải về và kiểm trên CPU bằng `weights_only=True`.
- SHA-256 checkpoint: `ea34137b9cfc92024a6c8db65368d461e930c563b92320d227ce07df8aac3749`. Trọng số/NPZ ở reports/local và Output Kaggle, không commit Git.
- R/W hữu hạn, shape256×256/256×60, không lưu train feature history và không chạy oracle trong đường benchmark. Các field sai số joint=null/not_run đúng phạm vi.
- 16 source/config hashes khớp core tại commitd3e7405; bundle5851fa…b6ca19c không đổi. [Source API Version3](../notebooks/archive/05_profile_base_api_v3.ipynb) lưu đúng variant đã chạy, gồm bước ghi W&B offline.
- Dùng client Kaggle2.2.5 trong môi trường tool riêng để đối chiếu output **theo version**. Client2.1 cũ bỏ qua suffix/version khi tải output; các lần tải cũ đã được đối chiếu lại với version-pinned bytes trước khi công bố.

## Tracking và lần lỗi trước

Version2 dừng ở tracking_setup vì lượt gửi API không được cấp secret W&B, trước khi train. Version3 train mới5 epochs, tắt online login và tạo offline run `pv82dy1f`. Numeric JSON gốc ghi not_logged/offline_pending_sync, không ghi logged trước khi đồng bộ.

Trạng thái sync hiện hành nằm trong `tracking_sync` của [manifest](BP5-KAGGLE-01-provenance.json). Chỉ sau khi SDK sync hoàn tất và API xác nhận finished/config/source/numeric khớp mới xuất logged metadata. Nếu đã sync, bản gốc trước sync được lưu riêng ở reports/archive; số liệu không thay đổi. Không lấy W&B run của analytic Version1 gán cho BP5.

## Dự trù160 epochs, d8000, K5, seed42

| Thành phần | Giây | Cách tính |
|---|---:|---|
| BP160 |2435,43 |BP5×32 |
| Analytic50.000 ảnh train |802,26 |Ngoại suy từ [profile d8000 T4](profile_d8000_kaggle_results.json) |
| Trích feature/eval và overhead |54,28 |Phần non-BP của lượt5 ×95.000/41.000 lượt ảnh |
| Setup và tests |120 |Khoản dự phòng, không phải số đo riêng |
| I/O state d8000/checkpoint |120 |Khoản dự phòng, không phải số đo riêng |
| Tổng dự trù |3531,97 (~58,87 phút) |Ước lượng, chưa phải runtime benchmark đo được |

Để còn ít nhất30% cửa sổ thời gian, cần tối thiểu3531,97/0,7=**84,09 phút**. Chuẩn bị timeout **2 giờ** có khoảng50,94% thời gian dự phòng so với ước lượng. Quota26 giờ là con số Sang cung cấp trước các lượt ngắn, không phải phép đọc tài khoản trực tiếp; job setup/download và hiệu năng host có thể biến động. Peak analytic3,20GB và peak BP5 nêu trên là các phép đo riêng, không phải cam kết peak của full benchmark.

[Budget JSON](benchmark_seed42_budget.json) lưu công thức và khoản dự phòng. [Notebook06](../notebooks/06_benchmark_seed42_ready.ipynb) đã chuẩn bị160 epochs/d8000/K5/seed42, fresh base, offline tracking và resume v2. Bootstrap06 trên CPU đạt54 tests với2 skip; guard chặn chạy160 epochs trên CPU và thay success cũ bằng failed khi lỗi. **Chưa khởi chạy160 epochs** trong lượt nghiệm thu BP5 này. Giữ checkpoint5 epochs làm bằng chứng/profile; không resume nó thành target160.
