# Resume và nghiệm thu trước benchmark dài

Cập nhật 04/11/2026. Dùng cùng [protocol v1.0](protocol_chot.md) và [metrics v1.0](metrics.md). Notebook mới là **`notebooks/04_resume_benchmark_profile.ipynb`**; notebook02/03 giữ nguyên source của các bằng chứng cũ.

**Cập nhật06/11:** analytic Kaggle, BP5 và CUDA resume đã đạt; [JSON BP5](../reports/profile_base_kaggle_ready.json), [review/budget](../reports/REVIEW_BP5_KAGGLE_20261105.md). Tracking BP5 [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) đã sync, Finished và khớp API công khai. Không cần chạy05 lại. [Notebook06](../notebooks/06_benchmark_seed42_ready.ipynb) đã chạy xong160/d8000/K5/seed42, Ā65,1646%, final55,87%, [Kaggle v1](https://www.kaggle.com/code/hsangg206/06-benchmark-seed42/output?scriptVersionId=365844618)/[W&B zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) Finished; [báo cáo](../reports/R1_SEED42_KAGGLE_20261106.md); R1 nay đã đủ42/43/44, Ā65,1196 ± 0,6104% (n=3, ddof=1), [báo cáo tổng hợp](../reports/R1_CIFAR100_K5_3SEEDS_20261106.md); xem [trạng thái ngắn](BUOC_TIEP_THEO_SANG.md). R2 đã đủ9 analytic runs mới/3 seeds,4 hàng gồm baseline R1 tái dùng ([báo cáo](../reports/R2_CIFAR100_K5_20261107.md)); R3 cũng đã hoàn tất9 phép đo/54 so sánh joint ([báo cáo](../reports/R3_CIFAR100_K5_BATCH_20261107.md)); đã có [đối chiếu tác giả](../reports/AUTHOR_CODE_COMPARISON_20261107.md); còn sản phẩm nộp. Các bước04 bên dưới giữ để tái lập analytic.

## 1. Đã thay đổi gì, vì sao

| Thay đổi | Lý do và hành vi |
|---|---|
| `src/base_training.py` | Ghi checkpoint v2 sau **mỗi epoch hoàn tất**, qua file tạm/fsync/replace. Lỗi giữa epoch giữ bản epoch trước. |
| Resume đầy đủ | Lưu model/BatchNorm, SGD momentum, scheduler, mục tiêu epochs, history, RNG Python/NumPy/Torch CPU/CUDA và generator của DataLoader. Chỉ tải model sẽ đổi LR, augmentation hoặc thứ tự mẫu. |
| Kiểm checkpoint | Đối chiếu seed, recipe, class order, normalization, revision dữ liệu, loader, runtime và hash source huấn luyện. Không âm thầm resume khi đổi điều kiện. |
| `src/streaming.py` + `initialize_batches` | Base tích lũy `H.T@H` và `H.T@Y` theo batch, cộng gamma **một lần**, dùng Cholesky giải R/W. Phase mới chỉ nhận batch hiện tại. Không giữ toàn bộ H hay chạy joint oracle trong benchmark. |
| `src/metrics.py` | Xuất A-bar gồm base, final accuracy, base forgetting và signed delta. Chuỗi chưa đủ K có full A-bar=null; SD giữa seeds=null khi mới có một seed. |
| `src/profile_analytic.py` | Đo d8000 trước training dài: thời gian và peak RSS/GPU, shape/state bytes, checkpoint/source hashes. Đây là subset resource probe, không phải accuracy benchmark. |
| `configs/cifar100_benchmark_k5.json` | Preset160 epochs/d8000/K5 đã chốt. CLI mặc định vẫn là smoke2 epochs; phải chọn preset benchmark một cách tường minh. |

No-FE của benchmark dùng trực tiếp **F(n,64)** từ backbone, giữ cả giá trị âm; không thay bằng random projection64. `joint_reference_status="not_run"` và các error/disagreement fields=null khi không chạy oracle. Smoke vẫn giữ oracle để kiểm chứng nhỏ.

## 2. Tái lập profile04 khi cần

1. Tải/import **[notebook04](../notebooks/04_resume_benchmark_profile.ipynb)** từ main lên một notebook Kaggle mới. Bật Internet và GPU.
2. Trong **Input / Add Input**, thêm Output của notebook03 đã chạy thành công. Cần file **`base_checkpoint.pt`** của base2 epochs; `results.json` không chứa trọng số. Nếu Output03 không có checkpoint, chạy lại smoke03 và lưu Output trước.
3. Mở **cell code đầu tiên** đang thu gọn, giữ:

   ```python
   ACTION = "profile_analytic"
   CHECKPOINT = None
   RESUME_CHECKPOINT = None
   LOG_WANDB = True
   PROFILE_DIMENSION = 8000
   ```

   Notebook tự tìm checkpoint nếu `/kaggle/input` có đúng một file tên `base_checkpoint.pt`. Nếu có nhiều file, copy đúng đường dẫn từ Input và điền vào `CHECKPOINT`. Bật quyền dùng secret `WANDB_API_KEY` trong Kaggle Secrets; không dán key vào cell.
4. **Save & Run All.** Cell tests có kiểm resume CUDA khi GPU khả dụng. Cell cuối phải in `RESUME-BENCHMARK-PROFILE COMPLETE`, status complete, mọi check true và W shape `[8000,100]`.
5. Lấy **Output → `profile_d8000/results.json`**, `profile_d8000/progress.json` và link version đã lưu. Kiểm `wandb_status="logged"`, run URL do JSON tự lấy. Gửi JSON và link version để chốt bằng chứng GPU/time/quota.

Profile mặc định dùng **1000 ảnh base + 200 ảnh từ 50 lớp mới + 600 ảnh test**, batch128, float64. Chuyển50→100 ở đây chỉ đo kích thước W100 và tài nguyên; không phải protocol K5 với năm phase10 lớp. Không dùng accuracy subset cho bảng paper. CPU và Kaggle T4 đã đo được d8000; [manifest Kaggle](../reports/PROFILE-KAGGLE-01-provenance.json) xác nhận profile analytic đạt. **BP5/CUDA đã nghiệm thu ở Version3; quota26h do Sang cung cấp, budget2 giờ đã lập. Tracking BP5 dùng offline payload riêng và trạng thái sync trong manifest**.

Sau profile analytic, đổi **`ACTION="profile_base"`**, đặt `CHECKPOINT=None`, `RESUME_CHECKPOINT=None`, `STOP_AFTER_EPOCH=None`, rồi Save & Run All một version mới. Action này train BP5 epochs, chạy một phase analytic d256 để kiểm pipeline, ghi history/time/checkpoint v2; LR milestones vẫn80/120 nên không giảm LR trong5 epochs. Dùng thời gian thực đo để dự trù160 epochs, cộng analytic/evaluation/I/O và dự phòng ít nhất30% theo protocol.

Chỉ sau khi kiểm resume CUDA, hai profile và quota đạt mới đổi **`ACTION="benchmark"`**. Giữ `CHECKPOINT=None` để train base160 epochs mới, hoặc dùng đúng checkpoint base160 epochs đã hoàn tất. Checkpoint2/5 epochs sẽ bị từ chối cho preset160. Không cần chạy160 epochs để kiểm tính năng resume.

## 3. Resume dùng như thế nào

Resume hiện hỗ trợ **BP tại biên epoch**, với `num_workers=0`. Chưa hỗ trợ resume R/W giữa chuỗi analytic; nếu ngắt ở analytic, reuse checkpoint base hoàn tất để chạy lại analytic từ đầu.

| Tình huống | Tham số |
|---|---|
| Train mới | `CHECKPOINT=None`, `RESUME_CHECKPOINT=None` |
| Tiếp tục base đang dở, checkpoint v2 | `CHECKPOINT=None`, `RESUME_CHECKPOINT="/kaggle/input/.../base_checkpoint.pt"` |
| Base hoàn tất, chạy lại analytic | `CHECKPOINT="/kaggle/input/.../base_checkpoint.pt"`, `RESUME_CHECKPOINT=None` |

Để thử ngắt: dùng `ACTION="profile_base"`, `STOP_AFTER_EPOCH=1`, target vẫn5 epochs. Lệnh chủ động báo lỗi sau khi lưu epoch1; attempt ghi failed vì pipeline chưa hoàn tất. Checkpoint ở `profile_base/attempts/<attempt_id>/base_checkpoint.pt`, đường dẫn xuất hiện trong lỗi/progress. Lưu và attach Output đó vào phiên tiếp theo, đặt `RESUME_CHECKPOINT` đúng file, **`STOP_AFTER_EPOCH=None`**, giữ target5 và source/runtime/config như cũ. Checkpoint cuối có `completed_epoch=5`, `target_epochs=5`, `training_complete=true`.

Checkpoint được ghi trên disk mỗi epoch; điều này không tự đồng bộ Output lên Kaggle giữa phiên. Cần giữ/download checkpoint hoặc lưu version Output trước khi kết thúc phiên. Không thể hứa khôi phục file chưa được Kaggle lưu khi phiên bị hard kill. Resume strict sẽ báo lỗi nếu môi trường Kaggle/source thay đổi; giữ đúng version notebook04 và package/runtime đã tạo checkpoint.

CLI tương đương, chạy từ thư mục gốc repo:

```bash
python -m pip install -r requirements-cifar.txt
python -m unittest discover -s tests -v
python -m src.verify_resume --data-dir ../data

# Target vẫn2 epochs, chủ động dừng sau epoch1 (exit lỗi là dự kiến).
python -m src.cifar_smoke --output-dir reports/local/resume_demo --stop-after-epoch 1
# Lấy đúng path từ progress/lỗi, thay <attempt_id> rồi chạy:
python -m src.cifar_smoke --output-dir reports/local/resume_demo --resume-checkpoint reports/local/resume_demo/attempts/<attempt_id>/base_checkpoint.pt

# Profile local từ checkpoint base2 epochs hoàn tất đã có:
python -m src.profile_analytic --checkpoint reports/local/cifar100_smoke/base_checkpoint.pt --data-dir ../data --output-dir reports/local/profile_d8000 --dimension 8000 --threads 4 --device cpu

# Chỉ dùng khi đã qua gate và đủ tài nguyên; đây là lượt160 epochs:
python -m src.cifar_smoke --config configs/cifar100_benchmark_k5.json --device cuda --output-dir reports/local/benchmark_seed42
```

Checkpoint cũ thiếu optimizer/RNG vẫn dùng **reuse/profile** khi base đã hoàn tất và metadata khớp; không dùng resume. Checkpoint kiểm tra subset200 ảnh bị từ chối khi reuse làm backbone cho full CIFAR benchmark.

## 4. Đọc kết quả và giới hạn phép đo

- [Resume CPU](../reports/resume_cpu_verification.json): ResNet32, CIFAR thật200 ảnh/epoch, augmentation crop/flip, hai epochs liên tục so với1+resume1. Model/BatchNorm và history ngoài thời gian phải giống. Unit test còn so từng tensor optimizer/scheduler/RNG, LR và sample order; có test riêng cho ngắt giữa epoch, ghi file lỗi, checkpoint sai điều kiện.
- [Streaming CPU100 lớp](../reports/cifar100_streaming_cpu_verification.json): reuse backbone2 epochs/d256, đủ6 mốc/10.000 test cuối; joint và full feature collector bị chặn khi chạy. So accuracy với [run oracle CPU lịch sử](../reports/cifar100_joint_cpu_reference.json) ở từng mốc. Đây là kiểm runner, không phải benchmark160 epochs.
- [Profile CPU d8000](../reports/profile_d8000_cpu_results.json): RSS là process memory, sampled20ms; không phải tổng RAM máy. CUDA memory=null trên CPU. `persistent_analytic_state_bytes=522496000` chỉ gồm R/W/FE (~498,29MiB); peak cao hơn do solver workspace, mảng tạm, CNN/dữ liệu.
- Profile cache F64 của **subset** để tách thời gian CNN khỏi analytic; không lưu expanded H lịch sử hay ma trận R lớn ra NPZ. Runner benchmark chính đọc CNN/analytic xen kẽ, không cache train cũ. Các trường `base_accumulation_seconds`/`last_incremental_seconds` của runner chính gồm thời gian đọc iterator/CNN; `feature_extraction_seconds` nằm trong wall time, không cộng lại. Profile riêng dùng F64 đã trích nên timing analytic không gồm CNN.
- Ngoại suy trong profile chỉ là tuyến tính theo số mẫu trên cùng host; chưa gồm full evaluation, BP, I/O. Không lấy thời gian CPU làm cam kết runtime Kaggle.

Manifest [RESUME-PROFILE-01](../reports/RESUME-PROFILE-01-provenance.json) giữ checksum các JSON và source bundle04. JSON giữ đúng source dirty/commit tại lúc chạy; không gán commit hậu nghiệm thành source đã thực thi. Các file JSON cũ vẫn giữ nguyên.
