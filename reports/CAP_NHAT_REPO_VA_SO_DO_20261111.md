# Báo cáo cập nhật chuẩn repository và bổ sung sơ đồ tự vẽ — 11/11/2026

Báo cáo này ghi nhận các hạng mục đã bổ sung vào repository **`24122050S/20-ML-Paper`** sau khi kiểm tra mức độ đáp ứng yêu cầu giữa kỳ (`Reading the Paper`, `Experimental Reproduction`, `5% Repo Quality`) và chuẩn `Critical Analysis`.

> **Lưu ý về tính toàn vẹn lịch sử thực nghiệm:** Không có số liệu JSON, manifest (`*-provenance.json`), notebook đã chạy (`01`–`12`), hoặc checkpoint/run W&B lịch sử nào bị sửa đổi hay chạy lại. Các lượt chạy Kaggle trước đây tiếp tục giữ nguyên commit/bundle SHA-256 đã pin trong manifest của chúng.

---

## 1. Đồng bộ repository

- Đồng bộ fast-forward nhánh local `main` từ `31de92e` lên `576014b` (`origin/main`, PR #22 ngày 08/11/2026).
- Giữ nguyên file untracked lịch sử `reports/REVIEW_GITHUB_20261103.md` tại local, không đưa nhầm vào thay đổi mới.

---

## 2. Hai sơ đồ tự vẽ mới trong `reports/figures/`

Được sinh tự động bằng script [`scripts/plot_acil_diagrams.py`](../scripts/plot_acil_diagrams.py) (sử dụng Matplotlib trong môi trường vẽ `requirements-plots.txt`, không cắt dán hình từ PDF bài báo gốc và không dump raw code):

1. **[`reports/figures/acil_pipeline_end_to_end.png`](figures/acil_pipeline_end_to_end.png)** — *Sơ đồ Pipeline End-to-End của ACIL:*
   - **Stage 1 (Phase 0 — Base BP Training & Analytic Re-alignment):** Dữ liệu 50 lớp base (`25,000` ảnh, augmentation + base-only norm) $\rightarrow$ `ResNet-32` + `FC50` (`160` epochs SGD, LR `0.1 -> 0.01 -> 0.001`) $\rightarrow$ nạp lại & đóng băng backbone (`eval()`, khóa BatchNorm, bỏ `FC50`) $\rightarrow$ trích đặc trưng sạch `F_0 (n, 64)` chuyển sang `float64` và mở rộng cố định `H_0 = ReLU(F_0 W_fe)` (`W_fe` $\sim \mathcal{N}(0, 1)$, `d=8000`) $\rightarrow$ tích lũy Gram/cross và giải Cholesky (`R_0`, `W_0`).
   - **Stage 2 (Phases $k = 1..K$ — Incremental Analytic Update):** Dữ liệu phase mới (`5,000` ảnh/phase, không lưu mẫu cũ) qua đúng backbone đóng băng + `W_fe` $\rightarrow$ đệm cột toàn cục `W_pad = [W_{k-1}, 0]` $\rightarrow$ cập nhật đệ quy Woodbury & phần dư theo phương trình (9)–(10) của bài báo.
   - **Stage 3 (All-Seen Inference):** Forward ảnh test mọi lớp đã thấy $0..C_k-1$ $\rightarrow$ `S = H_test W_k` $\rightarrow$ `argmax` không dùng Task ID.
2. **[`reports/figures/acil_problem_and_predecessors.png`](figures/acil_problem_and_predecessors.png)** — *Minh họa bài toán CIL và đối chiếu tiền nhiệm:*
   - **Panel A:** Bối cảnh học tăng dần theo phase ($50 \rightarrow 60 \rightarrow \dots \rightarrow 100$ lớp) với 3 ràng buộc (*không lưu dữ liệu cũ / privacy*, *ổn định–dẻo*, *cạnh tranh argmax khi số lớp tăng*).
   - **Panel B:** Đối chiếu cấu trúc 4 phần (*Core Idea*, *Stored State*, *Remaining Limitation*, *Practical Consequence*) giữa **Regularization-based (LwF, EWC)**, **Exemplar Replay-based (iCaRL, PODNet, AANets)** và **ACIL (NeurIPS 2022)**.

Cả hai file ảnh đã được kiểm tra trực quan sau khi xuất: không bị tràn khung, không đè chữ, căn lề và mũi tên rõ ràng để đưa trực tiếp vào slide deck `.pptx`.

Tái tạo hai hình từ gốc repo (với môi trường có Matplotlib):

```bash
python scripts/plot_acil_diagrams.py
```

---

## 3. Cập nhật `README.md` theo chuẩn 5% & `statement/reproduction/`

1. **Thêm mục "Sơ đồ phương pháp và bài toán (tự vẽ)"** nhúng trực tiếp 2 hình trên và dẫn script tái tạo.
2. **Thêm "Bảng ánh xạ toàn bộ thực nghiệm (Experiment Mapping)"** hợp nhất toàn bộ 24 run chính (R1, R2, R3, Author C1/C2/C1R) cùng các run verify/smoke/profile (`VERIFY-01`, `VERIFY-02`, `CIFAR-SMOKE-01`, `CIFAR-KAGGLE-01`, `CIFAR-KAGGLE-02`, `PROFILE-KAGGLE-01` [`gey0yop4`], `BP5-KAGGLE-01` [`pv82dy1f`]) theo đúng 4 cột: *Thí nghiệm ↔ Notebook / Script ↔ W&B run ↔ Kết quả chính (1 dòng)*.
3. **Thêm bảng "Giải thích các thư viện ngoài standard stack (`numpy`, `scipy`, `scikit-learn`, `torch`)"** giải trình lý do sử dụng từng gói trong `requirements-*.txt` (`wandb`, `huggingface_hub`, `pyarrow`, `Pillow`, `filelock`, `psutil`, `threadpoolctl`, `matplotlib`, `nbformat`, `nbclient`, `ipykernel`, `tqdm`, `PyYAML`, `torchvision`).

---

## 4. Chuẩn hóa docstring 3 phần trong `src/*.py`

Đã rà soát bằng `ast` và bổ sung docstring đầy đủ 3 thành phần (**mục đích**, **`Tham số:`**, **`Trả về:`**) cho toàn bộ các hàm/phương thức (kể cả helper nội bộ và dunder methods quan trọng) trên tất cả 14 module nguồn trong `src/` mà **không thay đổi logic thực thi hay comment giải thích nghiệp vụ sẵn có**:

- `src/analytic.py` (`_data`, `_classes`, `_gamma`, `_targets`, `joint_ridge`, `initialize`, `initialize_batches`, `incremental_update`)
- `src/base_training.py` (`atomic_torch_save`, `_runtime`, `capture_rng`, `restore_rng`, `_validate_resume`, `train_base`)
- `src/cifar_data.py` (`sha256_file`, `load_cifar100`, `make_protocol`, `audit_data`, `CifarSubset.__init__`, `__len__`, `__getitem__`)
- `src/cifar_smoke.py` (`fingerprint`, `seed_everything`, `make_loader`, `validate_checkpoint`, `extract_features`, `iter_feature_batches`, `validate_config`, `run`, `_compute_report`, `train_batches`, `test_batches`, `main`)
- `src/metrics.py` (`summarize`)
- `src/notebook_bundle.py` (`build`)
- `src/profile_analytic.py` (`MemoryMonitor.__init__`, `_sample`, `__enter__`, `__exit__`, `balanced_loader`, `profile`, `batches`, `main`)
- `src/reporting.py` (`collect_provenance`, `save_json`, `log_and_save_report`)
- `src/resnet.py` (`ResidualBlock.__init__`, `forward`, `ResNet32.__init__`, `_stage`, `forward_features`, `forward`, `freeze_backbone`)
- `src/run_attempt.py` (`_now`, `_read_json`, `_atomic_json`, `RunAttempt.__init__`, `_identity`, `_current`, `_archive_legacy`, `__enter__`, `_pending_report`, `progress`, `complete`, `fail`, `_finish`, `__exit__`)
- `src/streaming.py` (`StreamingHead.__init__`, `transform`, `fit_base`, `update_phase`, `evaluate`, `state_bytes`, `run_streaming_phases`)
- `src/verification.py` (`make_synthetic_phases`, `run_verification`)
- `src/verify_analytic.py` (`main`)
- `src/verify_resume.py` (`assert_state_equal`, `verify`, `setup`, `main`)

---

## 5. Kiểm tra xác nhận sau chỉnh sửa

Sau tất cả chỉnh sửa trong `src/`, `scripts/`, `README.md` và `reports/`:

1. **Kiểm tra AST docstring toàn bộ `src/*.py`:** Tất cả hàm và phương thức đều có docstring đủ 3 phần (*mục đích*, *Tham số*, *Trả về*).
2. **Tạo lại 2 hình sơ đồ bằng `scripts/plot_acil_diagrams.py`:** Thành công, kiểm tra trực quan 2 file PNG rõ đẹp.
3. **Unit tests (`tests/`):** Chạy `python -m unittest discover -s tests -v` — **54 tests chạy, 53 passed, 1 skipped** (CUDA test bỏ qua trên môi trường CPU local).
4. **Unit tests (`scripts/tests/`):** Chạy `python -m unittest discover -s scripts/tests -v` — **11/11 passed**.

---

## 6. Những việc còn lại để hoàn tất nộp bài

- **Bắt buộc cho Giữa kỳ (45% — hạn 21/11/2026 theo ghi chú trong repo):**
  1. Soạn nội dung đọc hiểu còn thiếu (*bài toán + ví dụ thực tế tự đặt*, *phân tích chi tiết >= 2 tiền nhiệm*, *phân loại đóng góp theoretical/technical/empirical & novelty*, *giải thích trực quan + giới hạn các định lý Theorem 3.1 / Absolute Memorization / Privacy*, *phân tích độ phức tạp time/space so sánh với tiền nhiệm*).
  2. Dựng file **Slide Deck `.pptx`** (20–40 slides gồm appendix, `< 20 MB`, speaker notes $\le 500$ từ trên **mọi** slide, trích dẫn IEEE dưới chân từng slide dùng nguồn, **không** có slide References riêng, gắn link W&B cho từng số liệu, nhúng 2 sơ đồ tự vẽ mới + 5 biểu đồ kết quả trong `reports/figures/`).
  3. Kiểm tra chế độ **Public** trên Kaggle cho các notebook benchmark (`06`–`12`) nếu dẫn link Kaggle trong slide/README; commit và push các cập nhật repo này lên GitHub trước khi nộp link.
- **Cho Cuối kỳ (`Critical Analysis` + `Empirical Demonstration`, hạn 05/12/2026):**
  - Viết tài liệu phản biện lý thuyết 4 trục + tối thiểu 3 strengths có bằng chứng + 2 theoretical limitations thiết kế được thí nghiệm + 2 future directions gắn 1-1, chạy thực nghiệm chứng minh trên W&B (tag riêng) và dựng Poster `.pptx` 1 slide khổ dọc A2.

