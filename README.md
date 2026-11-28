# ACIL — Reproduction and Analysis

Đồ án Nhập môn Máy học (**Nhóm 3**): đọc, tự triển khai, tái lập và phản biện **ACIL**, NeurIPS 2022.

| Thành viên | MSSV | Phụ trách chính |
|---|---|---|
| Nguyễn Hoàng Sang | `24122050` | Phương pháp, analytic head, kiểm chứng joint/phase/minibatch và R3 |
| Trần Hoàng Minh Vĩ | `24122026` | Dataset, protocol, backbone, Kaggle và R1 |
| Trần Đình Tấn Phát | `24122046` | Metrics, W&B, ablation R2, bằng chứng và tổng hợp kết quả |

## Hiện trạng

- [Tài liệu Reading the Paper — ACIL (NeurIPS 2022), 11/11/2026](docs/READING_THE_PAPER_ACIL.md): tổng hợp nội dung đọc hiểu lý thuyết gồm bài toán & ví dụ thực tế tự đặt, đối chiếu 3 hướng tiền nhiệm (LwF, iCaRL, PODNet/AANets), phân loại đóng góp & điểm mới, phân tích phương pháp 5 lớp (ký hiệu, pipeline, phân tích từng thành phần, chứng minh & điều kiện biên Định lý 3.1 / Absolute Memorization / Privacy, độ phức tạp thời gian–không gian) và danh sách trích dẫn IEEE cho slide giữa kỳ.
- [Kịch bản 36 Slides Giữa kỳ (Main 32 + Appendix 4), 12/11/2026](docs/KICH_BAN_36_SLIDE_GIUA_KY.md) & **[Slide Deck Giữa kỳ hoàn chỉnh (`ACIL_Nhom3_GiuaKy.pptx`, 36 slides, 4,91 MiB / 5,15 MB), 18/11/2026](slides/ACIL_Nhom3_GiuaKy.pptx)**: đầy đủ nội dung đọc hiểu và thực nghiệm tái lập `R1/R2/R3` + đối chiếu tác giả, kèm link W&B public, trích dẫn IEEE dưới chân từng slide và Speaker Notes trên đủ 36/36 slides (`≤ 500` từ/slide).
- [Khung Phản biện Lý thuyết & Thiết kế Thực nghiệm Cuối kỳ, 18/11/2026](docs/PHAN_BIEN_CUOI_KY_ACIL.md): phân tích 4 góc nhìn lý thuyết của ACIL, 3 điểm mạnh có bằng chứng (`S1/S2/S3` từ `R1/R2/R3`), 2 hạn chế lý thuyết chính cho Poster A2 (`P1` phụ thuộc tập Base đóng băng, `P2` bộ nhớ $\mathcal{O}(d^2)$ theo byte & độ nhạy số học Woodbury), thiết kế thực nghiệm kiểm chứng `E1`–`E2` và 2 hướng phát triển gắn 1–1 (`FD1`–`FD2`).
- [Kế hoạch Thực nghiệm Cuối kỳ (Pre-registered Protocol v1.0), 18/11/2026](docs/KE_HOACH_EMPIRICS_CUOI_KY.md): khóa trước giả định kiểm tra, biến thay đổi/cố định, đối chứng, metric, dự đoán & tiêu chí bác bỏ cho `E1a/E1b` và `E2a/E2b/FD2`, đi kèm các runner riêng [`scripts/e1_base_assumption_runner.py`](scripts/e1_base_assumption_runner.py), [`scripts/e2_memory_precision_runner.py`](scripts/e2_memory_precision_runner.py) và bộ kiểm thử đơn vị [`scripts/tests/test_final_empirics.py`](scripts/tests/test_final_empirics.py).
- **Thực nghiệm cuối kỳ `E2` hoàn tất ngày 18/11/2026 (kiểm chứng `P2`):** tái sử dụng 3 checkpoints ResNet-32 160 epochs của R1 (`seeds 42/43/44`, không train lại CNN), quét 11 mức chiều mở rộng $d \in \{64..8000\}$ kèm nguyên mẫu `FD2` và độ trôi số học Woodbury `float32` vs `float64`. Tại đúng trần ngân sách `6.144 MB` của 2.000 ảnh CIFAR-100 ($d = 798$ `fp64` / $d = 1160$ `fp32`), $A_{\mathrm{final}}$ giảm **$-5,19\text{ pp}$** (`56,13% → 50,94%`) / **$-3,95\text{ pp}$** (`52,18%`) và $\bar{A}$ giảm **$-2,95\text{ pp}$** / **$-2,30\text{ pp}$** so với chuẩn bài báo $d = 8000$ (`522,50 MB` = `85,04×` ngân sách kho mẫu); đồng thời hạ sang `float32` làm sai số tương đối trọng số Woodbury tăng hơn $4 \times 10^7$ lần và phát sinh mẫu đổi nhãn `argmax`. [Báo cáo E2](reports/E2_MEMORY_PRECISION_20261118.md) · [Summary](reports/e2_memory_precision_summary.json) · [Manifest](reports/E2-FINAL-EMPIRICS-provenance.json) · [Biểu đồ Poster](reports/figures/e2_memory_and_woodbury_precision.png) (6 W&B Finished/công khai).
- **Thực nghiệm cuối kỳ `E1` hoàn tất ngày 18/11/2026 (kiểm chứng `P1`):** chạy 6 kernels Kaggle GPU T4 (`notebooks/13..18`, 12 lượt huấn luyện ResNet-32 160 epochs + tầng giải tích $d=8000, \gamma=0.1$ trên 3 seeds `42/43/44`). Ở `E1a`, thu hẹp lớp Base $c_0: 50 \to 30 \to 20 \to 10$ làm $A_{\mathrm{final}}$ 100 lớp sụt từ **`56,13 ± 0,56%`** xuống **`48,14%`**, **`41,79%`** và **`33,57 ± 0,53%`** (**`-22,56 pp`**) dù nghiệm đệ quy vẫn khớp Joint Ridge (`0` mẫu lệch). Ở `E1b`, giữ nguyên $c_0 = 50$ (`25.000` ảnh Base) nhưng tách rời miền ngữ nghĩa (`10` superclasses tự nhiên/sinh vật ở Base $\to$ `10` superclasses phương tiện/đồ vật & nhóm còn lại ở Incremental) làm $A_{\mathrm{inc}}$ 50 lớp mới giảm **`-8,21 pp`** (`43,93% → 35,71%`), nới rộng $\Delta_{\text{gap}} = B_5 - A_{\mathrm{inc}}$ thêm **`+9,85 pp`** lên **`34,26 ± 0,62 pp`**. [Báo cáo E1](reports/E1_BASE_ASSUMPTION_20261118.md) · [Summary](reports/e1_base_assumption_summary.json) · [Manifest](reports/E1-FINAL-EMPIRICS-provenance.json) · [Biểu đồ Poster](reports/figures/e1_base_representation_bottleneck.png) (12 W&B Finished/công khai).
- **Thực nghiệm phụ cuối kỳ `E3` hoàn tất ngày 18/11/2026 (trình bày hạn chế thứ 3 `P3` trong README):** kiểm chứng trên cùng `d = 1024` và 3 checkpoints `R1` (`seeds 42/43/44`) để tách bạch **độ trôi số học `float32` (`E2b`)** khỏi **thiên lệch cấu trúc của hàm mục tiêu MSE One-Hot khi mất cân bằng lớp (`E3`)**. Khi số mẫu mỗi lớp tăng dần giảm từ $500$ xuống $100 \to 50 \to 10$ (so với $N_{\mathrm{base}} = 500$), dù nghiệm `float64` vẫn thỏa Định lý 3.1 tới $\le 7,74 \times 10^{-12}$ (`0` mẫu lệch Joint Ridge), chuẩn cột trọng số $\|w_{\mathrm{inc}}\|_2 / \|w_{\mathrm{base}}\|_2$ co từ `0,915` xuống **`0,103`** và khiến độ chính xác 50 lớp mới $A_{\mathrm{inc}}$ sụt từ `32,62%` xuống **`0,21%`** (`10×`) rồi **`0,00%`** (`50×`). Nguyên mẫu nghiệm đóng **Weighted RLS (`FD3`)** khôi phục **`+28,66 pp`** ở `10×` mà không lưu mẫu cũ hay tăng bộ nhớ $R_k$. [Báo cáo E3](reports/E3_MSE_IMBALANCE_20261118.md) · [Summary](reports/e3_mse_imbalance_summary.json) · [Manifest](reports/E3-FINAL-EMPIRICS-provenance.json) · [Biểu đồ `P3`](reports/figures/e3_mse_imbalance_vs_woodbury_drift.png) (3 W&B Finished/công khai).
- [Báo cáo tổng hợp thực nghiệm giữa kỳ, 07/11/2026](reports/BAO_CAO_THUC_NGHIEM_GIUA_KY_20261107.md): đầy đủ `R1/R2/R3`, [đối chiếu tác giả](reports/AUTHOR_CODE_COMPARISON_20261107.md), 2 sơ đồ tự vẽ trong `reports/figures/`, docstring 3 phần cho toàn bộ `src/`. Slide deck giữa kỳ ([`slides/ACIL_Nhom3_GiuaKy.pptx`](slides/ACIL_Nhom3_GiuaKy.pptx)), các bộ phản biện cuối kỳ (`E1/E2/E3`) và Poster A2 cuối kỳ ([`slides/ACIL_Nhom3_Poster_A2.pptx`](slides/ACIL_Nhom3_Poster_A2.pptx)) đều đã hoàn thiện; toàn bộ `15/15` notebook thực nghiệm chính (`03`–`18`) trên Kaggle và `56/56` runs trên W&B ở chế độ Public.
- Đã có module analytic hỗ trợ nhãn toàn cục và minibatch; notebook/CLI dùng cùng source.
- Đã có test độc lập cho phương pháp và thứ tự log W&B/xuất JSON.
- VERIFY-01 là mốc lịch sử trên dữ liệu tổng hợp. VERIFY-02 kiểm tra thêm minibatch.
- Pipeline CIFAR smoke đã có lượt CUDA đủ 100 lớp: [JSON](reports/cifar100_kaggle_full5_results.json), [Kaggle scriptVersionId 365084714](https://www.kaggle.com/code/hsangg206/03-cifar100-safe-attempts/output?scriptVersionId=365084714), [W&B blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) finished. W cuối 256×100, test 10.000 ảnh, accuracy 22,05%, mọi check đạt; source/config hashes đã đối chiếu. Đây là smoke 2 epochs/d256.
- Đã có resume CPU/CUDA, streaming analytic, profile d8000 và BP5; các bằng chứng lịch sử giữ riêng trong [danh mục](reports/README.md). **R1 hoàn tất 3 seeds 42/43/44**, mỗi seed 160 epochs/d8000/gamma0.1/K5. Ā **65,1196 ± 0,6104%**, final **56,1300 ± 0,5575%**, F_base **8,3067 ± 0,6201pp** (mean ± sample SD, n=3, ddof=1). Cả 3 W&B Finished/công khai; [báo cáo](reports/R1_CIFAR100_K5_3SEEDS_20261106.md), [summary JSON](reports/r1_cifar100_k5_summary.json), [manifest](reports/R1-C100-K5-3SEEDS-provenance.json). R2 và R3 cũng đã hoàn tất theo kế hoạch; đã có [đối chiếu tác giả](reports/AUTHOR_CODE_COMPARISON_20261107.md); đã hoàn thiện [Slide Deck `.pptx` giữa kỳ](slides/ACIL_Nhom3_GiuaKy.pptx), ba bộ thực nghiệm phản biện cuối kỳ (`E1`, `E2`, `E3`) và [Poster A2 cuối kỳ](slides/ACIL_Nhom3_Poster_A2.pptx).
- **R2 hoàn tất ngày 07/11/2026:** 9 lượt analytic mới trên seeds 42/43/44 và 3 baseline R1 tái dùng, đủ 4 hàng/3 seeds. [Báo cáo R2](reports/R2_CIFAR100_K5_20261107.md), [summary](reports/r2_cifar100_k5_summary.json), [manifest](reports/R2-C100-K5-provenance.json). Cả 9 W&B mới Finished/công khai; không train thêm backbone. Phạm vi 4/5 hàng Table 2, không có gamma 0.
- **R3 hoàn tất 07/11/2026:** batch 32/128/512 × 3 timing repeats trên một seed 42/d1024; 9 tiến trình CPU riêng, 54 so sánh phase khớp joint và 0 argmax khác. 9 W&B Finished/công khai. [Báo cáo](reports/R3_CIFAR100_K5_BATCH_20261107.md), [summary](reports/r3_cifar100_k5_summary.json), [manifest](reports/R3-C100-K5-provenance.json). Đây là 3 lần đo thời gian, không phải 3 seed benchmark.
- **Đã chạy code tác giả ngày 07/11/2026:** 2 lượt chính full CIFAR100/K5 trên cùng checkpoint 160 seed 42/d1024 và 1 diagnostic state base. [Báo cáo](reports/AUTHOR_CODE_COMPARISON_20261107.md), [raw JSON](reports/author_comparison_results.json), [manifest](reports/AUTHOR-COMPARE-01-provenance.json). C1 dự đoán khớp 6/6 mốc nhưng 2 phần tử score ngoài tolerance; C2 đạt 6/6; matched-base diagnostic đạt 5/5. Ba W&B Finished/public vẫn giữ flag C1=false. Chưa chạy recipe native 300 epochs/d8192.
- Project W&B: [hsangg-2006/acil-reproduction](https://wandb.ai/hsangg-2006/acil-reproduction), đã xác nhận Public ngày 02/11/2026.

## Cấu trúc và cách chạy

Mở thư mục gốc **`20-ML-Paper`** làm thư mục dự án trong VS Code. Cấu trúc đang dùng:

```text
20-ML-Paper/
├── src/                          # Code nhóm tự viết
│   ├── analytic.py               # Analytic head: phương trình (4), (9), (10) và joint ridge tham chiếu
│   ├── verification.py           # Dữ liệu tổng hợp, so joint/phase/minibatch
│   ├── verify_analytic.py        # CLI chạy VERIFY-02
│   ├── cifar_data.py             # Tải CIFAR-100, chia lớp theo seed, normalization từ base train
│   ├── resnet.py                 # ResNet-32 cho CIFAR
│   ├── base_training.py          # Base training bằng BP, checkpoint và resume theo epoch
│   ├── cifar_smoke.py            # Pipeline smoke có joint oracle
│   ├── streaming.py              # Pipeline benchmark không giữ dữ liệu phase cũ
│   ├── metrics.py                # Ā, final accuracy, F_base
│   ├── profile_analytic.py       # Đo thời gian/RAM cho analytic d=8000
│   ├── verify_resume.py          # Kiểm chạy liên tục so với resume
│   ├── run_attempt.py            # Thư mục và trạng thái riêng cho từng lượt chạy
│   ├── reporting.py              # Xuất JSON, source hashes, W&B
│   └── notebook_bundle.py        # Đóng gói source vào notebook Kaggle
├── tests/                        # Unit test cho các module trong src/
├── notebooks/
│   ├── 01_verify_analytic_update.ipynb      # VERIFY-01/02
│   ├── 02_cifar100_smoke.ipynb              # Smoke 50+10 lớp
│   ├── 03_cifar100_safe_attempts.ipynb      # Smoke đủ 100 lớp
│   ├── 04_resume_benchmark_profile.ipynb    # Profile d=8000 và BP5
│   ├── 05_profile_base_ready.ipynb          # Bản dựng lại của profile BP5
│   ├── 06_benchmark_seed42_ready.ipynb      # R1 seed 42
│   ├── 07_benchmark_seed43_ready.ipynb      # R1 seed 43
│   ├── 08_benchmark_seed44_ready.ipynb      # R1 seed 44
│   ├── 09_r2_ablation_seed42.ipynb          # R2 seed 42
│   ├── 10_r2_ablation_seed43.ipynb          # R2 seed 43
│   ├── 11_r2_ablation_seed44.ipynb          # R2 seed 44
│   ├── 12_r3_batch_analysis.ipynb           # R3
│   ├── 13_e1a_base_ratio_seed42.ipynb       # E1a (c0=10,20,30) seed 42
│   ├── 14_e1a_base_ratio_seed43.ipynb       # E1a (c0=10,20,30) seed 43
│   ├── 15_e1a_base_ratio_seed44.ipynb       # E1a (c0=10,20,30) seed 44
│   ├── 16_e1b_superclass_shift_seed42.ipynb # E1b superclass_disjoint seed 42
│   ├── 17_e1b_superclass_shift_seed43.ipynb # E1b superclass_disjoint seed 43
│   ├── 18_e1b_superclass_shift_seed44.ipynb # E1b superclass_disjoint seed 44
│   └── archive/
├── scripts/
│   ├── build_*.py                # Tạo notebook từ source hiện tại (gồm build_e1_notebooks.py)
│   ├── r2_runner.py, r3_runner.py, r3_worker.py   # Chạy R2/R3
│   ├── aggregate_r1.py, aggregate_r2.py, aggregate_r3.py   # Tổng hợp mean ± SD từ JSON
│   ├── plot_r1.py, plot_r2.py, plot_r3.py   # Vẽ hình giữa kỳ trong reports/figures/
│   ├── plot_acil_diagrams.py             # Tự vẽ sơ đồ pipeline ACIL và bài toán/tiền nhiệm
│   ├── compare_author_acil.py, diagnose_author_base_precision.py   # Chạy đối chiếu code tác giả
│   ├── aggregate_author_comparison.py, plot_author_comparison.py, track_author_comparison.py
│   ├── e1_base_assumption_runner.py, e2_memory_precision_runner.py   # Runner thực nghiệm cuối kỳ E1 & E2 (tách riêng với R1-R3)
│   ├── e3_mse_imbalance_runner.py   # Runner thực nghiệm phụ E3 (MSE mất cân bằng lớp & FD3 Weighted RLS)
│   ├── plot_final_empirics.py, plot_e3_mse_imbalance.py   # Tự vẽ biểu đồ thực nghiệm cuối kỳ E1, E2 & E3
│   ├── reconcile_r3_tracking.py
│   └── tests/                    # Test cho scripts R3, đối chiếu tác giả và thực nghiệm cuối kỳ E1/E2/E3
├── configs/
│   ├── verify_analytic_update.json
│   ├── cifar100_smoke.json
│   └── cifar100_benchmark_k5.json          # Cấu hình R1
├── slides/
│   ├── ACIL_Nhom3_GiuaKy.pptx        # Slide deck giữa kỳ hoàn chỉnh (36 slides + Speaker Notes + IEEE footers)
│   └── ACIL_Nhom3_Poster_A2.pptx     # Poster A2 dọc cuối kỳ (Critical Analysis & Empirical Demonstration)
├── docs/
│   ├── READING_THE_PAPER_ACIL.md     # Nội dung đọc hiểu học thuật đầy đủ (Reading the Paper)
│   ├── KICH_BAN_36_SLIDE_GIUA_KY.md  # Kịch bản 36 slides giữa kỳ kèm Speaker Notes & IEEE footers
│   ├── PHAN_BIEN_CUOI_KY_ACIL.md     # Phân tích 4 góc nhìn lý thuyết, 3 Strengths, P1-P2, E1-E2 & FD1-FD2 cuối kỳ
│   ├── KE_HOACH_EMPIRICS_CUOI_KY.md  # Kế hoạch thực nghiệm cuối kỳ khóa trước khi chạy (E1a/E1b, E2a/E2b, FD2)
│   ├── protocol_chot.md          # Protocol benchmark v1.0
│   ├── metrics.md                # Định nghĩa metric v1.0
│   ├── Dac_ta_ACIL.md            # Đặc tả module analytic
│   ├── R3_KE_HOACH_DO.md, DOI_CHIEU_CODE_TAC_GIA_KE_HOACH.md   # Kế hoạch đăng ký trước khi chạy
│   └── HUONG_DAN_*.md, CHAY_KAGGLE_BANG_API.md, RESUME_BENCHMARK_PROFILE.md   # Hướng dẫn chạy
├── reports/
│   ├── BAO_CAO_THUC_NGHIEM_GIUA_KY_20261107.md   # Báo cáo tổng hợp R1/R2/R3 và đối chiếu tác giả
│   ├── E1_BASE_ASSUMPTION_20261118.md, E2_MEMORY_PRECISION_20261118.md, E3_MSE_IMBALANCE_20261118.md   # Báo cáo thực nghiệm cuối kỳ E1, E2 & E3
│   ├── R1_*.md, R2_*.md, R3_*.md, AUTHOR_CODE_COMPARISON_*.md   # Báo cáo từng bộ giữa kỳ
│   ├── r1_*_summary.json, r2_*_summary.json, r3_*_summary.json, e1_*_summary.json, e2_*_summary.json, e3_*_summary.json   # Số liệu tổng hợp
│   ├── *_results.json, *_ready.json, *_progress.json   # JSON gốc từng lượt chạy
│   ├── *-provenance.json         # Manifest: source, version Kaggle, W&B, SHA-256
│   ├── figures/                  # Hình R1/R2/R3, đối chiếu tác giả, E1/E2 cuối kỳ và sơ đồ tự vẽ
│   ├── archive/                  # JSON gốc trước khi đồng bộ tracking và các lượt lỗi được giữ lại
│   └── README.md                 # Danh mục bằng chứng
├── requirements-verification.txt # Chỉ NumPy, cho module analytic
├── requirements-tracking.txt     # + wandb
├── requirements-cifar.txt        # + PyTorch, dữ liệu, SciPy
├── requirements-dev.txt          # + chạy notebook local
├── requirements-plots.txt        # matplotlib cho hình báo cáo
├── requirements-author-comparison.txt   # Môi trường riêng cho code tác giả
├── .gitattributes
├── .gitignore
└── README.md
```

`.git/`, `.venv/` và `reports/local/` là thư mục dùng trên máy; không đưa lên GitHub. `configs/`, các module reporting/verification và test tracking được giữ vì notebook VERIFY-02 đang dùng chúng.

| Đường dẫn | Nội dung |
|---|---|
| `src/` | Code nhóm tự viết: analytic head, dữ liệu, backbone, training, pipeline, metric (đủ docstring 3 phần) |
| `tests/`, `scripts/tests/` | Unit test cho phương pháp, pipeline, resume, metric, tracking, phân tích và thực nghiệm cuối kỳ `E1/E2/E3` |
| `notebooks/` | Notebook chạy trên Kaggle (`01..12` giữa kỳ, `13..18` cuối kỳ E1); từ notebook 02 trở đi có source bundle |
| `scripts/` | Tạo notebook, chạy R2/R3/E1/E2/E3, tổng hợp số liệu, vẽ hình tự dựng và đối chiếu code tác giả |
| `configs/` | Cấu hình VERIFY-02, smoke và benchmark R1 |
| `slides/` | File `.pptx` slide deck giữa kỳ hoàn chỉnh (36 slides kèm IEEE footers và Speaker Notes) |
| `docs/` | Nội dung đọc hiểu, kịch bản slide, khung phản biện cuối kỳ, kế hoạch khóa trước, protocol, metric, hướng dẫn chạy |
| `reports/` | Kết quả đã công bố, manifest SHA-256 và hình vẽ; `reports/local/` bị Git bỏ qua |

Từ thư mục gốc, Python 3.10 trở lên; môi trường local được kiểm tra bằng Python 3.12:

```bash
python -m pip install -r requirements-verification.txt
python -m unittest discover -s tests -v
python -m src.verify_analytic
```

NumPy được khóa 2.1.3. W&B là tùy chọn; khi cần tracking:

```bash
python -m pip install -r requirements-tracking.txt
python -m src.verify_analytic --wandb --entity hsangg-2006 --project acil-reproduction
```

`requirements-dev.txt` dành cho chạy/kiểm tra notebook cục bộ bằng nbclient và ipykernel; không bắt buộc để chạy core hoặc bộ unit test.

Local dùng thông tin đăng nhập/env đã cấu hình, không ghi credential vào source. JSON mặc định ở `reports/local/verify_02_results.json`; tracking tắt ghi `not_logged`, URL null. Khi init/log/finish lỗi, file ghi `failed` và lệnh trả lỗi. `logged` chỉ được ghi sau khi run context kết thúc thành công.

## Chạy notebook trên Kaggle

1. Import notebook 01, chọn Python, Accelerator **None**, bật Internet.
2. Notebook tự lấy source từ GitHub khi không chạy trong checkout của repo. `SOURCE_REF` mặc định main; có thể đặt commit cụ thể để tái lập. Commit và source hashes thật được lưu vào JSON.
3. Bật secret `WANDB_API_KEY` cho notebook nếu dùng tracking. Mặc định log vào project nhóm; SDK khóa 0.28.1. Có thể tắt `LOG_WANDB` ở cell đầu để chỉ kiểm chứng.
4. Run All hoặc Save & Run All; kiểm tra VERIFY-02 COMPLETE và test đạt.
5. Lấy `verify_analytic_update_v2_results.json` từ Output. URL trong file phải là URL từ chính run vừa hoàn thành.
6. Cập nhật bảng bằng chứng bằng đúng version, commit nguồn và JSON của cùng lượt chạy. Không điền run ID cố định vào source.

[Hướng dẫn Kaggle chi tiết](docs/HUONG_DAN_KAGGLE.md). Notebook tự chọn đường dẫn output cho Kaggle/local; không bắt máy cá nhân phải có `/kaggle/working`. Trên máy cá nhân muốn tắt tracking, đặt `ACIL_LOG_WANDB=0` trước khi mở kernel hoặc đổi `LOG_WANDB` trong cell thiết lập.

Bộ test phương pháp kiểm tra ba seed, joint/phase/minibatch, lớp xuất hiện lại, state đầu vào không bị sửa, feature thiếu hạng với gamma dương và input không hợp lệ. Test reporting giả lập SDK, không cần đăng nhập W&B. Config dùng chung giữ seed 42, gamma 1, input 16, feature 24, 3 phase × 2 lớp, minibatch 7, float64 cùng các ngưỡng `rtol`/`atol`.

## Chạy CIFAR-100 smoke (50+10 lớp)

[Hướng dẫn CIFAR 50+10](docs/HUONG_DAN_CIFAR_SMOKE.md) mô tả các bước chạy trên máy cá nhân / Kaggle và cách đọc kết quả.

```bash
python -m pip install -r requirements-cifar.txt
python -B -m src.cifar_smoke
```

Lệnh mặc định tải dữ liệu, train base 2 epochs, freeze ResNet-32/BatchNorm, expansion 256, chạy một phase mới và kiểm joint trên đủ 60 lớp. API dùng chính `src/analytic.py`; metadata mapping thuộc pipeline. Root output là `reports/local/cifar100_smoke/`; artifact đầy đủ nằm trong `attempts/<attempt_id>/`. Các test ảnh/attempt được skip rõ ràng nếu thiếu dependency CIFAR; 15 test core/tracking/metrics vẫn chạy với NumPy.

Analytic d8000 và BP5/CUDA resume đã đạt. BP5 chạy qua API trong [Version 3 /365703432](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432); không cần chạy 05 lại. **[Notebook 06](notebooks/06_benchmark_seed42_ready.ipynb)** đã đặt R1 seed 42/160 epochs/d8000/K5, checkpoint đầu vào=None, offline W&B để sync sau. [Hướng dẫn API](docs/CHAY_KAGGLE_BANG_API.md) và [budget](reports/benchmark_seed42_budget.json) giải thích timeout 2 giờ. R1 seed 42 đã chạy xong 160 epochs/đủ 5 phase: [JSON](reports/benchmark_seed42_kaggle_results.json), [W&B zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei); không cần chạy 06 lại để lấy bằng chứng. Script `python scripts/build_profile_base_notebook.py` tạo 05 từ cùng bundle 04; giữ 04 làm snapshot của bằng chứng mới. Generator cũ `python -m src.notebook_bundle` vẫn mặc định tạo 04: không dùng để ghi đè snapshot đã công bố. `benchmark` dùng preset 160 epochs khi đã qua gate. `--resume-checkpoint` tiếp tục BP đang dở; `--reuse-checkpoint` dùng base đã hoàn tất.

Mỗi lần gọi pipeline tạo một attempt riêng. Root `latest.json`, `progress.json`, `results.json` có cùng attempt ID và trạng thái mới nhất: `running`, `complete` hoặc `failed`. Chỉ nhận kết quả khi `status=complete` và `verification_passed=true`; lấy checkpoint/state từ `output_dir` trong report. Nếu lượt mới lỗi, root ghi failed/false; dữ liệu lượt trước vẫn ở thư mục riêng (xem [`src/run_attempt.py`](src/run_attempt.py) và [`tests/test_run_attempts.py`](tests/test_run_attempts.py)).

`filelock` bảo vệ metadata root khi nhiều lượt cùng ghi; JSON trạng thái được ghi qua file tạm rồi replace. Đây là hạ tầng lưu kết quả, không tham gia công thức analytic. [Tài liệu filelock](https://py-filelock.readthedocs.io/en/latest/).

## Sơ đồ phương pháp và bài toán (tự vẽ)

Hai sơ đồ minh họa phục vụ slide đọc hiểu giữa kỳ được tự dựng bằng Matplotlib qua [`scripts/plot_acil_diagrams.py`](scripts/plot_acil_diagrams.py) (cài môi trường vẽ `requirements-plots.txt` rồi chạy `python scripts/plot_acil_diagrams.py`; không cắt dán hình từ PDF bài báo):

- **Sơ đồ pipeline end-to-end của ACIL:** [`reports/figures/acil_pipeline_end_to_end.png`](reports/figures/acil_pipeline_end_to_end.png) — mô tả 3 giai đoạn từ Base BP Training (`ResNet-32` + `FC50`, 160 epochs) $\rightarrow$ đóng băng backbone & chiếu ngẫu nhiên $W_{\mathrm{fe}}$ $\rightarrow$ khởi tạo Ridge (`R_0`, `W_0`) $\rightarrow$ cập nhật đệ quy Woodbury/phần dư (`R_k`, `W_k`, không giữ mẫu cũ) $\rightarrow$ suy luận `argmax` trên toàn bộ lớp đã thấy.
- **Minh họa bài toán CIL và đối chiếu tiền nhiệm:** [`reports/figures/acil_problem_and_predecessors.png`](reports/figures/acil_problem_and_predecessors.png) — tóm tắt bài toán học tăng dần theo lớp, 3 thách thức chính (*không giữ dữ liệu cũ*, *ổn định–dẻo*, *không gian lớp mở rộng*) và so sánh giữa hướng Regularization (LwF/EWC), Exemplar Replay (iCaRL/PODNet/AANets) với ACIL.

![ACIL End-to-End Pipeline (tự vẽ)](reports/figures/acil_pipeline_end_to_end.png)

![CIL Problem & Predecessor Approaches vs ACIL (tự vẽ)](reports/figures/acil_problem_and_predecessors.png)

## Bằng chứng thí nghiệm

**Phạm vi tái lập đã chọn (*Core scope* — chi tiết tại [`docs/protocol_chot.md`](docs/protocol_chot.md)):** Do ràng buộc hạn mức GPU Kaggle T4 không đủ để chạy lại toàn bộ các bộ ảnh quy mô lớn như ImageNet-Full (`1.000` lớp, `d = 15.000`) hay quét đủ mọi giá trị `K ∈ {5, 10, 25, 50}`, nhóm chọn **Core scope** tập trung kiểm định trọn vẹn hai thí nghiệm trọng tâm của bài báo trên CIFAR-100 (`R1`: benchmark chính `K = 5`, `d = 8000`, `160` epochs trên **3 seeds `42/43/44`**; `R2`: ablation `4/5` hàng Table 2 về tầng mở rộng FE và độ nhạy $\gamma$ trên **3 seeds**), bổ sung **phân tích tự thiết kế `R3`** (quét kích thước minibatch `b ∈ {32, 128, 512}` đo thời gian, RAM và kiểm chứng khớp Joint Oracle trên đủ `54/54` mốc phase) cùng **đối chiếu trực tiếp source code tác giả (`C1/C2/C1R`)**.

### Bảng ánh xạ toàn bộ thực nghiệm (Experiment Mapping)

Bảng tổng hợp dưới đây ánh xạ từng lượt thí nghiệm tới notebook/script tạo ra kết quả, link W&B công khai và 1 dòng tóm tắt kết quả/ý nghĩa của lượt chạy:

| Thí nghiệm | Notebook / Script tạo kết quả | W&B run | Kết quả chính (1 dòng) |
|---|---|---|---|
| `VERIFY-01` | [`notebooks/01_verify_analytic_update.ipynb`](notebooks/01_verify_analytic_update.ipynb) | [7aeaupq7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7aeaupq7) Finished | Khớp nghiệm joint ridge trên dữ liệu tổng hợp theo phase (max abs W error `~1.84e-14`). |
| `VERIFY-02` | [`notebooks/01_verify_analytic_update.ipynb`](notebooks/01_verify_analytic_update.ipynb) / [`src/verify_analytic.py`](src/verify_analytic.py) | [wi4wau2d](https://wandb.ai/hsangg-2006/acil-reproduction/runs/wi4wau2d) Finished | Đạt 11/11 tests; cập nhật minibatch khớp joint trong `rtol=1e-8` (max rel W `~3.43e-13`), 0 argmax khác. |
| `CIFAR-SMOKE-01` | [`src/cifar_smoke.py`](src/cifar_smoke.py) (local CPU) | Tắt tracking (offline) | Kiểm tra end-to-end 2 epochs/d256 (50+10 lớp) trên CPU local trước khi chạy GPU Kaggle. |
| `CIFAR-KAGGLE-01` | [`notebooks/02_cifar100_smoke.ipynb`](notebooks/02_cifar100_smoke.ipynb) | [o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2) Finished | Smoke GPU 2 epochs/d256 (50→60 lớp): base 32,58%, 60 lớp 29,70%, khớp joint oracle. |
| `CIFAR-KAGGLE-02` | [`notebooks/03_cifar100_safe_attempts.ipynb`](notebooks/03_cifar100_safe_attempts.ipynb) | [blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) Finished | Smoke GPU đủ 5 phase (50→100 lớp, W 256×100, 10.000 ảnh test), final 22,05%, khớp joint 6/6 mốc. |
| `PROFILE-KAGGLE-01` | [`notebooks/04_resume_benchmark_profile.ipynb`](notebooks/04_resume_benchmark_profile.ipynb) (V1) | [gey0yop4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/gey0yop4) Finished | Đo tài nguyên d8000 trên subset Kaggle T4 (cửa sổ profile 34,07s; peak RSS ~3,20 GB) trước benchmark dài. |
| `BP5-KAGGLE-01` | [`notebooks/04_resume_benchmark_profile.ipynb`](notebooks/04_resume_benchmark_profile.ipynb) (V3) / [`notebooks/05_profile_base_ready.ipynb`](notebooks/05_profile_base_ready.ipynb) | [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) Finished | Xác nhận 5 epochs BP thật (76,11s), checkpoint v2 và CUDA resume proof đạt (53/54 tests, 1 snapshot skip). |
| `R1 / seed 42` | [`notebooks/06_benchmark_seed42_ready.ipynb`](notebooks/06_benchmark_seed42_ready.ipynb) | [zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) Finished | Benchmark chính seed 42 (160 epochs, d8000, K5): Ā `65,1646%`, final `55,87%`, F_base `8,32 pp`. |
| `R1 / seed 43` | [`notebooks/07_benchmark_seed43_ready.ipynb`](notebooks/07_benchmark_seed43_ready.ipynb) | [v51m2dan](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan) Finished | Benchmark chính seed 43 (160 epochs, d8000, K5): Ā `65,7062%`, final `56,77%`, F_base `8,92 pp`. |
| `R1 / seed 44` | [`notebooks/08_benchmark_seed44_ready.ipynb`](notebooks/08_benchmark_seed44_ready.ipynb) | [vqcs1ql4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4) Finished | Benchmark chính seed 44 (160 epochs, d8000, K5): Ā `64,4880%`, final `55,75%`, F_base `7,68 pp`. |
| `R2 / seed 42 (3 ca mới)` | [`notebooks/09_r2_ablation_seed42.ipynb`](notebooks/09_r2_ablation_seed42.ipynb) | [xdrm0d27](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xdrm0d27) · [qcgdklxh](https://wandb.ai/hsangg-2006/acil-reproduction/runs/qcgdklxh) · [3yp99ewh](https://wandb.ai/hsangg-2006/acil-reproduction/runs/3yp99ewh) | Ablation seed 42: No-FE/γ=0.1 (`53,3970%`), FE8000/γ=0.01 (`65,1606%`), FE8000/γ=0.001 (`65,1576%`). |
| `R2 / seed 43 (3 ca mới)` | [`notebooks/10_r2_ablation_seed43.ipynb`](notebooks/10_r2_ablation_seed43.ipynb) | [7wgme4so](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7wgme4so) · [xfleip26](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xfleip26) · [aler6nj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/aler6nj2) | Ablation seed 43: No-FE/γ=0.1 (`53,4313%`), FE8000/γ=0.01 (`65,7034%`), FE8000/γ=0.001 (`65,7034%`). |
| `R2 / seed 44 (3 ca mới)` | [`notebooks/11_r2_ablation_seed44.ipynb`](notebooks/11_r2_ablation_seed44.ipynb) | [81aci8h1](https://wandb.ai/hsangg-2006/acil-reproduction/runs/81aci8h1) · [dub202jk](https://wandb.ai/hsangg-2006/acil-reproduction/runs/dub202jk) · [l9c50ezb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/l9c50ezb) | Ablation seed 44: No-FE/γ=0.1 (`52,9357%`), FE8000/γ=0.01 (`64,4880%`), FE8000/γ=0.001 (`64,4880%`). |
| `R3 / batch 32 (3 repeats)` | [`notebooks/12_r3_batch_analysis.ipynb`](notebooks/12_r3_batch_analysis.ipynb) / [`scripts/r3_runner.py`](scripts/r3_runner.py) | [luf7uv40](https://wandb.ai/hsangg-2006/acil-reproduction/runs/luf7uv40) · [yqxph1bb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/yqxph1bb) · [59vlfxzv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/59vlfxzv) | Batch 32 (d1024, seed 42): tổng analytic `17,3334 ± 0,0784 s`, peak RSS `151,35 MiB`, 18/18 mốc khớp joint. |
| `R3 / batch 128 (3 repeats)` | [`notebooks/12_r3_batch_analysis.ipynb`](notebooks/12_r3_batch_analysis.ipynb) / [`scripts/r3_runner.py`](scripts/r3_runner.py) | [o54hbdfp](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o54hbdfp) · [4481bnn1](https://wandb.ai/hsangg-2006/acil-reproduction/runs/4481bnn1) · [3gv0moj7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/3gv0moj7) | Batch 128 (d1024, seed 42): tổng analytic `6,9247 ± 0,1098 s`, peak RSS `146,10 MiB` (thấp nhất), 18/18 mốc khớp joint. |
| `R3 / batch 512 (3 repeats)` | [`notebooks/12_r3_batch_analysis.ipynb`](notebooks/12_r3_batch_analysis.ipynb) / [`scripts/r3_runner.py`](scripts/r3_runner.py) | [lnk0grbq](https://wandb.ai/hsangg-2006/acil-reproduction/runs/lnk0grbq) · [5ehmjvh3](https://wandb.ai/hsangg-2006/acil-reproduction/runs/5ehmjvh3) · [xesh2kjl](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xesh2kjl) | Batch 512 (d1024, seed 42): tổng analytic `4,9975 ± 0,1123 s` (nhanh nhất), peak RSS `164,55 MiB`, 18/18 mốc khớp joint. |
| `Author C1 controlled` | [`scripts/compare_author_acil.py`](scripts/compare_author_acil.py) | [0tlcy6id](https://wandb.ai/hsangg-2006/acil-reproduction/runs/0tlcy6id) Finished (`equivalence=false`) | Đối chiếu code tác giả cùng FE/order R3: prediction khớp 6/6, 2 phần tử score ở phase 3–4 vượt `allclose`. |
| `Author C2 native FE/main` | [`scripts/compare_author_acil.py`](scripts/compare_author_acil.py) | [8muywodv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/8muywodv) Finished | Chạy `main.main` tác giả với native FE/shuffle trên cache cố định: R/W/score/prediction khớp 6/6. |
| `Author C1R posthoc` | [`scripts/diagnose_author_base_precision.py`](scripts/diagnose_author_base_precision.py) | [yeb0jqfb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/yeb0jqfb) Finished | Nạp base state nhóm rồi cập nhật tác giả 5 phase: khớp 5/5 (max score error `6,65e-13`), định vị drift về bước base. |
| `E2a + FD2 (3 seeds)` | [`scripts/e2_memory_precision_runner.py`](scripts/e2_memory_precision_runner.py) | [givauiwx](https://wandb.ai/hsangg-2006/acil-reproduction/runs/givauiwx) · [e3z5fld4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/e3z5fld4) · [x9uxqlxu](https://wandb.ai/hsangg-2006/acil-reproduction/runs/x9uxqlxu) | Quét 11 mức `d` + `FD2`: trần iso-byte `6.144 MB` (`d=798` fp64 / `d=1160` fp32) làm `A_final` giảm `-5,19 pp` / `-3,95 pp` so với `d=8000` (`522,5 MB`). |
| `E2b Woodbury fp32/fp64 (3 seeds)` | [`scripts/e2_memory_precision_runner.py`](scripts/e2_memory_precision_runner.py) | [bf8zhv34](https://wandb.ai/hsangg-2006/acil-reproduction/runs/bf8zhv34) · [luzl13ud](https://wandb.ai/hsangg-2006/acil-reproduction/runs/luzl13ud) · [dwmcy1ob](https://wandb.ai/hsangg-2006/acil-reproduction/runs/dwmcy1ob) | Tại `d=1024`, `float64` giữ `max_rel_W ~ 8e-11` (`0` lệch nhãn); `float32` trôi lên `1,5e-3..4,0e-3` và phát sinh `3..9` mẫu đổi `argmax`. |
| `E1a / seed 42 (c0=10,20,30)` | [`notebooks/13_e1a_base_ratio_seed42.ipynb`](notebooks/13_e1a_base_ratio_seed42.ipynb) | [c1c00gop](https://wandb.ai/hsangg-2006/acil-reproduction/runs/c1c00gop) · [1sxua0j4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/1sxua0j4) · [teevqsnt](https://wandb.ai/hsangg-2006/acil-reproduction/runs/teevqsnt) | Thu hẹp lớp Base seed 42 (160ep, d8000): `A_final` 100 lớp giảm `55,87% (R1) → 48,76% (c0=30) → 43,16% (c0=20) → 33,32% (c0=10)`. |
| `E1a / seed 43 (c0=10,20,30)` | [`notebooks/14_e1a_base_ratio_seed43.ipynb`](notebooks/14_e1a_base_ratio_seed43.ipynb) | [4mwubtt2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/4mwubtt2) · [t65feqkw](https://wandb.ai/hsangg-2006/acil-reproduction/runs/t65feqkw) · [ysgblws2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/ysgblws2) | Thu hẹp lớp Base seed 43 (160ep, d8000): `A_final` 100 lớp giảm `56,77% (R1) → 48,08% (c0=30) → 40,59% (c0=20) → 33,20% (c0=10)`. |
| `E1a / seed 44 (c0=10,20,30)` | [`notebooks/15_e1a_base_ratio_seed44.ipynb`](notebooks/15_e1a_base_ratio_seed44.ipynb) | [pbct2s6d](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pbct2s6d) · [q3t7kvyb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/q3t7kvyb) · [lpps6veq](https://wandb.ai/hsangg-2006/acil-reproduction/runs/lpps6veq) | Thu hẹp lớp Base seed 44 (160ep, d8000): `A_final` 100 lớp giảm `55,75% (R1) → 47,58% (c0=30) → 41,62% (c0=20) → 34,18% (c0=10)`. |
| `E1b superclass_disjoint (3 seeds)` | [`notebooks/16_e1b_superclass_shift_seed42.ipynb`](notebooks/16_e1b_superclass_shift_seed42.ipynb) · [`17`](notebooks/17_e1b_superclass_shift_seed43.ipynb) · [`18`](notebooks/18_e1b_superclass_shift_seed44.ipynb) | [db4wwmvz](https://wandb.ai/hsangg-2006/acil-reproduction/runs/db4wwmvz) · [2cy9asyu](https://wandb.ai/hsangg-2006/acil-reproduction/runs/2cy9asyu) · [e9wono5t](https://wandb.ai/hsangg-2006/acil-reproduction/runs/e9wono5t) | Lệch miền ngữ nghĩa tại `c0=50` (Sinh vật/Tự nhiên → Đồ vật/Phương tiện): `A_inc` sụt `-8,21 pp` (`35,71 ± 0,40%`), `Δ_gap` tăng lên `34,26 ± 0,62 pp`. |
| `E3 + FD3 MSE Imbalance (3 seeds, P3)` | [`scripts/e3_mse_imbalance_runner.py`](scripts/e3_mse_imbalance_runner.py) | [18y2c7pg](https://wandb.ai/hsangg-2006/acil-reproduction/runs/18y2c7pg) · [7uqw193s](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7uqw193s) · [rfu8q4pv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/rfu8q4pv) | Tại `d=1024` `float64` (khớp Joint `≤ 7,7e-12`), mất cân bằng lớp `10×/50×` làm `A_inc` sụp `32,62% → 0,21% → 0,00%`; `FD3-WRLS` khôi phục `+28,66 pp` ở `10×`. |

> **Đối chiếu đủ `56 / 56` runs trên W&B [`hsangg-2006/acil-reproduction`](https://wandb.ai/hsangg-2006/acil-reproduction):** Ngoài **51 runs chính thức** được ánh xạ trong bảng trên, project công khai còn lưu giữ **5 runs chạy thử sơ khởi ngày 02–03/11/2026** (đã gắn nhãn `draft` và `superseded` trên W&B, không dùng làm số liệu báo cáo): 4 lượt kiểm thử tổng hợp trước `VERIFY-01` ([`z8yhkxii`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/z8yhkxii), [`b4shifg6`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/b4shifg6), [`eyg1alwi`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/eyg1alwi), [`d7bmnw50`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/d7bmnw50)) và 1 lượt smoke CIFAR-100 thử nghiệm trước `CIFAR-KAGGLE-02` ([`i6ryerr4`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/i6ryerr4)).

### Chi tiết các lượt đối chiếu và provenance lịch sử

| ID              | Source snapshot                                                                                    | Kết quả                                                                 | Kaggle version                                                                                                           | W&B                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| VERIFY-01       | [23af76f](https://github.com/24122050S/20-ML-Paper/commit/23af76f9615a0b7e81ff5d8cc373acab45ec4398) | [JSON đã đối chiếu log](reports/verify_analytic_update_results.json)  | [Version 8](https://www.kaggle.com/code/hsangg206/01-verify-analytic-update6bf7e5ced7/notebook?scriptVersionId=364773332) | [7aeaupq7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7aeaupq7) |
| VERIFY-02       | [217684f](https://github.com/24122050S/20-ML-Paper/commit/217684f02566c9d0a34cea0071a26dbfda7b5582) | [JSON nguyên bản Kaggle](reports/verify_analytic_update_v2_results.json) | [Version 9](https://www.kaggle.com/code/hsangg206/01-verify-analytic-update6bf7e5ced7/output?scriptVersionId=364788888)   | [wi4wau2d](https://wandb.ai/hsangg-2006/acil-reproduction/runs/wi4wau2d) |
| CIFAR-SMOKE-01  | Local working tree có source hashes; chưa commit tại thời điểm chạy                         | [JSON local đã kiểm tra](reports/cifar100_smoke_results.json)           | Run local CPU                                                                                                            | Tắt tracking, ID/URL null                                              |
| CIFAR-KAGGLE-01 | Source bundle; hashes trong JSON đã đối chiếu                                                 | [JSON Kaggle](reports/cifar100_kaggle_results.json)                        | [Version 2](https://www.kaggle.com/code/hsangg206/notebook7e7946d4ee?scriptVersionId=364942764)                           | [o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2) |
| CIFAR-KAGGLE-02 | [Bundle notebook 03 tại 9972be9](https://github.com/24122050S/20-ML-Paper/blob/9972be9db06b60ff1abbf0c7483325543cda8b1a/notebooks/03_cifar100_safe_attempts.ipynb) | [JSON 100 lớp](reports/cifar100_kaggle_full5_results.json) | [scriptVersionId 365084714](https://www.kaggle.com/code/hsangg206/03-cifar100-safe-attempts/output?scriptVersionId=365084714) | [blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) |
| BP5-KAGGLE-01 | [Source API Version 3](notebooks/archive/05_profile_base_api_v3.ipynb), core 16 hashes trong manifest | [JSON BP5 sau sync](reports/profile_base_kaggle_ready.json) | [Version 3 /365703432](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432) | [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) Finished |
| R1-SEED42-KAGGLE-01 | [Notebook 06 tại cb9c0f0](https://github.com/24122050S/20-ML-Paper/blob/cb9c0f03571768af41fde4f8835aac41ac3421e3/notebooks/06_benchmark_seed42_ready.ipynb), 16 source hashes đã kiểm | [JSON R1 seed 42](reports/benchmark_seed42_kaggle_results.json) | [Version 1 /365844618](https://www.kaggle.com/code/hsangg206/06-benchmark-seed42/output?scriptVersionId=365844618) | [zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) Finished |
| R1-SEED43-KAGGLE-01 | [Notebook 07](notebooks/07_benchmark_seed43_ready.ipynb), core 16 hashes khớp | [JSON seed 43](reports/benchmark_seed43_kaggle_results.json) | [Version 1 /366266785](https://www.kaggle.com/code/hsangg206/07-benchmark-seed43/output?scriptVersionId=366266785) | [v51m2dan](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan) Finished |
| R1-SEED44-KAGGLE-01 | [Notebook 08](notebooks/08_benchmark_seed44_ready.ipynb), core 16 hashes khớp | [JSON seed 44](reports/benchmark_seed44_kaggle_results.json) | [Version 1 /366266800](https://www.kaggle.com/code/hsangg206/08-benchmark-seed44/output?scriptVersionId=366266800) | [vqcs1ql4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4) Finished |

CIFAR-KAGGLE-01: Python 3.13.15, NumPy 2.1.3, PyTorch 2.11.0+cu128, CUDA, train mới 2 epochs, expansion 256, seed 42. Base 50 lớp đạt 32,58%; sau phase mới 60 lớp đạt 29,70% trên đủ 6.000 ảnh test. W là 256×50 rồi 256×60, dự đoán khớp joint, mọi check đạt. Đã đối chiếu output và JSON của Version 2 với run W&B Finished. JSON ghi `logged`; git_commit null do source được giải nén từ bundle. [Manifest](reports/CIFAR-KAGGLE-01-provenance.json) liên kết version, source bundle và checksum JSON. Kaggle hiển thị môi trường GPU T4 x2, runtime notebook 1m42s; notebook được ghi nhận Private lúc đối chiếu. Đây là smoke test.

CIFAR-KAGGLE-02: train mới 2 epochs, seed 42, d256, gamma 0.1 trên CUDA; Python 3.13.16, NumPy 2.1.3, PyTorch 2.11.0+cu130. Sáu mốc 50→60→70→80→90→100 lớp có accuracy 33,46%→29,9167%→27,50%→25,2250%→23,3111%→22,05%; dự đoán khớp joint ở mọi phase. W&B API công khai xác nhận finished/config/summary khớp; [manifest](reports/CIFAR-KAGGLE-02-provenance.json) giữ checksum JSON và source bundle (giao diện/số version hiển thị và loại GPU ghi nhận theo metadata thu thập lúc đối chiếu).

VERIFY-01: Python 3.13.15, NumPy 2.1.3, float64, seed 42, gamma 1, 3 phase, 2 lớp/phase, feature 24. Kết quả ở hàng này được đối chiếu từ output lưu của Version 8, sửa metadata URL cũ mà không đổi numeric rows. Sai lệch W tuyệt đối lớn nhất khoảng 1.84e-14; đây không phải sai lệch tương đối. [Version 5 gốc](reports/archive/VERIFY-01-version5.json) được giữ riêng.

VERIFY-02 giữ bài toán tổng hợp tương ứng, so sánh thêm cập nhật minibatch với joint sau từng phase; đánh giá W/R/score/argmax, lớp lặp, state không bị sửa và lỗi tracking. Các con số nhỏ có thể khác bitwise giữa môi trường; áp dụng ngưỡng đã ghi trong config.

VERIFY-02 Version 9 đã Save & Run All thành công trên Kaggle CPU ngày 02/11/2026: 11 tests đạt; W qua ba phase là `(24,2)`, `(24,4)`, `(24,6)`; sai lệch tương đối W lớn nhất của minibatch so với joint là khoảng `3.43e-13`, không có nhãn dự đoán khác nhau. W&B xác nhận Finished. Notebook có output được Kaggle push ở [eddd50d](https://github.com/24122050S/20-ML-Paper/commit/eddd50d20a3d7f11bf6b24efd7d15916a1581bd2); commit này khác source commit `217684f` mà notebook đã fetch để tính toán. [Manifest đối chiếu](reports/VERIFY-02-provenance.json) ghi cả hai và SHA-256 của JSON.

**Trạng thái quyền truy cập công khai (kiểm tra ẩn danh ngày 22/11/2026):**
- **GitHub repo** ([`24122050S/20-ML-Paper`](https://github.com/24122050S/20-ML-Paper)) và **W&B project** ([`hsangg-2006/acil-reproduction`](https://wandb.ai/hsangg-2006/acil-reproduction), đủ **56/56** runs `Finished` không cần xác thực: `51` runs chính thức + `5` runs thử nghiệm sơ khởi `02–03/11` đã gắn nhãn `draft`/`superseded`) đều **Public (`HTTP 200`)**.
- **Kaggle Notebooks:** toàn bộ **15/15** notebook thực nghiệm chính từ **`03` đến `18`** (`CIFAR-KAGGLE-02`, `PROFILE-KAGGLE-01`, `BP5`, `R1 06–08`, `R2 09–11`, `R3 12`, `E1a 13–15`, `E1b 16–18`) đã ở chế độ **Public (`is_private=false`, truy cập ẩn danh `HTTP 200`)**. Riêng 2 notebook sơ khởi lịch sử **`01-verify-analytic-update6bf7e5ced7`** (`VERIFY-01/02`) và **`02-cifar-100-smoke`** (trước đổi tên là `notebook7e7946d4ee`, `CIFAR-KAGGLE-01`) hiện vẫn ở chế độ **Private (`HTTP 404`)** trên Kaggle; toàn bộ mã nguồn `.ipynb` và JSON kết quả của 2 lượt này đã được lưu đầy đủ kèm manifest SHA-256 trên GitHub và W&B Public.

Mốc tổng hợp **07/11/2026 (lịch sử giữa kỳ)** & cập nhật **18/11/2026**: suite kiểm thử hiện tại đạt **54 tests trong `tests/`** (53 đạt, 1 skip CUDA trên CPU) và **24 tests trong `scripts/tests/`** (gồm R3, đối chiếu tác giả, `E1/E2` và `E3`). [Protocol benchmark v1.0](docs/protocol_chot.md) và [metrics v1.0](docs/metrics.md) đã chốt: CIFAR-100/K5, base 160 epochs, d8000, gamma 0.1; R1/R2/R3 theo core scope. Resume CPU đã so model/optimizer/scheduler/RNG/history chính xác; streaming đủ 100 lớp khớp accuracy oracle CPU ở cả 6 mốc. [Bằng chứng resume/streaming](reports/RESUME-PROFILE-01-provenance.json) ghi phạm vi từng phép kiểm. Analytic d8000 trên Kaggle T4 đã đạt qua [gey0yop4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/gey0yop4); BP5/CUDA đạt qua [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f). R1 đủ 42/43/44: Ā 65,1196 ± 0,6104% (n=3, ddof=1), [summary](reports/r1_cifar100_k5_summary.json) / [báo cáo tổng hợp](reports/R1_CIFAR100_K5_3SEEDS_20261106.md); R2/R3, [đối chiếu tác giả](reports/AUTHOR_CODE_COMPARISON_20261107.md), [Slide Deck `.pptx` giữa kỳ](slides/ACIL_Nhom3_GiuaKy.pptx), ba bộ thực nghiệm phản biện cuối kỳ (`E1`, `E2`, `E3`) và [Poster A2 cuối kỳ](slides/ACIL_Nhom3_Poster_A2.pptx) đều đã hoàn tất. Checksum JSON CPU đã khớp Git blob ([manifest CPU](reports/CIFAR-SMOKE-01-provenance.json)); số liệu lịch sử giữ nguyên. [Danh mục bằng chứng](reports/README.md).

Analytic của notebook 04 là Public Version 1 với source bundle khớp và W&B finished; cùng kernel này hiện có BP5 ở Version 3. Version-pinned output API đã đối chiếu raw bytes của analytic V1 /365647614 ([manifest analytic](reports/PROFILE-KAGGLE-01-provenance.json)) và BP5 V3 /365703432 ([manifest BP5](reports/BP5-KAGGLE-01-provenance.json)). BP5 cung cấp executed CUDA test proof trong JSON.

BP5: [Kaggle Version 3](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432), [JSON tổng hợp sau sync](reports/profile_base_kaggle_ready.json), [manifest/source archive](reports/BP5-KAGGLE-01-provenance.json), [W&B pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) Finished. Đồng bộ đúng payload offline, không train lại; API không đăng nhập xác nhận config, 16 source hashes, summary và CUDA proof khớp. [JSON gốc trước sync](reports/archive/BP5-KAGGLE-01-original-ready.json) giữ nguyên `not_logged/offline_pending_sync`; bản tổng hợp chỉ bổ sung metadata đồng bộ thực tế. Run gey0yop4 thuộc analytic Version 1.

## Kiểm resume và profile bằng CLI

```bash
python -m src.verify_resume --data-dir ../data
python -m src.profile_analytic --checkpoint reports/local/cifar100_smoke/base_checkpoint.pt --data-dir ../data --dimension 8000 --threads 4 --device cpu
```

Các lệnh trên kiểm subset CPU và tài nguyên; JSON giữ trạng thái tracking `not_logged`. Notebook 04 hỗ trợ logging trên Kaggle. D8000 vẫn dùng NumPy/SciPy CPU cho analytic; GPU dùng cho CNN.

## Quy ước phương pháp

- Nhãn đầu vào là ID toàn cục. Một lớp gặp lại ở batch sau vẫn dùng cột cũ.
- Cùng backbone/feature expansion và gamma trong một chuỗi cập nhật. Module hiện chỉ hỗ trợ gamma > 0; không gọi một gamma rất nhỏ là ablation gamma=0.
- Joint giữ lịch sử chỉ để tạo nghiệm tham chiếu trong bộ kiểm tra. Hàm incremental không nhận dữ liệu train cũ.
- Accuracy tổng hợp không dùng so với accuracy benchmark của paper; tương đương joint analytic không có nghĩa tương đương joint CNN training hoặc accuracy lớp cũ luôn bất biến.

## Nguồn và hỗ trợ

**Bài báo tái lập**

- H. Zhuang, Z. Weng, H. Wei, R. Xie, K.-A. Toh và Z. Lin, [ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract-Conference.html), NeurIPS 2022. Analytic head trong `src/analytic.py` tự cài từ mục 3.1–3.2, phương trình (4), (9), (10).

**Thành phần dùng lại, không phải đóng góp của nhóm**

| Thành phần | Nguồn | Cách dùng trong repo |
|---|---|---|
| Kiến trúc ResNet-32, shortcut option A | K. He, X. Zhang, S. Ren và J. Sun, "Deep Residual Learning for Image Recognition", CVPR 2016, mục 4.2 | `src/resnet.py` viết lại theo mô tả kiến trúc; không chép code có sẵn |
| Dataset CIFAR-100 | A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", Tech. Rep., University of Toronto, 2009 | Dữ liệu cho mọi thí nghiệm CIFAR |
| Bản phân phối CIFAR-100 | Hugging Face [`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100), revision `aadb3af77e9048adbea6b47c21a81e47dd092ae5` | `src/cifar_data.py` tải tự động và ghi SHA-256 từng file vào JSON kết quả |
| Protocol CIL: 50 lớp base + K phase | A. Douillard và cộng sự, PODNet, ECCV 2020; Y. Liu, B. Schiele và Q. Sun, AANets, CVPR 2021; theo ACIL mục 4.1 | Định nghĩa chia lớp trong `docs/protocol_chot.md`; sai khác về thứ tự lớp ghi ở mục 6 |
| Code tác giả ACIL | [ZHUANGHP/Analytic-continual-learning](https://github.com/ZHUANGHP/Analytic-continual-learning/tree/0afd6bf87622e808ce7f730f8556e78229b9e758), commit `0afd6bf` | Chỉ chạy trong môi trường riêng để đối chiếu; không chép vào `src/` |

**Thư viện và nền tảng**

- Tính toán chuẩn (`standard stack`): NumPy 2.1.3, SciPy 1.17.0, PyTorch 2.11.0 (và scikit-learn 1.5.2 trong môi trường đối chiếu code tác giả). Phiên bản khóa trong các file `requirements-*.txt`.
- Nền tảng: Weights & Biases (`wandb` 0.28.1), Kaggle Notebooks (GPU T4), Hugging Face Datasets và GitHub.

**Giải thích các thư viện ngoài standard stack (`numpy`, `scipy`, `scikit-learn`, `torch`)**

| Thư viện | File cấu hình | Lý do sử dụng ngoài standard scientific stack |
|---|---|---|
| `wandb` (0.28.1) | `requirements-tracking.txt` | Bắt buộc theo quy định môn học để theo dõi công khai toàn bộ run, cấu hình, metric và đồ thị. |
| `huggingface_hub` (1.10.1), `pyarrow` (23.0.1), `Pillow` (11.1.0) | `requirements-cifar.txt` | Phục vụ tải tự động CIFAR-100 từ Hugging Face (`uoft-cs/cifar100`), đọc file Parquet (bao gồm cột `coarse_label` trong `E1b`) theo revision pin và giải mã byte ảnh PNG sang mảng RGB uint8. |
| `filelock` (4.2.6) | `requirements-cifar.txt` | Khóa tệp an toàn khi cập nhật `latest.json`/`results.json` ở thư mục root giữa các attempt, tránh race condition. |
| `psutil` (7.2.2) | `requirements-cifar.txt`, `requirements-author-comparison.txt` | Lấy mẫu bộ nhớ RSS của tiến trình CPU trong [`MemoryMonitor`](src/profile_analytic.py) cho profile d8000, thí nghiệm `R3` và cửa sổ đo bộ nhớ `E2a`/`FD2`. |
| `threadpoolctl` (3.6.0) | `requirements-cifar.txt`, `requirements-author-comparison.txt` | Giới hạn nhất quán số luồng BLAS/OpenMP khi trích xuất đặc trưng và đo thời gian analytic `R3`, `E1`, `E2`, `E3` trên CPU. |
| `matplotlib` (3.9.2) | `requirements-plots.txt` | Chỉ dùng để tự vẽ các biểu đồ kết quả `R1/R2/R3`, đối chiếu tác giả, 2 sơ đồ phương pháp/bài toán và 3 biểu đồ phản biện cuối kỳ (`E1/E2/E3`) trong `reports/figures/`, tách khỏi môi trường train. |
| `nbformat` (5.10.4), `nbclient` (0.11.0), `ipykernel` (7.2.0) | `requirements-dev.txt` | Chỉ dùng cục bộ để đóng gói, xác thực schema notebook (`tests/test_notebook_distribution.py`, `scripts/build_e1_notebooks.py`) và chạy kiểm thử notebook. |
| `tqdm` (4.67.1), `PyYAML` (6.0.2) | `requirements-author-comparison.txt` | Thư viện tiện ích tiến trình và đọc config YAML cho repo đối chiếu của tác giả (`ZHUANGHP/Analytic-continual-learning`). |
| `torchvision` (0.20.1+cpu) | Hướng dẫn ở dòng đầu `requirements-author-comparison.txt` (pin tại [`reports/author_comparison_environment_freeze.txt`](reports/author_comparison_environment_freeze.txt)) | Cài riêng từ PyTorch CPU index trong `.venv-author` để chạy `main.py`/`DatasetWrapper` gốc của tác giả mà không ảnh hưởng môi trường `src/`. |

[Đặc tả analytic](docs/Dac_ta_ACIL.md) · [Hướng dẫn tự chạy CIFAR smoke](docs/HUONG_DAN_CIFAR_SMOKE.md) · [Hướng dẫn resume & profile](docs/RESUME_BENCHMARK_PROFILE.md) · [Hướng dẫn Kaggle](docs/HUONG_DAN_KAGGLE.md)

## Tổng hợp R1

Đã đủ 3 seed độc lập 42/43/44. Xem [bảng theo seed/phase và hình](reports/R1_CIFAR100_K5_3SEEDS_20261106.md). Tái tạo bằng `python scripts/aggregate_r1.py`; vẽ bằng môi trường riêng với `requirements-plots.txt` và `python scripts/plot_r1.py`. Các script này không gửi GPU job. Summary chỉ complete khi đủ 3 input/provenance hợp lệ; SD dùng ddof=1. R2/R3 đã hoàn tất theo [protocol](docs/protocol_chot.md); các báo cáo riêng ghi đúng phạm vi từng bộ.

## R2: bỏ expansion và thay gamma — hoàn tất 07/11/2026

| Cấu hình | Ā: mean ± sample SD (%) |
|---|---:|
| Bỏ expansion, gamma 0.1 (H64) | 53,2547 ± 0,2768 |
| FE8000, gamma 0.1 (baseline R1) | 65,1196 ± 0,6104 |
| FE8000, gamma 0.01 | 65,1173 ± 0,6089 |
| FE8000, gamma 0.001 | 65,1163 ± 0,6088 |

n=3 seeds 42/43/44, sample SD với ddof=1. [Báo cáo và hình](reports/R2_CIFAR100_K5_20261107.md) giải thích kết quả, sai lệch với paper và paired differences. Baseline FE8000/gamma0.1 dùng 3 run R1 ở bảng trên; 9 dòng sau là 9 run analytic mới.

| Experiment | Source snapshot | Cấu hình | Kết quả | Kaggle version | W&B |
|---|---|---|---|---|---|
| R2/42/no_fe_g0p1 | [Notebook 09 @ c30ec15](https://github.com/24122050S/20-ML-Paper/blob/c30ec15a7cdd1ac49ad87bbd3cff2bba6d835e5a/notebooks/09_r2_ablation_seed42.ipynb) | Bỏ expansion, gamma 0.1 (H64) | [Raw JSON](reports/r2_seed42_kaggle_results.json) | [v1 /366341660](https://www.kaggle.com/code/hsangg206/09-r2-ablation-seed42/output?scriptVersionId=366341660) | [xdrm0d27](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xdrm0d27) Finished |
| R2/42/fe8000_g0p01 | [Notebook 09 @ c30ec15](https://github.com/24122050S/20-ML-Paper/blob/c30ec15a7cdd1ac49ad87bbd3cff2bba6d835e5a/notebooks/09_r2_ablation_seed42.ipynb) | FE8000, gamma 0.01 | [Raw JSON](reports/r2_seed42_kaggle_results.json) | [v1 /366341660](https://www.kaggle.com/code/hsangg206/09-r2-ablation-seed42/output?scriptVersionId=366341660) | [qcgdklxh](https://wandb.ai/hsangg-2006/acil-reproduction/runs/qcgdklxh) Finished |
| R2/42/fe8000_g0p001 | [Notebook 09 @ c30ec15](https://github.com/24122050S/20-ML-Paper/blob/c30ec15a7cdd1ac49ad87bbd3cff2bba6d835e5a/notebooks/09_r2_ablation_seed42.ipynb) | FE8000, gamma 0.001 | [Raw JSON](reports/r2_seed42_kaggle_results.json) | [v1 /366341660](https://www.kaggle.com/code/hsangg206/09-r2-ablation-seed42/output?scriptVersionId=366341660) | [3yp99ewh](https://wandb.ai/hsangg-2006/acil-reproduction/runs/3yp99ewh) Finished |
| R2/43/no_fe_g0p1 | [Notebook 10 @ c30ec15](https://github.com/24122050S/20-ML-Paper/blob/c30ec15a7cdd1ac49ad87bbd3cff2bba6d835e5a/notebooks/10_r2_ablation_seed43.ipynb) | Bỏ expansion, gamma 0.1 (H64) | [Raw JSON](reports/r2_seed43_kaggle_results.json) | [v1 /366341726](https://www.kaggle.com/code/hsangg206/10-r2-ablation-seed43/output?scriptVersionId=366341726) | [7wgme4so](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7wgme4so) Finished |
| R2/43/fe8000_g0p01 | [Notebook 10 @ c30ec15](https://github.com/24122050S/20-ML-Paper/blob/c30ec15a7cdd1ac49ad87bbd3cff2bba6d835e5a/notebooks/10_r2_ablation_seed43.ipynb) | FE8000, gamma 0.01 | [Raw JSON](reports/r2_seed43_kaggle_results.json) | [v1 /366341726](https://www.kaggle.com/code/hsangg206/10-r2-ablation-seed43/output?scriptVersionId=366341726) | [xfleip26](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xfleip26) Finished |
| R2/43/fe8000_g0p001 | [Notebook 10 @ c30ec15](https://github.com/24122050S/20-ML-Paper/blob/c30ec15a7cdd1ac49ad87bbd3cff2bba6d835e5a/notebooks/10_r2_ablation_seed43.ipynb) | FE8000, gamma 0.001 | [Raw JSON](reports/r2_seed43_kaggle_results.json) | [v1 /366341726](https://www.kaggle.com/code/hsangg206/10-r2-ablation-seed43/output?scriptVersionId=366341726) | [aler6nj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/aler6nj2) Finished |
| R2/44/no_fe_g0p1 | [Notebook 11 @ 16a0fa0](https://github.com/24122050S/20-ML-Paper/blob/16a0fa0fff53c33fda34dab7a9fe7f6915587466/notebooks/11_r2_ablation_seed44.ipynb) | Bỏ expansion, gamma 0.1 (H64) | [Raw JSON](reports/r2_seed44_kaggle_results.json) | [v1 /366348428](https://www.kaggle.com/code/hsangg206/11-r2-ablation-seed44/output?scriptVersionId=366348428) | [81aci8h1](https://wandb.ai/hsangg-2006/acil-reproduction/runs/81aci8h1) Finished |
| R2/44/fe8000_g0p01 | [Notebook 11 @ 16a0fa0](https://github.com/24122050S/20-ML-Paper/blob/16a0fa0fff53c33fda34dab7a9fe7f6915587466/notebooks/11_r2_ablation_seed44.ipynb) | FE8000, gamma 0.01 | [Raw JSON](reports/r2_seed44_kaggle_results.json) | [v1 /366348428](https://www.kaggle.com/code/hsangg206/11-r2-ablation-seed44/output?scriptVersionId=366348428) | [dub202jk](https://wandb.ai/hsangg-2006/acil-reproduction/runs/dub202jk) Finished |
| R2/44/fe8000_g0p001 | [Notebook 11 @ 16a0fa0](https://github.com/24122050S/20-ML-Paper/blob/16a0fa0fff53c33fda34dab7a9fe7f6915587466/notebooks/11_r2_ablation_seed44.ipynb) | FE8000, gamma 0.001 | [Raw JSON](reports/r2_seed44_kaggle_results.json) | [v1 /366348428](https://www.kaggle.com/code/hsangg206/11-r2-ablation-seed44/output?scriptVersionId=366348428) | [l9c50ezb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/l9c50ezb) Finished |

Các cấu hình dùng đúng checkpoint 160 cùng seed, cùng class order/normalization/HF revision; hai gamma mới giữ FE hash giống R1. Không train lại backbone. No-FE dùng trực tiếp F64. JSON numeric giữ nguyên trạng thái lúc capture offline; ready/manifest bổ sung actual run ID/URL sau khi sync và kiểm công khai. NPZ mới được load/kiểm đầy đủ trên Kaggle CPU, lưu tại Output đúng version; local chỉ tải JSON/proof/log, không tải NPZ lớn.

Tái tạo bảng: `python scripts/aggregate_r2.py`. Tái tạo hình: cài `requirements-plots.txt` trong môi trường riêng rồi `python scripts/plot_r2.py`. Để chạy lại thực nghiệm, import notebook 09/10/11, bật Internet/GPU và gắn đúng R1 Input version 1 theo cell cấu hình; `CHECKPOINT=None` tìm checkpoint theo SHA đã pin và sẽ lỗi nếu không có, không tự train. W&B tạo offline trước, cần đồng bộ bằng `wandb sync` trên payload đúng lượt và xác minh xong mới công bố tracking. Giới hạn hai phiên batch GPU của tài khoản tại lúc chạy khiến seed 44 được gửi sau 42/43. R2 tái lập 4/5 hàng Table 2; gamma 0 ngoài scope v1.0.

## R3: phân tích batch size — hoàn tất 07/11/2026

| Batch | Analytic mean ± SD (s) | Peak RSS mean ± SD (MiB) | Argmax khác joint |
|---:|---:|---:|---:|
| 32 | 17,3334 ± 0,0784 | 151,3503 ± 0,9749 | 0 |
| 128 | 6,9247 ± 0,1098 | 146,1029 ± 0,9158 | 0 |
| 512 | 4,9975 ± 0,1123 | 164,5521 ± 4,3843 | 0 |

n=3 timing repeats trên **một seed 42**, sample SD với ddof=1. Thời gian chỉ gồm 6 cửa sổ học analytic (có FE); CNN/cache/evaluation/oracle/IO/log nằm ngoài. RSS là toàn process CPU lấy mẫu 5ms, gồm runtime/input/allocator; R/W/FE luôn 9.28125 MiB. [Báo cáo/hình](reports/R3_CIFAR100_K5_BATCH_20261107.md) có bảng base/incremental, sai số, 6 mốc accuracy, giới hạn phép đo và nguồn run. Cả 54 so sánh R/W/score đạt `rtol=1e-6`/`atol=1e-8`, không có argmax khác joint.

| Experiment | Source snapshot | Cấu hình | Kết quả | Kaggle version | W&B |
|---|---|---|---|---|---|
| R3/b32_r1 | [Notebook 12 @ eaf2566](https://github.com/24122050S/20-ML-Paper/blob/eaf256660fa896ca16ebe3a908868244f07f07d6/notebooks/12_r3_batch_analysis.ipynb) | batch 32, repeat1, seed 42/d1024 | [Raw JSON](reports/r3_cifar100_k5_results.json) | [v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048) | [luf7uv40](https://wandb.ai/hsangg-2006/acil-reproduction/runs/luf7uv40) Finished |
| R3/b128_r1 | [Notebook 12 @ eaf2566](https://github.com/24122050S/20-ML-Paper/blob/eaf256660fa896ca16ebe3a908868244f07f07d6/notebooks/12_r3_batch_analysis.ipynb) | batch 128, repeat1, seed 42/d1024 | [Raw JSON](reports/r3_cifar100_k5_results.json) | [v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048) | [o54hbdfp](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o54hbdfp) Finished |
| R3/b512_r1 | [Notebook 12 @ eaf2566](https://github.com/24122050S/20-ML-Paper/blob/eaf256660fa896ca16ebe3a908868244f07f07d6/notebooks/12_r3_batch_analysis.ipynb) | batch 512, repeat1, seed 42/d1024 | [Raw JSON](reports/r3_cifar100_k5_results.json) | [v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048) | [lnk0grbq](https://wandb.ai/hsangg-2006/acil-reproduction/runs/lnk0grbq) Finished |
| R3/b128_r2 | [Notebook 12 @ eaf2566](https://github.com/24122050S/20-ML-Paper/blob/eaf256660fa896ca16ebe3a908868244f07f07d6/notebooks/12_r3_batch_analysis.ipynb) | batch 128, repeat2, seed 42/d1024 | [Raw JSON](reports/r3_cifar100_k5_results.json) | [v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048) | [4481bnn1](https://wandb.ai/hsangg-2006/acil-reproduction/runs/4481bnn1) Finished |
| R3/b512_r2 | [Notebook 12 @ eaf2566](https://github.com/24122050S/20-ML-Paper/blob/eaf256660fa896ca16ebe3a908868244f07f07d6/notebooks/12_r3_batch_analysis.ipynb) | batch 512, repeat2, seed 42/d1024 | [Raw JSON](reports/r3_cifar100_k5_results.json) | [v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048) | [5ehmjvh3](https://wandb.ai/hsangg-2006/acil-reproduction/runs/5ehmjvh3) Finished |
| R3/b32_r2 | [Notebook 12 @ eaf2566](https://github.com/24122050S/20-ML-Paper/blob/eaf256660fa896ca16ebe3a908868244f07f07d6/notebooks/12_r3_batch_analysis.ipynb) | batch 32, repeat2, seed 42/d1024 | [Raw JSON](reports/r3_cifar100_k5_results.json) | [v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048) | [yqxph1bb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/yqxph1bb) Finished |
| R3/b512_r3 | [Notebook 12 @ eaf2566](https://github.com/24122050S/20-ML-Paper/blob/eaf256660fa896ca16ebe3a908868244f07f07d6/notebooks/12_r3_batch_analysis.ipynb) | batch 512, repeat3, seed 42/d1024 | [Raw JSON](reports/r3_cifar100_k5_results.json) | [v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048) | [xesh2kjl](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xesh2kjl) Finished |
| R3/b32_r3 | [Notebook 12 @ eaf2566](https://github.com/24122050S/20-ML-Paper/blob/eaf256660fa896ca16ebe3a908868244f07f07d6/notebooks/12_r3_batch_analysis.ipynb) | batch 32, repeat3, seed 42/d1024 | [Raw JSON](reports/r3_cifar100_k5_results.json) | [v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048) | [59vlfxzv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/59vlfxzv) Finished |
| R3/b128_r3 | [Notebook 12 @ eaf2566](https://github.com/24122050S/20-ML-Paper/blob/eaf256660fa896ca16ebe3a908868244f07f07d6/notebooks/12_r3_batch_analysis.ipynb) | batch 128, repeat3, seed 42/d1024 | [Raw JSON](reports/r3_cifar100_k5_results.json) | [v1 /366444048](https://www.kaggle.com/code/hsangg206/12-r3-batch-analysis/output?scriptVersionId=366444048) | [3gv0moj7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/3gv0moj7) Finished |

[Kế hoạch chốt trước run](docs/R3_KE_HOACH_DO.md) · [Summary](reports/r3_cifar100_k5_summary.json) · [Manifest](reports/R3-C100-K5-provenance.json). Đã tải/kiểm đủ fixture và 60 NPZ ở local, giải lại joint từ full fixed features để so 54 trạng thái. Raw JSON giữ trạng thái capture offline; ready/manifest ghi actual tracking sau sync. Tái tạo bảng bằng `python scripts/aggregate_r3.py`, vẽ với môi trường riêng có `requirements-plots.txt` rồi `python scripts/plot_r3.py`. Để chạy lại, import notebook 12 và gắn Input R1 seed 42/version 1, bật GPU/Internet; cache F64 chỉ phục vụ nghiên cứu/oracle, không phải replay storage của learner. Các notebook/script R1/R2 và bằng chứng lịch sử được giữ nguyên.

Sau `wandb sync`, dùng `python scripts/reconcile_r3_tracking.py --suite <raw-r3_ready.json>` để chốt summary từ raw report và thêm alias `r3/learning_seconds_total`, `r3/max_abs_score_error_all_phases`, tránh nhầm giá trị phase cuối với tổng/max cả chuỗi. Helper chỉ nhận run đã Finished và ghi hash raw suite; các run công bố đã qua bước này và được kiểm API công khai. History/numeric JSON giữ nguyên.

**Cả hai giai đoạn giữa kỳ và cuối kỳ đều đã hoàn tất:** bộ thực nghiệm giữa kỳ (`R1/R2/R3`, đối chiếu code tác giả `C1/C2/C1R`), tài liệu đọc hiểu [`docs/READING_THE_PAPER_ACIL.md`](docs/READING_THE_PAPER_ACIL.md), kịch bản 36 slides [`docs/KICH_BAN_36_SLIDE_GIUA_KY.md`](docs/KICH_BAN_36_SLIDE_GIUA_KY.md), file [`slides/ACIL_Nhom3_GiuaKy.pptx`](slides/ACIL_Nhom3_GiuaKy.pptx), phần phân tích phản biện lý thuyết & thực nghiệm kiểm chứng cuối kỳ (`E1`, `E2`, `E3`) và **Poster A2 dọc** ([`slides/ACIL_Nhom3_Poster_A2.pptx`](slides/ACIL_Nhom3_Poster_A2.pptx)).

## Đối chiếu source tác giả — thực thi 07/11/2026

[Kế hoạch có phiên bản](docs/DOI_CHIEU_CODE_TAC_GIA_KE_HOACH.md) · [Cách chạy riêng môi trường/input](docs/HUONG_DAN_CHAY_CODE_TAC_GIA.md) · [Báo cáo phân tích](reports/AUTHOR_CODE_COMPARISON_20261107.md).

| Case | Script thực thi | Điều kiểm tra / kết quả | Bằng chứng | W&B |
|---|---|---|---|---|
| C1 controlled | [Runner @2d74c46](https://github.com/24122050S/20-ML-Paper/blob/2d74c46/scripts/compare_author_acil.py) | Author source cùng FE/order như R3; prediction giống 6/6, score fail 2 phase (2 elements), không nới tolerance | [Raw](reports/author_comparison_results.json) | [0tlcy6id](https://wandb.ai/hsangg-2006/acil-reproduction/runs/0tlcy6id) Finished; equivalence=false |
| C2 native FE/main | [Runner @2d74c46](https://github.com/24122050S/20-ML-Paper/blob/2d74c46/scripts/compare_author_acil.py) | Gọi main.main thật; cùng cache/checkpoint, native FE/shuffle; R/W/score/prediction đạt 6/6 | [Raw](reports/author_comparison_results.json), [IL.csv](reports/author_comparison_native_IL.csv) | [8muywodv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/8muywodv) Finished |
| C1R posthoc | [Diagnostic @5429ad9](https://github.com/24122050S/20-ML-Paper/blob/5429ad9/scripts/diagnose_author_base_precision.py) | Nạp base state nhóm rồi cập nhật tác giả 5 phase; đạt 5/5, không thay thế C1 cold-start | [Raw diagnostic](reports/author_comparison_base_diagnostic.json) | [yeb0jqfb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/yeb0jqfb) Finished |

Source tác giả pin `0afd6bf`, checkout riêng và không sửa core. Một seed 42/d1024, không có 3 seed SD; không dùng accuracy này thay R1 d8000. `status=failed` trong raw suite giữ gate C1 không đạt; 3 run Finished chỉ là tracking hoàn tất. Đã tái hiện lỗi nhãn khi bật class shuffle và định vị sai khác FP32 tại pooling; phần lớn score drift của cold-start được định vị vào state base qua diagnostic riêng. R1/R2/R3 và các JSON/manifests lịch sử giữ nguyên.

## Thực nghiệm phản biện cuối kỳ `E1` & `E2` (cho Poster A2) — hoàn tất 18/11/2026

[Khung phản biện lý thuyết](docs/PHAN_BIEN_CUOI_KY_ACIL.md) · [Kế hoạch khóa trước khi chạy](docs/KE_HOACH_EMPIRICS_CUOI_KY.md) · [Báo cáo `E1`](reports/E1_BASE_ASSUMPTION_20261118.md) · [Báo cáo `E2`](reports/E2_MEMORY_PRECISION_20261118.md) · [Manifest `E1`](reports/E1-FINAL-EMPIRICS-provenance.json) · [Manifest `E2`](reports/E2-FINAL-EMPIRICS-provenance.json).

- **`E1` (kiểm chứng hạn chế chính `P1` — Nghẽn biểu diễn từ tập lớp Base đóng băng, `n=3` seeds `42/43/44` trên 6 kernels Kaggle GPU T4 Public):**
  - `E1a` (*Base-Class Ratio Sweep*): giảm quy mô Phase 0 qua $c_0 \in \{50, 30, 20, 10\}$ làm $A_{\mathrm{final}}$ trên 100 lớp giảm đơn điệu **`56,13 ± 0,56%` → `48,14 ± 0,59%` → `41,79 ± 1,29%` → `33,57 ± 0,53%`** (**`-22,56 pp`**) dù nghiệm đệ quy ở mọi lượt đều khớp Joint Ridge (`≤ 3,6e-14`, `0` mẫu lệch).
  - `E1b` (*Semantic Superclass Shift* tại cố định $c_0 = 50$, `25.000` ảnh Base): tách 10 coarse superclasses Tự nhiên/Sinh vật ở Phase 0 khỏi 10 superclasses Phương tiện/Đồ vật & nhóm còn lại ở Phases `1..5` làm $A_{\mathrm{inc}}$ 50 lớp mới giảm **`-8,21 pp`** (`43,93% → 35,71 ± 0,40%`), $A_{\mathrm{final}}$ giảm **`-3,29 pp`** (`52,84 ± 0,17%`) và khoảng lệch chuyển giao $\Delta_{\text{gap}} = B_5 - A_{\mathrm{inc}}$ mở rộng từ `24,41 pp` lên **`34,26 ± 0,62 pp`**.

| Run `E1` | Notebook nguồn | Cấu hình | Raw JSON | Kaggle version (Public) | W&B (Finished) |
|---|---|---|---|---|---|
| `E1a/42/c0_10` | [`13_e1a_base_ratio_seed42.ipynb`](notebooks/13_e1a_base_ratio_seed42.ipynb) | `uniform`, `c0=10`, `K=9`, `d=8000` | [JSON](reports/e1a_c0_10_seed42_results.json) | [v1 /370801804](https://www.kaggle.com/code/hsangg206/13-e1a-base-ratio-seed42/output?scriptVersionId=370801804) | [c1c00gop](https://wandb.ai/hsangg-2006/acil-reproduction/runs/c1c00gop) |
| `E1a/42/c0_20` | [`13_e1a_base_ratio_seed42.ipynb`](notebooks/13_e1a_base_ratio_seed42.ipynb) | `uniform`, `c0=20`, `K=8`, `d=8000` | [JSON](reports/e1a_c0_20_seed42_results.json) | [v1 /370801804](https://www.kaggle.com/code/hsangg206/13-e1a-base-ratio-seed42/output?scriptVersionId=370801804) | [1sxua0j4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/1sxua0j4) |
| `E1a/42/c0_30` | [`13_e1a_base_ratio_seed42.ipynb`](notebooks/13_e1a_base_ratio_seed42.ipynb) | `uniform`, `c0=30`, `K=7`, `d=8000` | [JSON](reports/e1a_c0_30_seed42_results.json) | [v1 /370801804](https://www.kaggle.com/code/hsangg206/13-e1a-base-ratio-seed42/output?scriptVersionId=370801804) | [teevqsnt](https://wandb.ai/hsangg-2006/acil-reproduction/runs/teevqsnt) |
| `E1a/43/c0_10` | [`14_e1a_base_ratio_seed43.ipynb`](notebooks/14_e1a_base_ratio_seed43.ipynb) | `uniform`, `c0=10`, `K=9`, `d=8000` | [JSON](reports/e1a_c0_10_seed43_results.json) | [v1 /370801839](https://www.kaggle.com/code/hsangg206/14-e1a-base-ratio-seed43/output?scriptVersionId=370801839) | [4mwubtt2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/4mwubtt2) |
| `E1a/43/c0_20` | [`14_e1a_base_ratio_seed43.ipynb`](notebooks/14_e1a_base_ratio_seed43.ipynb) | `uniform`, `c0=20`, `K=8`, `d=8000` | [JSON](reports/e1a_c0_20_seed43_results.json) | [v1 /370801839](https://www.kaggle.com/code/hsangg206/14-e1a-base-ratio-seed43/output?scriptVersionId=370801839) | [t65feqkw](https://wandb.ai/hsangg-2006/acil-reproduction/runs/t65feqkw) |
| `E1a/43/c0_30` | [`14_e1a_base_ratio_seed43.ipynb`](notebooks/14_e1a_base_ratio_seed43.ipynb) | `uniform`, `c0=30`, `K=7`, `d=8000` | [JSON](reports/e1a_c0_30_seed43_results.json) | [v1 /370801839](https://www.kaggle.com/code/hsangg206/14-e1a-base-ratio-seed43/output?scriptVersionId=370801839) | [ysgblws2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/ysgblws2) |
| `E1a/44/c0_10` | [`15_e1a_base_ratio_seed44.ipynb`](notebooks/15_e1a_base_ratio_seed44.ipynb) | `uniform`, `c0=10`, `K=9`, `d=8000` | [JSON](reports/e1a_c0_10_seed44_results.json) | [v1 /370817341](https://www.kaggle.com/code/hsangg206/15-e1a-base-ratio-seed44/output?scriptVersionId=370817341) | [pbct2s6d](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pbct2s6d) |
| `E1a/44/c0_20` | [`15_e1a_base_ratio_seed44.ipynb`](notebooks/15_e1a_base_ratio_seed44.ipynb) | `uniform`, `c0=20`, `K=8`, `d=8000` | [JSON](reports/e1a_c0_20_seed44_results.json) | [v1 /370817341](https://www.kaggle.com/code/hsangg206/15-e1a-base-ratio-seed44/output?scriptVersionId=370817341) | [q3t7kvyb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/q3t7kvyb) |
| `E1a/44/c0_30` | [`15_e1a_base_ratio_seed44.ipynb`](notebooks/15_e1a_base_ratio_seed44.ipynb) | `uniform`, `c0=30`, `K=7`, `d=8000` | [JSON](reports/e1a_c0_30_seed44_results.json) | [v1 /370817341](https://www.kaggle.com/code/hsangg206/15-e1a-base-ratio-seed44/output?scriptVersionId=370817341) | [lpps6veq](https://wandb.ai/hsangg-2006/acil-reproduction/runs/lpps6veq) |
| `E1b/42/disjoint` | [`16_e1b_superclass_shift_seed42.ipynb`](notebooks/16_e1b_superclass_shift_seed42.ipynb) | `superclass_disjoint`, `c0=50`, `K=5` | [JSON](reports/e1b_superclass_disjoint_seed42_results.json) | [v1 /370817350](https://www.kaggle.com/code/hsangg206/16-e1b-superclass-shift-seed42/output?scriptVersionId=370817350) | [db4wwmvz](https://wandb.ai/hsangg-2006/acil-reproduction/runs/db4wwmvz) |
| `E1b/43/disjoint` | [`17_e1b_superclass_shift_seed43.ipynb`](notebooks/17_e1b_superclass_shift_seed43.ipynb) | `superclass_disjoint`, `c0=50`, `K=5` | [JSON](reports/e1b_superclass_disjoint_seed43_results.json) | [v1 /370827515](https://www.kaggle.com/code/hsangg206/17-e1b-superclass-shift-seed43/output?scriptVersionId=370827515) | [2cy9asyu](https://wandb.ai/hsangg-2006/acil-reproduction/runs/2cy9asyu) |
| `E1b/44/disjoint` | [`18_e1b_superclass_shift_seed44.ipynb`](notebooks/18_e1b_superclass_shift_seed44.ipynb) | `superclass_disjoint`, `c0=50`, `K=5` | [JSON](reports/e1b_superclass_disjoint_seed44_results.json) | [v1 /370833058](https://www.kaggle.com/code/hsangg206/18-e1b-superclass-shift-seed44/output?scriptVersionId=370833058) | [e9wono5t](https://wandb.ai/hsangg-2006/acil-reproduction/runs/e9wono5t) |

![Biểu đồ E1 — Nghẽn biểu diễn tập lớp Base đóng băng](reports/figures/e1_base_representation_bottleneck.png)

- **`E2` (kiểm chứng hạn chế chính `P2` — Ngân sách Byte $\mathcal{O}(d^2)$ & độ nhạy số học Woodbury, `n=3` seeds `42/43/44`):**
  - Tái sử dụng nguyên vẹn 3 checkpoints ResNet-32 160 epochs của `R1` (không huấn luyện lại CNN), đối chiếu ngân sách `6.144 MB` của kho 2.000 ảnh gốc CIFAR-100: cấu hình chuẩn bài báo `d = 8000` chiếm `522,50 MB` (`85,04×` kho mẫu, đạt $\bar{A} = 65,12 \pm 0,61\%$, $A_{\mathrm{final}} = 56,13 \pm 0,56\%$), trong khi ép về đúng trần ngân sách byte tại **`d = 798` (`float64`, `6,141 MB`)** làm $\bar{A}$ giảm còn **`62,17 ± 0,47%` (`-2,95 pp`)**, $A_{\mathrm{final}}$ còn **`50,94 ± 0,43%` (`-5,19 pp`)** và $A_{\mathrm{inc}}$ 50 lớp mới giảm **`-13,64 pp`** (`43,93% → 30,29%`).
  - Nguyên mẫu `FD2` (*Block-Diagonal RFAM*, `d=8000, B=16`, `42,50 MB`) đạt $\bar{A} = 61,70 \pm 0,52\%$, thấp hơn cấu hình ma trận dày đặc `d = 2048` (`36,24 MB`, $\bar{A} = 63,80 \pm 0,50\%$) do triệt tiêu các khối ngoài đường chéo làm mất khả năng khử cộng tuyến chéo giữa 8.000 đặc trưng ngẫu nhiên cùng sinh từ không gian 64 chiều.
  - Ở `E2b` (`d = 1024`), chuyển từ `float64` sang `float32` làm sai số tương đối trọng số Woodbury tăng hơn $4 \times 10^7\times$ (từ $\sim 8 \times 10^{-11}$ lên $1,5 \times 10^{-3}$ ở `200` bước và $4,0 \times 10^{-3}$ ở `1.575` bước `[K=25, b=16]`), phát sinh `3..9` mẫu đổi nhãn `argmax`.

| Run `E2` | Script nguồn | Phạm vi kiểm chứng | Raw JSON | W&B (Finished) |
|---|---|---|---|---|
| `E2a+FD2 / seed 42` | [`scripts/e2_memory_precision_runner.py`](scripts/e2_memory_precision_runner.py) | Quét 11 mức `d ∈ [64..8000]` + `FD2 (B=16, 80)` trên ckpt `R1/42` | [JSON](reports/e2_seed42_results.json) | [givauiwx](https://wandb.ai/hsangg-2006/acil-reproduction/runs/givauiwx) |
| `E2a+FD2 / seed 43` | [`scripts/e2_memory_precision_runner.py`](scripts/e2_memory_precision_runner.py) | Quét 11 mức `d ∈ [64..8000]` + `FD2 (B=16, 80)` trên ckpt `R1/43` | [JSON](reports/e2_seed43_results.json) | [e3z5fld4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/e3z5fld4) |
| `E2a+FD2 / seed 44` | [`scripts/e2_memory_precision_runner.py`](scripts/e2_memory_precision_runner.py) | Quét 11 mức `d ∈ [64..8000]` + `FD2 (B=16, 80)` trên ckpt `R1/44` | [JSON](reports/e2_seed44_results.json) | [x9uxqlxu](https://wandb.ai/hsangg-2006/acil-reproduction/runs/x9uxqlxu) |
| `E2b / seed 42` | [`scripts/e2_memory_precision_runner.py`](scripts/e2_memory_precision_runner.py) | Woodbury `fp32` vs `fp64` (`d=1024`, 4 mức `γ`, `200` & `1575` bước) | [JSON](reports/e2_seed42_results.json) | [bf8zhv34](https://wandb.ai/hsangg-2006/acil-reproduction/runs/bf8zhv34) |
| `E2b / seed 43` | [`scripts/e2_memory_precision_runner.py`](scripts/e2_memory_precision_runner.py) | Woodbury `fp32` vs `fp64` (`d=1024`, 4 mức `γ`, `200` & `1575` bước) | [JSON](reports/e2_seed43_results.json) | [luzl13ud](https://wandb.ai/hsangg-2006/acil-reproduction/runs/luzl13ud) |
| `E2b / seed 44` | [`scripts/e2_memory_precision_runner.py`](scripts/e2_memory_precision_runner.py) | Woodbury `fp32` vs `fp64` (`d=1024`, 4 mức `γ`, `200` & `1575` bước) | [JSON](reports/e2_seed44_results.json) | [dwmcy1ob](https://wandb.ai/hsangg-2006/acil-reproduction/runs/dwmcy1ob) |

![Biểu đồ E2 — Đánh đổi Ngân sách Byte và Độ trôi số học Woodbury](reports/figures/e2_memory_and_woodbury_precision.png)

Tái tạo biểu đồ Poster `E1/E2` từ JSON tổng hợp đã khóa SHA-256: `python scripts/plot_final_empirics.py --mode all`. Chạy bộ kiểm thử đơn vị của `E1/E2` bằng `python -m unittest discover -s scripts/tests -p "test_final_empirics.py" -v`.

---

## Hạn chế phụ thứ 3 (`P3` — Trình bày bổ sung trong README ngoài 2 hạn chế chính trên Poster A2)

Theo quy định môn học, **Poster A2 cuối kỳ** chỉ trình bày 2 hạn chế lý thuyết mạnh nhất (`P1` và `P2`). Hạn chế lý thuyết thứ 3 (**`P3` — Thiên lệch mất cân bằng lớp của hàm mục tiêu MSE One-Hot, tách bạch với độ trôi số học `float32` Woodbury**) được kiểm chứng độc lập trên 3 seeds (`42/43/44`) tại [`reports/E3_MSE_IMBALANCE_20261118.md`](reports/E3_MSE_IMBALANCE_20261118.md) ([Summary](reports/e3_mse_imbalance_summary.json) · [Manifest SHA-256](reports/E3-FINAL-EMPIRICS-provenance.json)) và trình bày tóm tắt kèm bằng chứng ngay tại đây.

### 1. Cơ chế lý thuyết & Đối chiếu tách bạch tại cùng chiều `d = 1024`

Trong ACIL, hàm mục tiêu hồi quy Frobenius trên nhãn one-hot tách độc lập theo từng cột lớp $c$:

$$
w_c = R_k \left( \sum_{i:\, y_i = c} h_i \right), \qquad R_k = \left( \sum_{j=1}^{N_{0:k}} h_j h_j^T + \gamma I_d \right)^{-1}
$$

Do tổng vế phải tỷ lệ thuận với số mẫu huấn luyện $N_c$ của lớp $c$ trong khi $R_k$ và phạt $\gamma \|w_c\|_2^2$ tác động như nhau lên mọi cột, thiết lập chuẩn CIFAR-100 ($N_c = 500$ đồng đều ở cả 100 lớp) che khuất hoàn toàn điểm yếu này. Khi triển khai trên luồng tăng dần mất cân bằng (*long-tailed / few-shot incremental classes*, $N_{\mathrm{inc}} \ll N_{\mathrm{base}}$), chuẩn cột $\|w_c\|_2$ của các lớp mới bị co nhỏ có hệ thống khiến điểm dự đoán $h^T w_c$ thua áp đảo trước 50 lớp Base trong phép `argmax`.

Tái sử dụng đúng 3 checkpoints `R1` (`seeds 42/43/44`) và cùng chiều mở rộng $d = 1024$ ($\gamma = 0.1$) để so sánh ngang bằng hai cơ chế:

| Cơ chế kiểm chứng (cùng `d = 1024`, 3 seeds) | Khớp Định lý 3.1 (`max_rel_W` vs Joint `fp64`) | Tổng mẫu lệch `argmax` (trên `30.000` mẫu) | Ảnh hưởng lên độ chính xác | Kết luận tách bạch |
|---|---:|---:|---|---|
| **`E2b`: `float32` Woodbury (`200` bước, `N_inc=500`)** | Không (`1,49e-03`) | `3` | $\Delta A_{\mathrm{final}} = 0,00\text{ pp}$ | Phá tính khớp đại số chính xác, buộc trả chi phí RAM `float64`, nhưng chưa phá accuracy ở `d=1024`. |
| **`E2b`: `float32` Woodbury (`1.575` bước, `γ=1e-3`)** | Không (`3,94e-03`) | `9` | $|\Delta A_{\mathrm{final}}| < 0,01\text{ pp}$ | Tích tụ sai số theo số bước cập nhật, tạo thêm mẫu đảo nhãn sát biên quyết định. |
| **`E3`: `float64` MSE One-Hot (`10×`, `N_inc=50`)** | **Khớp máy (`6,50e-12`)** | **`0`** | **$A_{\mathrm{inc}}$ sụt `-32,41 pp`** (`32,62% → 0,21%`) | Dù Định lý 3.1 đúng tuyệt đối, hàm mục tiêu MSE làm sụp đổ khả năng nhận diện lớp mới (`‖w_inc‖/‖w_base‖ = 0,283`). |
| **`FD3`: Weighted RLS (`ω_c = N_base/N_inc`, `10×`)** | **Khớp máy (`6,23e-12`)** | **`0`** | Khôi phục **`+28,66 pp`** $A_{\mathrm{inc}}$ (`→ 28,87%`) | Không lưu mẫu cũ, không tốn thêm byte cho $R_k$, giữ nguyên cập nhật một lượt và khôi phục 88% hiệu năng lớp mới. |

### 2. Đường cong sụp đổ `E3` và khôi phục nghiệm đóng `FD3-WRLS` (`n = 3` seeds, `ddof = 1`)

| Tỷ số mất cân bằng $\rho = N_{\mathrm{base}}/N_{\mathrm{inc}}$ | ACIL $A_{\mathrm{final}}$ (%) | **ACIL $A_{\mathrm{inc}}$ 50 lớp mới (%)** | Tỷ số $\|w_{\mathrm{inc}}\|_2 / \|w_{\mathrm{base}}\|_2$ | **FD3-WRLS $A_{\mathrm{final}}$ (%)** | **FD3-WRLS $A_{\mathrm{inc}}$ 50 lớp mới (%)** |
|---|---:|---:|---:|---:|---:|
| `1×` (`N_inc = 500`, cân bằng) | `51,90 ± 0,48` | **`32,62 ± 1,57`** | `0,915` | `51,90 ± 0,48` | `32,62 ± 1,57` |
| `2×` (`N_inc = 250`) | `46,57 ± 0,57` | `17,87 ± 1,20` | `0,674` | `51,67 ± 0,37` | `32,29 ± 1,35` (`+14,42 pp`) |
| `5×` (`N_inc = 100`) | `39,94 ± 0,55` | `3,31 ± 0,89` | `0,431` | `50,96 ± 0,37` | `31,05 ± 1,30` (`+27,74 pp`) |
| `10×` (`N_inc = 50`) | `38,38 ± 0,13` | **`0,21 ± 0,10`** | **`0,283`** | **`49,96 ± 0,62`** | **`28,87 ± 1,69` (`+28,66 pp`)** |
| `20×` (`N_inc = 25`) | `38,29 ± 0,12` | `0,01 ± 0,01` | `0,178` | `47,81 ± 0,64` | `24,70 ± 1,71` (`+24,69 pp`) |
| `50×` (`N_inc = 10`) | `38,28 ± 0,05` | **`0,00 ± 0,00`** | **`0,103`** | **`44,16 ± 0,12`** | **`16,81 ± 0,40` (`+16,81 pp`)** |

| Run `E3 + FD3` | Script nguồn | Cấu hình | Raw JSON | W&B (Finished, Public) |
|---|---|---|---|---|
| `E3+FD3 / seed 42` | [`scripts/e3_mse_imbalance_runner.py`](scripts/e3_mse_imbalance_runner.py) | `d=1024`, `γ=0.1`, `N_inc ∈ {500..10}`, `fp64` | [JSON](reports/e3_seed42_results.json) | [18y2c7pg](https://wandb.ai/hsangg-2006/acil-reproduction/runs/18y2c7pg) |
| `E3+FD3 / seed 43` | [`scripts/e3_mse_imbalance_runner.py`](scripts/e3_mse_imbalance_runner.py) | `d=1024`, `γ=0.1`, `N_inc ∈ {500..10}`, `fp64` | [JSON](reports/e3_seed43_results.json) | [7uqw193s](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7uqw193s) |
| `E3+FD3 / seed 44` | [`scripts/e3_mse_imbalance_runner.py`](scripts/e3_mse_imbalance_runner.py) | `d=1024`, `γ=0.1`, `N_inc ∈ {500..10}`, `fp64` | [JSON](reports/e3_seed44_results.json) | [rfu8q4pv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/rfu8q4pv) |

![Biểu đồ E3 — Tách bạch MSE mất cân bằng lớp với độ trôi Woodbury float32](reports/figures/e3_mse_imbalance_vs_woodbury_drift.png)

### Cách chạy lại trọn bộ thực nghiệm cuối kỳ (`E1`, `E2`, `E3`), kiểm thử và vẽ biểu đồ

Từ thư mục gốc `20-ML-Paper` sau khi cài môi trường CIFAR (`python -m pip install -r requirements-cifar.txt`):

```bash
# 1. Chạy toàn bộ unit tests của thực nghiệm cuối kỳ (E1, E2, E3 & FD2/FD3)
python -m unittest discover -s scripts/tests -p "test_final_empirics.py" -v
python -m unittest discover -s scripts/tests -p "test_e3_mse_imbalance.py" -v

# 2. Chạy lại E1 (P1 — huấn luyện ResNet-32 trên GPU hoặc import notebooks/13..18 lên Kaggle T4)
python scripts/e1_base_assumption_runner.py run --branch E1a --seed 42 --base-classes 10 --split-mode uniform --epochs 160 --expansion-dim 8000 --gamma 0.1 --verify-joint-oracle --output-dir reports/local/e1a_c0_10_s42 --wandb-mode disabled
python scripts/e1_base_assumption_runner.py run --branch E1b --seed 42 --base-classes 50 --split-mode superclass_disjoint --epochs 160 --expansion-dim 8000 --gamma 0.1 --verify-joint-oracle --output-dir reports/local/e1b_s42 --wandb-mode disabled

# 3. Chạy lại E2 (P2 — tái dùng checkpoint R1 seed 42/43/44 tải từ Kaggle Output v1, không train lại CNN)
python scripts/e2_memory_precision_runner.py run --seed 42 --checkpoint <duong_dan_ckpt_R1_seed42.pt> --output-dir reports/local/e2_s42 --threads 4 --wandb-mode disabled

# 4. Chạy lại E3 & FD3-WRLS (P3 — tái dùng fixture đặc trưng F64 đã trích từ bước E2)
python scripts/e3_mse_imbalance_runner.py --seeds 42 43 44 --fixture-root reports/local/final_empirics_e2_20261118 --output-root reports/local/e3_rerun --wandb-mode disabled

# 5. Tái tạo toàn bộ 3 biểu đồ cuối kỳ từ các tệp JSON tổng hợp đã khóa SHA-256 (môi trường requirements-plots.txt)
python scripts/plot_final_empirics.py --mode all
python scripts/plot_e3_mse_imbalance.py
```

*(Đổi `--wandb-mode disabled` thành `online` hoặc `offline` nếu muốn đẩy run mới lên W&B; toàn bộ kết quả cục bộ ghi trong `reports/local/` được `.gitignore` loại trừ để không ghi đè các báo cáo và manifest đã công bố.)*
