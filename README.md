# ACIL — Reproduction and Analysis

Đồ án Nhập môn Máy học: đọc, tự triển khai, tái lập và phản biện **ACIL**, NeurIPS 2022.

| Thành viên | Phụ trách chính                                                       |
| ------------ | ------------------------------------------------------------------------ |
| Sang         | Phương pháp, analytic head, kiểm chứng joint/phase/minibatch và R3 |
| Vĩ          | Dataset, protocol, backbone, Kaggle và R1                               |
| Phát        | Metrics, W&B, ablation R2, bằng chứng và tổng hợp kết quả         |

## Hiện trạng

- Đã có module analytic hỗ trợ nhãn toàn cục và minibatch; notebook/CLI dùng cùng source.
- Đã có test độc lập cho phương pháp và thứ tự log W&B/xuất JSON.
- VERIFY-01 là mốc lịch sử trên dữ liệu tổng hợp. VERIFY-02 kiểm tra thêm minibatch.
- Pipeline CIFAR smoke đã có lượt CUDA đủ 100 lớp: [JSON](reports/cifar100_kaggle_full5_results.json), [Kaggle scriptVersionId 365084714](https://www.kaggle.com/code/hsangg206/03-cifar100-safe-attempts/output?scriptVersionId=365084714), [W&B blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) finished. W cuối 256×100, test 10.000 ảnh, accuracy 22,05%, mọi check đạt; source/config hashes đã đối chiếu. Đây là smoke 2 epochs/d256.
- Đã có checkpoint resume tại biên epoch, benchmark streaming tách oracle, no-FE, metric export và profile d8000 trên CPU/Kaggle T4. [Run gey0yop4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/gey0yop4) Finished:34,07s/peak RSS3,20GB trên subset; [JSON](reports/profile_d8000_kaggle_results.json) và [manifest](reports/PROFILE-KAGGLE-01-provenance.json) đã đối chiếu API. **BP5 đã chạy thay Sang trên Kaggle Version3**:76,11s training, CUDA resume passed, W256×60/test6.000/accuracy41,35%; [JSON](reports/profile_base_kaggle_ready.json) và [manifest](reports/BP5-KAGGLE-01-provenance.json). Tracking BP5 đã đồng bộ: [W&B pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) Finished; API công khai xác nhận config/source/summary khớp. [Notebook06](notebooks/06_benchmark_seed42_ready.ipynb) và [budget](reports/benchmark_seed42_budget.json) đã chuẩn bị cho160 epochs/d8000/seed42; chưa khởi chạy. [Trạng thái ngắn cho Sang](docs/BUOC_TIEP_THEO_SANG.md).
- Project W&B: [hsangg-2006/acil-reproduction](https://wandb.ai/hsangg-2006/acil-reproduction), đã xác nhận Public ngày 02/11/2026.

## Cấu trúc và cách chạy

Mở thư mục gốc **`20-ML-Paper`** làm thư mục dự án trong VS Code. Cấu trúc đang dùng:

```text
20-ML-Paper/
├── docs/
│   ├── Dac_ta_ACIL.md
│   ├── protocol_chot.md
│   ├── metrics.md
│   ├── RESUME_BENCHMARK_PROFILE.md
│   ├── BUOC_TIEP_THEO_SANG.md
│   ├── CHAY_KAGGLE_BANG_API.md
│   ├── HUONG_DAN_KAGGLE.md
│   ├── HUONG_DAN_CIFAR_SMOKE.md
│   ├── SUA_LOI_TRANG_THAI_RUN.md
│   ├── SUA_LOI_CHECKSUM_JSON.md
│   ├── PHAN_CONG_NHOM.md
│   ├── THAY_DOI_VERIFY02.md
│   └── THAY_DOI_CIFAR_SMOKE.md
├── src/
│   ├── __init__.py
│   ├── analytic.py
│   ├── verification.py
│   ├── reporting.py
│   ├── verify_analytic.py
│   ├── cifar_data.py
│   ├── resnet.py
│   ├── cifar_smoke.py
│   ├── base_training.py
│   ├── streaming.py
│   ├── metrics.py
│   ├── profile_analytic.py
│   ├── verify_resume.py
│   ├── run_attempt.py
│   └── notebook_bundle.py
├── tests/
│   ├── test_analytic_update.py
│   ├── test_reporting.py
│   ├── test_cifar_pipeline.py
│   ├── test_run_attempts.py
│   ├── test_notebook_distribution.py
│   ├── test_resume.py
│   ├── test_streaming_metrics.py
│   └── test_profile.py
├── notebooks/
│   ├── 01_verify_analytic_update.ipynb
│   ├── 02_cifar100_smoke.ipynb
│   ├── 03_cifar100_safe_attempts.ipynb
│   ├── 04_resume_benchmark_profile.ipynb
│   ├── 05_profile_base_ready.ipynb
│   ├── 06_benchmark_seed42_ready.ipynb
│   └── archive/05_profile_base_api_v3.ipynb
├── configs/
│   ├── verify_analytic_update.json
│   ├── cifar100_smoke.json
│   └── cifar100_benchmark_k5.json
├── reports/
│   ├── archive/
│   ├── README.md
│   ├── VERIFY-02-provenance.json
│   ├── CIFAR-SMOKE-01-provenance.json
│   ├── cifar100_smoke_results.json
│   ├── cifar100_kaggle_results.json
│   ├── CIFAR-KAGGLE-01-provenance.json
│   ├── cifar100_kaggle_full5_results.json
│   ├── CIFAR-KAGGLE-02-provenance.json
│   ├── resume_cpu_verification.json
│   ├── cifar100_streaming_cpu_verification.json
│   ├── profile_d8000_cpu_results.json
│   ├── RESUME-PROFILE-01-provenance.json
│   ├── profile_d8000_kaggle_results.json
│   ├── profile_d8000_kaggle_progress.json
│   ├── PROFILE-KAGGLE-01-provenance.json
│   ├── profile_base_kaggle_ready.json
│   ├── BP5-KAGGLE-01-provenance.json
│   ├── benchmark_seed42_budget.json
│   ├── verify_analytic_update_results.json
│   └── verify_analytic_update_v2_results.json
├── scripts/
│   ├── build_profile_base_notebook.py
│   └── build_benchmark_notebook.py
├── .gitattributes
├── .gitignore
├── README.md
├── requirements-verification.txt
├── requirements-tracking.txt
├── requirements-cifar.txt
└── requirements-dev.txt
```

`.git/`, `.venv/` và `reports/local/` là thư mục dùng trên máy; không đưa lên GitHub. `configs/`, các module reporting/verification và test tracking được giữ vì notebook VERIFY-02 đang dùng chúng.

| Đường dẫn                                 | Nội dung thực sự có                                                |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| `src/analytic.py`                           | Joint ridge, khởi tạo R/W, cập nhật theo nhãn toàn cục          |
| `src/verification.py`                       | Dữ liệu tổng hợp và kiểm chứng ba cách học                    |
| `src/reporting.py`                          | Source hashes, run ID/URL động, xuất JSON sau logging               |
| `configs/verify_analytic_update.json`       | Config VERIFY-02                                                       |
| `tests/`                                    | Test phương pháp và lỗi logging/finish của tracking              |
| `notebooks/01_verify_analytic_update.ipynb` | Notebook dùng module, chạy local hoặc import riêng trên Kaggle    |
| `docs/`                                     | Đặc tả và phân công theo hiện trạng repo                       |
| `reports/`                                  | Kết quả lịch sử/đã công bố; `reports/local/` bị Git bỏ qua |

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

## Chạy CIFAR một mình

Sang hiện thực hiện một mình; không cần chờ bàn giao từ Vĩ để chạy smoke. [Hướng dẫn CIFAR 50+10](docs/HUONG_DAN_CIFAR_SMOKE.md) có bước chạy local/Kaggle và cách đọc kết quả.

```bash
python -m pip install -r requirements-cifar.txt
python -B -m src.cifar_smoke
```

Lệnh mặc định tải dữ liệu, train base 2 epochs, freeze ResNet-32/BatchNorm, expansion 256, chạy một phase mới và kiểm joint trên đủ 60 lớp. API dùng chính `src/analytic.py`; metadata mapping thuộc pipeline. Root output là `reports/local/cifar100_smoke/`; artifact đầy đủ nằm trong `attempts/<attempt_id>/`. Các test ảnh/attempt được skip rõ ràng nếu thiếu dependency CIFAR; 15 test core/tracking/metrics vẫn chạy với NumPy.

Analytic d8000 và BP5/CUDA resume đã đạt. BP5 chạy qua API trong [Version3 /365703432](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432); không cần chạy05 lại. **[Notebook06](notebooks/06_benchmark_seed42_ready.ipynb)** đã đặt R1 seed42/160 epochs/d8000/K5, checkpoint đầu vào=None, offline W&B để sync sau. [Hướng dẫn API](docs/CHAY_KAGGLE_BANG_API.md) và [budget](reports/benchmark_seed42_budget.json) giải thích timeout2 giờ. Chưa có run160 epochs. Script `python scripts/build_profile_base_notebook.py` tạo05 từ cùng bundle04; giữ04 làm snapshot của bằng chứng mới. Generator cũ `python -m src.notebook_bundle` vẫn mặc định tạo04: không dùng để ghi đè snapshot đã công bố. `benchmark` dùng preset160 epochs khi đã qua gate. `--resume-checkpoint` tiếp tục BP đang dở; `--reuse-checkpoint` dùng base đã hoàn tất.

Mỗi lần gọi pipeline tạo một attempt riêng. Root `latest.json`, `progress.json`, `results.json` có cùng attempt ID và trạng thái mới nhất: `running`, `complete` hoặc `failed`. Chỉ nhận kết quả khi `status=complete` và `verification_passed=true`; lấy checkpoint/state từ `output_dir` trong report. Nếu lượt mới lỗi, root ghi failed/false; dữ liệu lượt trước vẫn ở thư mục riêng. [Giải thích và test hồi quy](docs/SUA_LOI_TRANG_THAI_RUN.md).

`filelock` bảo vệ metadata root khi nhiều lượt cùng ghi; JSON trạng thái được ghi qua file tạm rồi replace. Đây là hạ tầng lưu kết quả, không tham gia công thức analytic. [Tài liệu filelock](https://py-filelock.readthedocs.io/en/latest/).

## Bằng chứng thí nghiệm

| ID              | Source snapshot                                                                                    | Kết quả                                                                 | Kaggle version                                                                                                           | W&B                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| VERIFY-01       | [23af76f](https://github.com/24122050S/20-ML-Paper/commit/23af76f9615a0b7e81ff5d8cc373acab45ec4398) | [JSON đã đối chiếu log](reports/verify_analytic_update_results.json)  | [Version 8](https://www.kaggle.com/code/hsangg206/01-verify-analytic-update6bf7e5ced7/notebook?scriptVersionId=364773332) | [7aeaupq7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7aeaupq7) |
| VERIFY-02       | [217684f](https://github.com/24122050S/20-ML-Paper/commit/217684f02566c9d0a34cea0071a26dbfda7b5582) | [JSON nguyên bản Kaggle](reports/verify_analytic_update_v2_results.json) | [Version 9](https://www.kaggle.com/code/hsangg206/01-verify-analytic-update6bf7e5ced7/output?scriptVersionId=364788888)   | [wi4wau2d](https://wandb.ai/hsangg-2006/acil-reproduction/runs/wi4wau2d) |
| CIFAR-SMOKE-01  | Local working tree có source hashes; chưa commit tại thời điểm chạy                         | [JSON local đã kiểm tra](reports/cifar100_smoke_results.json)           | Run local CPU                                                                                                            | Tắt tracking, ID/URL null                                              |
| CIFAR-KAGGLE-01 | Source bundle; hashes trong JSON đã đối chiếu                                                 | [JSON Kaggle](reports/cifar100_kaggle_results.json)                        | [Version 2](https://www.kaggle.com/code/hsangg206/notebook7e7946d4ee?scriptVersionId=364942764)                           | [o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2) |
| CIFAR-KAGGLE-02 | [Bundle notebook 03 tại 9972be9](https://github.com/24122050S/20-ML-Paper/blob/9972be9db06b60ff1abbf0c7483325543cda8b1a/notebooks/03_cifar100_safe_attempts.ipynb) | [JSON 100 lớp](reports/cifar100_kaggle_full5_results.json) | [scriptVersionId 365084714](https://www.kaggle.com/code/hsangg206/03-cifar100-safe-attempts/output?scriptVersionId=365084714) | [blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) |
| BP5-KAGGLE-01 | [Source API Version3](notebooks/archive/05_profile_base_api_v3.ipynb), core16 hashes trong manifest | [JSON BP5 sau sync](reports/profile_base_kaggle_ready.json) | [Version3 /365703432](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432) | [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) Finished |

CIFAR-KAGGLE-01: Python 3.13.15, NumPy 2.1.3, PyTorch 2.11.0+cu128, CUDA, train mới 2 epochs, expansion 256, seed 42. Base 50 lớp đạt 32,58%; sau phase mới 60 lớp đạt 29,70% trên đủ 6.000 ảnh test. W là 256×50 rồi 256×60, dự đoán khớp joint, mọi check đạt. Đã đối chiếu output và JSON của Version 2 với run W&B Finished. JSON ghi `logged`; git_commit null do source được giải nén từ bundle. [Manifest](reports/CIFAR-KAGGLE-01-provenance.json) liên kết version, source bundle và checksum JSON. Kaggle hiển thị môi trường GPU T4 x2, runtime notebook 1m42s; notebook được ghi nhận Private lúc đối chiếu. Đây là smoke test.

CIFAR-KAGGLE-02: train mới 2 epochs, seed 42, d256, gamma 0.1 trên CUDA; Python 3.13.16, NumPy 2.1.3, PyTorch 2.11.0+cu130. Sáu mốc 50→60→70→80→90→100 lớp có accuracy 33,46%→29,9167%→27,50%→25,2250%→23,3111%→22,05%; dự đoán khớp joint ở mọi phase. W&B API công khai xác nhận finished/config/summary khớp; [manifest](reports/CIFAR-KAGGLE-02-provenance.json) giữ checksum JSON và source bundle. Link Kaggle do Sang cung cấp; giao diện/số version hiển thị và loại GPU chưa xác minh độc lập. [Review lượt chạy](reports/REVIEW_CIFAR_KAGGLE_FULL5_20261104.md).

VERIFY-01: Python 3.13.15, NumPy 2.1.3, float64, seed 42, gamma1, 3 phase, 2 lớp/phase, feature24. Kết quả ở hàng này được đối chiếu từ output lưu của Version 8, sửa metadata URL cũ mà không đổi numeric rows. Sai lệch W tuyệt đối lớn nhất khoảng 1.84e-14; đây không phải sai lệch tương đối. [Version 5 gốc](reports/archive/VERIFY-01-version5.json) được giữ riêng.

VERIFY-02 giữ bài toán tổng hợp tương ứng, so sánh thêm cập nhật minibatch với joint sau từng phase; đánh giá W/R/score/argmax, lớp lặp, state không bị sửa và lỗi tracking. Các con số nhỏ có thể khác bitwise giữa môi trường; áp dụng ngưỡng đã ghi trong config.

VERIFY-02 Version 9 đã Save & Run All thành công trên Kaggle CPU ngày 02/11/2026: 11 tests đạt; W qua ba phase là `(24,2)`, `(24,4)`, `(24,6)`; sai lệch tương đối W lớn nhất của minibatch so với joint là khoảng `3.43e-13`, không có nhãn dự đoán khác nhau. W&B xác nhận Finished. Notebook có output được Kaggle push ở [eddd50d](https://github.com/24122050S/20-ML-Paper/commit/eddd50d20a3d7f11bf6b24efd7d15916a1581bd2); commit này khác source commit `217684f` mà notebook đã fetch để tính toán. [Manifest đối chiếu](reports/VERIFY-02-provenance.json) ghi cả hai và SHA-256 của JSON.

Notebook Kaggle được ghi nhận **Private** trong các lần đối chiếu version; cần kiểm quyền truy cập trước khi nộp. Source, notebook và JSON đã lưu trên GitHub; notebook 01 có output, còn bản 02 mang source bundle của Version 2. W&B project đã được ghi nhận Public.

Trạng thái cập nhật **05/11/2026**: smoke 100 lớp đã có JSON và W&B finished; source/config của lượt mới khớp bundle 03. Suite hiện tại: **54 tests, 53 đạt và 1 skip CUDA** trên local CPU; bootstrap04 từ bundle trong kernel mới đạt52/54, skip CUDA và test snapshot02/03 không phân phối trong bundle. [Protocol benchmark v1.0](docs/protocol_chot.md) và [metrics v1.0](docs/metrics.md) đã chốt: CIFAR-100/K5, base160 epochs, d8000, gamma0.1; R1/R2/R3 theo core scope. Resume CPU đã so model/optimizer/scheduler/RNG/history chính xác; streaming đủ100 lớp khớp accuracy oracle CPU ở cả6 mốc. [Bằng chứng mới](reports/RESUME-PROFILE-01-provenance.json) và [báo cáo thay đổi](reports/REVIEW_RESUME_PROFILE_20261104.md) ghi phạm vi từng phép kiểm. Analytic d8000 trên Kaggle T4 đã đạt qua [gey0yop4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/gey0yop4); BP5/CUDA cũng đã đạt: GPU bundle54 tests,53 passed/1 historical-snapshot skip. [Review BP5](reports/REVIEW_BP5_KAGGLE_20261105.md) ghi checkpoint v2, RAM/GPU và ngân sách~59 phút; chuẩn bị2 giờ để còn hơn30% dự phòng. [Notebook06](notebooks/06_benchmark_seed42_ready.ipynb) đã qua bootstrap/guard trên CPU; chưa chạy R1. [Checksum CPU](docs/SUA_LOI_CHECKSUM_JSON.md) đã khớp Git blob; số liệu lịch sử giữ nguyên. [Review tổng quan](reports/REVIEW_TONG_QUAN_20261102.md) · [Danh mục bằng chứng](reports/README.md).

Analytic của notebook04 là Public Version1 với source bundle khớp và W&B finished; cùng kernel này hiện có BP5 ở Version3. Version-pinned output API đã đối chiếu raw bytes của analytic V1 /365647614 và BP5 V3 /365703432. BP5 cung cấp executed CUDA test proof trong JSON. [Review analytic Kaggle](reports/REVIEW_PROFILE_KAGGLE_20261104.md) và [review BP5](reports/REVIEW_BP5_KAGGLE_20261105.md) ghi phạm vi từng lượt.

BP5: [Kaggle Version3](https://www.kaggle.com/code/hsangg206/04-resume-benchmark-profile/output?scriptVersionId=365703432), [JSON tổng hợp sau sync](reports/profile_base_kaggle_ready.json), [manifest/source archive](reports/BP5-KAGGLE-01-provenance.json), [W&B pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) Finished. Đồng bộ đúng payload offline, không train lại; API không đăng nhập xác nhận config, 16 source hashes, summary và CUDA proof khớp. [JSON gốc trước sync](reports/archive/BP5-KAGGLE-01-original-ready.json) giữ nguyên `not_logged/offline_pending_sync`; bản tổng hợp chỉ bổ sung metadata đồng bộ thực tế. Run gey0yop4 thuộc analytic Version1.

## Kiểm resume và profile bằng CLI

```bash
python -m src.verify_resume --data-dir ../data
python -m src.profile_analytic --checkpoint reports/local/cifar100_smoke/base_checkpoint.pt --data-dir ../data --dimension 8000 --threads 4 --device cpu
```

Các lệnh trên kiểm subset CPU và tài nguyên; JSON giữ trạng thái tracking `not_logged`. Notebook04 hỗ trợ logging trên Kaggle. D8000 vẫn dùng NumPy/SciPy CPU cho analytic; GPU dùng cho CNN.

## Quy ước phương pháp

- Nhãn đầu vào là ID toàn cục. Một lớp gặp lại ở batch sau vẫn dùng cột cũ.
- Cùng backbone/feature expansion và gamma trong một chuỗi cập nhật. Module hiện chỉ hỗ trợ gamma > 0; không gọi một gamma rất nhỏ là ablation gamma=0.
- Joint giữ lịch sử chỉ để tạo nghiệm tham chiếu trong bộ kiểm tra. Hàm incremental không nhận dữ liệu train cũ.
- Accuracy tổng hợp không dùng so với accuracy benchmark của paper; tương đương joint analytic không có nghĩa tương đương joint CNN training hoặc accuracy lớp cũ luôn bất biến.

## Nguồn và hỗ trợ

- Zhuang et al., [ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract-Conference.html), NeurIPS 2022, mục 3.1–3.2, phương trình (4), (9), (10).
- NumPy cho đại số; W&B cho tracking; GitHub/Kaggle cho source và môi trường chạy.
- Notebook/code khởi đầu và các sửa đổi được hỗ trợ bởi ChatGPT. Nhóm chịu trách nhiệm đọc, kiểm chứng và giải thích toàn bộ sản phẩm.

[Phân công Sang/Vĩ/Phát](docs/PHAN_CONG_NHOM.md) · [Đặc tả analytic](docs/Dac_ta_ACIL.md) · [Báo cáo thay đổi](docs/THAY_DOI_VERIFY02.md)

[Báo cáo thay đổi CIFAR smoke](docs/THAY_DOI_CIFAR_SMOKE.md) · [Hướng dẫn tự chạy](docs/HUONG_DAN_CIFAR_SMOKE.md)
