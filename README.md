# ACIL — Reproduction and Analysis

Đồ án Nhập môn Máy học: đọc, tự triển khai, tái lập và phản biện **ACIL**, NeurIPS 2022.

| Thành viên | Phụ trách chính |
|---|---|
| Sang | Phương pháp, analytic head, kiểm chứng joint/phase/minibatch và R3 |
| Vỹ | Dataset, protocol, backbone, Kaggle và R1 |
| Phát | Metrics, W&B, ablation R2, bằng chứng và tổng hợp kết quả |

## Hiện trạng

- Đã có module analytic hỗ trợ nhãn toàn cục và minibatch; notebook/CLI dùng cùng source.
- Đã có test độc lập cho phương pháp và thứ tự log W&B/xuất JSON.
- VERIFY-01 là mốc lịch sử trên dữ liệu tổng hợp. VERIFY-02 kiểm tra thêm minibatch.
- Pipeline CIFAR smoke đã có kết quả local và Kaggle CUDA Version 2 (`364942764`), W&B run `o2r63pj2` Finished. Source/config hashes trong JSON Kaggle đã được đối chiếu; xem bảng bằng chứng và manifest bên dưới.
- Project W&B: [hsangg-2006/acil-reproduction](https://wandb.ai/hsangg-2006/acil-reproduction), đã xác nhận Public ngày 02/11/2026.

## Cấu trúc và cách chạy

Mở thư mục gốc **`20-ML-Paper`** làm thư mục dự án trong VS Code. Cấu trúc đang dùng:

```text
20-ML-Paper/
├── docs/
│   ├── Dac_ta_ACIL.md
│   ├── HUONG_DAN_KAGGLE.md
│   ├── HUONG_DAN_CIFAR_SMOKE.md
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
│   └── notebook_bundle.py
├── tests/
│   ├── test_analytic_update.py
│   ├── test_reporting.py
│   └── test_cifar_pipeline.py
├── notebooks/
│   ├── 01_verify_analytic_update.ipynb
│   └── 02_cifar100_smoke.ipynb
├── configs/
│   ├── verify_analytic_update.json
│   └── cifar100_smoke.json
├── reports/
│   ├── archive/
│   ├── README.md
│   ├── VERIFY-02-provenance.json
│   ├── CIFAR-SMOKE-01-provenance.json
│   ├── cifar100_smoke_results.json
│   ├── cifar100_kaggle_results.json
│   ├── CIFAR-KAGGLE-01-provenance.json
│   ├── verify_analytic_update_results.json
│   └── verify_analytic_update_v2_results.json
├── .gitattributes
├── .gitignore
├── README.md
├── requirements-verification.txt
├── requirements-tracking.txt
├── requirements-cifar.txt
└── requirements-dev.txt
```

`.git/`, `.venv/` và `reports/local/` là thư mục dùng trên máy; không đưa lên GitHub. `configs/`, các module reporting/verification và test tracking được giữ vì notebook VERIFY-02 đang dùng chúng.

| Đường dẫn | Nội dung thực sự có |
|---|---|
| `src/analytic.py` | Joint ridge, khởi tạo R/W, cập nhật theo nhãn toàn cục |
| `src/verification.py` | Dữ liệu tổng hợp và kiểm chứng ba cách học |
| `src/reporting.py` | Source hashes, run ID/URL động, xuất JSON sau logging |
| `configs/verify_analytic_update.json` | Config VERIFY-02 |
| `tests/` | Test phương pháp và lỗi logging/finish của tracking |
| `notebooks/01_verify_analytic_update.ipynb` | Notebook dùng module, chạy local hoặc import riêng trên Kaggle |
| `docs/` | Đặc tả và phân công theo hiện trạng repo |
| `reports/` | Kết quả lịch sử/đã công bố; `reports/local/` bị Git bỏ qua |

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

Sang hiện thực hiện một mình; không cần chờ bàn giao từ Vỹ để chạy smoke. [Hướng dẫn CIFAR 50+10](docs/HUONG_DAN_CIFAR_SMOKE.md) có bước chạy local/Kaggle và cách đọc kết quả.

```bash
python -m pip install -r requirements-cifar.txt
python -B -m src.cifar_smoke
```

Lệnh mặc định tải dữ liệu, train base 2 epochs, freeze ResNet-32/BatchNorm, expansion 256, chạy một phase mới theo minibatch và kiểm tra joint trên đủ 60 lớp. API dùng chính `src/analytic.py`; metadata mapping nằm trong pipeline. Kết quả local ở `reports/local/cifar100_smoke/`; đây là smoke test. Khi chưa cài dependency CIFAR, 9 test ảnh được skip có thông báo; 11 test core/tracking vẫn chạy với NumPy.

Notebook 02 có source bundle để upload độc lập lên Kaggle. Tạo lại bằng `python -m src.notebook_bundle` sau mỗi lần sửa source/config/test; không cần đưa source lên GitHub trước để chạy thử.

## Bằng chứng thí nghiệm

| ID | Source snapshot | Kết quả | Kaggle version | W&B |
|---|---|---|---|---|
| VERIFY-01 | [23af76f](https://github.com/24122050S/20-ML-Paper/commit/23af76f9615a0b7e81ff5d8cc373acab45ec4398) | [JSON đã đối chiếu log](reports/verify_analytic_update_results.json) | [Version 8](https://www.kaggle.com/code/hsangg206/01-verify-analytic-update6bf7e5ced7/notebook?scriptVersionId=364773332) | [7aeaupq7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7aeaupq7) |
| VERIFY-02 | [217684f](https://github.com/24122050S/20-ML-Paper/commit/217684f02566c9d0a34cea0071a26dbfda7b5582) | [JSON nguyên bản Kaggle](reports/verify_analytic_update_v2_results.json) | [Version 9](https://www.kaggle.com/code/hsangg206/01-verify-analytic-update6bf7e5ced7/output?scriptVersionId=364788888) | [wi4wau2d](https://wandb.ai/hsangg-2006/acil-reproduction/runs/wi4wau2d) |
| CIFAR-SMOKE-01 | Local working tree có source hashes; chưa commit | [JSON local đã kiểm tra](reports/cifar100_smoke_results.json) | Run local CPU | Tắt tracking, ID/URL null |
| CIFAR-KAGGLE-01 | Source bundle; hashes trong JSON đã đối chiếu | [JSON Kaggle](reports/cifar100_kaggle_results.json) | [Version 2](https://www.kaggle.com/code/hsangg206/notebook7e7946d4ee?scriptVersionId=364942764) | [o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2) |

CIFAR-KAGGLE-01: Python 3.13.15, NumPy 2.1.3, PyTorch 2.11.0+cu128, CUDA, train mới 2 epochs, expansion 256, seed42. Base 50 lớp đạt 32,58%; sau phase mới 60 lớp đạt 29,70% trên đủ 6.000 ảnh test. W là 256×50 rồi 256×60, dự đoán khớp joint, mọi check đạt. Đã đối chiếu output và JSON của Version 2 với run W&B Finished. JSON ghi `logged`; git_commit null do source được giải nén từ bundle. [Manifest](reports/CIFAR-KAGGLE-01-provenance.json) liên kết version, source bundle và checksum JSON. Kaggle hiển thị môi trường GPU T4 x2, runtime notebook 1m42s; notebook hiện Private. Đây là smoke test.

VERIFY-01: Python 3.13.15, NumPy 2.1.3, float64, seed42, gamma1, 3 phase, 2 lớp/phase, feature24. Kết quả ở hàng này được đối chiếu từ output lưu của Version 8, sửa metadata URL cũ mà không đổi numeric rows. Sai lệch W tuyệt đối lớn nhất khoảng 1.84e-14; đây không phải sai lệch tương đối. [Version 5 gốc](reports/archive/VERIFY-01-version5.json) được giữ riêng.

VERIFY-02 giữ bài toán tổng hợp tương ứng, so sánh thêm cập nhật minibatch với joint sau từng phase; đánh giá W/R/score/argmax, lớp lặp, state không bị sửa và lỗi tracking. Các con số nhỏ có thể khác bitwise giữa môi trường; áp dụng ngưỡng đã ghi trong config.

VERIFY-02 Version 9 đã Save & Run All thành công trên Kaggle CPU ngày 02/11/2026: 11 tests đạt; W qua ba phase là `(24,2)`, `(24,4)`, `(24,6)`; sai lệch tương đối W lớn nhất của minibatch so với joint là khoảng `3.43e-13`, không có nhãn dự đoán khác nhau. W&B xác nhận Finished. Notebook có output được Kaggle push ở [eddd50d](https://github.com/24122050S/20-ML-Paper/commit/eddd50d20a3d7f11bf6b24efd7d15916a1581bd2); commit này khác source commit `217684f` mà notebook đã fetch để tính toán. [Manifest đối chiếu](reports/VERIFY-02-provenance.json) ghi cả hai và SHA-256 của JSON.

Notebook Kaggle hiện vẫn **Private**; người ngoài chưa xem được link Kaggle. Source, notebook có output và JSON đã lưu trên GitHub để kiểm tra; W&B project đã Public.

## Quy ước phương pháp

- Nhãn đầu vào là ID toàn cục. Một lớp gặp lại ở batch sau vẫn dùng cột cũ.
- Cùng backbone/feature expansion và gamma trong một chuỗi cập nhật. Module hiện chỉ hỗ trợ gamma > 0; không gọi một gamma rất nhỏ là ablation gamma=0.
- Joint giữ lịch sử chỉ để tạo nghiệm tham chiếu trong bộ kiểm tra. Hàm incremental không nhận dữ liệu train cũ.
- Accuracy tổng hợp không dùng so với accuracy benchmark của paper; tương đương joint analytic không có nghĩa tương đương joint CNN training hoặc accuracy lớp cũ luôn bất biến.

## Nguồn và hỗ trợ

- Zhuang et al., [ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract-Conference.html), NeurIPS 2022, mục 3.1–3.2, phương trình (4), (9), (10).
- NumPy cho đại số; W&B cho tracking; GitHub/Kaggle cho source và môi trường chạy.
- Notebook/code khởi đầu và các sửa đổi được hỗ trợ bởi ChatGPT. Nhóm chịu trách nhiệm đọc, kiểm chứng và giải thích toàn bộ sản phẩm.

[Phân công Sang/Vỹ/Phát](docs/PHAN_CONG_NHOM.md) · [Đặc tả analytic](docs/Dac_ta_ACIL.md) · [Báo cáo thay đổi](docs/THAY_DOI_VERIFY02.md)

[Báo cáo thay đổi CIFAR smoke](docs/THAY_DOI_CIFAR_SMOKE.md) · [Hướng dẫn tự chạy](docs/HUONG_DAN_CIFAR_SMOKE.md)
