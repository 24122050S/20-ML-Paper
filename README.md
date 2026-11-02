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
- **Chưa công bố pipeline CIFAR-100/backbone trong repo này.** Kết quả hỗ trợ cục bộ ở workspace khác không mặc nhiên là artifact của repo.
- Project W&B: [hsangg-2006/acil-reproduction](https://wandb.ai/hsangg-2006/acil-reproduction), đã xác nhận Public ngày 02/11/2026.

## Cấu trúc và cách chạy

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

Local dùng thông tin đăng nhập/env đã cấu hình, không ghi credential vào source. JSON mặc định ở `reports/local/verify_02_results.json`; tracking tắt ghi `not_logged`, URL null. Khi init/log/finish lỗi, file ghi `failed` và lệnh trả lỗi. `logged` chỉ được ghi sau khi run context kết thúc thành công.

## Chạy notebook trên Kaggle

1. Import notebook 01, chọn Python, Accelerator **None**, bật Internet.
2. Notebook tự lấy source từ GitHub khi không chạy trong checkout của repo. `SOURCE_REF` mặc định main; có thể đặt commit cụ thể để tái lập. Commit và source hashes thật được lưu vào JSON.
3. Bật secret `WANDB_API_KEY` cho notebook nếu dùng tracking. Mặc định log vào project nhóm; SDK khóa 0.28.1. Có thể tắt `LOG_WANDB` ở cell đầu để chỉ kiểm chứng.
4. Run All hoặc Save & Run All; kiểm tra VERIFY-02 COMPLETE và test đạt.
5. Lấy `verify_analytic_update_v2_results.json` từ Output. URL trong file phải là URL từ chính run vừa hoàn thành.
6. Cập nhật bảng bằng chứng bằng đúng version, commit nguồn và JSON của cùng lượt chạy. Không điền run ID cố định vào source.

[Hướng dẫn Kaggle chi tiết](kaggle/README.md). Notebook tự chọn đường dẫn output cho Kaggle/local; không bắt máy cá nhân phải có `/kaggle/working`.

## Bằng chứng thí nghiệm

| ID | Source snapshot | Kết quả | Kaggle version | W&B |
|---|---|---|---|---|
| VERIFY-01 | [23af76f](https://github.com/24122050S/20-ML-Paper/commit/23af76f9615a0b7e81ff5d8cc373acab45ec4398) | [JSON đã đối chiếu log](reports/verify_analytic_update_results.json) | [Version 8](https://www.kaggle.com/code/hsangg206/01-verify-analytic-update6bf7e5ced7/notebook?scriptVersionId=364773332) | [7aeaupq7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7aeaupq7) |
| VERIFY-02 | Module hiện tại, có source hashes trong report | Local tests/CLI đạt; chưa gắn với run Kaggle/W&B mới | Chờ chạy bản mới | Chưa ghi run cho bản module mới |

VERIFY-01: Python 3.13.15, NumPy 2.1.3, float64, seed42, gamma1, 3 phase, 2 lớp/phase, feature24. Kết quả ở hàng này được đối chiếu từ output lưu của Version 8, sửa metadata URL cũ mà không đổi numeric rows. Sai lệch W tuyệt đối lớn nhất khoảng 1.84e-14; đây không phải sai lệch tương đối. [Version 5 gốc](reports/archive/VERIFY-01-version5.json) được giữ riêng.

VERIFY-02 giữ bài toán tổng hợp tương ứng, so sánh thêm cập nhật minibatch với joint sau từng phase; đánh giá W/R/score/argmax, lớp lặp, state không bị sửa và lỗi tracking. Các con số nhỏ có thể khác bitwise giữa môi trường; áp dụng ngưỡng đã ghi trong config.

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
