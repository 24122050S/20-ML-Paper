# ACIL — Reproduction and Analysis

Đồ án môn Nhập môn Máy học.

## Paper

ACIL: Analytic Class-Incremental Learning with
Absolute Memorization and Privacy Protection.

Hội nghị: NeurIPS 2022.

## Thành viên

| Thành viên | Phụ trách chính |
|---|---|
| Sang | Phương pháp, analytic head, kiểm chứng Joint/Incremental và R3 |
| Vỹ | Dataset, protocol, backbone, Kaggle và R1 |
| Phát | Metrics, W&B, ablation R2 và tổng hợp kết quả |

## Công cụ

- GitHub: quản lý mã nguồn và phối hợp nhóm.
- Kaggle: chạy notebook và thí nghiệm.
- Weights & Biases: theo dõi cấu hình và kết quả.

## Trạng thái

Đang thiết lập repository và kiểm chứng công thức cập nhật ACIL.

## Phạm vi hiện tại

Nhóm đang kiểm chứng phần cập nhật analytic của ACIL trên dữ liệu tổng hợp.

Notebook 01 chưa phải thực nghiệm tái lập benchmark ảnh của paper.
Phạm vi tái lập benchmark sẽ được ghi sau khi nhóm thống nhất.

## Cấu trúc repository

| Đường dẫn | Nội dung |
|---|---|
| docs/ | Đặc tả, phân công, phương pháp, protocol và metrics |
| notebooks/ | Notebook kiểm chứng và thực nghiệm |
| src/ | Mã nguồn phương pháp |
| configs/ | Cấu hình thí nghiệm |
| tests/ | Kiểm tra tính đúng |
| kaggle/ | Hướng dẫn và notebook chạy Kaggle |
| reports/ | Kết quả nhỏ và phân tích được chọn để lưu |

## Môi trường kiểm chứng

- Python: 3.13.15
- NumPy: 2.1.3
- Kiểu tính toán: float64
- Seed: 42
- Không cần GPU cho notebook 01.

Phiên bản trên được ghi từ lần chạy kiểm chứng của nhóm.

Cài thư viện khi chạy trên máy cá nhân:

```bash
python -m pip install -r requirements-verification.txt
```

Lệnh trên chỉ cài thư viện; cần môi trường mở notebook để chạy file ipynb.

## Chạy notebook 01 trên Kaggle

1. Import notebooks/01_verify_analytic_update.ipynb vào Kaggle.
2. Chọn Python và Accelerator: None.
3. Giữ LOG_WANDB = False nếu chưa cấu hình W&B.
4. Chạy toàn bộ notebook từ đầu.
5. Kiểm tra thông báo PASS.
6. Dùng Save & Run All để lưu phiên bản đã chạy.

## Thí nghiệm VERIFY-01

- Dữ liệu tổng hợp gồm 3 phase.
- Mỗi phase thêm 2 lớp.
- Đặc trưng mở rộng có 24 chiều.
- Cùng feature map cố định và gamma = 1.0.
- Nhánh Joint giải ridge trực tiếp bằng numpy.linalg.solve.
- Nhánh Incremental cập nhật từ R, W và dữ liệu phase hiện tại.
- So sánh trọng số, ma trận R, điểm dự đoán và nhãn dự đoán.

| ID | Mã nguồn | Kết quả | Kaggle version | Code commit | W&B |
|---|---|---|---|---|---|
| VERIFY-01 | notebooks/01_verify_analytic_update.ipynb | reports/verify_analytic_update_results.json | Chưa điền | Chưa điền | Chưa đồng bộ |

## Kết luận kiểm chứng

Các kiểm tra Joint và Incremental đạt trên cấu hình tổng hợp đã chạy.
Sai số trọng số lớn nhất khoảng 1.84e-14.

Đây là kiểm chứng số học trên một cấu hình nhỏ, không thay thế chứng minh
lý thuyết và không phải kết quả benchmark ảnh của ACIL.

## So sánh với paper

Chưa hoàn thành thực nghiệm benchmark.
Không dùng accuracy trên dữ liệu tổng hợp để so sánh với accuracy của paper.

## Nguồn và hỗ trợ

- Zhuang et al., ACIL: Analytic Class-Incremental Learning with
  Absolute Memorization and Privacy Protection, NeurIPS 2022.
- Công thức tham chiếu: mục 3.1–3.2, phương trình (4), (9), (10).
- NumPy được dùng cho các phép toán ma trận và giải hệ tuyến tính.
- Notebook khởi đầu được tạo với hỗ trợ ChatGPT; nhóm chịu trách nhiệm
  đọc, kiểm chứng và giải thích nội dung trước khi sử dụng trong đồ án.
