# Metrics ACIL v1.0

Chốt **04/11/2026**, dùng cùng [protocol benchmark v1.0](protocol_chot.md). Mọi accuracy trong JSON/bảng dùng thang **0..100**; chênh lệch giữa hai accuracy dùng **điểm phần trăm (pp)**. Giữ giá trị chưa làm tròn khi tính; chỉ làm tròn ở bước hiển thị.

## 1. Ký hiệu và tập đánh giá

K=5 là số phase incremental, ngoài base phase0; có6 điểm0..5. C_k=50+10k lớp. M_k là analytic classifier sau cập nhật phase k. T_j là test của nhóm lớp j; T_0 chứa5000 ảnh base, T_j với j>0 chứa1000 ảnh. T_0:k có100×C_k ảnh.

Mọi dự đoán của M_k đều argmax trên **C_k cột đã thấy**, kể cả khi chỉ đo trên ảnh base hoặc ảnh mới. Không đưa task ID, không chọn riêng head của task, không che các cột mới khi đo base. Label mapping train/test phải giống protocol.

## 2. Accuracy từng phase và trung bình incremental

`A_k = 100 * số dự đoán đúng của M_k trên T_0:k / |T_0:k|`.

- Field hiện có: `results[k].accuracy_pct`; cao hơn tốt hơn.
- `test_samples_all_seen` phải bằng100×C_k, không chỉ test ảnh của phase mới.
- `final_accuracy_pct = A_K`: accuracy của classifier cuối trên toàn bộ100 lớp.
- **`average_incremental_accuracy_pct = sum(A_0,...,A_K)/(K+1)`**. Bao gồm base; mỗi phase có trọng số bằng nhau theo mục4.2 của paper.

Không trung bình có trọng số theo số ảnh của mỗi phase; không bỏ A0; không dùng A_K thay A-bar. Muốn có average đủ K phải có K+1 điểm liên tiếp, đúng config. Run thiếu/failed không được mang nhãn kết quả hoàn chỉnh.

Base A0 được đo sau analytic re-alignment. Accuracy của FC head trong BP training chỉ là diagnostic và không được ghép vào chuỗi A.

## 3. Base/new accuracy và dấu của forgetting

`B_k = 100 * số đúng của M_k trên T_0 / 5000`.

- Field hiện có: `base_accuracy_pct`. Đây vẫn là classifier phân loại trong toàn bộ C_k lớp.
- `new_class_accuracy_pct = N_k` chỉ đo nhóm lớp mới **của phase k**, không phải tất cả lớp ngoài base. Ở phase0 ghi null.
- Metric chính nhóm chốt: **`base_forgetting_pp = B_0 - B_K`**; dương là giảm khả năng nhận diện base, nhỏ hơn tốt hơn. Giữ giá trị âm nếu base cải thiện; không abs, không clamp0.
- Lưu thêm **`base_accuracy_delta_pp = B_K - B_0`**; âm là giảm. Hai field phải đối dấu.

[Paper mục4.2, trang PDF7](https://proceedings.neurips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf#page=7) in F=B_K−B_0, nhưng Table1 và phần diễn giải trang9 trình bày forgetting giảm chất lượng bằng số dương và nói F>0. Có bất nhất dấu ở nguồn. Nhóm giữ **cả hai field**, và khi đối chiếu cột F=9,00 của Table1 dùng **độ giảm `base_forgetting_pp`**, ghi rõ quy ước trong chú thích bảng. Không nói độ giảm là nguyên xi công thức đã in ở mục4.2.

Đây là forgetting của **nhóm base**, không phải average forgetting của toàn bộ task cũ theo ma trận accuracy từng task. JSON hiện chỉ có B_k và accuracy của task mới tại thời điểm nó xuất hiện; chưa đủ để suy ra forgetting của mọi nhóm cũ. Muốn metric đó phải thu ma trận a[k,j] và chốt định nghĩa riêng.

Tương đương joint analytic không kéo theo B_K=B_0 hoặc F=0: số lớp cạnh tranh trong argmax thay đổi. F=0 không phải tiêu chí pass của numerical verification.

## 4. Ví dụ thật từ smoke100

Nguồn: [JSON CIFAR-KAGGLE-02](../reports/cifar100_kaggle_full5_results.json), [manifest](../reports/CIFAR-KAGGLE-02-provenance.json), W&B blhw74ff. Run2 epochs/d256 chỉ dùng để kiểm cách tính dưới đây.

| k | C_k | A_k (%) | B_k (%) | Số ảnh test seen |
|---|---:|---:|---:|---:|
| 0 | 50 | 33.4600 | 33.4600 | 5000 |
| 1 | 60 | 29.9167 | 30.6000 | 6000 |
| 2 | 70 | 27.5000 | 28.7400 | 7000 |
| 3 | 80 | 25.2250 | 27.3200 | 8000 |
| 4 | 90 | 23.3111 | 26.5800 | 9000 |
| 5 | 100 | 22.0500 | 25.5600 | 10000 |

Tính từ JSON chưa làm tròn: **A-bar=26,91046296296%**, **A_final=22,05%**, **F_base=7,90pp**, **base delta=−7,90pp**. Một seed nên SD giữa seeds là null, không điền0. Không lấy A_final22,05% so với A-bar66,30% của paper; không dùng ví dụ smoke làm dòng benchmark.

## 5. Tổng hợp các seed và so với paper

R1/R2 có seed42/43/44 theo protocol. Tính metric đầy đủ **trong từng seed trước**, rồi báo `mean ± sample_SD` với `ddof=1`, n=3 và danh sách seed/run. Các seed của protocol đồng thời đổi class order, khởi tạo BP và FE; SD không chỉ phản ánh khởi tạo CNN. Một run có6 phase không được coi là6 seed.

Nếu mới có1 seed: báo số đơn và n=1, SD=null. Nếu một seed lỗi, giữ failed run và báo số completed thực tế; không im lặng chọn seed tốt nhất. R2 có thể báo thêm paired delta với baseline gamma0.1 trong **cùng seed/checkpoint**. Ba lần lặp đo thời gian của R3 chỉ ước lượng biến thiên timing, không thay thế3 seed accuracy.

| Field | Công thức | Đơn vị / ý nghĩa |
|---|---|---|
| `deviation_pp` | ours − paper | pp; giữ dấu |
| `relative_deviation_pct` | 100×(ours−paper)/abs(paper) | % tương đối, giữ dấu |
| `absolute_relative_deviation_pct` | 100×abs(ours−paper)/abs(paper) | % tương đối không dấu, nếu cần |

Paper reference bằng0 thì hai relative fields là null và giải thích; dùng deviation_pp, không thay denominator bằng epsilon. Đối với A-bar, deviation dương là tốt hơn; đối với F_base độ giảm, deviation dương là quên nhiều hơn.

R1 Table1 CIFAR100/K5 có A-bar66,30% và F9,00 theo bảng. Table1 không cung cấp một cột A_final để gán làm tham chiếu cuối; để paper_final=null nếu chưa có nguồn thích hợp. R2 đối chiếu các A-bar52,99 /66,30 /66,25 /66,23 theo [protocol](protocol_chot.md); không trộn score README tác giả mới vào các số này.

## 6. Kiểm tương đương và R3

Với oracle joint dùng đúng feature map, gamma, dữ liệu, precision và mapping:

- `relative_w_error = ||W_inc−W_joint||_F / max(||W_joint||_F, eps64)`.
- Lưu `max_abs_r_error`, `max_abs_score_error` và `prediction_disagreements` trên cùng test seen.
- So W/R/score bằng `rtol=1e-6`, `atol=1e-8`; R3 yêu cầu argmax disagreements=0 ở từng phase. Nếu lỗi, lưu failure và dữ liệu chẩn đoán, không đổi tolerance sau khi thấy kết quả.
- `eps64` chỉ bảo vệ tỷ số báo cáo nếu norm bằng0; không phải regularization gamma và không thay điều kiện allclose.

R3 reset R/W cho từng batch-size/repeat, dùng cùng checkpoint160/seed42, cùng FE1024/gamma0.1 và sample order. Joint oracle/cache historical features chỉ phục vụ kiểm chứng và nằm **ngoài thời gian analytic được so sánh**. Incremental update chỉ nhận batch hiện tại và trạng thái; log cache/oracle riêng khi báo memory.

Ở mode benchmark không chạy oracle, các error/disagreement fields phải null với `joint_reference_status="not_run"`; không ghi0 để ám chỉ đã so khớp. `checks` và verification_passed phải dựa trên những điều kiện bắt buộc của mode đã khai báo, không gọi check chưa chạy là passed. Đó là hợp đồng của benchmark runner cần triển khai; smoke hiện tại vẫn chạy oracle.

## 7. Thời gian và bộ nhớ

Tách `base_training_seconds`, `feature_extraction_seconds`, `base_analytic_seconds`, `incremental_analytic_seconds`, `evaluation_seconds`, `tracking_seconds` và `pipeline_seconds`. Không suy GPU time từ CPU time. Khi đo đoạn CUDA phải đồng bộ trước/sau; NumPy analytic hiện ở CPU. Thời gian notebook gồm cài dependency/download/tests khác thời gian pipeline JSON.

R3 báo wall time analytic trên cùng host/threads, 3 repeats và thứ tự batch-size cố định32→128→512 trong mỗi repeat; mỗi cấu hình khởi tạo lại. Lưu từng lần đo rồi mean/sample_SD, không chỉ số nhanh nhất. Tách feature extraction và oracle khỏi cửa sổ timing.

`persistent_analytic_state_bytes = R.nbytes + W.nbytes + W_fe.nbytes` với FE; no-FE bỏ W_fe. Báo thêm backbone state bytes riêng. Peak CPU RSS và peak CUDA allocation phải được đo ở khoảng thực thi được nêu rõ; kích thước ma trận không thay thế peak memory. Không đếm cache diagnostic vào thuật toán mà không chú thích, và cũng không bỏ nó khỏi peak thực tế khi cùng nằm trong process.

## 8. Schema summary và điều kiện xuất bản

Summary dẫn tới JSON gốc bằng checksum, giữ config/source/version/attempt/W&B; không sửa JSON lịch sử chỉ để thêm số suy ra. Tên metric chốt:

```json
{
  "protocol_version": "ACIL-C100-v1.0",
  "metric_version": "ACIL-metrics-v1.0",
  "mode": "smoke",
  "seed": 42,
  "n_incremental_phases": 5,
  "n_evaluations": 6,
  "average_incremental_accuracy_pct": 26.910462962962963,
  "final_accuracy_pct": 22.05,
  "base_forgetting_pp": 7.900000000000002,
  "base_accuracy_delta_pp": -7.900000000000002,
  "n_completed_seeds": 1,
  "between_seed_sd": null
}
```

Đây là **ví dụ summary suy ra của smoke**, không phải file mới đã được runner xuất. `mode` benchmark/ablation/r3 phải có định danh và provenance riêng; không đổi tên run2 epochs thành benchmark.

Một summary hoàn chỉnh chỉ được xuất khi source report status complete, verification true, đủ phase0..K, đúng số lớp/test/shape, các A/B hữu hạn trong[0,100] và tracking thật của thí nghiệm đã hoàn tất. Root/latest/progress/results phải cùng attempt ID khi chọn artifact của pipeline mới. Run thiếu điểm thì báo partial; không tính average K đầy đủ từ các điểm còn lại.

Định dạng bảng/deck: phase accuracies2 chữ số thập phân, A-bar/F2 chữ số, relative numerical error dạng khoa học; giữ JSON đầy đủ precision. Mọi số nhóm có W&B link và source/result checksum. Số paper có nguồn/table/setting; các sai khác class order, normalization, shortcut, FE initialization và environment dẫn lại [protocol](protocol_chot.md).
