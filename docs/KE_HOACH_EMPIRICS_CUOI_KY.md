# Kế hoạch Thực nghiệm Cuối kỳ (Pre-registered Protocol v1.0) — ACIL

- **Cơ sở lý thuyết đối chiếu:** [`docs/PHAN_BIEN_CUOI_KY_ACIL.md`](PHAN_BIEN_CUOI_KY_ACIL.md) (chốt ngày 18/11/2026).
- **Phạm vi tài liệu:** Khóa trước toàn bộ thiết kế thực nghiệm cuối kỳ (**Empirical Demonstration**) kiểm chứng hai hạn chế lý thuyết trọng tâm **`P1`** (qua tổ hợp thực nghiệm **`E1a`**, **`E1b`**) và **`P2`** (qua tổ hợp thực nghiệm **`E2a`**, **`E2b`** và nguyên mẫu thay thế **`FD2`**) trước khi chạy benchmark đầy đủ.
- **Trạng thái lúc chốt tài liệu:** Đã khóa giao thức, mã nguồn runner và bộ kiểm thử đơn vị (`unit tests`); **chưa khởi chạy các lượt đo đầy đủ để lấy số liệu báo cáo**.
- **Cập nhật sau khi hoàn tất chạy thực nghiệm (18/11/2026):** Toàn bộ nội dung kế hoạch bên dưới được giữ nguyên theo bản khóa trước khi chạy; kết quả thực chạy 3 seeds (`42, 43, 44`), 18 runs W&B công khai và biểu đồ Poster được công bố tại [`reports/E2_MEMORY_PRECISION_20261118.md`](../reports/E2_MEMORY_PRECISION_20261118.md) ([manifest `E2`](../reports/E2-FINAL-EMPIRICS-provenance.json)) và [`reports/E1_BASE_ASSUMPTION_20261118.md`](../reports/E1_BASE_ASSUMPTION_20261118.md) ([manifest `E1`](../reports/E1-FINAL-EMPIRICS-provenance.json)).

---

## 1. Nguyên tắc Bảo toàn Lịch sử & Quản trị Tái lập

1. **Bất biến tuyệt đối với phần Giữa kỳ (`R1–R3` & đối chiếu code tác giả):**
   - Không chỉnh sửa bất kỳ tệp nào trong `src/*.py`, `configs/*.json`, các script điều phối/vẽ hình của `R1–R3` hay các tệp báo cáo và `*-provenance.json` đã khóa mã băm SHA-256 trong `reports/`.
   - Dấu vân tay `provenance["source_sha256"]` giữ nguyên theo `HEAD` hiện tại (sau đợt bổ sung docstring 3 phần cho `src/` ngày 11/11/2026), đồng thời các checkpoint `R1` tái dùng trong `E2` được xác minh trực tiếp qua cả `checkpoint_sha256` của tệp lẫn `backbone_sha256` của toàn bộ trọng số và BatchNorm.
2. **Tách module thực thi riêng trong `scripts/`:**
   - Thực nghiệm `E1` (`E1a`, `E1b`): [`scripts/e1_base_assumption_runner.py`](../scripts/e1_base_assumption_runner.py).
   - Thực nghiệm `E2` (`E2a`, `E2b`, `FD2`): [`scripts/e2_memory_precision_runner.py`](../scripts/e2_memory_precision_runner.py).
   - Bộ kiểm thử đơn vị kiểm tra tính đúng trước khi chạy: [`scripts/tests/test_final_empirics.py`](../scripts/tests/test_final_empirics.py).
3. **Cách ly lượt chạy và đồng bộ W&B minh bạch:**
   - Mỗi cấu hình ghi kết quả vào một thư mục con độc lập qua `src.run_attempt.RunAttempt` (`status: running -> complete | failed`, ghi JSON nguyên tử chuẩn UTF-8/LF).
   - Toàn bộ lượt chạy cuối kỳ đẩy lên cùng project công khai [`hsangg-2006/acil-reproduction`](https://wandb.ai/hsangg-2006/acil-reproduction) nhưng bắt buộc gắn nhãn **`final-empirics`** kèm nhãn vấn đề riêng (**`P1-base-assumption`** hoặc **`P2-memory-precision`**) để tách bạch hoàn toàn với các bảng `R1`, `R2`, `R3` của giữa kỳ.
4. **Quy tắc thống kê khóa trước:**
   - Các phép đo so sánh độ chính xác tổng quát hóa báo cáo theo **`mean ± sample SD`** trên **3 seeds độc lập (`42, 43, 44`)** với bậc tự do hiệu chỉnh `ddof = 1`.
   - Không chọn lọc seed có kết quả thuận lợi, không điều chỉnh lại danh sách nhóm lớp hay lưới chiều $d$, hệ số $\gamma$ sau khi quan sát kết quả trên tập `test`.

---

## 2. Thiết kế Chi tiết Thực nghiệm `E1` — Kiểm chứng Hạn chế `P1` (Giả định Tập Base Đóng băng)

### 2.1. Bảng khóa cấu hình `E1a` và `E1b`

| Hạng mục khóa trước | Nhánh `E1a` — Thu hẹp quy mô lớp Base (*Base-Class Ratio Sweep*) | Nhánh `E1b` — Dịch chuyển miền ngữ nghĩa (*Semantic Superclass Shift*) |
|---|---|---|
| **1. Giả định lý thuyết bị kiểm tra** | **Giả định A1 (phần quy mô):** Tập lớp Base $\mathcal{C}_0$ chiếm $50\%$ tổng số lớp là đủ rộng để backbone đóng băng $\theta_0$ tạo không gian đặc trưng 64 chiều phân tách tuyến tính được các lớp đến sau qua phép chiếu ngẫu nhiên $W_{\mathrm{fe}}$. | **Giả định A1 (phần đồng phân phối):** Tập lớp Base $\mathcal{C}_0$ có độ phủ ngữ nghĩa đồng đều với các lớp tăng dần $\mathcal{C}_{1:K}$, nên các bộ lọc tích chập học ở Phase 0 tổng quát hóa sang các pha sau mà không cần cập nhật đạo hàm. |
| **2. Biến độc lập thay đổi** | Số lượng lớp Base ở Phase 0: $c_0 \in \{10, 20, 30, 50\}$. Các lớp còn lại $(100 - c_0)$ luôn vào thêm theo từng pha cố định **10 lớp/phase**, tương ứng $K \in \{9, 8, 7, 5\}$ pha tăng dần (tổng số mốc đánh giá $K + 1 \in \{10, 9, 8, 6\}$). | Quy tắc phân bổ 20 nhóm ngữ nghĩa lớn (*coarse superclasses*) của CIFAR-100 vào thiết lập chuẩn **$c_0 = 50$ lớp Base + $K = 5 \times 10$ lớp mới**:<br>• `uniform` (hoán vị ngẫu nhiên đồng nhất)<br>• `stratified` (mỗi superclass đóng góp 2–3 lớp vào Base và 2–3 lớp vào Incremental)<br>• `superclass_disjoint` (10 superclass Tự nhiên/Sinh vật ở Base $\to$ 10 superclass Đồ vật nhân tạo/Phương tiện & nhóm còn lại ở Incremental; hỗ trợ thêm chiều ngược `superclass_disjoint_reverse`). |
| **3. Biến kiểm soát cố định** | • Tập dữ liệu CIFAR-100 pin revision `aadb3af77e9048adbea6b47c21a81e47dd092ae5` (`500` train / `100` test mỗi lớp).<br>• Trên từng seed (`42, 43, 44`), giữ **cùng một vector hoán vị `class_order`** của 100 lớp; mọi cấu hình đều kết thúc trên **đúng cùng 100 lớp và 10,000 ảnh test** ở pha cuối.<br>• Kiến trúc ResNet-32 (`classifier = Linear(64, c_0)` ở Phase 0), SGD 160 epochs (`lr=0.1`, milestones `[80, 120]`, `momentum=0.9`, `weight_decay=5e-4`, `batch_size=128`), chuẩn hóa `mean/std` tính **chỉ từ $500 \cdot c_0$ ảnh train của Phase 0**.<br>• Tầng giải tích: $d = 8000$ (`RNG seed + 1000`), $\gamma = 0.1$, minibatch $b = 128$, `float64`, 4 luồng CPU/BLAS. | • Giữ nguyên quy mô dữ liệu ở mọi pha như bài báo gốc: Phase 0 luôn có **đúng 50 lớp (`25,000` ảnh train)**, mỗi pha $1\dots 5$ có **đúng 10 lớp (`5,000` ảnh train)**.<br>• Cùng kiến trúc ResNet-32 (`Linear(64, 50)`), cùng lịch huấn luyện SGD 160 epochs, chuẩn hóa `base_train_only` trên đúng 25,000 ảnh Phase 0.<br>• Cùng tầng giải tích $d = 8000$, $\gamma = 0.1$, minibatch $b = 128$, `float64`, 3 seeds (`42, 43, 44`). |
| **4. Đối chứng chuẩn (*Control*)** | Cấu hình chuẩn bài báo **$c_0 = 50, K = 5$** (`uniform`): đối chiếu trực tiếp với 3 runs `R1` đã nghiệm thu ($\bar{A} = 65.12 \pm 0.61\%$, $A_{\mathrm{final}} = 56.13 \pm 0.56\%$). | Cấu hình đồng phân phối ngữ nghĩa: `uniform` (3 runs `R1` chuẩn) và `stratified` (50 lớp Base phủ đủ cả 20/20 coarse superclasses). |
| **5. Hệ thống Metric & Đơn vị thống kê** | Báo `mean ± SD` ($n = 3$ seeds, `ddof = 1`):<br>• $A_{\mathrm{final}}$ (`final_accuracy_pct`, trên đủ 100 lớp ở pha cuối — thước đo đối chiếu ngang bằng chính)<br>• $B_{\mathrm{final}}$ (`final_base_accuracy_pct`, trên `labels < c_0`)<br>• $A_{\mathrm{inc}}$ (`final_inc_accuracy_pct`, trên toàn bộ $100 - c_0$ lớp học giải tích `labels >= c_0`)<br>• Khoảng lệch chuyển giao $\Delta_{\text{gap}} = B_{\mathrm{final}} - A_{\mathrm{inc}}$ (pp)<br>• $\bar{A}$ (`average_incremental_accuracy_pct`, ghi kèm chú thích số mốc $K+1$ khác nhau giữa các mức $c_0$). | Báo `mean ± SD` ($n = 3$ seeds, `ddof = 1`) trên cùng lưới 6 mốc đánh giá (`50, 60, 70, 80, 90, 100` lớp):<br>• $\bar{A}$ (`average_incremental_accuracy_pct`) và $A_{\mathrm{final}}$ (100 lớp)<br>• $B_5$ (50 lớp Base) và $A_{\mathrm{inc}}$ (50 lớp Incremental ở Phase 5)<br>• Độ chênh chuyển giao $\Delta_{\text{gap}} = B_5 - A_{\mathrm{inc}}$ (pp)<br>• Phân rã chẩn đoán nội bộ $A_{\mathrm{inc}}$: độ chính xác trên **30 lớp đồ vật nhân tạo/phương tiện thuần túy** (`inc_manmade_accuracy_pct`) so với **20 lớp sinh học/con người còn lại** (`inc_biological_accuracy_pct`). |
| **6. Kết quả dự đoán & Tiêu chí bác bỏ** | • **Dự đoán:** Khi hạ $c_0: 50 \to 30 \to 20 \to 10$, dù nghiệm đệ quy vẫn khớp nghiệm Joint Ridge (`relative_w_error` $\approx 10^{-12}$ khi kiểm oracle), $A_{\mathrm{final}}$ và $A_{\mathrm{inc}}$ giảm đơn điệu rõ rệt (dự kiến $A_{\mathrm{final}}$ giảm $\ge 6\text{–}15\text{ pp}$ tại $c_0 = 10$ so với $c_0 = 50$), đồng thời $\Delta_{\text{gap}}$ nới rộng mạnh.<br>• **Bác bỏ giả thuyết `P1` (nhánh quy mô) nếu:** $A_{\mathrm{final}}$ tại $c_0 = 10$ chênh lệch dưới $2.0\text{ pp}$ so với $c_0 = 50$. | • **Dự đoán:** Dù cùng học từ 25,000 ảnh của 50 lớp ở Phase 0, chế độ `superclass_disjoint` làm giảm $A_{\mathrm{inc}}$ và $A_{\mathrm{final}}$ so với `uniform`/`stratified`, đẩy $\Delta_{\text{gap}} = B_5 - A_{\mathrm{inc}}$ tăng thêm rõ rệt; trong đó nhóm 30 lớp phương tiện/đồ vật nhân tạo chịu mức sụt mạnh hơn nhóm sinh học gần miền.<br>• **Bác bỏ giả thuyết `P1` (nhánh ngữ nghĩa) nếu:** $A_{\mathrm{inc}}$ và $\Delta_{\text{gap}}$ của `superclass_disjoint` xấp xỉ `stratified` (chênh lệch $< 1.0\text{ pp}$). |

### 2.2. Phân hoạch tường minh 20 Coarse Superclasses của CIFAR-100 cho `E1b`

Dữ liệu Parquet gốc của `uoft-cs/cifar100` (`aadb3af77e9048adbea6b47c21a81e47dd092ae5`) định nghĩa chính xác 20 nhóm lớp thô (`coarse_label` `0..19`), mỗi nhóm chứa đúng 5 lớp chi tiết (`fine_label` `0..99`). Để tránh mọi tùy tiện khi chia miền, hai tập 10 superclass rời nhau được khóa cứng như sau:

- **Tập miền Tự nhiên, Thực vật & Động vật (`Group A` — 10 superclasses = 50 lớp fine, dùng cho Phase 0 của `superclass_disjoint`):**
  - `0: aquatic_mammals` (`4, 30, 55, 72, 95`), `1: fish` (`1, 32, 67, 73, 91`), `2: flowers` (`54, 62, 70, 82, 92`), `4: fruit_and_vegetables` (`0, 51, 53, 57, 83`), `7: insects` (`6, 7, 14, 18, 24`), `8: large_carnivores` (`3, 42, 43, 88, 97`), `10: large_natural_outdoor_scenes` (`23, 33, 49, 60, 71`), `11: large_omnivores_and_herbivores` (`15, 19, 21, 31, 38`), `12: medium_mammals` (`34, 63, 64, 66, 75`), `17: trees` (`47, 52, 56, 59, 96`).
- **Tập miền Đồ vật nhân tạo, Phương tiện & Nhóm ngoại miền (`Group B` — 10 superclasses = 50 lớp fine, dùng cho Phases `1..5` của `superclass_disjoint`):**
  - *6 nhóm đồ vật cứng / công trình / phương tiện nhân tạo thuần túy (30 lớp fine):* `3: food_containers` (`9, 10, 16, 28, 61`), `5: household_electrical_devices` (`22, 39, 40, 86, 87`), `6: household_furniture` (`5, 20, 25, 84, 94`), `9: large_man-made_outdoor_things` (`12, 17, 37, 68, 76`), `18: vehicles_1` (`8, 13, 48, 58, 90`), `19: vehicles_2` (`41, 69, 81, 85, 89`).
  - *4 nhóm con người & nhánh sinh vật chưa xuất hiện ở Phase 0 (20 lớp fine):* `13: non-insect_invertebrates` (`26, 45, 77, 79, 99`), `14: people` (`2, 11, 35, 46, 98`), `15: reptiles` (`27, 29, 44, 78, 93`), `16: small_mammals` (`36, 50, 65, 74, 80`).

Thứ tự xuất hiện cụ thể của 50 lớp trong nội bộ Phase 0 và 50 lớp qua Phases `1..5` được hoán vị xác định bằng `np.random.default_rng(seed)` cho từng `seed in {42, 43, 44}`. Khi chạy trên dữ liệu thật CIFAR-100, runner tự động đối chiếu bảng ánh xạ `fine_label -> coarse_label` trực tiếp từ tệp Parquet trong cache với bảng hằng số trên trước khi huấn luyện.

---

## 3. Thiết kế Chi tiết Thực nghiệm `E2` — Kiểm chứng Hạn chế `P2` (Ngân sách Bộ nhớ Byte, Độ chính xác Số học & Nguyên mẫu `FD2`)

### 3.1. Bảng khóa cấu hình `E2a` (kèm `FD2`) và `E2b`

| Hạng mục khóa trước | Nhánh `E2a` (+ `FD2`) — Đường cong Đánh đổi theo Ngân sách Byte (*Iso-Memory Budget Sweep*) | Nhánh `E2b` — Độ nhạy Số học của Woodbury (`float32` vs `float64`) |
|---|---|---|
| **1. Giả định lý thuyết bị kiểm tra** | **Giả định A4:** Kích thước trạng thái $R_k \in \mathbb{R}^{d \times d}$ là chi phí cố định nhỏ và kết quả của ACIL ở $d=8000$ giữ nguyên giá trị khi đặt cạnh các đối thủ lưu tối đa $M = 2{,}000$ ảnh mẫu trong thiết lập *Strict-Memory* ([1-Supp], Bảng A). | **Giả định A3:** Công thức đệ quy Woodbury (Phương trình 10) duy trì tương đương đại số với học gộp Ridge trên số học dấu phẩy động hữu hạn khi hạ kiểu lưu trữ từ `float64` xuống `float32` để giảm $50\%$ bộ nhớ, kể cả khi $\gamma$ nhỏ hoặc số bước cập nhật lớn. |
| **2. Biến độc lập thay đổi** | • **Lưới ACIL chuẩn:** Chiều mở rộng $d \in \{64\text{ (No-FE)}, 128, 256, 512, 768, \mathbf{798}, 1024, \mathbf{1160}, 2048, 4096, 8000\}$.<br>• **Nguyên mẫu thay thế `FD2` (*Block-Diagonal RFAM*):** Tại $d = 8000$, chia không gian đặc trưng mở rộng thành $B \in \{16, 80\}$ khối đường chéo độc lập cỡ $m = d / B \in \{500, 100\}$. | • Kiểu dấu phẩy động của toàn bộ luồng tính toán analytic: `dtype` $\in \{\text{float32}, \text{float64}\}$.<br>• Hệ số điều chuẩn Ridge: $\gamma \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}\}$.<br>• Chế độ chia bước cập nhật Woodbury trên 50 lớp tăng dần: `(K=5, b=128)` (`200` bước RLS) so với `(K=25, b=16)` (`1,575` bước RLS, tương ứng thiết lập $K=25$ ở mục 4.3 bài báo). |
| **3. Biến kiểm soát cố định** | • **Tái sử dụng nguyên vẹn 3 checkpoints ResNet-32 160 epochs của `R1`** (`seeds 42, 43, 44`, kiểm cứng SHA-256 từ `reports/R1-SEED*-KAGGLE-01-provenance.json`, khóa cứng không gọi `train_base`).<br>• Trích xuất đặc trưng 64 chiều `F64` của đủ 50,000 ảnh train và 10,000 ảnh test một lần cho mỗi seed vào fixture có kiểm SHA-256.<br>• Cố định $\gamma = 0.1$, $K = 5$, minibatch $b = 128$, `float64`, `4` luồng CPU/BLAS (`threadpool_limits`), ma trận $W_{\mathrm{fe}}$ sinh từ `default_rng(seed + 1000)`. | • Dùng cùng fixture đặc trưng `F64` đã kiểm SHA-256 của checkpoint `R1` (`seed 42` làm hạt nhân khảo sát chi tiết; có thể lặp kiểm chứng trên `seeds 43, 44` ở chiều $d = 1024$).<br>• Cố định chiều mở rộng $d = 1024$ (và kiểm tra bổ sung tại $d = 2048/4096$ nếu cần), cùng ma trận gốc $W_{\mathrm{fe}}$ ép kiểu tường minh sang `dtype` đích trước khi nhân.<br>• Chặn mọi phép tự động nâng kiểu ngầm (*silent upcasting*) của NumPy sang `float64` trong luồng `float32`. |
| **4. Đối chứng chuẩn (*Control*)** | • Mốc trên không ràng buộc bộ nhớ: $d = 8000$ (`522.50 MB` `float64`, khớp tuyệt đối kết quả `R1`).<br>• Mốc dưới không mở rộng đặc trưng: $d = 64$ No-FE (`0.084 MB`, khớp tuyệt đối `R2` `no_fe_g0p1`).<br>• **Hai điểm neo công bằng theo đúng ngân sách `6,144,000 bytes` của 2,000 ảnh CIFAR-100:** $d_{\max}^{(\text{fp64})} = 798$ (`6.141 MB` ở `float64`) và $d_{\max}^{(\text{fp32})} = 1160$ (`6.143 MB` ở `float32`). | Nghiệm chuẩn **Joint Ridge Oracle `float64`** giải trực tiếp hệ phương trình tuyến tính $(\tilde{H}_k^T \tilde{H}_k + \gamma I_d) \tilde{W}_k = \tilde{H}_k^T \tilde{Y}_k$ trên toàn bộ dữ liệu đã thấy tại từng phase. |
| **5. Hệ thống Metric & Đơn vị thống kê** | Báo `mean ± SD` ($n = 3$ seeds, `ddof = 1`):<br>• Dung lượng trạng thái lưu trữ chính xác `persistent_state_bytes` (Byte / MB / MiB) ở cả `float64` và quy đổi `float32`, cùng tỷ số so với kho 2,000 ảnh gốc (`6.144 MB`)<br>• Đỉnh bộ nhớ tiến trình quan sát trong cửa sổ học `peak_learning_rss_bytes` và mức tăng RSS<br>• Thời gian học giải tích thuần túy `learning_seconds = base_seconds + incremental_seconds`<br>• $\bar{A}$, $A_{\mathrm{final}}$, $B_5$, $A_{\mathrm{inc}}$ và $F_{\mathrm{base}}$. | Trên từng tổ hợp `(dtype, gamma, K, batch_size)`:<br>• Sai số tương đối trọng số cực đại qua các phase: `max_relative_w_error` $= \max_k \|W_k - \tilde{W}_k^{(\text{fp64})}\|_F / \|\tilde{W}_k^{(\text{fp64})}\|_F$<br>• Sai số tương đối ma trận $R_k$: `max_relative_r_error`<br>• Độ lệch đối xứng: `max_symmetry_frobenius_error` $= \max_k \|R_k - R_k^T\|_F$<br>• Số mẫu test bị đổi nhãn $\operatorname{argmax}$ so với Joint Oracle `float64`: `final_prediction_disagreements` (trên 10,000 ảnh) và tổng qua mọi phase<br>• Độ lệch $\Delta\bar{A}$ và $\Delta A_{\mathrm{final}}$ (pp) so với Joint Oracle `float64`. |
| **6. Kết quả dự đoán & Tiêu chí bác bỏ** | • **Dự đoán `E2a`:** Tại ngưỡng ngang bằng bộ nhớ byte với 2,000 ảnh CIFAR-100 ($d = 798$ ở `float64` hoặc $d = 1160$ ở `float32`), $\bar{A}$ thấp hơn cấu hình $d = 8000$ khoảng $1.8\text{–}3.0\text{ pp}$ và $A_{\mathrm{final}}$ thấp hơn khoảng $3.5\text{–}5.5\text{ pp}$.<br>• **Dự đoán `FD2`:** Xấp xỉ khối đường chéo ($d=8000, B=16$) giảm $16\times$ bộ nhớ của $R_k$ (`512 MB -> 32 MB`) nhưng giữ được phần lớn lợi ích của 8,000 chiều chiếu ngẫu nhiên tốt hơn cắt cụt thẳng $d$ về cùng dung lượng.<br>• **Bác bỏ giả thuyết `P2` (nhánh bộ nhớ) nếu:** ACIL tại $d = 798$ đạt $\bar{A}$ và $A_{\mathrm{final}}$ chênh lệch dưới $0.5\text{ pp}$ so với $d = 8000$. | • **Dự đoán `E2b`:** Ở `float64`, `max_relative_w_error` luôn giữ ở mức $\le 10^{-9}$ với `0` mẫu lệch dự đoán trên mọi mức $\gamma$. Ở `float32`, khi $\gamma$ giảm về $10^{-3}\text{–}10^{-4}$ và đặc biệt ở chế độ `1,575` bước `(K=25, b=16)`, sai số tích lũy làm `max_relative_w_error` tăng vọt nhiều bậc độ lớn ($10^{-3}\text{–}10^{0}$), phát sinh bất đồng dự đoán so với nghiệm học gộp hoặc suy biến xác định dương.<br>• **Bác bỏ giả thuyết `P2` (nhánh số học) nếu:** `float32` vẫn giữ `0` bất đồng dự đoán và `max_relative_w_error` $< 10^{-5}$ ngay cả tại $\gamma = 10^{-4}, K = 25, b = 16$. |

### 3.2. Bảng định lượng chính xác ngân sách Byte trên lưới quét `E2a` & `FD2`

Với $d_{\text{cnn}} = 64$, $C_5 = 100$ lớp ở pha cuối và kho mẫu đối chiếu $M = 2{,}000 \times 32 \times 32 \times 3 = \mathbf{6{,}144{,}000\text{ bytes}}$ (`6.144 MB` hệ thập phân / `5.859 MiB` hệ nhị phân):

| Cấu hình mô hình | Số phần tử trạng thái $(R_k, W_{\mathrm{fe}}, W_5)$ | Dung lượng `float64` (`8 B/số`) | Dung lượng `float32` (`4 B/số`) | Tỷ số Byte (`float64` / `2,000` ảnh) | Vai trò trong thực nghiệm `E2a` |
|---|---:|---:|---:|---:|---|
| `d = 64` (No-FE) | `10,496` | `0.084 MB` (`0.080 MiB`) | `0.042 MB` (`0.040 MiB`) | `0.01x` | Cận dưới không mở rộng đặc trưng (khớp `R2`) |
| `d = 128` | `37,376` | `0.299 MB` (`0.285 MiB`) | `0.150 MB` (`0.143 MiB`) | `0.05x` | Miền tiết kiệm bộ nhớ cực nhỏ |
| `d = 256` | `107,520` | `0.860 MB` (`0.820 MiB`) | `0.430 MB` (`0.410 MiB`) | `0.14x` | Ngang tầm $\sim 280$ ảnh CIFAR-100 |
| `d = 512` | `346,112` | `2.769 MB` (`2.641 MiB`) | `1.384 MB` (`1.320 MiB`) | `0.45x` | Dưới một nửa ngân sách kho mẫu |
| `d = 768` | `715,776` | `5.726 MB` (`5.461 MiB`) | `2.863 MB` (`2.730 MiB`) | `0.93x` | Điểm lưới chẵn sát dưới trần `float64` |
| **`d = 798` (Iso-Byte fp64)** | **`767,676`** | **`6.141 MB` (`5.857 MiB`)** | `3.071 MB` (`2.928 MiB`) | **`0.9996x`** | **Trần chính xác cùng ngân sách `2,000` ảnh ở `float64`** |
| `d = 1024` | `1,216,512` | `9.732 MB` (`9.281 MiB`) | `4.866 MB` (`4.641 MiB`) | `1.58x` | Điểm lưới chẵn đối chiếu `R3` (`4.87 MB` ở `fp32`) |
| **`d = 1160` (Iso-Byte fp32)** | **`1,535,840`** | `12.287 MB` (`11.718 MiB`) | **`6.143 MB` (`5.859 MiB`)** | `2.00x` (`0.9999x` fp32) | **Trần chính xác cùng ngân sách `2,000` ảnh ở `float32`** |
| `d = 2048` | `4,530,176` | `36.241 MB` (`34.562 MiB`) | `18.121 MB` (`17.281 MiB`) | `5.90x` | Miền chuyển tiếp bậc trung |
| `d = 4096` | `17,448,960` | `139.592 MB` (`133.125 MiB`) | `69.796 MB` (`66.562 MiB`) | `22.72x` | Gấp $\sim 23\times$ kho mẫu |
| `d = 8000` (Chuẩn bài báo) | `65,312,000` | `522.496 MB` (`498.291 MiB`) | `261.248 MB` (`249.146 MiB`) | **`85.04x`** | Cấu hình Bảng 1 & Bảng A [1-Supp] (khớp `R1`) |
| `FD2: d = 8000, B = 16` (`m=500`) | `5,312,000` | `42.496 MB` (`40.527 MiB`) | `21.248 MB` (`20.264 MiB`) | `6.92x` | Giảm $16\times$ ma trận $R_k$ (`512 -> 32 MB`) |
| `FD2: d = 8000, B = 80` (`m=100`) | `2,112,000` | `16.896 MB` (`16.113 MiB`) | `8.448 MB` (`8.057 MiB`) | `2.75x` | Giảm $80\times$ ma trận $R_k$ (`512 -> 6.4 MB`) |

### 3.3. Quy tắc vùng đo thời gian và bộ nhớ trong `E2`

Tuân thủ cùng kỷ luật đo của `docs/R3_KE_HOACH_DO.md`:
- Bước nạp checkpoint ResNet-32 và trích xuất đặc trưng 64 chiều `F64` chạy tách riêng trước vòng quét và không cộng vào `learning_seconds` của tầng giải tích.
- Trước mỗi cấu hình đo, thực hiện bước làm nóng BLAS/LAPACK nhỏ cố định (`32 × 32`) và thu gom rác bộ nhớ (`gc.collect()`).
- Chỉ các lệnh biến đổi đặc trưng mở rộng `ReLU(F @ W_fe)`, tích lũy/giải hệ ở Phase 0 và cập nhật Woodbury qua các phase tăng dần nằm trong ngữ cảnh `LearningWindow` (đo `perf_counter` và lấy mẫu RSS định kỳ mỗi `5 ms`). Bước đánh giá trên tập test, giải nghiệm đối chiếu Joint Oracle và lưu JSON/W&B nằm hoàn toàn ngoài cửa sổ đo.

---

## 4. Định dạng Tệp Đầu ra, Nhãn W&B và Danh mục Kiểm thử Trước Khi Chạy

### 4.1. Bảng ánh xạ Nhãn W&B và Đầu ra JSON

| Nhánh thí nghiệm | Script điều phối | Tiền tố Run W&B | Bộ W&B Tags bắt buộc | Tệp tổng hợp dự kiến (sau khi chạy ở Ưu tiên 3) |
|---|---|---|---|---|
| **`E1a`** (Base Ratio) | `scripts/e1_base_assumption_runner.py` | `E1a-c0_{10,20,30}-seed{s}` | `["final-empirics", "P1-base-assumption", "E1a-base-ratio", "seed{s}"]` | `reports/e1_base_assumption_summary.json` |
| **`E1b`** (Superclass Shift) | `scripts/e1_base_assumption_runner.py` | `E1b-{mode}-seed{s}` | `["final-empirics", "P1-base-assumption", "E1b-superclass-shift", "{mode}", "seed{s}"]` | `reports/e1_base_assumption_summary.json` |
| **`E2a` & `FD2`** (Memory Sweep) | `scripts/e2_memory_precision_runner.py` | `E2a-d{d}-seed{s}` / `FD2-d8000-B{B}-seed{s}` | `["final-empirics", "P2-memory-precision", "E2a-memory-budget", "seed{s}"]` | `reports/e2_memory_precision_summary.json` |
| **`E2b`** (Woodbury `fp32/fp64`) | `scripts/e2_memory_precision_runner.py` | `E2b-{dtype}-g{gamma}-K{K}b{b}-seed{s}` | `["final-empirics", "P2-memory-precision", "E2b-woodbury-precision", "{dtype}", "seed{s}"]` | `reports/e2_memory_precision_summary.json` |

### 4.2. Danh mục Kiểm thử Đơn vị Bắt buộc (`scripts/tests/test_final_empirics.py`)

Mọi kiểm thử dưới đây phải vượt qua 100% trước khi phát lệnh chạy thực nghiệm dài trên GPU/CPU:

1. **Kiểm tra giao thức & chống rò rỉ thống kê của `E1a`:**
   - Với mọi $c_0 \in \{10, 20, 30, 50\}$, các phase rời nhau từng đôi một, hợp lại phủ đúng `0..99`, giữ nguyên `class_order` của cùng một seed và khớp hệt `src.cifar_data.make_protocol` tại $c_0 = 50$.
   - Thống kê chuẩn hóa `mean/std` chỉ phụ thuộc vào các mẫu `labels < c_0` (khi thay đổi giá trị điểm ảnh của các lớp tương lai `>= c_0` hoặc tập `test`, `mean/std` không đổi đến từng bit).
2. **Kiểm tra phân hoạch ngữ nghĩa `E1b` & thực thi smoke trọn vòng:**
   - Ở `superclass_disjoint`, tập 10 coarse superclasses của Phase 0 và tập 10 coarse superclasses của Phases `1..5` giao nhau bằng rỗng (`shared_superclass_count == 0`, mỗi bên đúng 50 lớp fine); ở `stratified`, cả 20/20 superclasses đều hiện diện ở cả hai phía (`shared_superclass_count == 20`).
   - Chạy thử trọn vòng pipeline `E1` trên tập ảnh giả lập thu nhỏ (2 epochs BP + cập nhật giải tích + kiểm chứng đối chiếu Joint Oracle): xác nhận `RunAttempt` ghi trạng thái `complete`, trọng số backbone không đổi qua các pha tăng dần, `prediction_disagreements == 0` so với Joint Ridge và tính đúng các metric tổng quát ($B_k, A_{\mathrm{inc}}, \Delta_{\text{gap}}$, phân rã `manmade` vs `biological`).
3. **Kiểm tra công thức ngân sách Byte, bảo vệ checkpoint `R1` & nguyên mẫu `FD2` của `E2a`:**
   - Hàm tính `state_bytes` khớp chính xác từng byte với thuộc tính `.nbytes` thực tế của các mảng NumPy ở mọi cấu hình (`No-FE 64`, `798`, `1160`, `8000` và `BlockDiagonalStreamingHead`).
   - Khi đặt số khối $B = 1$, `BlockDiagonalStreamingHead` cho ra ma trận $R_k$, $W_k$ và điểm dự đoán trùng khớp với `StreamingHead` chuẩn tới cấp độ chính xác máy (`rtol=1e-12, atol=1e-12`); khi $B > 1$, dung lượng lưu trữ của $\hat{R}_k$ giảm đúng $B$ lần và nghiệm trong từng khối con khớp nghiệm Joint Ridge của khối đó.
   - Chặn tuyệt đối hành vi chạy khi thiếu checkpoint hoặc sai mã băm SHA-256 của `R1` (không bao giờ tự động huấn luyện lại mạng CNN).
4. **Kiểm tra tính chặt kiểu dữ liệu `float32` vs `float64` của `E2b`:**
   - Xác nhận luồng `float32` giữ đúng `dtype == np.float32` trên mọi mảng trung gian và trạng thái đầu ra `(R, W)` ở từng minibatch (không bị nâng kiểu ngầm lên `float64`).
   - Trên bài toán có điều kiện xấu ($\gamma = 10^{-4}$, nhiều bước cập nhật nhỏ), luồng `float64` vẫn bám sát nghiệm Joint Oracle (`relative_w_error` $< 10^{-9}$), trong khi luồng `float32` bộc lộ độ lệch trọng số lớn hơn nhiều bậc độ lớn đúng như dự đoán giải tích số.
