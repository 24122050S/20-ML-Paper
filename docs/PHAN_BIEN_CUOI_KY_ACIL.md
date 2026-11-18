# Phân tích Phản biện Lý thuyết & Khung Thực nghiệm Cuối kỳ — ACIL (NeurIPS 2022)

- **Bài báo phân tích:** H. Zhuang, Z. Weng, H. Wei, R. Xie, K.-A. Toh, and Z. Lin, *"ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection"*, NeurIPS, vol. 35, pp. 11602–11614, 2022 [1] (kèm Tài liệu bổ sung chính thức — *Supplementary Material* [1-Supp]).
- **Vai trò của tài liệu:** Chốt khung lập luận lý thuyết cho giai đoạn Cuối kỳ (**Critical Analysis** & **Empirical Demonstration**), bao gồm:
  1. Phân tích 4 góc nhìn lý thuyết của bài báo (giả định ẩn, độ vững của tuyên bố, mô hình hóa & hàm mục tiêu, độ phức tạp & tính tổng quát);
  2. Hệ thống hóa **3 Điểm mạnh (Strengths)** đi kèm bằng chứng định lượng từ quá trình đọc hiểu và tái lập (`R1`, `R2`, `R3`, `VERIFY-01/02`);
  3. Phát biểu toán học cho **2 Hạn chế lý thuyết trọng tâm (`P1`, `P2`)**, khóa công thức và giao thức thực nghiệm kiểm chứng tương ứng (`E1`, `E2`) trên codebase hiện tại, cùng **2 Hướng phát triển tương lai (`FD1`, `FD2`)** gắn trực tiếp 1–1 với từng hạn chế.
- **Kết quả thực nghiệm kiểm chứng đã hoàn tất (18/11/2026):** Kế hoạch khóa trước [`KE_HOACH_EMPIRICS_CUOI_KY.md`](KE_HOACH_EMPIRICS_CUOI_KY.md) · Báo cáo `E1` [`reports/E1_BASE_ASSUMPTION_20261118.md`](../reports/E1_BASE_ASSUMPTION_20261118.md) · Báo cáo `E2` [`reports/E2_MEMORY_PRECISION_20261118.md`](../reports/E2_MEMORY_PRECISION_20261118.md).

---

## 1. Phân tích Lý thuyết theo 4 Góc nhìn (Four Theoretical Axes)

### 1.1. Các giả định ẩn của phương pháp (Hidden Assumptions)

Trong Bảng kiểm cuối bài báo gốc ([1], trang 12, mục 2(a): *"Did you state the full set of assumptions of all theoretical results?"*), nhóm tác giả đánh dấu **`[N/A]`** dù bài báo phát biểu Định lý 3.1 như một kết quả lý thuyết trung tâm. Khi đối chiếu công thức từ mục 3.1–3.2 [1] và chứng minh ở mục 1 Tài liệu bổ sung [1-Supp] với quá trình cài đặt thực tế, hiệu lực và chất lượng của ACIL phụ thuộc vào bốn giả định ngầm dưới đây:

1. **Giả định A1 — Tính đủ đầy và đồng phân phối ngữ nghĩa của tập lớp Base ($50\%$ số lớp đầu):**
   - Toàn bộ các pha tăng dần $k \ge 1$ chỉ học một bộ phân loại tuyến tính trên đặc trưng $\phi_{\theta_0}(x) = \operatorname{ReLU}(f_{\text{cnn}}(x; \theta_0) W_{\mathrm{fe}})$, trong đó tham số tích chập $\theta_0$ được học duy nhất từ tập dữ liệu Phase 0 ($\mathcal{D}_0^{\text{train}}$) rồi đóng băng vĩnh viễn.
   - Do $W_{\mathrm{fe}} \sim \mathcal{N}(0, I)$ là phép chiếu ngẫu nhiên đẳng hướng không phụ thuộc dữ liệu, tầng mở rộng đặc trưng chỉ có thể tách tuyến tính những mẫu vốn đã khác biệt trong không gian đầu ra $f_{\text{cnn}}(x; \theta_0) \in \mathbb{R}^{d_{\text{cnn}}}$. Nếu hai lớp mới $c_a, c_b$ ở phase sau bị ánh xạ chồng lấn lên nhau do các bộ lọc của $\theta_0$ chưa từng học các đặc trưng thị giác phân biệt chúng ở Phase 0 ($f_{\text{cnn}}(x_a; \theta_0) \approx f_{\text{cnn}}(x_b; \theta_0)$), thì không tồn tại chiều mở rộng $d$ hay nghiệm Ridge $W_k$ nào tách được hai lớp đó.
   - Mọi bảng thực nghiệm của bài báo ([1], mục 4.1, Bảng 1–2) đều giữ cố định quy mô Phase 0 ở mức **một nửa tổng số lớp ($c_0 = 50$ trên CIFAR-100, $c_0 = 500$ trên ImageNet-Full)** lấy mẫu trải đều trên các nhóm đối tượng. Giả định này dễ bị vi phạm trong thực tế khi hệ thống khởi đầu từ ít lớp (*cold-start CIL*, $c_0 \ll C_K / 2$) hoặc khi các lớp xuất hiện ở giai đoạn sau thuộc một miền ngữ nghĩa khác hẳn giai đoạn đầu (*semantic/domain shift*).

2. **Giả định A2 — Tính tĩnh của thống kê tiền xử lý và Batch Normalization qua mọi phase:**
   - Để tổng tự tương quan $\sum_{i=0}^k H_i^T H_i$ có nghĩa hình học nhất quán, hàm $\phi_{\theta_0}(x)$ phải là một hàm xác định không đổi theo thời gian. Điều này đòi hỏi cả hằng số chuẩn hóa ảnh đầu vào (`mean`, `std`) lẫn các vector thống kê `running_mean`, `running_var` trong mọi tầng BatchNorm của ResNet phải khóa cố định theo dữ liệu Phase 0 (`model.eval()`). Khi phân phối cường độ sáng hoặc màu sắc của ảnh ở các phase sau dịch chuyển so với Phase 0, việc chuẩn hóa bằng thống kê cũ làm lệch phân phối kích hoạt trước $\operatorname{ReLU}$.

3. **Giả định A3 — Số học thực chính xác vô hạn trong phép cập nhật hạng thấp lặp lại:**
   - Chứng minh Định lý 3.1 ([1-Supp], mục 1) biến đổi đại số trên trường số thực chính xác $\mathbb{R}$, với điều kiện ngầm $\gamma > 0$ cố định để $H_0^T H_0 + \gamma I_d \succ 0$ (lưu ý thêm một lỗi in ấn nhỏ tại dòng `(a)` trang 1 của [1-Supp] khi viết thiếu số mũ nghịch đảo `^{-1}` trên cụm $(I + X_k^{(\text{fe})} R_{k-1} X_k^{(\text{fe})T})$, dù dòng `(c)` và Phương trình (10) bản chính viết đúng).
   - Trên phần cứng thực tế, phép trừ lặp lại $R_k = R_{k-1} - R_{k-1} H_b^T (I_b + H_b R_{k-1} H_b^T)^{-1} H_b R_{k-1}$ qua hàng trăm đến hàng nghìn minibatch có thể làm tích tụ sai số dấu phẩy động, làm mất dần tính đối xứng hoặc tính xác định dương của $R_k$ nếu dùng kiểu `float32` hoặc khi ma trận có số điều kiện lớn ($\gamma$ rất nhỏ). Ngay trong mục 4.3 ([1], trang 8), tác giả cũng ghi nhận $\bar{A}$ trên CIFAR-100 giảm nhẹ từ $66.30\%$ ($K=5$) xuống $65.95\%$ ($K=25$) do sai số lượng tử hóa tích lũy theo số vòng tính toán.

4. **Giả định A4 — Đếm dung lượng theo "số phần tử" và tách trạng thái $R_k$ khỏi ngân sách bộ nhớ:**
   - Bài báo coi kích thước ma trận $R_k \in \mathbb{R}^{d \times d}$ là một hằng số nhẹ không phụ thuộc số mẫu và không tính $R_k, W_k, W_{\mathrm{fe}}$ vào ngân sách giới hạn bộ nhớ khi so sánh với các phương pháp lưu mẫu trong thiết lập *Strict-Memory* ([1-Supp], mục 2, Bảng A). Trên thực tế phần cứng, bộ nhớ RAM/VRAM bị ràng buộc theo **số byte thực tế** (phụ thuộc trực tiếp vào kiểu `float32`/`float64` và các bộ đệm trung gian khi nhân ma trận), chứ không phải chỉ theo số lượng ảnh gốc lưu lại.

---

### 1.2. Độ vững của các tuyên bố trong bài báo (Soundness of the Claims)

Bảng dưới đây đối chiếu bốn tuyên bố nổi bật của bài báo [1] với phạm vi thực sự được chứng minh bởi toán học và thực nghiệm:

| Tuyên bố trong bài báo [1] | Phần có cơ sở vững | Phần nói quá hoặc phụ thuộc điều kiện ẩn chưa nêu hết |
|---|---|---|
| **1. Tương đương với học gộp (*Identical to joint learning*, Định lý 3.1)** | Đúng về mặt đại số đối với **hồi quy tuyến tính Ridge trên đặc trưng cố định** ($\hat{W}_{\text{FCN}}^{(k)}$ từ Eq. 9 trùng với Eq. 7 khi giữ nguyên $\theta_0, W_{\mathrm{fe}}, \gamma > 0$). Kiểm chứng R3 của nhóm xác nhận sai lệch tương đối $\le 1.16 \times 10^{-11}$ trên `float64`. | Không tương đương với huấn luyện gộp toàn mạng CNN (*BP-based joint learning*). Mục 3.2 trang 6 có thừa nhận ngắn gọn rằng bộ trích đặc trưng trở nên "phần nào lỗi thời" (*somewhat obsolete*), nhưng phần Tóm tắt và Giới thiệu dễ gây ấn tượng rằng mô hình đạt chất lượng như học lại trên toàn bộ dữ liệu lịch sử. |
| **2. Ghi nhớ tuyệt đối (*Absolute Memorization*)** | Đúng theo nghĩa nghiệm phân loại tuyến tính không phụ thuộc thứ tự hay số lượng chia nhỏ phase $K$ (khi giữ nguyên tập Base Phase 0 và dùng đủ độ chính xác số học). | Không đồng nghĩa với không suy giảm độ chính xác trên các lớp cũ ($F = 0$). Bài báo giải thích đúng ở trang 9 rằng $F > 0$ xảy ra ngay cả khi học gộp do không gian cạnh tranh mở rộng từ 50 lên 100 lớp; tuy nhiên, bài báo chưa tách bạch việc giữ vững lớp cũ ($B_5 = 68.33\%$ ở R1) đi kèm với độ chính xác thấp hơn hẳn trên các lớp mới học sau ($N_5 = 42.00\%$ ở R1). |
| **3. Bảo vệ quyền riêng tư dữ liệu (*Data Privacy Protection*)** | Không lưu trữ ảnh gốc hay vector đặc trưng từng mẫu cũ; phương trình $H_k^T H_k = R_k^{-1} - R_{k-1}^{-1}$ bất định dưới phép nhân ma trận trực giao $Q \in \mathbb{R}^{N_k \times N_k}$ ($N_k > 1$). | Tác giả có lưu ý ở trang 2 rằng "privacy" ở đây chỉ hiểu theo nghĩa không dùng lại mẫu cũ trong CIL. Xét nghiêm ngặt, $(R_k, W_k)$ vẫn lưu chính xác thống kê bậc hai của dữ liệu huấn luyện (có thể downdate hạng-1 nếu biết mẫu ứng viên) và không cung cấp bất kỳ bảo đảm $(\varepsilon, \delta)$-Differential Privacy nào trước tấn công suy luận thành viên (*membership inference*). |
| **4. Tiết kiệm bộ nhớ & bất biến trong *Strict-Memory Setting*** | Trên tập ảnh độ phân giải lớn và nhiều lớp như ImageNet ($1000$ lớp $\times 20$ ảnh $\times 224^2 \times 3 \approx 3.01\text{ GB}$ uint8), ma trận $R_k$ ở $d=15\text{k}$ (`900 MB` `float32` / `1.8 GB` `float64`) nhỏ hơn kho ảnh thô. | Trên CIFAR-100 (trang 9), bài báo so sánh `64M` phần tử của $R_k$ với `6.1M` phần tử của 2,000 ảnh mẫu mà bỏ qua hệ số byte (`8 bytes/float64` hoặc `4 bytes/float32` so với `1 byte/uint8`), làm ẩn đi chênh lệch thực tế **$42.5\times\text{–}85\times$**. Nghiêm trọng hơn, ở Bảng A ([1-Supp], trang 2), trong khi các đối thủ bị giới hạn cứng ở kho 2,000 mẫu, ACIL vẫn giữ nguyên $d=8000$ và chép lại nguyên si kết quả của Bảng 1 thay vì giảm $d$ về cùng ngân sách bộ nhớ. |

---

### 1.3. Mô hình hóa bài toán & Hàm mục tiêu (Formulation and Objective)

Để có được nghiệm đóng và công thức cập nhật đệ quy Woodbury, ACIL thay hàm mất mát phân loại chuẩn Cross-Entropy kèm Softmax bằng bài toán **hồi quy bình phương tối thiểu có điều chuẩn Frobenius (MSE-Ridge)** trên ma trận nhãn one-hot $\tilde{Y}_k \in \{0, 1\}^{N_{0:k} \times C_k}$ (Phương trình 5 [1]):

$$
\min_{W \in \mathbb{R}^{d \times C_k}} \sum_{i=1}^{N_{0:k}} \sum_{c=1}^{C_k} \left( y_{i,c} - h_i^T w_c \right)^2 + \gamma \sum_{c=1}^{C_k} \| w_c \|_2^2
$$

Cách thay thế hàm mục tiêu (*surrogate loss*) này mang lại tính chất cộng tính bậc hai cho phép giải đệ quy, nhưng đồng thời kéo theo ba hạn chế bản chất của hàm mục tiêu:

1. **Phạt sai số hai phía đối xứng trên biên phân loại:**
   - Trong phân loại đa lớp, một mẫu $x_i$ thuộc lớp đúng $y_i = c$ có điểm dự đoán $h_i^T w_c = 3.0$ và $h_i^T w_{c'} = -0.5$ ($\forall c' \neq c$) là một mẫu được phân tách rất tốt (loss Cross-Entropy tiến sát $0$). Ngược lại, hàm MSE phạt $(1 - 3.0)^2 = 4.0$ và $(-0.5)^2 = 0.25$, tức ép đầu ra tuyến tính phải co sát về đúng hai mức cứng $\{0, 1\}$ thay vì tối đa hóa biên cách biệt (*margin*) giữa logit lớp đúng và logit lớp sai lớn nhất. Các mẫu nằm xa ranh giới nhưng có chuẩn đặc trưng $\|h_i\|_2$ lớn có thể chi phối gradient bậc hai mạnh hơn các mẫu khó nằm sát biên quyết định.

2. **Tách rời các cột phân loại và độ nhạy với mất cân bằng số mẫu giữa các lớp:**
   - Hàm mục tiêu Frobenius tách thành tổng độc lập trên từng cột lớp $c$:
     $$
     w_c = R_k \left( \sum_{i:\, y_i = c} h_i \right), \qquad R_k = \left( \sum_{j=1}^{N_{0:k}} h_j h_j^T + \gamma I_d \right)^{-1}
     $$
   - Vế phải $\tilde{H}_k^T \tilde{y}^{(c)} = \sum_{i:\, y_i = c} h_i$ có độ lớn tỷ lệ thuận với số lượng mẫu huấn luyện $N^{(c)}$ của lớp $c$, trong khi ma trận $R_k$ và hệ số phạt đồng nhất $\gamma \|w_c\|_2^2$ tác động như nhau lên mọi cột. Trong các benchmark chuẩn như CIFAR-100, mọi lớp đều có đúng $500$ mẫu train nên nhược điểm này bị che khuất. Nếu áp dụng vào luồng dữ liệu thực tế mất cân bằng (*long-tailed / imbalanced CIL*, ví dụ các bệnh hiếm ở pha sau chỉ có $20\text{–}50$ ca so với $500$ ca/bệnh phổ biến ở pha đầu), chuẩn $\|w_c\|_2$ và điểm số $h_{\text{test}}^T w_c$ của lớp ít mẫu sẽ bị co nhỏ có hệ thống, gây thiên lệch dự đoán về các lớp nhiều mẫu.

3. **Sự phụ thuộc vào chiều mở rộng lớn $d$ và điều chuẩn $\gamma > 0$:**
   - Vì hàm mục tiêu chỉ là hồi quy tuyến tính một tầng trên nhãn $\{0, 1\}$, mô hình dễ bị thiếu năng lực biểu diễn (*under-fitting*) nếu đặt trực tiếp trên $d_{\text{cnn}} = 64$ chiều (như R2 đã chỉ ra: $\bar{A}$ giảm $11.86\text{ pp}$, $A_{\mathrm{final}}$ giảm $17.80\text{ pp}$). Ngược lại, khi đẩy $d$ lên rất lớn ($d = 8000 \dots 15000$) để bù tính phi tuyến, ma trận $\tilde{H}_k^T \tilde{H}_k$ có nhiều trị riêng nhỏ sát $0$; nếu bỏ điều chuẩn ($\gamma \to 0^+$), bài toán trở nên xấu điều kiện (*ill-conditioned*), làm độ chính xác sụt từ $66.30\%$ xuống $51.12\%$ (hàng cuối Bảng 2 [1]).

---

### 1.4. Độ phức tạp & Tính tổng quát (Complexity and Generality)

1. **Khi nào câu chuyện độ phức tạp của ACIL thực sự có lợi, và khi nào bất lợi?**
   - *Về thời gian:* Tại mỗi phase mới $k \ge 1$, ACIL tốn $\mathcal{O}(N_k C_{\text{fwd}} + N_k d^2)$ với đúng $E=1$ lượt duyệt dữ liệu, nhanh hơn rõ rệt so với $\mathcal{O}(E (N_k + M)(2 C_{\text{fwd}} + C_{\text{bwd}}))$ ($E \approx 70\text{–}160$) của các phương pháp distillation + replay. Tuy nhiên, chi phí suy luận cho mỗi mẫu test tăng thêm hạng tử chiếu và phân loại $\mathcal{O}(d_{\text{cnn}} d + d C_k)$ (với $d=8000, C_k=100$: khoảng $1.31 \times 10^6$ phép nhân-cộng mỗi mẫu ở tầng đầu ra).
   - *Về không gian lưu trữ:* Gọi $b_{\text{float}} \in \{4, 8\}$ là số byte mỗi số thực của trạng thái analytic và $D_{\text{img}} = H \times W \times 3$ (bytes `uint8`) là kích thước một ảnh gốc. Trạng thái giữ giữa các phase của ACIL tốn:
     $$
     M_{\text{ACIL}}(d) = b_{\text{float}} \cdot \left( d^2 + d_{\text{cnn}} d + C_k d \right) \text{ (bytes)}
     $$
     so với kho $M$ ảnh mẫu của replay tốn $M_{\text{replay}} = M \cdot D_{\text{img}}$ bytes (cộng thêm một bản sao mạng Teacher kích thước $|\theta_0|$ lúc huấn luyện).
     - Điểm hòa vốn lưu trữ xảy ra khi $M \cdot D_{\text{img}} \approx b_{\text{float}} d^2$. Với ảnh nhỏ như CIFAR-100 ($D_{\text{img}} = 3072\text{ bytes}$) và $M = 2000$ ($6.14\text{ MB}$), lựa chọn $d=8000$ của bài báo tốn **$522.50\text{ MB}$** ở `float64` (hoặc **$261.25\text{ MB}$** ở `float32`), tức tốn bộ nhớ gấp hàng chục lần kho ảnh mẫu. Ngay cả khi tăng $d > 10\text{k}$ trên ImageNet, chính nhóm tác giả cũng gặp giới hạn tràn bộ nhớ trên GPU 11 GB ([1], trang 9).

2. **Giới hạn về tính tổng quát hóa ngoài thiết lập đánh giá của bài báo:**
   - Thiết lập đánh giá trong [1] thuận lợi cho ACIL trên cả ba phương diện: (i) tập Base chiếm tới $50\%$ tổng số lớp và đồng nhất miền với các lớp sau; (ii) số mẫu mỗi lớp hoàn toàn cân bằng ($500$ mẫu/lớp trên CIFAR-100); (iii) toàn bộ dữ liệu thuộc bài toán phân loại ảnh tĩnh một miền. Khi bước ra ngoài vùng giả định này — chẳng hạn ít lớp nền ban đầu, dịch chuyển miền ngữ nghĩa giữa các phase, hoặc ràng buộc bộ nhớ biên chặt trên thiết bị nhúng — các ưu điểm được quảng bá của ACIL không còn giữ nguyên.

---

## 2. Ba Điểm mạnh Cốt lõi Có Bằng chứng Kiểm chứng (Verified Strengths)

Theo yêu cầu của phần **Critical Analysis**, dưới đây là 3 điểm mạnh thực chất của ACIL được lập luận kèm bằng chứng trực tiếp từ đọc hiểu và kết quả tái lập công khai của nhóm trên project W&B [`hsangg-2006/acil-reproduction`](https://wandb.ai/hsangg-2006/acil-reproduction):

### Điểm mạnh S1 — Tương đương đại số với học gộp Ridge và không phụ thuộc kích thước minibatch
- **Lập luận lý thuyết:** Nhờ hàm mục tiêu bình phương tối thiểu bậc hai trên đặc trưng tĩnh $\phi_{\theta_0}(x)$ có thống kê đủ cộng tính $(\sum H_j^T H_j, \sum H_j^T Y_j)$, phép cập nhật đệ quy qua bổ đề Sherman–Morrison–Woodbury (Phương trình 10) và công thức phần dư toàn cục (tương đương Phương trình 9) bảo toàn nghiệm tối ưu toàn cục $\tilde{W}_k$ ở mọi bước mà không phụ thuộc vào cách chia phase hay cỡ lô dữ liệu $b$.
- **Bằng chứng từ tái lập:**
  - Trên dữ liệu tổng hợp (`VERIFY-02`, W&B [`wi4wau2d`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/wi4wau2d)): cập nhật theo minibatch lẻ ($b=7$) và lớp lặp lại khớp nghiệm Joint Ridge với sai số tương đối trọng số tối đa $3.43 \times 10^{-13}$, `0` dự đoán lệch.
  - Trên dữ liệu thật CIFAR-100 đủ 50,000 ảnh train và 10,000 ảnh test (`R3`, $d=1024, \gamma=0.1$, 9 runs W&B [`luf7uv40`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/luf7uv40)..[`xesh2kjl`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xesh2kjl), xem [`reports/R3_CIFAR100_K5_BATCH_20261107.md`](../reports/R3_CIFAR100_K5_BATCH_20261107.md)): khi quét $b \in \{32, 128, 512\}$, cả **54/54 mốc đối chiếu theo phase** đều khớp nghiệm Joint Oracle độc lập với `max_rel_w_error` $\le 1.159 \times 10^{-11}$, `max_rel_r_error` $\le 3.823 \times 10^{-15}$, và **`0 / 10,000` mẫu test bị đổi nhãn dự đoán** ($\bar{A} = 63.10\%$ trùng khớp trên mọi cấu hình batch).

### Điểm mạnh S2 — Cập nhật phase mới chỉ qua một lượt duyệt dữ liệu (*single-pass*), không lưu ảnh gốc và chi phí thời gian thấp
- **Lập luận lý thuyết:** Khác với các phương pháp distillation/replay phải lặp $70\text{–}160$ epochs lan truyền tiến và ngược qua mạng CNN ở mỗi phase mới, giai đoạn tăng dần của ACIL không tính gradient, không giữ kho ảnh cũ và chỉ trích xuất đặc trưng tiến đúng 1 lần ($E=1$) trước khi thực hiện các phép nhân ma trận BLAS.
- **Bằng chứng từ tái lập:**
  - Trong bộ benchmark chính `R1` ($d=8000, K=5$, 3 seeds `42/43/44`, W&B [`zt1014ei`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei), [`v51m2dan`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan), [`vqcs1ql4`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4), xem [`reports/R1_CIFAR100_K5_3SEEDS_20261106.md`](../reports/R1_CIFAR100_K5_3SEEDS_20261106.md)): trong khi huấn luyện 160 epochs SGD ở Phase 0 mất **`2423.48–2537.35 s` ($\approx 40.4\text{–}42.3$ phút)** trên GPU T4, bước cập nhật giải tích cho một phase mới 5,000 ảnh ở $d=8000$ chỉ mất **`41.79–44.74 s`** trên CPU 4 luồng mà vẫn đạt $\bar{A} = 65.12 \pm 0.61\%$ (cạnh tranh với mức $64.83\%$ của PODNet có lưu 20 ảnh/lớp trong Bảng 1 [1]).
  - Ở `R3` ($d=1024, b=512$, W&B [`lnk0grbq`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/lnk0grbq), [`5ehmjvh3`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/5ehmjvh3), [`xesh2kjl`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xesh2kjl)), toàn bộ quá trình học giải tích của cả Phase 0 lẫn 5 phases tăng dần (50,000 ảnh) hoàn tất trong **$5.00 \pm 0.11\text{ s}$** trên CPU.

### Điểm mạnh S3 — Khắc phục giới hạn thiếu chiều tuyến tính nhờ Feature Expansion và rất ít nhạy cảm với hệ số điều chuẩn $\gamma$
- **Lập luận lý thuyết:** Việc ghép phép chiếu ngẫu nhiên cố định $W_{\mathrm{fe}}$ với phi tuyến $\operatorname{ReLU}$ mở rộng không gian đặc trưng từ $d_{\text{cnn}} = 64$ lên $d = 8000 \gg C_5$, nâng hạng ma trận đặc trưng và khả năng phân tách tuyến tính theo Định lý Cover mà không thêm bất kỳ tham số nào cần tối ưu bằng gradient. Đồng thời, công thức chỉ có một siêu tham số điều chuẩn $\gamma$ thay vì nhiều hệ số cân bằng loss nhạy cảm như hướng distillation.
- **Bằng chứng từ tái lập:**
  - Trong thực nghiệm `R2` trên cùng checkpoint ResNet-32 của từng seed (`42/43/44`, 9 runs mới [`xdrm0d27`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xdrm0d27)..[`l9c50ezb`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/l9c50ezb) đối chiếu ghép cặp với 3 runs `R1`, xem [`reports/R2_CIFAR100_K5_20261107.md`](../reports/R2_CIFAR100_K5_20261107.md)):
    1. Bật tầng FE ($d=8000$) giúp $\bar{A}$ tăng **$+11.86 \pm 0.37\text{ pp}$** (`53.25%` $\to$ `65.12%`) và độ chính xác 100 lớp cuối $A_{\mathrm{final}}$ tăng **$+17.80\text{ pp}$** (`38.33%` $\to$ `56.13%`).
    2. Khi quét $\gamma \in \{0.1, 0.01, 0.001\}$ (thay đổi gấp $100$ lần), độ lệch ghép cặp $\Delta\bar{A}$ chỉ là **$-0.0022 \pm 0.0020\text{ pp}$** (tại $\gamma=0.01$) và **$-0.0032 \pm 0.0035\text{ pp}$** (tại $\gamma=0.001$), xác nhận vùng làm việc ổn định rộng của $\gamma$.

---

## 3. Hai Cặp Hạn chế Lý thuyết – Thiết kế Thực nghiệm – Hướng Phát triển (P1 & P2)

Hai vấn đề dưới đây được chọn làm hai trục chính cho Poster Cuối kỳ vì: (i) bám trực tiếp vào khoảng trống lý thuyết mà bài báo [1] chưa định lượng hoặc xử lý chưa công bằng; (ii) đã có tín hiệu gợi ý rõ từ số liệu giữa kỳ; và (iii) có thể kiểm chứng sạch, có đối chứng chặt ngay trên codebase hiện tại.

---

### Vấn đề 1 (`P1`) — Giả định ẩn về tính đủ đầy biểu diễn của tập Base đóng băng (*Frozen Base Representation Bottleneck*)

#### A. Phát biểu và lập luận lý thuyết của Hạn chế `P1`
Định lý 3.1 chỉ chứng minh rằng nghiệm đệ quy $W_K$ của ACIL đạt cực tiểu toàn cục của hàm mất mát hồi quy trên không gian đặc trưng cố định $\mathcal{H}_{\theta_0} = \{ \operatorname{ReLU}(f_{\text{cnn}}(x; \theta_0) W_{\mathrm{fe}}) \}$:

$$
W_K = \tilde{W}_K = \arg\min_{W \in \mathbb{R}^{d \times C_K}} \left\| \tilde{Y}_K - \tilde{H}_K(\theta_0) W \right\|_F^2 + \gamma \| W \|_F^2
$$

Tuy nhiên, sai số tổng quát hóa thực sự trên toàn bộ phân phối dữ liệu $\mathcal{D}_{0:K}$ bị chặn dưới bởi **sai số xấp xỉ biểu diễn của chính tham số đóng băng $\theta_0$** học từ duy nhất tập lớp đầu $\mathcal{C}_0$:

$$
\mathcal{R}_{\text{ACIL}}(\theta_0) \ge \inf_{W \in \mathbb{R}^{d \times C_K}} \mathbb{E}_{(x, y) \sim \mathcal{D}_{0:K}} \left\| \mathbf{e}_y - \phi_{\theta_0}(x) W \right\|_2^2
$$

Vì $\theta_0$ được tối ưu bằng Cross-Entropy để tách riêng nội bộ $c_0 = |\mathcal{C}_0|$ lớp của Phase 0 (hiện tượng thu hẹp biểu diễn theo nhãn — *Neural Collapse / task-specific feature compression*), các hướng biến thiên không cần thiết cho $\mathcal{C}_0$ nhưng cần thiết để phân biệt các lớp tương lai $\mathcal{C}_{1:K}$ bị triệt tiêu ngay từ tầng tích chập trước khi đi vào $W_{\mathrm{fe}}$.
- **Dấu hiệu định lượng ban đầu từ R1 của nhóm:** Ngay cả ở thiết lập thuận lợi $c_0 = 50$ lớp ngẫu nhiên trên CIFAR-100 ([`reports/R1_CIFAR100_K5_3SEEDS_20261106.md`](../reports/R1_CIFAR100_K5_3SEEDS_20261106.md)), tại Phase 5 (100 lớp), độ chính xác trên 50 lớp Base đạt **$B_5 = 68.33 \pm 0.50\%$**, trong khi độ chính xác trên 10 lớp mới vừa học chỉ đạt **$N_5 = 42.00 \pm 4.96\%$** (thấp hơn lớp Base tới **$-26.33\text{ pp}$**).
- **Hệ quả lý thuyết có thể kiểm chứng:** Hiệu năng của ACIL không chỉ phụ thuộc vào tổng dữ liệu $\bigcup_{k=0}^K \mathcal{D}_k$, mà **nhạy cảm mạnh với (1) tỷ lệ lớp được nhìn thấy ở Phase 0 ($c_0 / C_K$) và (2) độ phủ ngữ nghĩa giữa $\mathcal{C}_0$ và $\mathcal{C}_{1:K}$** — hai chiều biến thiên mà bài báo gốc hoàn toàn cố định ở mức $c_0 = 50\%$ ngẫu nhiên đồng nhất.

#### B. Thiết kế thực nghiệm kiểm chứng `E1` (Khóa cấu hình trước khi chạy)
Sử dụng trực tiếp tập CIFAR-100 đã pin SHA-256 trong `src/cifar_data.py` (lưu ý file Parquet gốc `uoft-cs/cifar100` đã có sẵn cả hai cột `fine_label` $0\dots 99$ và `coarse_label` $0\dots 19$ tương ứng 20 nhóm ngữ nghĩa lớn *superclasses*). Thực nghiệm `E1` gồm 2 nhánh bổ trợ nhau:

1. **Nhánh `E1a` — Đường cong suy thoái khi thu hẹp quy mô tập Base (*Base-Class Ratio Degradation Curve*):**
   - **Biến thay đổi:** Số lượng lớp Base ở Phase 0: $c_0 \in \{10, 20, 30, 50\}$, các lớp còn lại $(100 - c_0)$ luôn được đưa vào theo từng đợt **10 lớp/phase** (tương ứng số phase tăng dần $K \in \{9, 8, 7, 5\}$; mọi cấu hình đều kết thúc tại **cùng tập 100 lớp đầy đủ** ở phase cuối).
   - **Biến cố định:** Cùng danh sách hoán vị `class_order` trên từng seed (`42, 43, 44`), cùng kiến trúc ResNet-32, cùng lịch huấn luyện Base 160 epochs SGD, cùng $d = 8000, \gamma = 0.1$, minibatch $b = 128$, `float64`.
   - **Thước đo chính & Đối chứng:**
     - Đối chứng chuẩn là cấu hình $c_0 = 50, K = 5$ (tái dùng trực tiếp 3 runs `R1` đã nghiệm thu).
     - Đo tại phase cuối (đủ 100 lớp, trên **cùng 10,000 ảnh test** để so sánh ngang bằng tuyệt đối): $A_{\mathrm{final}}$ (100 lớp), độ chính xác trung bình trên các lớp Base $B_{\mathrm{final}}$ (`labels < c_0`), độ chính xác trung bình trên toàn bộ $(100 - c_0)$ lớp học tăng dần $A_{\mathrm{inc}}$ (`labels >= c_0`), cùng $\bar{A}$ qua chuỗi phase.
   - **Dự đoán kiểm định & Tiêu chí bác bỏ:** Khi giảm $c_0$ từ $50 \to 30 \to 20 \to 10$, dù Định lý 3.1 vẫn giữ nghiệm khớp Joint Ridge ở mọi cấu hình, $A_{\mathrm{final}}$ và $A_{\mathrm{inc}}$ sẽ suy giảm đơn điệu rõ rệt (đặc biệt ở $c_0 = 10, 20$) do không gian đặc trưng 64 chiều của ResNet-32 chỉ học từ 10–20 lớp đầu bị suy giảm hạng biểu diễn. Giả thuyết bị bác bỏ nếu $A_{\mathrm{final}}$ giảm không đáng kể ($< 2\text{ pp}$) khi giảm $c_0$ từ $50$ xuống $10$.

2. **Nhánh `E1b` — Phá giả định đồng phân phối ngữ nghĩa giữa Base và Incremental (*Semantic Superclass Shift*):**
   - **Biến thay đổi:** Giữ nguyên quy mô chuẩn **$c_0 = 50$ lớp Base + $K = 5 \times 10$ lớp mới** (để loại bỏ hoàn toàn yếu tố thiếu số lượng mẫu hay thiếu số lớp ở Phase 0!), chỉ thay đổi **cấu trúc phân bố ngữ nghĩa** giữa Phase 0 và Phases $1\dots 5$ trên 3 seeds (`42, 43, 44`):
     - *Đối chứng đồng phân phối (`Uniform / Stratified Base 50`):* 50 lớp Base trải đều qua các nhóm ngữ nghĩa (đối chiếu với `R1` và/hoặc chia phân tầng đúng 2–3 lớp từ mỗi superclass).
     - *Lệch miền ngữ nghĩa (`Superclass-Disjoint Base 50` — Tự nhiên/Sinh vật $\to$ Nhân tạo/Phương tiện/Đồ vật):* Phase 0 gồm đúng **50 lớp thuộc 10 superclass tự nhiên & sinh vật** (`aquatic_mammals`, `fish`, `flowers`, `fruit_and_vegetables`, `insects`, `large_carnivores`, `large_natural_outdoor_scenes`, `large_omnivores_and_herbivores`, `medium_mammals`, `trees`), còn Phases $1\dots 5$ gồm **50 lớp thuộc 10 superclass còn lại** (nặng về phương tiện giao thông, thiết bị điện, nội thất, vật chứa, công trình nhân tạo: `vehicles_1`, `vehicles_2`, `household_electrical_devices`, `household_furniture`, `food_containers`, `large_man-made_outdoor_things`, `people`, `reptiles`, `small_mammals`, `non-insect_invertebrates` — hoặc đảo chiều *Nhân tạo/Đồ vật $\to$ Sinh vật*).
   - **Thước đo chính:** So sánh $A_{\mathrm{final}}$, $\bar{A}$, $B_5$ (50 lớp Base), $A_{\mathrm{inc}}$ (50 lớp mới ở Phase 5) và khoảng chênh lệch khả năng chuyển giao $\Delta_{\text{gap}} = B_5 - A_{\mathrm{inc}}$.
   - **Dự đoán kiểm định:** Ở chế độ lệch superclass, $A_{\mathrm{inc}}$ trên các nhóm lớp nhân tạo/phương tiện ở các pha sau sẽ tụt thấp hơn đáng kể so với thiết lập chia đều ngữ nghĩa, làm tăng mạnh độ lệch $\Delta_{\text{gap}} = B_5 - A_{\mathrm{inc}}$ dù số lớp và số ảnh huấn luyện ở Phase 0 hoàn toàn bằng nhau ($25{,}000$ ảnh).

#### C. Hướng phát triển tương lai `FD1` (Gắn 1–1 với Hạn chế `P1`)
Để giải quyết trực tiếp nút thắt biểu diễn đóng băng của `P1`, có hai hướng thiết kế có cơ sở toán học rõ ràng (đi kèm phân tích tác động lên Định lý 3.1):

1. **Làm giàu biểu diễn Phase 0 bằng mục tiêu Tự giám sát / Tương phản (*Self-Supervised / Supervised Contrastive Base Learning*) — Bảo toàn Định lý 3.1:**
   - Thay vì chỉ tối ưu Cross-Entropy trên $c_0$ nhãn (vốn ép các mẫu cùng lớp co cụm thành một điểm và triệt tiêu các đặc trưng phụ như kết cấu, hình học bộ phận), hàm mất mát ở Phase 0 được bổ sung thành phần tương phản tự giám sát không phụ thuộc nhãn lớp:
     $$
     \mathcal{L}_{\text{Base}}(\theta) = \mathcal{L}_{\text{CE}}(\theta; \mathcal{D}_0) + \lambda_{\text{ssl}} \, \mathcal{L}_{\text{InfoNCE}}(\theta; \mathcal{D}_0)
     $$
   - **Ưu điểm lý thuyết:** Sau khi kết thúc Phase 0, $\theta_0$ vẫn được đóng băng hoàn toàn trước khi chạy ARaBT và cập nhật Woodbury; do đó **toàn bộ giả thiết hàm $\phi_{\theta_0}$ cố định và kết luận $W_k = \tilde{W}_k$ của Định lý 3.1 được giữ nguyên 100%**, trong khi phổ trị riêng của ma trận đặc trưng $F_0$ trải đều hơn, giữ lại các đặc trưng hình thái chung giúp phân tách tốt hơn các lớp chưa gặp ở $\mathcal{C}_{1:K}$.

2. **Mở rộng nhánh Adapter hạng thấp đóng băng theo từng Phase (*Frozen Phase-Specific Low-Rank Adapters*):**
   - Lưu ý rằng nếu cập nhật trực tiếp $\theta_{k-1} \to \theta_k$ trên backbone chung mà vẫn cộng tiếp vào $R_{k-1}$ cũ thì Định lý 3.1 sẽ **mất hiệu lực** (do $\tilde{H}_{k-1}(\theta_{k-1})$ và $H_k(\theta_k)$ không còn chung hệ tọa độ). Để mở rộng biểu diễn mà không lưu mẫu cũ, có thể giữ cố định thân chính $\theta_0$, chỉ học nhẹ một nhánh phần dư hạng thấp nhỏ $\Delta\phi_k(x)$ trên dữ liệu phase mới hoặc ước lượng độ trôi đặc trưng (*feature drift compensation*) để chiếu xấp xỉ thống kê hiệp phương sai cũ sang không gian mới — chấp nhận đánh đổi từ tương đương đại số tuyệt đối sang xấp xỉ có sai số chặn trên.

---

### Vấn đề 2 (`P2`) — Chi phí bộ nhớ $\mathcal{O}(d^2)$ theo Byte thực tế và Độ nhạy số học của cập nhật MSE-Woodbury

#### A. Phát biểu và lập luận lý thuyết của Hạn chế `P2`
Trong ACIL, chất lượng phân loại phụ thuộc mạnh vào việc chiếu đặc trưng lên không gian chiều cao $d \gg d_{\text{cnn}}$ trước khi giải hồi quy MSE-Ridge. Tuy nhiên, cấu trúc này tạo ra một ràng buộc kép về mặt tài nguyên và giải tích số mà bài báo chưa đánh giá công bằng:

1. **Chi phí bộ nhớ trạng thái tăng bậc hai $\mathcal{O}(d^2)$ theo byte và sự thiếu nhất quán trong so sánh *Strict-Memory*:**
   - Để thực hiện cập nhật đệ quy ở mọi phase $k \ge 1$, hệ thống bắt buộc phải giữ trong bộ nhớ ba ma trận $(R_k, W_k, W_{\mathrm{fe}})$ với tổng dung lượng tính bằng byte là:
     $$
     M_{\text{state}}(d, C_k, b_{\text{elem}}) = b_{\text{elem}} \cdot \left( \underbrace{d^2}_{R_k} + \underbrace{d_{\text{cnn}} d}_{W_{\mathrm{fe}}} + \underbrace{C_k d}_{W_k} \right)
     $$
     trong đó $b_{\text{elem}} = 8$ bytes với `float64` và $b_{\text{elem}} = 4$ bytes với `float32` (chưa tính bộ nhớ đệm tạm thời $\mathcal{O}(b d + d^2)$ khi thực hiện phép nhân $R_{k-1} H_b^T$ và cập nhật $R_k$).
   - Tại cấu hình CIFAR-100 của bài báo ($d = 8000, d_{\text{cnn}} = 64, C_5 = 100$):
     - Với `float64`: $M_{\text{state}} = 8 \times (64{,}000{,}000 + 512{,}000 + 800{,}000) = \mathbf{522{,}496{,}000\text{ bytes}} \approx \mathbf{522.50\text{ MB}}$ (`498.29 MiB`).
     - Với `float32`: $M_{\text{state}} = \mathbf{261{,}248{,}000\text{ bytes}} \approx \mathbf{261.25\text{ MB}}$ (`249.15 MiB`).
     - Trong khi đó, kho $M = 2000$ ảnh gốc CIFAR-100 (`32×32×3 uint8`) mà iCaRL/PODNet/AANets được cấp trong thiết lập cố định mẫu chỉ chiếm $2000 \times 3072 = \mathbf{6{,}144{,}000\text{ bytes}} \approx \mathbf{6.14\text{ MB}}$ (`5.86 MiB`).
   - Như đã phân tích ở mục 1.2, Bảng A của Tài liệu bổ sung ([1-Supp], trang 2) đặt tên là *Strict-Memory Setting* nhưng vẫn cho phép ACIL dùng $d=8000$ (`261–522 MB`) để so với các phương pháp replay bị khóa ở `6.14 MB`. Nếu áp đặt **cùng ngân sách bộ nhớ thực $M_{\text{state}}(d) \le 6.144\text{ MB}$**, số chiều mở rộng tối đa mà ACIL được phép dùng trên CIFAR-100 bị ép xuống chỉ còn:
     $$
     d_{\max}^{(\text{fp64})} = \left\lfloor \sqrt{\frac{6{,}144{,}000}{8} + 82^2} - 82 \right\rfloor = \mathbf{798}, \qquad
     d_{\max}^{(\text{fp32})} = \left\lfloor \sqrt{\frac{6{,}144{,}000}{4} + 82^2} - 82 \right\rfloor = \mathbf{1160}
     $$
     — tức nhỏ hơn cấu hình $d=8000$ của bài báo từ **$7\times$ đến $10\times$** về số chiều (và $48\times\text{–}100\times$ về kích thước ma trận $R_k$).

2. **Đánh đổi độ chính xác số học khi hạ kiểu dữ liệu từ `float64` xuống `float32` để giảm một nửa RAM:**
   - Một cách tự nhiên để cắt giảm $50\%$ dung lượng của $R_k$ là lưu và cập nhật $(R_k, W_k)$ ở kiểu đơn `float32` ($\epsilon_{\text{mach}} \approx 1.19 \times 10^{-7}$) thay vì `float64` ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$).
   - Tuy nhiên, số điều kiện của ma trận Gram có điều chuẩn là $\kappa(\tilde{H}_k^T \tilde{H}_k + \gamma I_d) = \frac{\sigma_{\max}^2(\tilde{H}_k) + \gamma}{\sigma_{\min}^2(\tilde{H}_k) + \gamma} \le \frac{\|\tilde{H}_k\|_2^2}{\gamma} + 1$. Khi $N_{0:k} = 50{,}000$, $d$ lớn và $\gamma$ nhỏ (ví dụ $\gamma = 10^{-3}$ trong Bảng 2 [1]), $\kappa$ có thể lên tới $10^6\text{–}10^8$, tiệm cận nghịch đảo epsilon máy của `float32`. Qua hàng trăm phép trừ hạng thấp liên tiếp của công thức Woodbury (Phương trình 10), sai số làm tròn tích lũy làm $R_k$ trượt xa dần nghiệm Joint Ridge và có thể làm đảo nhãn dự đoán $\operatorname{argmax}$.

#### B. Thiết kế thực nghiệm kiểm chứng `E2` (Tái dùng 3 checkpoints R1 — không tốn GPU huấn luyện lại CNN)
Ưu điểm thực thi lớn của `E2` là **tái sử dụng nguyên vẹn 3 checkpoints ResNet-32 160 epochs đã kiểm chứng của `R1` (`seeds 42, 43, 44`)**, giữ cố định đặc trưng CNN `F` ($64$ chiều) để đo thuần túy quy luật đánh đổi của tầng giải tích qua 2 nhánh:

1. **Nhánh `E2a` — Đường cong đánh đổi Chất lượng – Bộ nhớ Byte – Thời gian theo Ngân sách Chiều mở rộng $d$ (*Iso-Memory Budget Trade-off Curve*):**
   - **Biến thay đổi:** Quét chiều mở rộng $d \in \{64\text{ (No-FE)}, 128, 256, 512, 768, 798, 1024, 1160, 2048, 4096, 8000\}$ trên đủ **3 seeds (`42, 43, 44`)**, với $\gamma = 0.1$, $K = 5$, minibatch $b = 128$, `float64` (kèm nhánh thử nghiệm nguyên mẫu `FD2` *Block-Diagonal RFAM* tại $d = 8000, B \in \{16, 80\}$).
   - **Điểm neo đối chiếu công bằng (*Equal-Byte Anchors*):**
     - $d = 768$ (`float64`, `state_bytes` $= 5.73\text{ MB}$) và điểm trần chính xác $d_{\max}^{(\text{fp64})} = 798$ (`float64`, `state_bytes` $= 6.141\text{ MB} \le 6.144\text{ MB}$): cấu hình lớn nhất nằm gọn trong ngân sách byte của 2,000 ảnh CIFAR-100 ở `float64`.
     - $d = 1024$ (`float32` tương đương `4.87 MB`, `float64` `9.73 MB`) và điểm trần chính xác $d_{\max}^{(\text{fp32})} = 1160$ (`float32` tương đương $6.143\text{ MB} \le 6.144\text{ MB}$): trên seed 42 ở R3 đã ghi nhận tại $d = 1024$ mức $\bar{A} = 63.10\%$ so với $65.16\%$ ở $d=8000$, tức hụt $-2.06\text{ pp}$ $\bar{A}$ và $-4.14\text{ pp}$ $A_{\mathrm{final}}$.
   - **Thước đo thu thập:** Dung lượng trạng thái lý thuyết & thực tế `state_bytes` (MB), `peak_rss_mib`, thời gian tính Base + 5 phases tăng dần (giây), $\bar{A}$, $A_{\mathrm{final}}$, $F_{\mathrm{base}}$ và $N_5$ (`mean ± sample SD`, $n=3$, `ddof=1`).
   - **Dự đoán kiểm định:** Khi buộc ACIL tuân thủ đúng ngân sách bộ nhớ bằng byte của 2,000 ảnh mẫu trên CIFAR-100 ($d \approx 768\text{–}1024$), $\bar{A}$ giảm khoảng $2.0\text{–}3.0\text{ pp}$ và $A_{\mathrm{final}}$ giảm khoảng $4.0\text{–}5.5\text{ pp}$ so với con số $d=8000$ mà bài báo giữ nguyên trong Bảng A *Strict-Memory* — chứng minh ưu thế ở Bảng A trên CIFAR-100 có phần đóng góp đáng kể từ việc dùng ngân sách bộ nhớ thực tế lớn gấp hàng chục lần.

2. **Nhánh `E2b` — Khảo sát sai số số học của cập nhật Woodbury giữa `float32` và `float64` theo $\gamma$ và số bước lặp:**
   - **Biến thay đổi:** Cài một nhánh kiểm tra kiểu dữ liệu tường minh (`dtype` $\in \{\text{float32}, \text{float64}\}$, tách riêng trong module/script thực nghiệm cuối kỳ, không sửa mặc định `float64` an toàn của `src/analytic.py`) tại $d = 1024$ (và/hoặc $d = 4096$) với $\gamma \in \{0.1, 10^{-2}, 10^{-3}, 10^{-4}\}$ và hai chế độ chia bước cập nhật: $(K=5, b=128 \implies 200\text{ bước})$ so với $(K=25\text{ hoặc }b=16 \implies 1565\text{ bước})$.
   - **Thước đo đối chiếu với nghiệm chuẩn Joint Ridge `float64`:** Sai số tương đối $\|W_k - \tilde{W}_k^{(\text{fp64})}\|_F / \|\tilde{W}_k^{(\text{fp64})}\|_F$, độ mất đối xứng $\|R_k - R_k^T\|_F$, tỷ lệ mẫu test bị lệch nhãn $\operatorname{argmax}$ và mức suy giảm $\bar{A}$ (nếu có).
   - **Dự đoán kiểm định:** Ở `float64`, sai số giữ ổn định quanh $10^{-11}\text{–}10^{-9}$ trên mọi mức $\gamma$; ngược lại ở `float32`, khi $\gamma$ giảm xuống $10^{-3}\text{–}10^{-4}$ và số bước Woodbury tăng lên, sai số tương đối của $W_k$ tăng vọt nhiều bậc độ lớn và phát sinh bất đồng nhãn dự đoán so với nghiệm học gộp — chỉ ra lý do thực tế buộc ACIL phải trả chi phí `float64` ($8$ bytes/phần tử) ở miền điều chuẩn nhỏ hoặc nhiều bước cập nhật.

#### C. Hướng phát triển tương lai `FD2` (Gắn 1–1 với Hạn chế `P2`)
Để giảm chi phí bộ nhớ từ $\mathcal{O}(d^2)$ xuống tuyến tính theo $d$ mà vẫn giữ được tính cập nhật một lượt không lưu mẫu cũ, hướng phát triển gắn trực tiếp với `P2` là **Xấp xỉ Cấu trúc Khối Đường chéo cho Ma trận Tương quan Nghịch đảo (*Block-Diagonal / Subspace-Ensemble RFAM*)**:

- **Công thức đề xuất:** Chia không gian đặc trưng mở rộng $d$ chiều thành $B$ khối con rời nhau, mỗi khối có kích thước $m = d / B$ ($H_k = [H_k^{(1)}, H_k^{(2)}, \dots, H_k^{(B)}]$ với $H_k^{(j)} \in \mathbb{R}^{N_k \times m}$). Thay vì lưu ma trận dày đặc $R_k \in \mathbb{R}^{d \times d}$ chứa mọi tương quan chéo giữa toàn bộ $d$ nơ-ron ngẫu nhiên, ta xấp xỉ ma trận Gram bằng dạng khối đường chéo:
  $$
  \hat{R}_k = \operatorname{diag}\!\left( R_k^{(1)}, R_k^{(2)}, \dots, R_k^{(B)} \right), \qquad R_k^{(j)} = \left( \sum_{i=0}^{k} H_i^{(j)T} H_i^{(j)} + \gamma I_m \right)^{-1} \in \mathbb{R}^{m \times m}
  $$
  và giải/cập nhật đệ quy độc lập từng khối trọng số $W_k^{(j)} \in \mathbb{R}^{m \times C_k}$ bằng đúng công thức Woodbury (Phương trình 9–10) trên kích thước nhỏ $m \times m$, với điểm dự đoán tổng hợp $S(x) = \frac{1}{B} \sum_{j=1}^B H_{\text{test}}^{(j)} W_k^{(j)}$ (hoặc tổng trực tiếp).
- **Lợi ích định lượng trực tiếp cho `P2`:**
  1. **Giảm bộ nhớ $B$ lần:** Dung lượng lưu trữ $\hat{R}_k$ giảm từ $d^2$ phần tử xuống $B \cdot (d/B)^2 = \frac{d^2}{B}$ phần tử. Chẳng hạn với $d = 8000$ chia thành $B = 16$ khối cỡ $m = 500$, bộ nhớ của $\hat{R}_k$ ở `float64` giảm $16$ lần từ **$512\text{ MB}$ xuống chỉ còn $32\text{ MB}$** (hoặc $16\text{ MB}$ ở `float32`), đồng thời thời gian cập nhật mỗi mẫu giảm từ $\mathcal{O}(d^2)$ xuống $\mathcal{O}(B m^2) = \mathcal{O}(d^2 / B)$ và có thể tính song song hoàn toàn trên $B$ khối.
  2. **Bảo toàn tính xác định dương & bất biến thứ tự phase trong từng khối:** Mỗi khối con $(R_k^{(j)}, W_k^{(j)})$ vẫn thỏa mãn chính xác Định lý 3.1 đối với bài toán Ridge con trên khối đặc trưng $H^{(j)}$ (không gặp rủi ro suy biến dấu như xấp xỉ cắt cụt SVD hạng thấp khi cập nhật trực tuyến).
  3. **Đánh đổi lý thuyết minh bạch:** Việc đặt các khối ngoài đường chéo của $\tilde{H}_k^T \tilde{H}_k$ bằng $0$ bỏ qua tương quan chéo giữa các nhóm cột chiếu ngẫu nhiên khác khối; sai số xấp xỉ so với nghiệm Ridge đầy đủ tỷ lệ với chuẩn của phần tương quan ngoài khối $\|\tilde{H}_k^T \tilde{H}_k - \operatorname{blkdiag}(\tilde{H}_k^T \tilde{H}_k)\|_2$. *(Ngoài ra, chính công thức Block-Diagonal RFAM này có thể cài đặt thử rất gọn như một baseline thay thế — Replacement Baseline — ngay trong phần thực nghiệm cuối kỳ nếu còn dư thời gian).*

---

## 4. Ma trận Nối Trực tiếp Lý thuyết $\leftrightarrow$ Thực nghiệm $\leftrightarrow$ Bố cục Poster A2

Theo quy định của trang **Empirical Demonstration**, Poster cuối kỳ (`1 slide .pptx`, khổ dọc **Portrait A2: `420 × 594 mm`**) phải đặt lập luận lý thuyết nằm ngay cạnh bằng chứng thực nghiệm của từng vấn đề (*wired issue-by-issue*). Bảng dưới đây khóa trước cấu trúc ánh xạ 1–1 để nhóm chuyển thẳng sang bước viết code (`Ưu tiên 2–3`) và dựng Poster (`Ưu tiên 5`):

| Khu vực trên Poster A2 dọc | Phía Lý thuyết (*Critical Analysis*) | Phía Bằng chứng Thực nghiệm (*Empirical Demonstration*) | W&B Tag / Nguồn truy nguyên |
|---|---|---|---|
| **Khối 1 (Đầu trang — ~18% chiều cao)** | Tóm tắt 3 dòng cơ chế ACIL (Freeze CNN + FE + Woodbury RLS) & tổng hợp nhanh 4 góc nhìn phản biện (Giả định ẩn, Độ vững tuyên bố, Hàm mục tiêu MSE, Độ phức tạp). | **3 Điểm mạnh có bằng chứng (`S1–S3`):**<br>• `S1`: 54/54 mốc khớp Joint $\le 1.16\times 10^{-11}$ (`R3`)<br>• `S2`: Cập nhật 1 lượt ~43 s/phase vs ~41 phút BP (`R1`)<br>• `S3`: FE tăng $+11.86\text{ pp}$, $\gamma$ biến thiên $0.0032\text{ pp}$ (`R2`) | Runs giữa kỳ đã công bố (`R1`, `R2`, `R3`, `VERIFY-02`) |
| **Khối 2 (Giữa trên — ~36% chiều cao): Trục Vấn đề `P1`** | **Hạn chế `P1`:** Định lý 3.1 chỉ tối ưu trên không gian con $\phi_{\theta_0}$ đóng băng; sai số bị chặn dưới bởi độ phủ của tập lớp Base $\mathcal{C}_0$.<br>**Hướng khắc phục `FD1`:** Tiền huấn luyện bổ sung mục tiêu tự giám sát $\mathcal{L}_{\text{InfoNCE}}$ ở Phase 0 (giữ nguyên hiệu lực Định lý 3.1). | **Biểu đồ `E1` (2 panels tự vẽ):**<br>• *Panel A (`E1a`):* Đường cong suy thoái $A_{\mathrm{final}}, A_{\mathrm{inc}}$ khi giảm quy mô lớp Base $c_0 \in \{50, 30, 20, 10\}$ (3 seeds).<br>• *Panel B (`E1b`):* So sánh `Uniform Base 50` vs `Superclass-Disjoint Base 50` (cùng 25,000 ảnh Base nhưng lệch miền ngữ nghĩa). | Project `acil-reproduction`, tag riêng: `["final-empirics", "P1-base-assumption"]` |
| **Khối 3 (Giữa dưới — ~36% chiều cao): Trục Vấn đề `P2`** | **Hạn chế `P2`:** Bộ nhớ trạng thái $\mathcal{O}(d^2)$ tính theo byte (`522.5 MB` ở $d=8000$ `float64` vs `6.14 MB` của 2,000 ảnh CIFAR-100); Bảng A [1-Supp] chưa giới hạn $d$ theo cùng ngân sách byte; hạ xuống `float32` gây trôi sai số Woodbury khi $\gamma$ nhỏ.<br>**Hướng khắc phục `FD2`:** Xấp xỉ khối đường chéo $\hat{R}_k = \operatorname{diag}(R_k^{(1)}, \dots, R_k^{(B)})$ giảm bộ nhớ $B$ lần. | **Biểu đồ `E2` (2 panels tự vẽ):**<br>• *Panel A (`E2a`):* Đường cong Pareto $\bar{A}, A_{\mathrm{final}}$ theo dung lượng `state_bytes` (MB) với $d \in [64, 8000]$ (3 seeds), đánh dấu vạch ngân sách `6.14 MB` ($d \approx 768\text{–}1024$).<br>• *Panel B (`E2b`):* Sai số tương đối $W_k$ & bất đồng dự đoán giữa `float32` và `float64` theo $\gamma$ và số bước lặp. | Project `acil-reproduction`, tag riêng: `["final-empirics", "P2-memory-precision"]` |
| **Khối 4 (Chân trang — ~10% chiều cao)** | Kết luận tổng hợp: Điều kiện thực tế để chọn ACIL thay cho Replay/Distillation + Danh mục trích dẫn IEEE gọn. | Bảng truy nguyên mã Run W&B cho mọi con số trên Poster + Link/QR dẫn tới GitHub `24122050S/20-ML-Paper`. | Đồng bộ bảng `Experiment Mapping` trong `README.md` |

---

## 5. Tài liệu tham khảo (Chuẩn IEEE)

- **[1]** H. Zhuang, Z. Weng, H. Wei, R. Xie, K.-A. Toh, and Z. Lin, "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, pp. 11602–11614, 2022.
- **[1-Supp]** H. Zhuang, Z. Weng, H. Wei, R. Xie, K.-A. Toh, and Z. Lin, "Supplementary material for ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," in *NeurIPS*, vol. 35, 2022.
- **[2]** S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, "iCaRL: Incremental classifier and representation learning," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2001–2010.
- **[3]** A. Douillard, M. Cord, C. Ollion, T. Robert, and E. Valle, "PODNet: Pooled outputs distillation for small-tasks incremental learning," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 86–102.
- **[4]** Y. Liu, B. Schiele, and Q. Sun, "Adaptive aggregation networks for class-incremental learning," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 2544–2553.
- **[5]** Y. Liu, B. Schiele, and Q. Sun, "RMM: Reinforced memory management for class-incremental learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34, pp. 3478–3490, 2021.
