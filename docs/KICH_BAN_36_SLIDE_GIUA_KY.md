# Kịch bản Chi tiết 36 Slides — Báo cáo Giữa kỳ ACIL (NeurIPS 2022)

- **Mục tiêu sản phẩm:** Bản thiết kế hoàn chỉnh cho **01 file Slide Deck `.pptx` (36 slides: 32 slides chính + 4 slides Appendix)** đáp ứng trọn vẹn yêu cầu hai phần **Reading the Paper** + **Experimental Reproduction** của môn Nhập môn Học máy (HCMUS).
- **Tuân thủ quy chuẩn nộp bài Lab Book:**
  1. **Tổng số slide:** `36` slides (nằm trong ngưỡng `20–40 slides` tính cả Appendix).
  2. **Dung lượng mục tiêu:** `< 20 MB` (dùng 7 hình PNG tự vẽ tại `reports/figures/` tổng dung lượng `< 1.7 MB`).
  3. **Speaker Notes:** Mọi slide (`1` đến `36`) đều có sẵn lời bình đầy đủ (`~140–240 từ/slide`, tuyệt đối `<= 500 từ`), giải thích rõ từng biến công thức, số liệu và mạch chuyển tiếp.
  4. **Trích dẫn chuẩn IEEE:** Nguồn ngoài được đánh số `[1]`–`[8]` và đặt **ngay ở footer của từng slide sử dụng nguồn**. **Không tạo slide `References` riêng.**
  5. **Hình ảnh & Code:** Chỉ sử dụng 7 hình tự vẽ của nhóm + bảng/sơ đồ PowerPoint; **không cắt dán hình paper**, **không dump code thô**. Mọi số thực nghiệm của nhóm đều kèm link **W&B Public**.

---

## Mục lục phân bổ 36 Slides

| Nhóm Slide | Chủ đề chính | Số slides | Trọng tâm & Hình sử dụng |
|---|---|---:|---|
| **Slides 1–2** | Trang bìa, Mục lục & Core Scope | 2 | Tiêu đề, nhóm thực hiện, GitHub/W&B public links, tóm tắt kết quả |
| **Slides 3–7** | Context, Ví dụ thực tế, Tiền nhiệm, Đóng góp & Tính mới | 5 | Bài toán CIL, Ví dụ da liễu liên viện, LwF / iCaRL / PODNet & AANets + `acil_problem_and_predecessors.png`, Đóng góp 3 nhóm & Novelty |
| **Slides 8–16** | Method (5 lớp phân tích chuyên sâu) | 9 | Ký hiệu, Pipeline `acil_pipeline_end_to_end.png`, What/Why/How (CNN+FE, ARaBT, Woodbury $R_k$, Residual $W_k$), Định lý 3.1, Absolute Memorization & Privacy + Giới hạn, Độ phức tạp vs Tiền nhiệm |
| **Slides 17–20** | Thiết lập Thực nghiệm, Protocol, Metrics & Kiểm định Tiền kỳ | 4 | Core Scope R1–R3, CIFAR-100 & ResNet-32 Protocol v1.0, Metrics v1.0 & dấu $F$, Sanity Gates `VERIFY-01/02`, Smoke, Profile/BP5 |
| **Slides 21–23** | Kết quả R1 — Benchmark chính (3 seeds, $d=8000$) | 3 | Training curve (`r1_cifar100_k5_training.png`), Accuracy curve (`r1_cifar100_k5_accuracy.png`), Đối chiếu Table 1 & bối cảnh |
| **Slides 24–25** | Kết quả R2 — Ablation FE & Độ nhạy $\gamma$ | 2 | Biểu đồ & bảng 4/5 hàng Table 2 (`r2_cifar100_k5_ablation.png`), Phân tích nghịch lý $F_{\mathrm{base}}$ & giới hạn |
| **Slides 26–27** | Kết quả R3 — Phân tích Minibatch (tự thiết kế) | 2 | Tương đương đại số 54/54 mốc, Thời gian & Peak RSS (`r3_cifar100_k5_batch_analysis.png`) |
| **Slides 28–30** | Đối chiếu Implementation Tác giả & Phân tích Sai lệch | 3 | C1/C2/C1R tầng mapping–normalization–pooling–FE, Biểu đồ (`author_acil_comparison.png`), Tổng hợp sai lệch $-1.18\text{ pp}$ (Đã chứng minh vs Giả thuyết) |
| **Slides 31–32** | Bài học Kỹ thuật, Tổng kết & Định hướng Cuối kỳ | 2 | Quyết định kỹ thuật cốt lõi (`attempt_id`, streaming, checkpoint v2), Kết luận trung thực & Lộ trình Poster |
| **Slides 33–36** | Appendix kỹ thuật & Truy nguyên (Không có References slide) | 4 | A1: Chứng minh quy nạp Theorem 3.1 & Residual; A2: Bảng toàn bộ W&B runs; A3: Kiến trúc `src/`, Unit Tests & Thư viện; A4: Lệnh tái lập & Checklist nộp bài |

---

## Danh mục chuỗi trích dẫn IEEE đặt ở Footer từng Slide

- **[1]** H. Zhuang, Z. Weng, H. Wei, R. Xie, K.-A. Toh, and Z. Lin, "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, pp. 11602–11614, 2022.
- **[2]** S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, "iCaRL: Incremental classifier and representation learning," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2001–2010.
- **[3]** Z. Li and D. Hoiem, "Learning without forgetting," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 40, no. 12, pp. 2935–2947, Dec. 2018.
- **[4]** A. Douillard, M. Cord, C. Ollion, T. Robert, and E. Valle, "PODNet: Pooled outputs distillation for small-tasks incremental learning," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 86–102.
- **[5]** Y. Liu, B. Schiele, and Q. Sun, "Adaptive aggregation networks for class-incremental learning," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 2544–2553.
- **[6]** K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.
- **[7]** A. Krizhevsky, "Learning multiple layers of features from tiny images," Univ. Toronto, Toronto, ON, Canada, Tech. Rep., 2009.
- **[8]** S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, "Learning a unified classifier incrementally via rebalancing," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 831–839.

---

## PHẦN 1: TRANG BÌA, BỐI CẢNH, TIỀN NHIỆM & ĐÓNG GÓP (SLIDES 1–7)

### Slide 1 — Trang bìa (Title Slide)
- **Tiêu đề trên slide:** Đọc hiểu & Tái lập Thực nghiệm: **ACIL — Analytic Class-Incremental Learning** *(NeurIPS 2022)* [1]
- **Bố cục trực quan:** Nền trắng tối giản chuẩn học thuật, thanh nhấn xanh dương đậm bên trái, hộp thẻ thông tin nhóm & 2 thẻ liên kết Public GitHub / Public W&B ở nửa dưới.
- **Nội dung trên slide:**
  - **Bài báo gốc:** *ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection* (Zhuang et al., NeurIPS 2022) [1]
  - **Môn học:** Nhập môn Học máy (CS — Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM) — Báo cáo Giữa kỳ
  - **Thành viên thực hiện:** Sang (Phương pháp, Analytic Head, R3), Vĩ (Dataset, Protocol, ResNet-32, R1), Phát (Metrics, Ablation R2, W&B, Tổng hợp) *(Ghi chú thực tế triển khai theo `docs/PHAN_CONG_NHOM.md`: Sang chủ trì chạy & tổng hợp toàn bộ hệ thống R1–R3 và đối chiếu tác giả)*
  - **Mã nguồn công khai:** `https://github.com/24122050S/20-ML-Paper`
  - **Theo dõi thực nghiệm công khai (W&B):** `https://wandb.ai/hsangg-2006/acil-reproduction`
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 35, pp. 11602–11614, 2022.`
- **Speaker Notes (~160 từ):**
  > Chào thầy cô và các anh chị trợ giảng. Đây là báo cáo giữa kỳ môn Nhập môn Học máy của nhóm với bài báo mục tiêu ACIL, công bố tại NeurIPS 2022. Bài trình bày gồm 36 slides (32 slides nội dung chính và 4 slides Appendix kỹ thuật), thực hiện hai nhiệm vụ trọng tâm của giai đoạn giữa kỳ: (1) Đọc hiểu và hệ thống hóa cơ sở toán học của phương pháp học tăng dần giải tích ACIL; (2) Tự cài đặt mã nguồn từ mô tả bài báo, tái lập các thực nghiệm cốt lõi trên bộ dữ liệu CIFAR-100, mở rộng phân tích tác động kích thước minibatch và đối chiếu trực tiếp với mã nguồn tác giả công bố. Toàn bộ mã nguồn, kết quả JSON, kiểm chứng tự động và biểu đồ tự vẽ đều được công khai trên repository GitHub `24122050S/20-ML-Paper`, kèm nhật ký 31 lượt chạy hoàn tất công khai trên Weights & Biases (`hsangg-2006/acil-reproduction`). Xin mời thầy cô theo dõi từ tổng quan phạm vi ở Slide 2.

---

### Slide 2 — Tổng quan Cấu trúc & Phạm vi Đồ án (Roadmap & Core Scope)
- **Tiêu đề trên slide:** Lộ trình trình bày & Các cột mốc thực nghiệm đã hoàn tất trong Core Scope
- **Bố cục trực quan:** Sơ đồ 4 khối ngang (Phần I $\to$ IV) ở nửa trên; lưới 4 thẻ số liệu nổi bật ở nửa dưới (R1, R2, R3, Đối chiếu tác giả).
- **Nội dung trên slide:**
  - **Cấu trúc 4 phần (32 slides chính + 4 Appendix):**
    1. **Bối cảnh & Tiền nhiệm (Slides 3–7):** Bài toán CIL, ví dụ y tế, LwF / iCaRL / PODNet & AANets, đóng góp & tính mới.
    2. **Phương pháp 5 lớp (Slides 8–16):** Ký hiệu, Pipeline tổng thể, What/Why/How, Định lý 3.1 & Giới hạn, Độ phức tạp.
    3. **Thực nghiệm Tái lập (Slides 17–30):** Protocol/Metrics, Sanity Gates, **R1** (3 seeds), **R2** (ablation), **R3** (batch), **Author Comparison**.
    4. **Tổng kết & Định hướng Cuối kỳ (Slides 31–32):** Bài học kỹ thuật, giới hạn kết luận, kế hoạch `Critical Analysis` & `Empirical Demonstration`.
  - **Tóm tắt bằng chứng đã thực thi (Core Scope — CIFAR-100, $K=5$):**
    - **R1 (3 seeds, $d=8000$):** $\bar{A} = 65.12 \pm 0.61\%$ (so với paper $66.30\%$, lệch $-1.18\text{ pp}$).
    - **R2 (4/5 hàng Table 2 × 3 seeds):** FE giúp tăng $+11.86\text{ pp}$; $\gamma \in [0.001, 0.1]$ biến thiên chỉ $0.0032\text{ pp}$.
    - **R3 (tự thiết kế, $d=1024$):** 54/54 mốc khớp Joint Oracle (sai số $\le 1.16\times 10^{-11}$, 0 argmax khác); batch 512 nhanh gấp $3.47\times$ batch 32.
    - **Author Compare (`0afd6bf`):** C1/C2/C1R định vị sai khác chuẩn hóa, pooling FP32 và khởi tạo base.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022.`
- **Speaker Notes (~185 từ):**
  > Slide này phác họa bản đồ toàn bộ 36 slides. Do ràng buộc tài nguyên GPU Kaggle và quy mô thực hiện, nhóm đăng ký trước **Core Scope** theo đúng hướng dẫn của môn học, tập trung kiểm chứng sâu trên bộ dữ liệu CIFAR-100 với kịch bản $50$ lớp nền cộng $5$ phase mỗi phase $10$ lớp ($K=5$). Nhóm không tự chạy lại các baseline khác trong Bảng 1 hay mở rộng sang ImageNet hoặc $K=25, 50$; do đó mọi nhận định về ưu thế large-$K$ của bài báo chỉ được trích dẫn đối chiếu đúng nguồn và không gán thành kết quả tự chạy của nhóm. Toàn bộ bốn kết quả tóm tắt ở nửa dưới slide — gồm R1 ba seed độc lập tại $d=8000$, R2 ablation bốn cấu hình, R3 phân tích ảnh hưởng minibatch $32/128/512$, và phần chạy đối chiếu trực tiếp mã nguồn tác giả `0afd6bf` — đều đã chạy xong hoàn chỉnh với provenance SHA-256 và link W&B công khai trước khi tổng hợp deck. Trước khi đi vào thực nghiệm, Slides 3–7 sẽ giải quyết câu hỏi: vì sao bài toán Class-Incremental Learning khó và các phương pháp trước ACIL gặp bế tắc ở đâu.

---

### Slide 3 — Bài toán Class-Incremental Learning (CIL) & 3 Ràng buộc Cốt lõi
- **Tiêu đề trên slide:** Học tăng dần theo lớp (CIL): Phải phân loại mọi lớp đã gặp mà không được giữ dữ liệu cũ
- **Bố cục trực quan:** Thanh timeline tuần tự `Phase 0` $\to$ `Phase 1` $\to \dots \to$ `Phase K` $\to$ `Single-Head Eval` ở trên; 3 cột phân tích thách thức cốt lõi ở dưới.
- **Nội dung trên slide:**
  - **Thiết lập toán học CIL [1], [2]:**
    - Chuỗi phase $k = 0, 1, \dots, K$: mỗi phase $k$ chỉ cung cấp dữ liệu $\mathcal{D}_k^{\text{train}} = \{(X_k^{(i)}, y_k^{(i)})\}_{i=1}^{N_k}$ của $c_k$ lớp mới; tập lớp không trùng ($\mathcal{C}_i \cap \mathcal{C}_j = \emptyset$).
    - Sau phase $k$, mô hình phải dự đoán đúng trên **tất cả $C_k = \sum_{j=0}^{k} c_j$ lớp đã thấy** (*single-head, task-agnostic* — **không biết Task ID** khi test).
  - **Ba ràng buộc làm bế tắc học sâu truyền thống:**
    1. **Không dữ liệu cũ (No Raw Old Data / Exemplar-Free):** $\mathcal{D}_0, \dots, \mathcal{D}_{k-1}$ bị xóa vì giới hạn bộ nhớ hoặc quy định bảo mật dữ liệu gốc [1].
    2. **Tiến thoái lưỡng nan Ổn định – Dẻo (*Stability–Plasticity Dilemma*):** Huấn luyện SGD trên dữ liệu mới sẽ ghi đè trọng số, xóa sạch biên phân cách lớp cũ (*Catastrophic Forgetting*) [2], [3].
    3. **Cạnh tranh $\operatorname{argmax}$ mở rộng (*Argmax Competition*):** Số lớp ứng viên tăng từ $50 \to 100$ làm vùng quyết định của lớp cũ bị thu hẹp ngay cả khi biểu diễn không đổi.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., "iCaRL," CVPR, 2017. [3] Z. Li and D. Hoiem, "Learning without forgetting," IEEE TPAMI, 2018.`
- **Speaker Notes (~195 từ):**
  > Slide 3 định nghĩa chính xác bài toán Class-Incremental Learning, gọi tắt là CIL, mà ACIL giải quyết. Ở đây $k$ là chỉ số pha huấn luyện; pha $0$ là tập nền ban đầu có $c_0$ lớp, các pha $k \ge 1$ lần lượt đưa thêm $c_k$ lớp hoàn toàn mới và rời nhau từng đôi một. Điểm khắt khe nhất của CIL nằm ở chế độ đánh giá single-head không cung cấp Task ID lúc suy luận: một ảnh kiểm tra bất kỳ có thể thuộc bất kỳ lớp nào trong tổng số $C_k$ lớp đã xuất hiện từ đầu đến nay. Nếu ta làm theo cách ngây thơ là lấy mạng nơ-ron từ pha $k-1$ rồi chạy hạ gradient SGD trên tập mới $\mathcal{D}_k^{\text{train}}$, gradient từ Cross-Entropy sẽ kéo toàn bộ đầu ra về $c_k$ lớp mới và gây ra hiện tượng quên thảm khốc (*Catastrophic Forgetting*). Khó khăn nhân đôi khi ta áp đặt ràng buộc *exemplar-free*, tức cấm lưu lại ảnh gốc của các pha trước vì lý do bộ nhớ biên hoặc bảo mật. Ngoài ra, ngay cả khi đặc trưng giữ nguyên, việc mở rộng không gian $\operatorname{argmax}$ từ 50 lên 100 lớp tự nhiên làm tăng xác suất nhầm lẫn chéo. Sang Slide 4, nhóm minh họa sự cấp thiết của các ràng buộc này qua một ví dụ thực tế tự thiết kế.

---

### Slide 4 — Ví dụ Thực tế Tự đặt: Sàng lọc Tổn thương Da liễu Liên viện
- **Tiêu đề trên slide:** Ví dụ thực tế: Vì sao hệ thống AI y tế tuyến tỉnh cần CIL không lưu mẫu cũ?
- **Bố cục trực quan:** Sơ đồ luồng kịch bản bệnh viện (Trái: Quý 0 — 50 bệnh phổ biến $\to$ Quý $k$ — +10 biến thể mới nổi $\to$ Chẩn đoán hợp nhất); Phải: 3 thẻ nguyên nhân thất bại của các cách tiếp cận cổ điển.
- **Nội dung trên slide:**
  - **Bối cảnh thực tế (Nhóm tự đặt):** Hệ thống hỗ trợ sàng lọc tổn thương da liễu triển khai qua nhiều quý tại các bệnh viện tuyến tỉnh:
    - **Phase 0 (Khởi tạo):** Mạng thị giác máy tính học **50 bệnh da liễu phổ biến** (viêm da cơ địa, vảy nến, nấm da...).
    - **Phases $k = 1 \dots K$ (Mỗi quý):** Phát sinh thêm **10 biến thể bệnh mới nổi / hiếm gặp** (tổn thương hậu nhiễm virus mới, phản ứng thuốc, ký sinh trùng địa phương).
    - **Yêu cầu lâm sàng:** Khi bệnh nhân đến khám, bác sĩ cần mô hình phân loại đồng thời cả $50 + 10k$ bệnh (*không thể hỏi trước bệnh nhân thuộc "đợt dữ liệu quý nào"*).
  - **Vì sao cách tiếp cận thông thường thất bại?**
    - **Cấm lưu kho ảnh bệnh nhân cũ (Pháp lý & Quyền riêng tư):** Ảnh tổn thương da là dữ liệu nhạy cảm định danh bệnh nhân; quy chế bảo vệ dữ liệu y tế buộc **xóa ảnh gốc sau mỗi đợt huấn luyện**, không thể đóng gói exemplar buffer gửi liên viện.
    - **Fine-tuning thuần túy $\to$ Quên lâm sàng:** Học SGD chỉ trên 10 bệnh mới khiến mô hình dự đoán nhầm 50 bệnh nền phổ biến thành 10 bệnh mới.
    - **Joint Retrain từ đầu $\to$ Bất khả thi:** Vừa vi phạm lưu trữ hồ sơ ảnh, vừa tốn hàng chục giờ GPU mỗi lần cập nhật nhỏ tại bệnh viện tuyến dưới.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Để cụ thể hóa động lực nghiên cứu, nhóm xây dựng một kịch bản ứng dụng thực tế trong y tế: hệ thống hỗ trợ sàng lọc tổn thương da liễu liên viện. Tại thời điểm triển khai ban đầu, tức Phase 0, hệ thống nhận diện 50 bệnh lý da phổ biến. Sang mỗi quý tiếp theo, tương ứng mỗi Phase $k$, cơ sở y tế ghi nhận thêm 10 nhóm tổn thương hoặc biến thể bệnh mới nổi. Trong phòng khám, ảnh chụp da của bệnh nhân mới có thể rơi vào cả bệnh cũ lẫn bệnh mới, nên đầu ra bắt buộc phải gộp chung $50 + 10k$ lớp trên một head duy nhất. Bài toán này vấp phải hai rào cản sống còn: thứ nhất là quy định bảo vệ quyền riêng tư và dữ liệu cá nhân không cho phép giữ hoặc luân chuyển kho ảnh bệnh nhân từ các quý trước để chạy huấn luyện gộp hay phát lại (*exemplar replay*); thứ hai là hạ tầng tính toán tại bệnh viện tuyến tỉnh hạn chế, không thể chạy lại hàng trăm epoch backpropagation mỗi quý. Do đó, hệ thống cần một thuật toán học tăng dần không lưu ảnh thô, cập nhật nhanh và tuyệt đối không quên kiến thức phân biệt cũ. Vậy các phương pháp tiền nhiệm trước ACIL đã thử giải quyết như thế nào? Chúng ta xem Slides 5–6.

---

### Slide 5 — Tiền nhiệm 1 & 2: Regularization (LwF) và Exemplar Replay (iCaRL)
- **Tiêu đề trên slide:** Phân tích hai hướng tiền nhiệm kinh điển: Ý tưởng, Hạn chế & Hệ quả thực tế
- **Bố cục trực quan:** Chia 2 cột dọc đối xứng cho **LwF [3]** và **iCaRL [2]**, mỗi cột cấu trúc rõ đúng 3 tầng có nhãn màu: *Core Idea* (Xanh) $\to$ *Remaining Limitation* (Vàng) $\to$ *Practical Consequence* (Đỏ).
- **Nội dung trên slide:**
  - **Tiền nhiệm 1 — Learning without Forgetting (LwF, Li & Hoiem 2018) [3]** *(Regularization / Knowledge Distillation, không lưu mẫu)*
    - **Ý tưởng cốt lõi:** Đóng băng bản sao mạng phase $k-1$ làm *Teacher*. Khi học ảnh lớp mới $X_k$, tối ưu tổng *Cross-Entropy trên lớp mới + Knowledge Distillation* ép logit lớp cũ của Student bám sát logit của Teacher trên chính ảnh $X_k$.
    - **Hạn chế còn tồn tại:** Truy vấn mạng Teacher bằng **dữ liệu ngoài phân phối ($X_k$ của lớp mới)** thay vì ảnh lớp cũ thực sự; sai số chưng cất bị khuếch đại qua nhiều phase (*error compounding*) và không hiệu chỉnh được biên quyết định trực tiếp giữa lớp cũ–mới [1], [2].
    - **Hệ quả thực tế:** Độ chính xác sụt giảm nghiêm trọng khi chuỗi phase kéo dài; mạng bị thiên lệch nặng về các lớp học sau cùng và mất khả năng nhận diện lớp nền ban đầu.
  - **Tiền nhiệm 2 — iCaRL (Rebuffi et al. 2017) [2]** *(Exemplar Replay + Nearest-Mean Classifier)*
    - **Ý tưởng cốt lõi:** Lưu bộ nhớ đệm cố định $\mathcal{M}$ ảnh gốc cũ (vd. $M = 2000$ ảnh CIFAR-100 chọn bằng *herding*); gộp $\mathcal{M} \cup \mathcal{D}_k^{\text{train}}$ chạy SGD + distillation, phân loại bằng tâm lớp gần nhất trong không gian chuẩn hóa $L_2$.
    - **Hạn chế còn tồn tại:** Bắt buộc **lưu giữ ảnh gốc quá khứ**; khi tổng số lớp $C_k$ tăng mà $|\mathcal{M}|$ cố định, số mẫu/lớp cũ ($m = |\mathcal{M}|/C_k$) co hẹp tạo **mất cân bằng lớp nặng** với hàng trăm mẫu/lớp mới [1], [4].
    - **Hệ quả thực tế:** **Bị chặn triển khai** ở miền cấm lưu dữ liệu thô (y tế, tài chính); vẫn tốn chi phí lặp SGD nhiều epoch ở mọi phase.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., "iCaRL," CVPR, 2017. [3] Z. Li and D. Hoiem, "Learning without forgetting," IEEE TPAMI, 2018. [4] A. Douillard et al., "PODNet," ECCV, 2020.`
- **Speaker Notes (~210 từ):**
  > Trước khi ACIL ra đời, cộng đồng CIL phát triển hai trường phái chính. Trường phái thứ nhất là Regularization và Knowledge Distillation không lưu mẫu, tiêu biểu là **LwF** [3]. Điểm mạnh của LwF là không cần giữ ảnh cũ, mà dùng mạng phase $k-1$ làm giáo viên để hướng dẫn mạng hiện tại giữ nguyên logit trên nhóm lớp cũ. Tuy nhiên, do không có dữ liệu cũ, giáo viên buộc phải sinh nhãn mềm trên chính ảnh của lớp mới $X_k$ — tức chưng cất ngoài phân phối. Hạn chế này tích lũy sai số qua từng bước và thiếu hoàn toàn lực đẩy phân tách trực tiếp giữa mẫu lớp cũ với lớp mới; hệ quả thực tế là khi sang các pha sau, LwF bị thiên lệch dự đoán mạnh về lớp mới và quên sâu các lớp pha đầu. Để khắc phục, trường phái thứ hai như **iCaRL** [2] chấp nhận lưu một kho ảnh đại diện $\mathcal{M}$ giới hạn (ví dụ 2000 ảnh) để phát lại cùng dữ liệu mới. Hạn chế rõ ràng của iCaRL là phá vỡ hoàn toàn tính bảo mật dữ liệu gốc; đồng thời khi số lớp tăng lên 80–100 mà ngân sách bộ nhớ cố định, mỗi lớp cũ chỉ còn vỏn vẹn 20 ảnh đối chọi với 500 ảnh/lớp mới, gây lệch trọng số và buộc phải tinh chỉnh lại toàn bộ CNN mỗi pha. Tiếp theo, Slide 6 phân tích các phương pháp Replay hiện đại hơn và đối chiếu trực quan với ACIL.

---

### Slide 6 — Tiền nhiệm 3 (PODNet & AANets) & Đối chiếu Cấu trúc với ACIL
- **Tiêu đề trên slide:** Ngay cả các phương pháp Distillation + Replay SOTA vẫn suy thoái khi số phase $K$ lớn
- **Bố cục trực quan:** Nhúng sơ đồ tự vẽ `reports/figures/acil_problem_and_predecessors.png` chiếm ~65% diện tích bên phải/dưới; cột tóm tắt **PODNet [4] & AANets [5]** ở bên trái/trên.
- **Nội dung trên slide:**
  - **Tiền nhiệm 3 — PODNet (Douillard et al. 2020) [4] & AANets (Liu et al. 2021) [5]:**
    - **Ý tưởng cốt lõi:** Nâng cấp iCaRL bằng chưng cất đặc trưng không gian đa tầng theo chiều rộng–cao (*Pooled Outputs Distillation* — PODNet) hoặc ghép song song hai nhánh dư *plastic–stable* có trọng số thích nghi (AANets), kết hợp exemplar replay (vd. 20 ảnh/lớp) [4], [5].
    - **Hạn chế còn tồn tại:** Vẫn phụ thuộc lưu dữ liệu thô và vòng lặp SGD qua nhiều epoch ở mọi phase; đặc trưng liên tục bị trượt (*representation drift*) khi chia thành nhiều đợt nhỏ ($K = 25, 50$) [1].
    - **Hệ quả thực tế:** Tốn chi phí tính toán GPU tỷ lệ với $K \times E$ epochs; theo Table 1 bài báo [1], trên CIFAR-100 từ $K=5 \to K=25$, $\bar{A}$ của PODNet giảm $64.83\% \to 60.72\%$ ($-4.11\text{ pp}$), AANets giảm $66.31\% \to 62.31\%$ ($-4.00\text{ pp}$).
  - **Hình minh họa tự vẽ:** Chèn [`reports/figures/acil_problem_and_predecessors.png`](../reports/figures/acil_problem_and_predecessors.png) (Panel A: Bối cảnh CIL & 3 ràng buộc; Panel B: Đối chiếu 3 hướng *Regularization vs. Exemplar Replay vs. ACIL*).
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., CVPR, 2017. [3] Z. Li and D. Hoiem, IEEE TPAMI, 2018. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021.`
- **Speaker Notes (~200 từ):**
  > Slide 6 xét nhóm tiền nhiệm hiện đại hơn là **PODNet** và **AANets**, cũng chính là hai phương pháp đặt chuẩn protocol 50% lớp nền cho bài báo ACIL. PODNet cải thiện độ ổn định biểu diễn nhờ chưng cất đặc trưng không gian tại các tầng trung gian, còn AANets kết hợp một nhánh dư ổn định bảo tồn tri thức cũ với một nhánh dẻo học đặc trưng mới. Dù đạt độ chính xác rất cao ở bài toán ít pha ($K=5$, PODNet đạt $64.83\%$ và AANets đạt $66.31\%$ theo Table 1 [1]), cả hai vẫn gặp hai rào cản: vẫn phải lưu ảnh mẫu cũ (thường 20 ảnh mỗi lớp) và vẫn phải lặp SGD ở từng pha mới. Hệ quả là khi bài toán chia nhỏ thành nhiều pha liên tiếp như $K=25$, sự dịch chuyển không gian đặc trưng tích tụ khiến cả hai giảm khoảng $4\text{ điểm phần trăm}$ độ chính xác trung bình. Sơ đồ tự vẽ bên phải đặt cạnh nhau ba hướng tiếp cận: Regularization không giữ mẫu nhưng trôi lệch nặng; Exemplar Replay hiệu năng khá hơn nhưng vi phạm quyền riêng tư và giảm sút ở $K$ lớn; trong khi ACIL giữ trạng thái toán học $(R_k, W_k)$ không lưu ảnh gốc và đạt nghiệm tương đương học gộp giải tích.

---

### Slide 7 — Phân loại Đóng góp & Tính mới của ACIL (Contributions & Novelty)
- **Tiêu đề trên slide:** Đóng góp 3 chiều của ACIL: Đổi mô hình cập nhật từ SGD lặp sang Giải tích Tuyến tính Có Đáp án Đóng
- **Bố cục trực quan:** Bảng 3 hàng phân loại đóng góp (*Theoretical / Technical / Empirical*) bên trái; Khối so sánh *Tiền nhiệm chưa làm được vs. Điểm mới của ACIL* bên phải.
- **Nội dung trên slide:**
  - **1. Phân loại đóng góp của bài báo [1]:**
    - **Theoretical (Lý thuyết):** Phát biểu & chứng minh **Định lý 3.1** rằng cập nhật đệ quy không giữ mẫu cũ ($W_k$) cho nghiệm **bằng chính xác tuyệt đối** học gộp Ridge ($\tilde{W}_k$); đặt khái niệm *Absolute Memorization*.
    - **Technical / Algorithmic (Kỹ thuật):** Đề xuất chuỗi **Freeze CNN + Random Feature Expansion ($\operatorname{ReLU}(F W_{\mathrm{fe}})$) + ARaBT ($R_0, W_0$) + Cập nhật Woodbury RLS theo batch/phase ($R_k, W_k$)** chạy một lượt duy nhất (*single-pass*).
    - **Empirical (Thực nghiệm):** Trên CIFAR-100 & ImageNet (trích bài báo [1]), phương pháp exemplar-free này giữ $\bar{A}$ ổn định theo $K$ và vượt PODNet/AANets ở $K=25, 50$ ($65.95\%$ vs $60.72\% / 62.31\%$ trên CIFAR-100 $K=25$).
  - **2. Tính mới cốt lõi đối chiếu tiền nhiệm:**
    - *Tiền nhiệm (SGD + Backprop):* Quỹ đạo gradient cục bộ thiếu dữ liệu cũ tất yếu lệch khỏi nghiệm học gộp $\to$ phụ thuộc thứ tự phase và suy giảm theo $K$.
    - *ACIL (Closed-form Analytic Update):* Tách học biểu diễn (chỉ Phase 0) khỏi giải bộ phân loại lồi có **thống kê đủ bậc hai cộng tính** $\sum H_j^T H_j$ $\to$ nén trọn lịch sử vào duy nhất ma trận $R_k \in \mathbb{R}^{d\times d}$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021.`
- **Speaker Notes (~190 từ):**
  > Slide 7 hệ thống hóa toàn bộ đóng góp của ACIL theo ba trục học thuật bắt buộc. Thứ nhất về **lý thuyết**, ACIL chứng minh Định lý 3.1: nghiệm bộ phân loại sau chuỗi cập nhật đệ quy đồng nhất đại số với nghiệm của bài toán hồi quy Ridge khi gộp chung toàn bộ đặc trưng lịch sử — từ đó thiết lập tính chất Ghi nhớ Tuyệt đối ở tầng phân loại. Thứ hai về **kỹ thuật**, tác giả thiết kế sự kết hợp giữa đóng băng backbone sau Phase 0, tầng mở rộng đặc trưng ngẫu nhiên phi tuyến FE nhằm khôi phục tính tách tuyến tính, bước khởi tạo ARaBT thay lớp FC bằng nghiệm giải tích, và công thức cập nhật Woodbury chỉ đi qua dữ liệu mới đúng một lượt. Thứ ba về **thực nghiệm**, ACIL phá vỡ định kiến rằng phương pháp không lưu mẫu luôn thua kém exemplar replay khi chứng minh hiệu năng trên CIFAR-100 và ImageNet-Full ở các thiết lập nhiều pha ($K=25, 50$). Điểm mới mang tính bước ngoặt của ACIL so với toàn bộ tiền nhiệm là chuyển bài toán học tăng dần từ việc xấp xỉ quỹ đạo gradient phi lồi sang bài toán hồi quy lồi bậc hai, nơi thông tin quá khứ có thể tích lũy chính xác mà không giữ nguyên bản dữ liệu gốc. Phần tiếp theo sẽ mổ xẻ phương pháp ACIL qua 5 lớp chi tiết.

---

## PHẦN 2: METHOD — PHÂN TÍCH PHƯƠNG PHÁP THEO 5 LỚP (SLIDES 8–16)

### Slide 8 — Lớp 1: Hệ thống Ký hiệu & Hàm Mục tiêu Học Gộp
- **Tiêu đề trên slide:** Lớp 1 (Ký hiệu & Bài toán): Mô hình hóa CIL thành bài toán hồi quy Ridge trên đặc trưng cố định
- **Bố cục trực quan:** Bảng ký hiệu đồng nhất (kèm Shape & Vai trò) ở bên trái (60%); hộp công thức hàm mục tiêu Joint Analytic Ridge & Nghiệm đóng ở bên phải (40%).
- **Nội dung trên slide:**
  - **Hệ thống ký hiệu chuẩn (đồng bộ bài báo [1] & [`docs/Dac_ta_ACIL.md`](Dac_ta_ACIL.md)):**
    - $k \in \{0, \dots, K\}$: chỉ số phase ($k=0$ là Base phase); $c_k$: số lớp mới ở phase $k$; $C_k = \sum_{j=0}^k c_j$: tổng lớp đã học.
    - $F_k = f_{\text{cnn}}(X_k; \theta_0) \in \mathbb{R}^{N_k \times d_{\text{cnn}}}$: đặc trưng sau Global Avg Pool của CNN đóng băng ($d_{\text{cnn}} = 64$ với ResNet-32).
    - $W_{\mathrm{fe}} \in \mathbb{R}^{d_{\text{cnn}} \times d}$: ma trận chiếu ngẫu nhiên cố định ($\mathcal{N}(0,1)$, $d=8000$).
    - $H_k = \operatorname{ReLU}(F_k W_{\mathrm{fe}}) \in \mathbb{R}^{N_k \times d}$: đặc trưng mở rộng ở phase $k$; $\tilde{H}_k = [H_0^T, \dots, H_k^T]^T$: ghép dọc toàn bộ lịch sử.
    - $Y_k \in \mathbb{R}^{N_k \times c_k}$: one-hot cục bộ phase $k$; $\tilde{Y}_k = \operatorname{diag}(Y_0, \dots, Y_k)$: nhãn one-hot lịch sử đến phase $k$.
    - $R_k \in \mathbb{R}^{d \times d}$, $W_k \in \mathbb{R}^{d \times C_k}$, $\gamma > 0$: ma trận nghịch đảo tương quan (RFAM), trọng số phân loại và hệ số Ridge.
  - **Hàm mục tiêu học gộp tại phase $k$ [1]:**
    $$
    \min_{W \in \mathbb{R}^{d \times C_k}} \left\| \tilde{Y}_k - \tilde{H}_k W \right\|_F^2 + \gamma \| W \|_F^2 \implies \tilde{W}_k = \left( \tilde{H}_k^T \tilde{H}_k + \gamma I_d \right)^{-1} \tilde{H}_k^T \tilde{Y}_k
    $$
    *Thách thức thuật toán:* Tính đúng $\tilde{W}_k$ chỉ từ $(H_k, Y_k)$ và $(R_{k-1}, W_{k-1})$, **không truy cập $\tilde{H}_{k-1}, \tilde{Y}_{k-1}$**.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~205 từ):**
  > Bắt đầu Lớp 1 của phần Phương pháp, Slide 8 thống nhất toàn bộ hệ thống ký hiệu toán học dùng xuyên suốt từ bài báo [1] sang mã nguồn tái lập của nhóm. Ở mỗi pha $k$, $N_k$ là số ảnh huấn luyện mới (chẳng hạn $25{,}000$ ảnh cho pha $0$ và $5{,}000$ ảnh cho mỗi pha $k \ge 1$ trên CIFAR-100). Mỗi ảnh đi qua mạng ResNet-32 đã đóng băng ở pha $0$ cho vector đặc trưng $64$ chiều $F_k$, sau đó nhân với ma trận cố định $W_{\mathrm{fe}}$ và đi qua $\operatorname{ReLU}$ để tạo ma trận đặc trưng mở rộng $H_k$ cỡ $N_k \times d$ với $d = 8000$. Khi xét bài toán lý tưởng có đủ toàn bộ dữ liệu từ đầu đến pha $k$ — ký hiệu ma trận ghép dọc là $\tilde{H}_k$ và ma trận nhãn one-hot khối chéo là $\tilde{Y}_k$ — ACIL không dùng hàm mất mát Cross-Entropy phi tuyến mà dùng hàm mất mát bình phương tối thiểu có điều chuẩn $L_2$ ($\gamma > 0$), tức Ridge Regression với chuẩn Frobenius $\|\cdot\|_F^2$. Lợi ích lớn nhất của công thức bên phải là hàm mục tiêu lồi chặt nên có duy nhất một nghiệm toàn cục dạng đóng $\tilde{W}_k$. Câu hỏi cốt lõi ở các slide kế tiếp là: làm sao tính được $\tilde{W}_k$ tăng dần qua từng pha mà chỉ giữ trạng thái $(R_{k-1}, W_{k-1})$?

---

### Slide 9 — Lớp 2: Kiến trúc & Luồng Pipeline End-to-End của ACIL
- **Tiêu đề trên slide:** Lớp 2 (Pipeline tổng thể): Backpropagation chỉ chạy ở Phase 0; các phase sau thuần đại số tuyến tính
- **Bố cục trực quan:** Nhúng hình tự vẽ [`reports/figures/acil_pipeline_end_to_end.png`](../reports/figures/acil_pipeline_end_to_end.png) làm trung tâm (~85% chiều rộng), phía dưới thêm dải 3 gạch đầu dòng tóm tắt luồng chuyển giao giữa 3 Stages.
- **Nội dung trên slide:**
  - **Sơ đồ tự vẽ:** Chèn [`reports/figures/acil_pipeline_end_to_end.png`](../reports/figures/acil_pipeline_end_to_end.png) (13 khối xử lý chia thành 3 giai đoạn khớp với `src/resnet.py`, `src/base_training.py`, `src/analytic.py`, `src/streaming.py`).
  - **Ba giai đoạn nối tiếp:**
    - **Stage 1 (Phase 0 — Base BP & ARaBT):** Huấn luyện ResNet-32 + `FC50` (160 epochs SGD) $\to$ đóng băng (`eval()`) và bỏ `FC50` $\to$ trích $F_0 \to H_0$ $\to$ khởi tạo nghiệm đóng $(R_0, W_0)$.
    - **Stage 2 (Phases $k = 1 \dots K$ — Analytic Incremental):** Trích $F_k \to H_k$ bằng cùng backbone/FE đóng băng (0 gradient, 0 mẫu cũ) $\to$ đệm cột $W_{\mathrm{pad}}$ $\to$ cập nhật $(R_k, W_k)$ theo Phương trình (9)–(10).
    - **Stage 3 (All-Seen Inference):** Ảnh kiểm tra qua đúng backbone/FE $\to$ tính điểm $S = H_{\mathrm{test}} W_k \to \hat{y} = \operatorname{argmax}_c S_{:,c}$ trên toàn bộ $C_k$ lớp.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [6] K. He et al., "Deep residual learning for image recognition," CVPR, 2016.`
- **Speaker Notes (~195 từ):**
  > Slide 9 trình bày bức tranh kiến trúc đầu-cuối do nhóm tự vẽ lại, phản ánh đầy đủ luồng dữ liệu từ ảnh thô tới dự đoán lớp cuối cùng. Khác với các sơ đồ giản lược chỉ vẽ công thức cập nhật, hình này cho thấy rõ vai trò của từng giai đoạn trong thực tế triển khai. Ở **Stage 1**, chỉ duy nhất ở Phase 0, ảnh có augmentation (padding, crop ngẫu nhiên, lật ngang) mới được đưa vào huấn luyện ResNet-32 bằng backpropagation trong 160 epochs để học bộ trích xuất đặc trưng cơ sở cho 50 lớp đầu. Ngay sau khi kết thúc BP, đầu ra `FC50` tạm thời bị loại bỏ, toàn bộ trọng số và thống kê BatchNorm của ResNet-32 được khóa cứng (`eval()`), và ta trích xuất đặc trưng sạch (không crop/flip) qua tầng chiếu ngẫu nhiên $W_{\mathrm{fe}}$ để khởi tạo cặp trạng thái giải tích $(R_0, W_0)$. Sang **Stage 2**, khi các pha mới $k=1\dots K$ xuất hiện, ảnh mới chỉ đi xuôi một lượt qua ResNet-32 đóng băng rồi cập nhật trực tiếp $(R_k, W_k)$ mà không lưu ảnh quá khứ. Cuối cùng ở **Stage 3**, lúc đánh giá, mọi mẫu test từ pha $0$ đến pha $k$ cạnh tranh bình đẳng qua tích $H_{\mathrm{test}} W_k$ mà không cần Task ID. Slides 10–13 sẽ đi sâu *What / Why / How* của từng khối.

---

### Slide 10 — Lớp 3 (Thành phần 1 & 2): Đóng băng Backbone CNN & Feature Expansion (FE)
- **Tiêu đề trên slide:** Lớp 3a (*What / Why / How*): Vì sao phải đóng băng CNN và chiếu đặc trưng từ 64 lên 8000 chiều?
- **Bố cục trực quan:** Chia 2 hộp ngang lớn: Hộp trên trình bày **Khối Đóng băng CNN $\theta_0$**; Hộp dưới trình bày **Khối Feature Expansion $H_k = \operatorname{ReLU}(F_k W_{\mathrm{fe}})$**, mỗi hộp có đúng 3 dòng `What` / `Why` / `How` + thẻ bằng chứng R2 nhỏ ở góc phải.
- **Nội dung trên slide:**
  - **Thành phần 1 — Backbone CNN Đóng băng ($\operatorname{CNN}_{\theta_0}$):**
    - **What:** ResNet-32 CIFAR ($d_{\text{cnn}} = 64$ sau Global Avg Pool) huấn luyện bằng SGD + Cross-Entropy trên 50 lớp base, sau đó bỏ `FC50`, khóa `requires_grad = False` và chuyển `eval()`.
    - **Why:** (i) Tránh gradient mới làm trượt không gian đặc trưng các lớp cũ (*representation drift*); (ii) Điều kiện tiên quyết để các phase sau nằm trên **cùng ánh xạ đặc trưng tĩnh** với phase 0 [1].
    - **How:** Chuẩn hóa ảnh bằng `mean/std` cố định từ base train; chạy tiến ở `float32` rồi ép `float64` sang tầng analytic; xác minh fingerprint tham số & BatchNorm không đổi qua các phase.
  - **Thành phần 2 — Random Feature Expansion (FE):**
    - **What:** Chiếu tuyến tính ngẫu nhiên cố định kết hợp $\operatorname{ReLU}$: $H_k = \operatorname{ReLU}(F_k W_{\mathrm{fe}})$, $W_{\mathrm{fe}} \in \mathbb{R}^{64 \times 8000}$ ($\mathcal{N}(0,1)$, không bias) [1].
    - **Why:** Không gian $64$ chiều trực tiếp từ ResNet-32 nhỏ hơn cả tổng $100$ lớp (*under-parameterized*); theo **Định lý Cover**, phép chiếu phi tuyến lên $d=8000 \gg 64$ giúp các lớp tách tuyến tính tốt hơn hẳn. *(Thí nghiệm R2 của nhóm: bật FE giúp $\bar{A}$ tăng **$+11.86\text{ pp}$** từ $53.25\%$ lên $65.12\%$)*.
    - **How:** Sinh $W_{\mathrm{fe}}$ một lần bằng RNG riêng (`seed + 1000`), không cập nhật gradient, giữ cố định tuyệt đối suốt từ Phase 0 đến $K$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [6] K. He et al., "Deep residual learning for image recognition," CVPR, 2016.`
- **Speaker Notes (~215 từ):**
  > Slide 10 phân tích hai khối đầu tiên của Stage 1 theo khung *What – Why – How*. Ở khối thứ nhất là backbone ResNet-32 đóng băng. Nhiều câu hỏi đặt ra: tại sao không fine-tune nhẹ thêm ở pha sau để học lớp mới tốt hơn? Lý do toán học nằm ở chỗ toàn bộ công thức đệ quy phía sau dựa trên giả thiết $H_0$ và $H_k$ thuộc cùng một hệ tọa độ đặc trưng; nếu tham số tích chập hoặc `running_mean` của BatchNorm thay đổi dù chỉ một chút, thông tin tương quan lưu trong $R_{k-1}$ trở nên lỗi thời và không còn tương đương học gộp. Ở khối thứ hai là tầng mở rộng đặc trưng FE. Với CIFAR-100, đầu ra của ResNet-32 chỉ có $d_{\text{cnn}} = 64$ chiều — nghĩa là ngay cả 100 vector trọng số phân loại cho 100 lớp cũng không thể độc lập tuyến tính trong $\mathbb{R}^{64}$. Bằng cách nhân với ma trận ngẫu nhiên $W_{\mathrm{fe}}$ cỡ $64 \times 8000$ rồi qua phi tuyến $\operatorname{ReLU}$, ta biến biểu diễn $64$ chiều thành không gian thưa chiều cao $8000$ — tương tự ý tưởng Extreme Learning Machine và Định lý Cover — cho phép bộ phân loại tuyến tính Ridge tách biệt cả 100 lớp dễ dàng hơn rất nhiều. Nhóm kiểm chứng điều này ở thực nghiệm R2: bỏ FE khiến $\bar{A}$ tụt từ $65.12\%$ xuống còn $53.25\%$.

---

### Slide 11 — Lớp 3 (Thành phần 3): Khởi tạo Giải tích Base (ARaBT) & Streaming Batch
- **Tiêu đề trên slide:** Lớp 3b (*What / Why / How*): Căn chỉnh giải tích ARaBT thay thế lớp FC của Backpropagation ở Phase 0
- **Bố cục trực quan:** Trái: Khung *What / Why / How* cho ARaBT (Phương trình 3–4 [1]); Phải: Hộp công thức tích lũy Gram/Cross theo batch & So sánh tiết kiệm bộ nhớ RAM khi $N_0 = 25{,}000, d=8000$.
- **Nội dung trên slide:**
  - **What (Phương trình (4) trong bài báo [1]):**
    - Khởi tạo cặp trạng thái ban đầu trên dữ liệu Phase 0 ($H_0 \in \mathbb{R}^{N_0 \times d}, Y_0 \in \mathbb{R}^{N_0 \times c_0}$):
      $$
      R_0 = \left( H_0^T H_0 + \gamma I_d \right)^{-1}, \qquad W_0 = R_0 H_0^T Y_0
      $$
  - **Why (Vì sao không dùng tiếp trọng số `FC50` của SGD?):**
    - `FC50` tối ưu hàm Cross-Entropy nên không phải nghiệm cực tiểu của bài toán Ridge trên $H_0$, cũng không sinh ra ma trận tương quan nghịch đảo $R_0$. Để chuỗi đệ quy ở các phase $k \ge 1$ hội tụ về đúng nghiệm học gộp $\tilde{W}_k$, điểm xuất phát $k=0$ bắt buộc phải được **căn chỉnh lại bằng nghiệm đóng Ridge (ARaBT)** [1].
  - **How (Cơ chế giải an toàn bộ nhớ trong [`initialize_batches`](../src/analytic.py)):**
    - Nếu ghép toàn bộ $H_0$ ($25{,}000 \times 8000$ `float64`), chỉ riêng $H_0$ chiếm **$1.60\text{ GB}$ RAM** trước khi tạo bản sao giải hệ.
    - Giải pháp streaming của nhóm: cộng dồn theo từng batch $b$:
      $$
      A_0 = \gamma I_d + \sum_{b} H_{0,b}^T H_{0,b}, \qquad B_0 = \sum_{b} H_{0,b}^T Y_{0,b}
      $$
      rồi giải hệ đối xứng xác định dương bằng phân rã **Cholesky** ($A_0 = L L^T$) để thu $R_0 = A_0^{-1}$ và $W_0 = A_0^{-1} B_0$.
    - **Quy tắc bất biến:** $\gamma I_d$ chỉ cộng **đúng 1 lần duy nhất ở bước khởi tạo $R_0$**; không cộng lặp lại ở batch hoặc phase sau.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~205 từ):**
  > Slide 11 giải thích bước ARaBT — *Analytic Re-alignment on Base Training* — tương ứng Phương trình (4) của bài báo. Trong Phase 0, lớp `FC50` dùng hàm mất mát Cross-Entropy chỉ đóng vai trò dẫn đường để ResNet-32 học bộ lọc ảnh tốt; sau đó nó bị loại bỏ hoàn toàn. Thay vào đó, ta đi qua 25,000 ảnh base sạch một lượt, thu ma trận đặc trưng mở rộng $H_0$ và giải trực tiếp bài toán bình phương tối thiểu có điều chuẩn $\gamma > 0$ để lấy $R_0$ cỡ $8000 \times 8000$ và $W_0$ cỡ $8000 \times 50$. Lưu ý ở đây có hai điểm kỹ thuật rất quan trọng khi hiện thực: thứ nhất, thay vì tạo một ma trận $H_0$ khổng lồ tốn 1.6 GB RAM (và còn nhân đôi khi tính $H_0^T H_0$), ta cộng dồn ma trận Gram $H_{0,b}^T H_{0,b}$ cỡ $8000 \times 8000$ theo từng minibatch $b$ rồi dùng phân rã Cholesky giải hệ xác định dương, giúp giữ nguyên toán học mà RAM ổn định; thứ hai, ma trận điều chuẩn $\gamma I_d$ được cộng đúng một lần khi lập $A_0$, đảm bảo nếu chia 25,000 ảnh thành các batch 32, 128 hay 512 thì độ mạnh điều chuẩn $\gamma$ luôn đồng nhất.

---

### Slide 12 — Lớp 3 (Thành phần 4): Cập nhật Đệ quy $R_k$ qua Bổ đề Woodbury
- **Tiêu đề trên slide:** Lớp 3c (*What / Why / How*): Bổ đề Woodbury hạ bậc nghịch đảo từ $\mathcal{O}(d^3)$ xuống $\mathcal{O}(b^3 + b d^2)$
- **Bố cục trực quan:** Hộp công thức Phương trình (10) ở đỉnh; So sánh trực quan kích thước ma trận nghịch đảo giữa tính trực tiếp ($8000 \times 8000$) vs. Woodbury theo batch ($128 \times 128$) ở giữa; 3 gạch *What/Why/How* ở dưới.
- **Nội dung trên slide:**
  - **What (Phương trình (10) trong bài báo [1]):**
    - Khi nhận lô đặc trưng mới $H_b \in \mathbb{R}^{b \times d}$ tại phase $k \ge 1$, ma trận nghịch đảo tương quan được cập nhật đệ quy:
      $$
      R_{\text{new}} = R_{\text{old}} - R_{\text{old}} H_b^T \left( I_b + H_b R_{\text{old}} H_b^T \right)^{-1} H_b R_{\text{old}}
      $$
  - **Why (Động lực tính toán):**
    - Vì $R_{\text{new}}^{-1} = R_{\text{old}}^{-1} + H_b^T H_b$, nếu cộng $H_b^T H_b$ rồi nghịch đảo trực tiếp ma trận cỡ $d \times d$ ($8000 \times 8000$), chi phí mỗi batch là $\mathcal{O}(d^3)$ — cực kỳ chậm!
    - Áp dụng **bổ đề nghịch đảo Sherman–Morrison–Woodbury** biến phép nghịch đảo cỡ $d \times d$ thành nghịch đảo ma trận lõi $S_b = I_b + H_b R_{\text{old}} H_b^T$ chỉ cỡ **$b \times b$** (với minibatch $b = 128 \ll d = 8000$).
  - **How (Quy trình ổn định số học 4 bước theo từng minibatch $b$):**
    1. Tính $P = R_{\text{old}} H_b^T \in \mathbb{R}^{d \times b}$ ($\mathcal{O}(bd^2)$).
    2. Lập $S_b = I_b + H_b P \in \mathbb{R}^{b \times b}$ — luôn đối xứng xác định dương vì $R_{\text{old}} \succ 0$ và cộng $I_b$.
    3. Giải hệ tuyến tính $S_b K^T = P^T$ bằng Cholesky thay cho `inv()` tường minh để tăng ổn định số học.
    4. Cập nhật $R_{\text{new}} = R_{\text{old}} - K^T P^T$ và đối xứng hóa $\frac{1}{2}(R_{\text{new}} + R_{\text{new}}^T)$ chống trôi dấu phẩy động.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~205 từ):**
  > Slide 12 lý giải trái tim tính toán của giai đoạn tăng dần: làm thế nào cập nhật $R_k = (\tilde{H}_k^T \tilde{H}_k + \gamma I_d)^{-1}$ mà không lưu quá khứ và không nghịch đảo lại ma trận $8000 \times 8000$ ở mỗi bước. Về mặt đại số, khi có thêm $b$ mẫu mới biểu diễn bởi ma trận $H_b$, ma trận tương quan tự động được cộng thêm một phần tử bán xác định dương hạng thấp là $H_b^T H_b$. Bổ đề Sherman–Morrison–Woodbury cho phép đổi nghịch đảo của một tổng hai ma trận trong không gian $d$ chiều sang nghịch đảo của ma trận $S_b = I_b + H_b R_{\text{old}} H_b^T$ trong không gian kích thước batch $b$. Trên cấu hình chuẩn của nhóm với $d = 8000$ và minibatch $b = 128$, thay vì nghịch đảo ma trận 64 triệu phần tử, hệ thống chỉ phải giải một hệ Cholesky nhỏ kích thước $128 \times 128$ rồi trừ phần cập nhật vào $R_{\text{old}}$. Trong quá trình cài đặt thực tế, nhóm nhận thấy hai chi tiết số học quan trọng: giữ độ chính xác kép `float64`, giải hệ bằng `solve`/Cholesky thay vì gọi hàm nghịch đảo tường minh, và đối xứng hóa $R_{\text{new}}$ sau mỗi batch để chặn sai số làm tròn tích tụ khi đi qua 40 minibatch của mỗi pha.

---

### Slide 13 — Lớp 3 (Thành phần 5): Cập nhật $W_k$ — Ghép khối vs. Dạng Residual Toàn cục
- **Tiêu đề trên slide:** Lớp 3d (*What / Why / How*): Hiệu chỉnh đồng thời trọng số lớp cũ và khởi tạo trọng số lớp mới
- **Bố cục trực quan:** Trên: Công thức Phương trình (9) [1] bên trái $\iff$ Công thức Global Residual của nhóm bên phải; Dưới: Bảng *What / Why / How* và trực quan hóa số hạng $-R_k H_k^T H_k W_{k-1}$.
- **Nội dung trên slide:**
  - **What — Hai biểu diễn tương đương đại số:**
    - **Dạng ghép khối theo trọn phase (Eq. 9 [1]):**
      $$
      W_k = \left[\, \underbrace{W_{k-1} - R_k H_k^T H_k W_{k-1}}_{\text{Hiệu chỉnh các lớp cũ } (d \times C_{k-1})} \;\;\;\;\;\; \underbrace{R_k H_k^T Y_k}_{\text{Lớp mới } (d \times c_k)} \,\right]
      $$
    - **Dạng phần dư toàn cục theo minibatch ([`incremental_update`](../src/analytic.py)):**
      $$
      W_{\text{pad}} = \left[\, W_{\text{old}} \;\; \mathbf{0}_{d \times (C_k - C_{\text{old}})} \,\right], \qquad W_{\text{new}} = W_{\text{pad}} + R_{\text{new}} H_b^T \left( Y_{b,\text{global}} - H_b W_{\text{pad}} \right)
      $$
  - **Why (Ý nghĩa trực quan của số hạng $-R_k H_k^T H_k W_{k-1}$):**
    - Mẫu ảnh mới $H_k$ **không thuộc các lớp cũ**, tức nhãn kỳ vọng trên các cột lớp cũ bằng $\mathbf{0}$. Số hạng $-R_k H_k^T (H_k W_{k-1})$ đóng vai trò kéo phản hồi của cột trọng số lớp cũ về $0$ trên vùng đặc trưng của lớp mới, **ngăn lớp cũ kích hoạt nhầm trên ảnh lớp mới mà không cần xem lại ảnh cũ**.
  - **How (Lợi thế của dạng Residual khi chạy Minibatch):**
    - Khi nhãn $Y_{b,\text{global}}$ dùng ID toàn cục ($0\dots C_k-1$), trên các cột lớp cũ ($Y_{\text{old}} = 0$) công thức cho đúng $W_{\text{old}} - R_k H_b^T H_b W_{\text{old}}$; khi một lớp mới lặp lại ở minibatch thứ 2 trong cùng phase, nó **cập nhật tiếp vào đúng cột cũ** thay vì nối thêm cột mới dư thừa.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~210 từ):**
  > Slide 13 làm rõ cách cập nhật ma trận trọng số phân loại $W_k$. Ở Phương trình (9) trong bài báo, $W_k$ gồm hai khối ghép ngang: khối bên trái có kích thước $d \times C_{k-1}$ dành cho các lớp cũ, còn khối bên phải $R_k H_k^T Y_k$ dành cho $c_k$ lớp mới. Một điểm rất tinh tế cần nhấn mạnh là khối bên trái không đứng yên mà bị trừ đi số hạng $R_k H_k^T H_k W_{k-1}$. Trực quan hóa điều này: các mẫu $H_k$ hiện tại mang nhãn 0 ở toàn bộ cột lớp cũ; nếu $W_{k-1}$ tạo ra giá trị dự đoán khác 0 trên $H_k$, số hạng trừ này chính là bước điều chỉnh tối ưu giúp giảm sai số trên dữ liệu mới mà vẫn cân bằng qua độ cong $R_k$ để không làm hỏng phân loại dữ liệu cũ. Khi tự triển khai chia theo minibatch, bài báo viết Eq. (9) giả định cả phase có đủ cột mới; nếu gọi Eq. (9) lặp lại qua 40 minibatch của cùng Phase 1 thì số cột sẽ bị nở sai theo số batch! Do đó, trong `src/analytic.py`, nhóm sử dụng dạng phần dư toàn cục hoàn toàn tương đương về đại số: giữ nguyên số cột $C_k$ cho cả phase và cập nhật qua phần dư $(Y_{b,\text{global}} - H_b W_{\mathrm{pad}})$, bảo đảm dù xử lý 1 lượt hay chia nhỏ 40 minibatch kết quả vẫn giống hệt nhau.

---

### Slide 14 — Lớp 4a: Định lý 3.1 — Tương đương Học Gộp, Trực quan & 4 Điều kiện Giới hạn
- **Tiêu đề trên slide:** Lớp 4a (Lý thuyết): Định lý 3.1 bảo chứng tương đương đại số với Joint Ridge — nhưng có điều kiện biên chặt
- **Bố cục trực quan:** Trái: Phát biểu Theorem 3.1 [1] + Phác thảo 3 dòng chứng minh quy nạp + Thẻ R3 (54/54 mốc khớp); Phải: Bốn thẻ cảnh báo giới hạn/điều kiện hiệu lực của Định lý.
- **Nội dung trên slide:**
  - **Định lý 3.1 (Equivalence of Incremental & Joint Analytic Learning) [1]:**
    - Nếu khởi tạo bằng Eq. (4) và cập nhật đệ quy mọi phase $k \ge 1$ bằng Eq. (9)–(10) trên cùng đặc trưng cố định, thì:
      $$
      \forall k \ge 0:\quad W_k = \tilde{W}_k = \left(\tilde{H}_k^T \tilde{H}_k + \gamma I_d\right)^{-1} \tilde{H}_k^T \tilde{Y}_k
      $$
  - **Trực quan cốt lõi & Bằng chứng thực nghiệm:**
    - Nghiệm Ridge chỉ phụ thuộc vào hai **thống kê đủ bậc hai cộng tính**: $\sum H_j^T H_j$ và $\sum H_j^T Y_j$; thứ tự cộng batch/phase không đổi tổng.
    - *Xác nhận R3 (`d=1024`, [luf7uv40..3gv0moj7]):* Khớp Joint Oracle tại **54/54 mốc** (max rel $W$ error $\le 1.159\times 10^{-11}$, `0` argmax khác).
  - **4 Điều kiện biên & Giới hạn lý thuyết bắt buộc:**
    1. **$\phi(x)$ bất biến tuyệt đối:** Backbone CNN, thống kê BatchNorm, chuẩn hóa ảnh và $W_{\mathrm{fe}}$ phải giữ nguyên qua mọi phase.
    2. **$\gamma > 0$ cố định & cộng 1 lần:** Đảm bảo $H_0^T H_0 + \gamma I_d$ khả nghịch; cộng lặp $\gamma$ sẽ đổi bài toán mục tiêu.
    3. **Tương đương *Joint Ridge trên feature cố định* $\neq$ *Joint CNN End-to-End*:** Trần chất lượng bị khóa bởi biểu diễn Phase 0.
    4. **Sai số dấu phẩy động:** Cần `float64`; cập nhật quá nhiều bước trên `float32` dễ mất tính đối xứng xác định dương của $R_k$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~215 từ):**
  > Slide 14 phân tích Định lý 3.1 — đóng góp lý thuyết trụ cột của bài báo ACIL [1]. Định lý khẳng định trọng số $W_k$ thu được bằng thuật toán đệ quy không giữ mẫu cũ bằng chính xác nghiệm tối ưu toàn cục $\tilde{W}_k$ nếu gom toàn bộ dữ liệu từ Phase 0 đến Phase $k$. Lý do trực quan rất đẹp: bài toán bình phương tối thiểu không cần nhớ từng ảnh riêng lẻ mà chỉ cần hai ma trận tổng $\sum H_j^T H_j$ và $\sum H_j^T Y_j$, vốn có tính chất cộng tính giao hoán. Chúng tôi cũng kiểm chứng định lý này bằng thực nghiệm R3 qua 54 phép so sánh và thấy sai số chỉ ở ngưỡng $10^{-11}$ trên `float64`. Tuy nhiên, để hiểu đúng bài báo, ta phải chỉ rõ bốn điều kiện biên ở cột bên phải — đây là phần rất dễ bị hiểu nhầm. Thứ nhất, ánh xạ đặc trưng phải đóng băng tuyệt đối; nếu bất kỳ tham số CNN hay BatchNorm nào đổi giữa chừng, định lý mất hiệu lực. Thứ hai, hệ số $\gamma > 0$ chỉ cộng ở pha 0 để giữ tính khả nghịch. Thứ ba, "học gộp" ở đây là học gộp *chỉ ở lớp phân loại tuyến tính cuối* trên đặc trưng đã học từ 50 lớp nền, chứ **không** tương đương với huấn luyện lại toàn bộ ResNet-32 từ đầu trên cả 100 lớp. Thứ tư, phép đệ quy đòi hỏi tính toán trên `float64` để không trôi sai số số học.

---

### Slide 15 — Lớp 4b: Giải mã "Absolute Memorization" & "Privacy Protection" kèm Giới hạn
- **Tiêu đề trên slide:** Lớp 4b (Lý thuyết): Phân biệt rõ *Ghi nhớ nghiệm đại số* với *Không quên độ chính xác* và *Bảo mật không phải DP*
- **Bố cục trực quan:** Chia 2 cột song song so sánh *Claim trong tên bài báo [1]* vs *Bản chất toán học thật sự & Giới hạn cần thận trọng*.
- **Nội dung trên slide:**
  - **1. "Absolute Memorization" (Ghi nhớ tuyệt đối) [1]:**
    - **Điều đúng:** Ghi nhớ tuyệt đối **nghiệm đại số tối ưu** ($\|W_k - \tilde{W}_k\|_F = 0$), giúp $A_K$ **bất biến theo số phase $K$** và thứ tự minibatch (nếu tập Phase 0 giống nhau).
    - **Giới hạn & Hiểu lầm cần tránh:** **KHÔNG đồng nghĩa $F_{\mathrm{base}} = 0$ hay độ chính xác trên lớp cũ đứng yên!** Khi số lớp tăng từ $50 \to 100$, các vùng quyết định trong $\operatorname{argmax}$ phải chia nhỏ gấp đôi trên một backbone cố định $\to$ xác suất mẫu cũ rơi nhầm sang cột lớp mới tăng lên.
    - *Minh chứng R1 của nhóm:* Độ chính xác tập 50 lớp gốc vẫn giảm từ $B_0 = 76.64\%$ xuống $B_5 = 68.33\%$ ($F_{\mathrm{base}} = 8.31 \pm 0.62\text{ pp}$, khớp xu hướng $F = 9.00$ trong Table 1 [1]).
  - **2. "Privacy Protection" (Bảo vệ quyền riêng tư) [1]:**
    - **Điều đúng:** Không lưu ảnh gốc $X_k$ hay ma trận từng mẫu $H_k$; không thể khôi phục duy nhất $H_k$ từ $R_k^{-1} - R_{k-1}^{-1} = H_k^T H_k$ vì mọi ma trận trực giao $Q$ đều thỏa $(QH_k)^T(QH_k) = H_k^T H_k$ (bất định phép quay/hoán vị, cộng thêm mất thông tin qua GAP và $\operatorname{ReLU}$).
    - **Giới hạn lý thuyết:** Đây là bảo vệ ở mức **exemplar-free & bất khả nghịch tuyến tính từ ma trận Gram**, **KHÔNG phải** chuẩn **Differential Privacy ($(\varepsilon, \delta)$-DP)**; ma trận $R_k, W_k$ vẫn mã hóa chính xác thống kê bậc hai của tập dữ liệu đã gặp.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~200 từ):**
  > Ngay trong tiêu đề bài báo [1] có hai cụm từ rất mạnh: "Absolute Memorization" và "Privacy Protection". Slide 15 bóc tách phạm vi đúng của hai khái niệm này để tránh suy diễn quá mức. Về *Absolute Memorization*: ACIL chỉ "ghi nhớ tuyệt đối" nghiệm đại số $\tilde{W}_k$ của học gộp trên toàn bộ đặc trưng lịch sử — nghĩa là không hề làm rơi rớt thông tin thống kê bậc hai từ Phase 0. Tuy nhiên, điều đó không có nghĩa độ chính xác trên 50 lớp nền sẽ giữ nguyên $F_{\mathrm{base}} = 0$. Khi tổng số lớp cạnh tranh trong $\operatorname{argmax}$ tăng từ 50 lên 100 trên cùng không gian biểu diễn cố định, các siêu phẳng quyết định trở nên chật chội hơn, kéo $B_k$ giảm $8.31\text{ điểm phần trăm}$ trong thực nghiệm R1 (gần với con số $9.00\%$ ở Table 1 bài báo). Về *Privacy Protection*: lập luận của tác giả là nếu chỉ lưu $(R_k, W_k)$, kẻ tấn công biết được hiệu $H_k^T H_k$ cũng đối mặt bài toán vô định vô số nghiệm $Q H_k$ với mọi ma trận trực giao $Q$. Nhưng về chuẩn bảo mật hiện đại, điều này không phải Differential Privacy có biên $(\varepsilon, \delta)$ chống *membership inference*, mà là bảo mật thực dụng theo nghĩa không lưu trữ ảnh thô hay vector cá nhân.

---

### Slide 16 — Lớp 5: Độ phức tạp Thời gian, Bộ nhớ & Đánh đổi với Tiền nhiệm
- **Tiêu đề trên slide:** Lớp 5 (Complexity): Đổi vòng lặp SGD đa epoch lấy cập nhật một lượt $\mathcal{O}(N_k d^2)$ và bộ nhớ tĩnh $\mathcal{O}(d^2)$
- **Bố cục trực quan:** Bảng đối chiếu Complexity (5 cột: Tiêu chí, Joint CNN, LwF/EWC, iCaRL/PODNet/AANets, ACIL) ở nửa trên; Hai hộp phân tích đánh đổi định lượng thực đo (Thời gian R1/R3 vs Bộ nhớ `522.5 MB` ở $d=8000$) ở nửa dưới.
- **Nội dung trên slide:**
  - **Bảng đối chiếu Độ phức tạp tại một phase $k \ge 1$ ($N_k$ mẫu, batch $b$, $d=8000$, $E$ epochs SGD, $M$ mẫu buffer):**
| Tiêu chí | Joint CNN | LwF / EWC [3] | iCaRL / PODNet / AANets [2], [4], [5] | **ACIL [1]** |
|---|---|---|---|---|
| **Buffer ảnh cũ** | $\mathcal{O}(N_{\text{total}})$ | **0** | $M$ ảnh ($\mathcal{O}(M D_{\text{img}})$) | **0** |
| **State giữ giữa phase** | $\mathcal{O}(\|\theta\| + N_{\text{total}} D_{\text{img}})$ | $\mathcal{O}(2\|\theta\|)$ | $\mathcal{O}(2\|\theta\| + M D_{\text{img}})$ | $\mathcal{O}(\|\theta_0\| + d^2 + d C_k)$ |
| **Lượt duyệt CNN** | $E$ epochs (Fwd+Bwd) | $E$ epochs (2 Fwd+Bwd) | $E$ epochs trên $(N_k+M)$ | **1 lượt ($E=1$)**, 0 Bwd |
| **Thời gian phase $k$** | $\mathcal{O}(E \sum N_j (C_{\text{fwd}}+C_{\text{bwd}}))$ | $\mathcal{O}(E N_k (2C_{\text{fwd}}+C_{\text{bwd}}))$ | $\mathcal{O}(E (N_k+M) (2C_{\text{fwd}}+C_{\text{bwd}}))$ | $\mathcal{O}(N_k C_{\text{fwd}} + N_k d^2)$ |
| **Khớp nghiệm học gộp** | Chuẩn CNN | Không | Không (suy ở $K$ lớn) | **Có (Joint Analytic)** |
  - **Đánh đổi định lượng thực tế (từ R1/R3 của nhóm):**
    - **Tốc độ cập nhật:** Trong R1 ($d=8000$, [`zt1014ei`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei)), BP160 mất **$40.4\text{–}42.3\text{ phút}$** trên T4, còn 1 phase incremental (5,000 ảnh) chỉ mất **$41.8\text{–}44.7\text{ s}$** trên CPU 4 luồng; toàn bộ streaming 6 mốc mất **$11.1\text{–}11.4\text{ phút}$**. Ở $d=1024$ (R3, $b=512$), toàn bộ analytic 6 phase xong trong **$5.00\text{ s}$**.
    - **Đánh đổi $\mathcal{O}(d^2)$ của $R_k$:** Với `float64` ở $d=8000$, $R_k = 512.00\text{ MB}$, $W_5 = 6.40\text{ MB}$, $W_{\mathrm{fe}} = 4.10\text{ MB}$ $\implies$ tổng **$522.50\text{ MB}$** (lớn hơn $2000$ ảnh CIFAR-100 chỉ $\approx 6.14\text{ MB}$, nhưng bất biến khi $N_{\text{total}}$ tăng lên hàng triệu ảnh như ImageNet).
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., CVPR, 2017. [3] Z. Li and D. Hoiem, IEEE TPAMI, 2018. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021.`
- **Speaker Notes (~220 từ):**
  > Slide 16 khép lại 5 lớp phân tích phương pháp bằng bảng so sánh độ phức tạp thời gian và không gian với các tiền nhiệm. Ở các phương pháp replay như iCaRL, PODNet hay AANets, mỗi pha mới $k$ đều buộc mạng chạy lại $E$ epochs (thường từ 70 đến 160 epochs) gồm cả lan truyền tiến lẫn ngược qua cả ảnh mới và bộ nhớ đệm $M$. Ngược lại, ACIL chỉ chạy lan truyền tiến đúng 1 lần qua $N_k$ ảnh mới ($E=1$), không backpropagation, rồi tốn $\mathcal{O}(N_k d^2)$ phép nhân ma trận cho Woodbury nhờ minibatch $b \ll d$ (tránh được $\mathcal{O}(d^3)$ của nghịch đảo trực tiếp). Số liệu đo thực tế ở R1 cho thấy phase cuối 5,000 ảnh trên $d=8000$ chỉ tốn khoảng $42\text{–}45\text{ giây}$ trên CPU 4 luồng, và ở R3 ($d=1024, b=512$) toàn bộ base cộng 5 pha chỉ mất $5.00\text{ giây}$. Tuy nhiên, nhóm cũng chỉ ra đánh đổi quan trọng về bộ nhớ: ma trận $R_k$ kích thước $d \times d$ ở `float64` với $d=8000$ chiếm $512\text{ MB}$ cố định (tổng $R, W, W_{\mathrm{fe}}$ là $522.50\text{ MB}$). Con số này lớn hơn kho $2000$ ảnh thô $32 \times 32$ trên CIFAR-100 ($\approx 6.14\text{ MB}$), nên lợi thế nén không gian của $\mathcal{O}(d^2)$ chủ yếu phát huy ở các bài toán ảnh phân giải lớn hoặc hàng triệu mẫu mà kích thước $R_k$ vẫn giữ nguyên.

---

## PHẦN 3 (NỬA ĐẦU): THIẾT LẬP THỰC NGHIỆM, PROTOCOL, METRICS & KIỂM ĐỊNH TIỀN KỲ (SLIDES 17–18)

### Slide 17 — Phạm vi Tái lập (Core Scope) & Kiến trúc Hệ thống Thực nghiệm R1–R3
- **Tiêu đề trên slide:** Thiết kế thực nghiệm trong Core Scope: Trả lời 4 câu hỏi trọng tâm từ Benchmark đến Author Comparison
- **Bố cục trực quan:** Bảng ma trận 4 thí nghiệm chính (**R1**, **R2**, **R3**, **Author C1/C2/C1R**) gồm cột: *Câu hỏi nghiên cứu*, *Cấu hình chốt*, *Khối lượng chạy*, và hộp *Tuyên bố phạm vi trung thực* ở bên dưới.
- **Nội dung trên slide:**
  - **Lý do chọn Core Scope (`docs/protocol_chot.md`):** Tập trung tối đa nguồn lực Kaggle T4/CPU để chạy đủ số seed lặp lại có kiểm chứng thay vì chạy dàn trải nông trên nhiều dataset.
  - **Ma trận 4 khối thực nghiệm chính:**
    1. **R1 — So sánh chính (`CIFAR-100 / K=5`):** $d=8000, \gamma=0.1$, 3 seeds (`42, 43, 44`), mỗi seed train mới 160 epochs $\to$ Đường cong 6 mốc, $\bar{A}, F_{\mathrm{base}}$ đối chiếu Table 1 [1].
    2. **R2 — Ablation thành phần (`Table 2`):** Dùng lại đúng checkpoint 160 mỗi seed; 4/5 hàng (*No-FE $\gamma=0.1$*, *FE8000 $\gamma \in \{0.1, 0.01, 0.001\}$*) $\times$ 3 seeds (9 runs mới + 3 R1).
    3. **R3 — Phân tích Batch Size (Nhóm tự thiết kế):** Checkpoint seed 42, $d=1024, \gamma=0.1$; sweep $b \in \{32, 128, 512\} \times 3$ timing repeats (9 runs, 54 so sánh Joint Oracle).
    4. **Author Comparison (`0afd6bf`):** Chạy nguyên bản code tác giả (`C1`, `C2`, `C1R`) trong venv riêng trên checkpoint seed 42, $d=1024$.
  - **Ranh giới tuyên bố rõ ràng:** Chưa tự chạy các baseline khác trong Table 1; chưa chạy $K=10/25/50$, ImageNet hoặc hàng $\gamma=0$ của Table 2 (do solver hiện yêu cầu $\gamma > 0$).
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021.`
- **Speaker Notes (~185 từ):**
  > Bắt đầu phần Thực nghiệm Tái lập, Slide 17 tóm tắt kiến trúc 4 bộ thực nghiệm được nhóm đăng ký trước khi chạy dài trong tài liệu `docs/protocol_chot.md` — tuyệt đối không sửa siêu tham số theo kết quả test. Theo đúng tinh thần Core Scope của môn học, nhóm thực hiện đầy đủ phép so sánh chính R1 trên ba seed độc lập ở kích thước chuẩn $d=8000$; thực hiện thí nghiệm thành phần R2 cho 4 trên 5 hàng của Table 2 (loại trừ hàng $\gamma = 0$ từ đầu vì solver dựa trên bổ đề Woodbury đòi hỏi ma trận xác định dương với $\gamma > 0$, tránh thay bằng epsilon giả danh $\gamma=0$); thiết kế thêm thí nghiệm bổ sung R3 để kiểm chứng định lý tương đương ở ba cỡ batch $32, 128, 512$ và đo đánh đổi thời gian–bộ nhớ; đồng thời chạy song song chính xác mã nguồn của tác giả để định vị sai khác. Mọi kết luận trình bày ở các slide tiếp theo đều tuân thủ nghiêm ranh giới phạm vi: chỉ tuyên bố những gì đã đo trên CIFAR-100/K=5 và tách bạch rõ số liệu nhóm tự chạy với số trích tham khảo từ bài báo.

---

### Slide 18 — Dữ liệu, Chuẩn bị & Giao thức Huấn luyện (`Protocol v1.0`)
- **Tiêu đề trên slide:** Dữ liệu & Protocol chốt trước (`ACIL-C100-v1.0`): Khóa chặt hạt giống, chuẩn hóa chỉ trên Base và kiến trúc ResNet-32
- **Bố cục trực quan:** Chia 2 cột: Trái là bảng *Dữ liệu & Chia lớp theo Phase*; Phải là bảng *Cấu hình Backbone ResNet-32 & Training Phase 0* kèm thẻ lưu ý khác biệt thứ tự lớp so với PODNet/AANets.
- **Nội dung trên slide:**
  - **Dữ liệu & Ánh xạ nhãn toàn cục:**
    - Nguồn tự động: Hugging Face `uoft-cs/cifar100` [7] (pin revision `aadb3af...`, kiểm SHA-256 Parquet): 50,000 train / 10,000 test (`32x32` RGB).
    - **Chia phase ($K=5$):** Phase 0: 50 lớp (`25,000` train / `5,000` test); Phases $1\dots 5$: mỗi phase $+10$ lớp (`5,000` train / `+1,000` test/phase; tổng `10,000` test ở Phase 5).
    - **Thứ tự lớp & Normalization:** `class_order = default_rng(seed).permutation(100)` (`42, 43, 44`); `mean/std` tính **duy nhất từ 25,000 ảnh base train** rồi khóa cho mọi phase/test. *(Ghi nhận khác biệt: PODNet/AANets dùng thứ tự lớp cố định qua các lần chạy [4], [5])*.
  - **Backbone & Lịch huấn luyện Phase 0 ([1], [6]):**
    - **Backbone:** ResNet-32 CIFAR nhóm tự cài ($n=5$, kênh `16/32/64`, shortcut option A, feature $64$ chiều) [6] + head tạm `FC50`.
    - **Tối ưu BP (160 epochs, chọn epoch cuối):** SGD (`lr=0.1`, momentum `0.9`, weight decay `0.0005`, batch `128`), `MultiStepLR` milestones `[80, 120]` ($\times 0.1$ tại epoch 80 và 120).
    - **Augmentation:** Chỉ áp dụng lúc BP (`pad 4 -> random crop 32 -> flip p=0.5`); **tắt hẳn crop/flip ở giai đoạn analytic & test**.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021. [6] K. He et al., CVPR, 2016. [7] A. Krizhevsky, Tech. Rep., 2009.`
- **Speaker Notes (~210 từ):**
  > Slide 18 mô tả chi tiết giao thức tái lập `ACIL-C100-v1.0`. Đầu tiên về dữ liệu, pipeline không tải từ đường dẫn không kiểm soát mà tải tự động CIFAR-100 từ Hugging Face tại revision được ghim cố định và đối chiếu mã băm SHA-256 trên từng file Parquet. Tập 100 lớp được tách thành 50 lớp nền ở Phase 0 và năm pha liên tiếp mỗi pha 10 lớp theo hoán vị `class_order` phụ thuộc từng seed độc lập `42, 43, 44`. Một chi tiết phương pháp luận quan trọng được nhóm ghi nhận công khai: theo kiểm tra các repository tham chiếu của PODNet [4] và AANets [5], những repo đó cố định một thứ tự lớp cho nhiều lượt chạy, trong khi giao thức của nhóm xáo lại thứ tự lớp theo mỗi seed — do đó độ lệch chuẩn $SD$ giữa 3 seed của nhóm phản ánh cả biến thiên chọn tập lớp nền lẫn khởi tạo trọng số. Về chuẩn hóa ảnh, `mean` và `std` được tính hoàn toàn từ $25{,}000$ ảnh huấn luyện của riêng Phase 0 để bảo đảm không rò rỉ thống kê từ tương lai hay tập test. Mạng ResNet-32 sử dụng kiến trúc chuẩn CIFAR với shortcut Option A, huấn luyện 160 epochs trên GPU T4 trước khi bước vào căn chỉnh ARaBT.

---

### Slide 19 — Chuẩn hóa Hệ thống Metrics & Quy ước Dấu (`Metrics v1.0`)
- **Tiêu đề trên slide:** Định nghĩa Metric & Kỷ luật Thống kê (`ACIL-metrics-v1.0`): Xử lý bất nhất dấu $F$ và quy tắc tổng hợp 3 seeds
- **Bố cục trực quan:** Bên trái là 4 công thức metric chính ($A_k, \bar{A}, A_{\mathrm{final}}, B_k / F_{\mathrm{base}}$); Bên phải là hộp cảnh báo quy ước dấu $F$ của bài báo [1] & Quy tắc thống kê $n=3$ seeds (`ddof=1`).
- **Nội dung trên slide:**
  - **Các thước đo chuẩn (`docs/metrics.md`, đánh giá single-head trên $100 \times C_k$ ảnh test):**
    - **$A_k$ (All-seen Accuracy):** Độ chính xác (%) trên toàn bộ $C_k$ lớp đã thấy sau phase $k$; **mọi dự đoán đều $\operatorname{argmax}$ toàn cục, không đưa Task ID**.
    - **$\bar{A}$ (Average Incremental Accuracy):** $\bar{A} = \frac{1}{K+1}\sum_{k=0}^{K} A_k$ ($K=5 \implies$ trung bình đều **6 mốc**, gồm cả $A_0$ **sau ARaBT**, không dùng trọng số theo số ảnh hay lấy $A_5$ thay $\bar{A}$).
    - **$A_{\mathrm{final}} = A_5$:** Độ chính xác cuối cùng trên đủ $100$ lớp ($10{,}000$ ảnh test).
    - **$B_k$ (Base Accuracy under All-Seen Competition):** Độ chính xác trên $5{,}000$ ảnh test của 50 lớp nền (vẫn cạnh tranh trước cả 100 cột).
  - **Xử lý bất nhất dấu của Forgetting $F$ trong bài báo [1]:**
    - Bài báo mục 4.2 in công thức $F = B_K - B_0$, nhưng Table 1 báo giá trị dương ($F = 9.00\%$). Nhóm chốt: **$F_{\mathrm{base}} = B_0 - B_K\text{ (pp)}$** (dương = quên nhiều hơn) để so với Table 1, đồng thời lưu song song `base_accuracy_delta_pp` ($B_K - B_0$).
  - **Kỷ luật thống kê:** Tính trọn bộ metric trên từng seed trước rồi mới lấy `mean ± sample SD (n=3, ddof=1)`; không coi 6 phases là 6 seeds, không gộp timing repeats R3 thành SD accuracy.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~205 từ):**
  > Slide 19 chốt định nghĩa metric trước khi đi vào kết quả. Trong tài liệu học thuật, dễ gặp ba sai sót khi báo cáo CIL: thứ nhất là bỏ qua mốc nền $A_0$ khi tính $\bar{A}$, hoặc lấy accuracy của lớp `FC50` từ backpropagation thay cho $A_0$ sau căn chỉnh ARaBT; thứ hai là đánh giá tách riêng từng task thay vì $\operatorname{argmax}$ toàn cục trên tất cả các lớp đã gặp; thứ ba là nhầm lẫn dấu của chỉ số quên $F$. Khi rà soát bài báo gốc [1], nhóm phát hiện tác giả viết $F = B_K - B_0$ ở mục 4.2 (cho kết quả âm khi quên) nhưng lại ghi số dương $9.00$ ở Bảng 1. Để minh bạch hoàn toàn, nhóm định nghĩa `base_forgetting_pp` $= B_0 - B_K$ nhằm đối chiếu trực tiếp với cột $F$ của Bảng 1, đồng thời vẫn lưu trường đối dấu `base_accuracy_delta_pp` trong JSON. Cuối cùng, nhóm tuân thủ nguyên tắc đơn vị thống kê: ở R1 và R2, một đơn vị là một seed hoàn chỉnh ($n=3, \mathrm{ddof}=1$); còn ở R3, ba lần chạy là ba lần đo thời gian lặp lại trên cùng một seed, tuyệt đối không gắn nhãn thành $SD$ độ chính xác qua các seed.

---

### Slide 20 — Kiểm chứng Tiền kỳ Trước Khi Chạy Dài (G1–G6 Sanity Gates)
- **Tiêu đề trên slide:** Kiểm định từng bước trước benchmark 160 epochs: Toán học (`1e-13`), Pipeline 100 lớp & Ngân sách T4
- **Bố cục trực quan:** Lộ trình 4 chặng kiểm định nối tiếp nhau (*Toy Numeric $\to$ CIFAR Smoke 100 lớp $\to$ Resume CPU/CUDA $\to$ Profile d8000 & BP5*) kèm thẻ Run ID W&B công khai cho mỗi chặng.
- **Nội dung trên slide:**
| Chặng | Thử nghiệm & Mục tiêu | Kết quả đạt được & Ý nghĩa | W&B Run (Public) |
|---|---|---|---|
| **G1 — Toán học tổng hợp** | `VERIFY-01/02`: 3 phase $\times 2$ lớp, $d=24$, minibatch $b=7$, lớp lặp | **11/11 tests đạt**; minibatch khớp Joint Ridge (max rel $W \approx 3.43\times 10^{-13}$, `0` argmax lệch). | [7aeaupq7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7aeaupq7) · [wi4wau2d](https://wandb.ai/hsangg-2006/acil-reproduction/runs/wi4wau2d) |
| **G2 — CIFAR Smoke** | `CIFAR-KAGGLE-01/02`: 2 epochs, $d=256$ trên GPU T4 (50+10 và đủ 50→100 lớp) | Xác nhận freeze/eval, tải HF tự động, $W_{256\times 100}$, khớp Joint 6/6 mốc ($A_{\mathrm{final}}=22.05\%$). | [o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2) · [blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) |
| **G3 — Resume v2 & Streaming** | Checkpoint v2 (RNG/optimizer/scheduler) CPU + CUDA & streaming d256 | CPU resume khớp chính xác; CUDA TinyCNN đạt (**53/54 tests**, 1 snapshot skip); streaming khớp oracle 6/6. | Đính kèm trong [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) |
| **G4 — Profile d8000 & BP5** | `PROFILE-KAGGLE-01` (d8000 subset) & `BP5-KAGGLE-01` (5 epochs 25k ảnh) | Profile d8000: `34.07 s`, peak RSS `~3.20 GB`; BP5: `76.11 s` $\implies$ dự trù R1 `~59 phút` (< 120 phút T4). | [gey0yop4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/gey0yop4) · [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) |
- **Footer IEEE:** `Nội bộ nhóm (Không trích nguồn ngoài; chi tiết manifest tại docs/protocol_chot.md và reports/README.md).`
- **Speaker Notes (~195 từ):**
  > Trước khi tiêu tốn quota GPU Kaggle cho các lượt chạy 160 epochs ở $d=8000$, nhóm xây dựng chuỗi kiểm định kỹ thuật theo 4 cổng nhằm triệt tiêu rủi ro sai công thức hoặc sập bộ nhớ giữa chừng. Ở cổng G1 trên dữ liệu tổng hợp (`VERIFY-01` và `VERIFY-02`), module `src/analytic.py` vượt qua toàn bộ 11 test kiểm tra điều kiện lớp xuất hiện lại ở nhiều minibatch, bảo toàn mảng đầu vào và sai lệch tương đối của $W$ chỉ $3.43\times 10^{-13}$. Ở cổng G2, nhóm chạy smoke test thực tế trên CIFAR-100 ở cấu hình nhẹ 2 epochs, $d=256$ (lượt `o2r63pj2` kiểm 50+10 lớp và lượt `blhw74ff` kiểm trọn chuỗi 100 lớp với $10{,}000$ ảnh test), chứng minh pipeline tải dữ liệu, đóng băng ResNet-32 và cập nhật streaming không rò rỉ khớp hoàn toàn nghiệm Joint Oracle tại cả 6 mốc. Cuối cùng, trước khi mở R1, lượt profile `gey0yop4` đo mức RAM đỉnh của $d=8000$ ($\approx 3.20\text{ GB}$ hoàn toàn nằm trong RAM Kaggle) và lượt `BP5` `pv82dy1f` đo 5 epochs thật trên T4 để chốt dự trù ngân sách ~59 phút/seed — đủ dự phòng trên 30% dưới ngưỡng timeout 120 phút.

---

## PHẦN 3 (TIẾP THEO): KẾT QUẢ R1, R2, R3 & ĐỐI CHIẾU TÁC GIẢ (SLIDES 21–32)

### Slide 21 — Kết quả R1 (1/3): Huấn luyện Base 160 Epochs & Tổng hợp 3 Seeds Độc lập
- **Tiêu đề trên slide:** Kết quả R1: Huấn luyện 160 epochs hội tụ đồng đều cả 3 seeds — $\bar{A} = 65.12 \pm 0.61\%$
- **Bố cục trực quan:** Trái (60%): Chèn hình tự vẽ [`reports/figures/r1_cifar100_k5_training.png`](../reports/figures/r1_cifar100_k5_training.png); Phải (40%): Bảng tổng hợp 3 seeds (`42, 43, 44`) kèm thời gian T4 và link W&B từng run.
- **Nội dung trên slide:**
  - **Hình minh họa:** Chèn [`reports/figures/r1_cifar100_k5_training.png`](../reports/figures/r1_cifar100_k5_training.png) (*Loss Cross-Entropy Base 25,000 ảnh/epoch & lịch LR giảm bậc tại epoch 80, 120 cho seeds 42, 43, 44*).
  - **Bảng kết quả R1 theo từng seed (`d=8000`, $\gamma=0.1$, $K=5$, $n=3$, $\mathrm{ddof}=1$):**
| Seed | $\bar{A}$ (%) | $A_{\mathrm{final}}$ (%) | $F_{\mathrm{base}}$ (pp) | Numeric Runtime (phút) | W&B Run (Finished / Public) |
|---:|---:|---:|---:|---:|---|
| **42** | 65.16 | 55.87 | 8.32 | 53.78 *(BP 41.60m)* | [zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) |
| **43** | 65.71 | 56.77 | 8.92 | 52.25 *(BP 40.39m)* | [v51m2dan](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan) |
| **44** | 64.49 | 55.75 | 7.68 | 54.13 *(BP 42.29m)* | [vqcs1ql4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4) |
| **Mean ± SD** | **65.12 ± 0.61** | **56.13 ± 0.56** | **8.31 ± 0.62** | **~53.39 phút/seed** | `r1_cifar100_k5_summary.json` |
  - **Tài nguyên đo ở seed 42:** Peak CPU RSS $\approx 3.55\text{ GB}$; Peak CUDA allocated $\approx 378.42\text{ MiB}$; R/W/FE state $= 522.50\text{ MB}$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [6] K. He et al., "Deep residual learning for image recognition," CVPR, 2016.`
- **Speaker Notes (~205 từ):**
  > Slide 21 trình bày kết quả đầu tiên của bộ benchmark chính **R1** trên cả ba hạt giống độc lập `42, 43, 44`. Quan sát biểu đồ phía bên trái: cả ba đường training Cross-Entropy đều giảm mượt mà và hội tụ rất ổn định sau hai mốc hạ learning rate tại epoch $80$ ($0.1 \to 0.01$) và epoch $120$ ($0.01 \to 0.001$). Không có hiện tượng sụp gradient hay thiếu đồng nhất giữa các lần huấn luyện ResNet-32; mỗi seed mất khoảng 40.4–42.3 phút cho 160 epochs BP trên GPU T4 và tổng cộng 52.3–54.1 phút cho toàn bộ chuỗi huấn luyện lẫn 6 pha streaming giải tích $d=8000$. Qua bảng bên phải (được tính trước trên từng seed trước khi tổng hợp `ddof=1`), giá trị trung bình đạt $\bar{A} = 65.12 \pm 0.61\%$ (giá trị đầy đủ trong JSON là $65.1196 \pm 0.6104\%$), độ chính xác cuối trên đủ 100 lớp là $56.13 \pm 0.56\%$, và độ suy giảm trên tập 50 lớp nền là $8.31 \pm 0.62$ điểm phần trăm. Toàn bộ ba checkpoint và ba run W&B đều được lưu giữ công khai, không hề lọc bỏ seed thấp (`seed 44` đạt `64.49%` vẫn được giữ đầy đủ trong trung bình).

---

### Slide 22 — Kết quả R1 (2/3): Diễn biến Accuracy qua 6 Phases ($50 \to 100$ Lớp)
- **Tiêu đề trên slide:** Đường cong R1 qua 6 phases: Độ chính xác Base giữ vững trên $68\%$ dù tổng số lớp tăng gấp đôi
- **Bố cục trực quan:** Trái (58%): Chèn hình tự vẽ [`reports/figures/r1_cifar100_k5_accuracy.png`](../reports/figures/r1_cifar100_k5_accuracy.png); Phải (42%): Bảng 6 mốc ($k=0\dots 5$) chi tiết $A_k$ (All-seen), $B_k$ (Base 50), và $N_k$ (Lớp mới từng phase).
- **Nội dung trên slide:**
  - **Hình minh họa:** Chèn [`reports/figures/r1_cifar100_k5_accuracy.png`](../reports/figures/r1_cifar100_k5_accuracy.png) (*All-seen test accuracy của Seed 42, 43, 44 và Mean ± Sample SD qua số lớp đã thấy 50→100*).
  - **Bảng diễn biến theo từng phase (Mean ± SD qua 3 seeds: `zt1014ei`, `v51m2dan`, `vqcs1ql4`):**
| Phase ($k$) | Lớp seen ($C_k$) | Ảnh test | $A_k$ All-seen (%) | $B_k$ Base 50 (%) | $N_k$ Lớp mới (%) |
|---:|---:|---:|---:|---:|---:|
| **0 (Base)** | 50 | 5,000 | **76.64 ± 0.12** | **76.64 ± 0.12** | — |
| **1** | 60 | 6,000 | 70.28 ± 0.69 | 73.51 ± 0.16 | 54.13 ± 3.95 |
| **2** | 70 | 7,000 | 66.29 ± 1.62 | 72.02 ± 0.16 | 54.40 ± 7.07 |
| **3** | 80 | 8,000 | 62.45 ± 0.67 | 70.59 ± 0.37 | 47.90 ± 5.61 |
| **4** | 90 | 9,000 | 58.93 ± 0.13 | 69.22 ± 0.44 | 43.43 ± 4.23 |
| **5** | 100 | 10,000 | **56.13 ± 0.56** | **68.33 ± 0.50** | **42.00 ± 4.96** |
  - **Nhận xét cơ chế từ dữ liệu:** $B_k$ giảm đều nhẹ ($\approx 1.0\text{–}3.1\text{ pp}$/phase) và SD rất nhỏ ($\le 0.50\%$), trong khi $N_k$ biến động mạnh theo từng nhóm 10 lớp ($42.00\text{–}54.40\%$) vì ResNet-32 được đóng băng từ 50 lớp base.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~200 từ):**
  > Slide 22 phân rã đường cong R1 qua từng pha từ 50 đến 100 lớp. Đường màu đen trên đồ thị thể hiện trung bình cộng kèm thanh sai số `sample SD` qua ba seed. Ở điểm xuất phát $k=0$, sau căn chỉnh giải tích ARaBT trên 50 lớp nền, mô hình đạt trung bình $76.64\%$ và độ lệch chuẩn rất hẹp ($0.12\%$). Khi nạp thêm từng đợt 10 lớp mới, đường all-seen $A_k$ giảm mượt từ $76.64\%$ xuống $56.13\%$ ở pha cuối trên trọn bộ $10{,}000$ ảnh test. Bảng bên phải hé lộ hai cơ chế thú vị đằng sau sự sụt giảm đó: thứ nhất, độ chính xác riêng trên tập 50 lớp nền ($B_k$) dù đứng trước áp lực cạnh tranh $\operatorname{argmax}$ gấp đôi lớp vẫn giữ được $68.33\%$ ở pha cuối ($F_{\mathrm{base}} = 8.31\text{ pp}$); thứ hai, độ chính xác trên nhóm 10 lớp mới vừa học ($N_k$) chỉ dao động quanh mức $42\text{–}54\%$ và có độ lệch chuẩn lớn (lên tới $7.07\%$ ở Phase 2). Điều này chứng minh nhận định lý thuyết: vì CNN chỉ học biểu diễn từ 50 lớp ban đầu, khả năng tổng quát hóa sang nhóm 10 lớp mới phụ thuộc vào độ gần ngữ nghĩa của các lớp đó với tập base.

---

### Slide 23 — Kết quả R1 (3/3): Đối chiếu Paper Table 1 & Bối cảnh Thực nghiệm
- **Tiêu đề trên slide:** Đối chiếu với Table 1 của bài báo: R1 đạt $\bar{A} = 65.12\%$ vs $66.30\%$ (lệch $-1.18\text{ pp} / -1.78\%$ tương đối)
- **Bố cục trực quan:** Bảng so sánh chính giữa **ACIL Nhóm (R1)** và **ACIL Paper Table 1** ở trên; Bảng bối cảnh **các phương pháp Table 1 (trích nguyên văn paper)** kèm nhãn phân biệt rõ *Nhóm tự chạy vs Trích nguồn* ở dưới.
- **Nội dung trên slide:**
  - **1. Đối chiếu trực diện ACIL (CIFAR-100, $K=5$, $d=8000$, $\gamma=0.1$):**
| Metric | Nhóm R1 (Mean ± SD, $n=3$) | Paper Table 1 [1] | Ours − Paper (pp) | Sai lệch tương đối (%) | Nguồn W&B |
|---|---:|---:|---:|---:|---|
| **$\bar{A}$ (%)** | **65.12 ± 0.61** *(65.1196)* | **66.30** | **-1.18** *(-1.1804)* | **-1.78%** *(-1.7804%)* | [zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) · [v51m2dan](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan) · [vqcs1ql4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4) |
| **$A_{\mathrm{final}}$ (%)** | **56.13 ± 0.56** *(56.1300)* | *Không báo trong Table 1* | — | — | Như trên |
| **$F_{\mathrm{base}}$ (pp)** | **8.31 ± 0.62** *(8.3067)* | **9.00** *(độ giảm dương)* | **-0.69** *(-0.6933)* | **-7.70%** *(-7.7037%)* | Như trên |
  - **2. Bối cảnh Table 1 (CIFAR-100, $K=5$ — *Các baseline khác chỉ trích từ bài báo [1], nhóm KHÔNG tự chạy lại*):**
    - *Exemplar-free (Paper):* EWC ($16.04\%$) $\cdot$ LwF ($24.03\%$) $\cdot$ **ACIL Paper ($66.30\%$)** vs. **ACIL Nhóm R1 ($65.12\%$)**.
    - *Exemplar Replay 20/class (Paper):* iCaRL [2] ($57.12\%$) $\cdot$ LUCIR [8] ($63.17\%$) $\cdot$ PODNet [4] ($64.83\%$) $\cdot$ AANets [5] ($66.31\%$).
  - **Ghi chú trung thực:** Với $K=5$, kết quả R1 của nhóm vẫn nằm trong vùng cạnh tranh trực tiếp với PODNet ($64.83\%$), nhưng nhóm chưa tự chạy lại replay baselines hay $K=25$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., CVPR, 2017. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021. [8] S. Hou et al., CVPR, 2019.`
- **Speaker Notes (~210 từ):**
  > Slide 23 đặt kết quả tái lập R1 cạnh trực tiếp Bảng 1 của bài báo NeurIPS 2022. Trên cùng thiết lập CIFAR-100 năm pha với $d=8000$ và $\gamma=0.1$, độ chính xác trung bình tăng dần của nhóm là $65.12\%$ (giá trị đầy đủ trong file tổng hợp là $65.1196\%$), thấp hơn con số $66.30\%$ do tác giả công bố đúng $1.18$ điểm phần trăm, tương đương sai lệch tương đối $-1.78\%$. Về độ quên trên tập nền $F_{\mathrm{base}}$, nhóm đo được mức giảm $8.31\text{ pp}$, thấp hơn một chút so với $9.00$ trong Bảng 1 (tức giảm quên hơn $-0.69\text{ pp}$). Bài báo không công bố cột $A_{\mathrm{final}}$ trong Bảng 1 nên nhóm để trống phần tham chiếu này thay vì tự suy đoán. Nhằm cung cấp ngữ cảnh khoa học đầy đủ, phần dưới slide đặt kết quả R1 cạnh các baseline từ Bảng 1 như LwF ($24.03\%$), iCaRL ($57.12\%$) hay PODNet ($64.83\%$), đồng thời ghi chú rõ ràng rằng những số liệu baseline đó được trích nguyên văn từ bài báo chứ nhóm chưa chạy lại. Nguyên nhân có thể giải thích khoảng cách $-1.18\text{ pp}$ sẽ được phân tích kỹ qua R2 và phần đối chiếu code tác giả ở các slide sau.

---

### Slide 24 — Kết quả R2 (1/2): Tác động của Feature Expansion & Nhạy cảm với $\gamma$
- **Tiêu đề trên slide:** Kết quả R2 (4/5 hàng Table 2): Bật FE tăng **$+11.86\text{ pp}$**, còn $\gamma \in [0.001, 0.1]$ hầu như bất biến
- **Bố cục trực quan:** Trái (55%): Chèn hình tự vẽ [`reports/figures/r2_cifar100_k5_ablation.png`](../reports/figures/r2_cifar100_k5_ablation.png); Phải (45%): Bảng kết quả 4 hàng × 3 seeds kèm cột Paper Table 2, Deviation và Paired $\Delta\bar{A}$ theo cùng checkpoint seed.
- **Nội dung trên slide:**
  - **Hình minh họa:** Chèn [`reports/figures/r2_cifar100_k5_ablation.png`](../reports/figures/r2_cifar100_k5_ablation.png) (Trái: cột $\bar{A}$ vs hình thoi Paper Table 2; Phải: Paired $\Delta\bar{A}$ theo seed so với $\gamma=0.1$).
  - **Bảng đối chiếu 4/5 hàng Table 2 (Tái dùng checkpoint R1 cùng seed — 9 runs mới + 3 R1):**
| Cấu hình | Ours $\bar{A}$ ± SD (%) | Paper Table 2 [1] | Ours − Paper (pp) | Paired $\Delta\bar{A}$ vs R1 (pp) | W&B Runs (Seeds 42 / 43 / 44) |
|---|---:|---:|---:|---:|---|
| **No-FE ($d=64$), $\gamma=0.1$** | **53.25 ± 0.28** | 52.99 | **+0.26** | **-11.86 ± 0.37** | [xdrm0d27](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xdrm0d27) · [7wgme4so](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7wgme4so) · [81aci8h1](https://wandb.ai/hsangg-2006/acil-reproduction/runs/81aci8h1) |
| **FE 8000, $\gamma=0.1$ (R1)** | **65.12 ± 0.61** | 66.30 | -1.18 | 0.00 | [zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) · [v51m2dan](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan) · [vqcs1ql4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4) |
| **FE 8000, $\gamma=0.01$** | **65.12 ± 0.61** *(65.1173)* | 66.25 | -1.13 | -0.0022 ± 0.0020 | [qcgdklxh](https://wandb.ai/hsangg-2006/acil-reproduction/runs/qcgdklxh) · [xfleip26](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xfleip26) · [dub202jk](https://wandb.ai/hsangg-2006/acil-reproduction/runs/dub202jk) |
| **FE 8000, $\gamma=0.001$** | **65.12 ± 0.61** *(65.1163)* | 66.23 | -1.11 | -0.0032 ± 0.0035 | [3yp99ewh](https://wandb.ai/hsangg-2006/acil-reproduction/runs/3yp99ewh) · [aler6nj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/aler6nj2) · [l9c50ezb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/l9c50ezb) |
  - **Phát hiện chính:** (1) Bỏ FE làm $\bar{A}$ sụp **$-11.86\text{ pp}$**, khẳng định vai trò sống còn của FE; (2) Trong dải $\gamma \in [0.001, 0.1]$, biên độ $\bar{A}$ chỉ **$0.0032\text{ pp}$**.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~205 từ):**
  > Slide 24 chuyển sang thí nghiệm thành phần **R2**, tái lập 4 trên 5 hàng của Table 2 trong bài báo [1]. Điểm chặt chẽ của thiết kế là mọi hàng đều tái dùng chính xác cùng checkpoint ResNet-32 đã huấn luyện xong của mỗi seed (chặn hẳn hàm train bằng mock để đảm bảo công bằng tuyệt đối) và cùng ma trận $W_{\mathrm{fe}}$ cho cả ba mức $\gamma$. Hai kết luận nổi bật hiện ra ngay trên biểu đồ và bảng số liệu. Thứ nhất, khi bỏ hoàn toàn tầng Feature Expansion — nghĩa là dùng trực tiếp vector 64 chiều $F$ làm đầu vào cho analytic head — $\bar{A}$ rơi từ $65.12\%$ xuống còn $53.25\%$, sụt mất $11.86$ điểm phần trăm (khớp rất sát con số $52.99\%$ ở Table 2, chỉ chênh $+0.26\text{ pp}$). Điều này chứng minh trực tiếp cho Định lý Cover đã nêu ở Slide 10: thiếu tầng mở rộng lên $d=8000$, bộ phân loại tuyến tính không đủ sức tách 100 lớp. Thứ hai, biểu đồ bên phải vẽ hiệu cặp theo từng seed khi giảm $\gamma$ từ $0.1$ xuống $0.01$ và $0.001$: độ thay đổi trung bình chỉ ở mức $-0.0022$ và $-0.0032\text{ pp}$ — chứng tỏ ACIL cực kỳ ổn định và không nhạy siêu tham số $\gamma$ trong miền này.

---

### Slide 25 — Kết quả R2 (2/2): Giải mã Nghịch lý "$F_{\mathrm{base}}$ của No-FE chỉ $1.56\text{ pp}$" & Giới hạn Ablation
- **Tiêu đề trên slide:** Cẩn trọng diễn giải R2: Vì sao No-FE có $F_{\mathrm{base}}$ nhỏ hơn ($1.56\text{ pp}$) nhưng chất lượng lại kém hơn hẳn?
- **Bố cục trực quan:** Bảng đối chiếu $A_{\mathrm{final}}$ & $F_{\mathrm{base}}$ giữa No-FE và 3 cấu hình FE ở trên; Hai cột giải thích cơ chế & 2 giới hạn thiết kế R2 ở dưới.
- **Nội dung trên slide:**
  - **Bảng đối chiếu $A_{\mathrm{final}}$ & $F_{\mathrm{base}}$ (3 seeds, `ddof=1`):**
| Cấu hình | $\bar{A}$ (%) | $A_{\mathrm{final}}$ (100 lớp, %) | $F_{\mathrm{base}}$ (pp) | Nhận định toàn diện |
|---|---:|---:|---:|---|
| **No-FE ($d=64$), $\gamma=0.1$** | 53.25 ± 0.28 | **38.33 ± 0.37** | **1.56 ± 0.64** | Trần ban đầu & lớp mới quá yếu $\to$ $A_{\mathrm{final}}$ thấp kỷ lục |
| **FE 8000, $\gamma \in \{0.1, 0.01, 0.001\}$** | 65.12 ± 0.61 | **56.13 ± 0.56** | **8.31 ± 0.62** | Cao hơn No-FE **+17.80 pp** ở $A_{\mathrm{final}}$ |
  - **Giải mã nghịch lý:** Nếu chỉ nhìn cột $F_{\mathrm{base}}$, dễ ngộ nhận No-FE "ít quên hơn" ($1.56 < 8.31\text{ pp}$). Thực chất ở không gian $64$ chiều không có FE, bộ phân loại tuyến tính không biểu diễn nổi các lớp mới, nên các lớp mới rất ít khi thắng được $\operatorname{argmax}$ để cạnh tranh với 50 lớp cũ $\implies$ $A_{\mathrm{final}}$ chỉ còn **$38.33\%$**! Do đó **không bao giờ đọc $F_{\mathrm{base}}$ tách rời khỏi $\bar{A}$ và $A_{\mathrm{final}}$**.
  - **Hai giới hạn được ghi nhận trung thực trong R2:**
    1. Bỏ FE đồng thời bỏ cả phép chiếu tuyến tính $64\to 8000$ và phi tuyến $\operatorname{ReLU}$ — R2 chưa tách riêng đóng góp của số chiều $d$ với phi tuyến $\operatorname{ReLU}$.
    2. Loại hàng $\gamma=0$ khỏi scope (báo **4/5 hàng Table 2**) vì thuật toán RLS + Woodbury hiện tại yêu cầu $\gamma > 0$ để bảo đảm $A_0 \succ 0$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Slide 25 đưa ra một điểm phân tích rất đáng giá mà nếu không quan sát kỹ rất dễ đọc nhầm số liệu R2. Khi nhóm đo $F_{\mathrm{base}}$ cho cấu hình No-FE, độ sụt accuracy trên 50 lớp gốc chỉ là $1.56\text{ pp}$, thấp hơn nhiều so với mức $8.31\text{ pp}$ của cấu hình đầy đủ FE 8000. Liệu có thể kết luận bỏ FE giúp giảm quên không? Câu trả lời chắc chắn là không: trên $100$ lớp ở pha cuối, độ chính xác tổng $A_{\mathrm{final}}$ của No-FE rớt xuống chỉ còn $38.33\%$ so với $56.13\%$ của FE 8000 — chênh lệch tới gần $18$ điểm phần trăm! Lý do là đặc trưng 64 chiều bị quá tải không thể mở ra các siêu phẳng phân tách cho 50 lớp mới sau này, nên các lớp mới gần như "không có tiếng nói" trong $\operatorname{argmax}$, gián tiếp khiến 50 lớp cũ ít bị cướp phiếu. Vì vậy bài học phương pháp là phải đọc $F_{\mathrm{base}}$ song hành cùng $\bar{A}$ và $A_{\mathrm{final}}$. Cuối slide, nhóm cũng minh bạch hai giới hạn: chưa tách riêng tác động của chiều $d$ với hàm ReLU, và chưa chạy hàng $\gamma = 0$ do đặc thù solver yêu cầu $\gamma > 0$.

---

### Slide 26 — Kết quả R3 (1/2): Kiểm chứng Tương đương Đại số Theo Kích thước Minibatch
- **Tiêu đề trên slide:** Thí nghiệm tự thiết kế R3: Đổi kích thước batch ($32 / 128 / 512$) cho nghiệm **khớp Joint Oracle ở 54/54 mốc**
- **Bố cục trực quan:** Hộp tóm tắt thiết lập chuẩn hóa công bằng (*checkpoint 160 seed 42, $d=1024, \gamma=0.1$, 9 fresh CPU workers, thứ tự xoay*) ở trên; Bảng sai số số học 54/54 mốc đối chiếu Joint Oracle ở dưới.
- **Nội dung trên slide:**
  - **Thiết kế thực nghiệm kiểm chứng công bằng (`docs/R3_KE_HOACH_DO.md`):**
    - Trên cùng checkpoint seed 42 ($d=1024, \gamma=0.1$, thứ tự mẫu cố định): quét 3 batch sizes $b \in \{32, 128, 512\} \times 3$ timing repeats = **9 tiến trình CPU độc lập (4 threads BLAS)** theo thứ tự xoay `32→128→512`, `128→512→32`, `512→32→128`.
    - **Joint Oracle** chạy riêng sau khi đo giờ, ghép đủ $H_{0:k}, Y_{0:k}$ để giải nghiệm chuẩn tham chiếu ở từng phase ($9 \times 6 = \mathbf{54\text{ mốc kiểm tra}}$).
  - **Bảng đối chiếu sai số với Joint Oracle (54/54 đạt `rtol=1e-6, atol=1e-8`):**
| Batch ($b$) | Số mốc | Max rel $W$ error | Max rel $R$ error | Max abs score error | Argmax khác Joint | W&B Runs (3 timing repeats) |
|---:|---:|---:|---:|---:|---:|---|
| **32** | 18 | `1.159e-11` | `3.823e-15` | `2.475e-12` | **0 / 10,000** | [luf7uv40](https://wandb.ai/hsangg-2006/acil-reproduction/runs/luf7uv40) · [yqxph1bb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/yqxph1bb) · [59vlfxzv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/59vlfxzv) |
| **128** | 18 | `9.679e-12` | `3.399e-15` | `2.302e-12` | **0 / 10,000** | [o54hbdfp](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o54hbdfp) · [4481bnn1](https://wandb.ai/hsangg-2006/acil-reproduction/runs/4481bnn1) · [3gv0moj7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/3gv0moj7) |
| **512** | 18 | `9.782e-12` | `3.272e-15` | `2.178e-12` | **0 / 10,000** | [lnk0grbq](https://wandb.ai/hsangg-2006/acil-reproduction/runs/lnk0grbq) · [5ehmjvh3](https://wandb.ai/hsangg-2006/acil-reproduction/runs/5ehmjvh3) · [xesh2kjl](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xesh2kjl) |
  - **Accuracy đồng nhất trên cả 9 runs & Joint (`d=1024`, seed 42):** $\bar{A} = 63.10\%$, $A_{\mathrm{final}} = 51.73\%$, $F_{\mathrm{base}} = 5.96\text{ pp}$. *(Lưu ý: $d=1024$ là cấu hình phân tích riêng, không thay thế $d=8000$ của R1).*
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~205 từ):**
  > Slide 26 giới thiệu **R3** — thí nghiệm mở rộng do nhóm tự thiết kế nhằm kiểm chứng một tuyên bố rất mạnh của ACIL trên dữ liệu ảnh CIFAR-100 đầy đủ: liệu việc chia nhỏ phase thành các minibatch $32$, $128$ hay $512$ có làm thay đổi nghiệm cuối cùng so với học gộp hay không? Để có thể giải Joint Oracle trên toàn bộ $50{,}000$ ảnh huấn luyện trong giới hạn RAM và thời gian Kaggle, nhóm đăng ký trước cấu hình $d = 1024$ trên cùng checkpoint epoch 160 của seed 42, giữ nguyên thứ tự mẫu và chạy 9 worker CPU độc lập theo lịch xoay nhằm loại bỏ thiên lệch thứ tự chạy. Kết quả hoàn toàn xác nhận lý thuyết: trên trọn bộ $54$ phép so sánh (9 runs × 6 mốc đánh giá), sai số tương đối cực đại của ma trận trọng số $W$ đều $\le 1.159\times 10^{-11}$, sai số của $R$ ở cỡ $10^{-15}$, và không có dù chỉ một mẫu ảnh test trong $10{,}000$ ảnh bị đổi nhãn dự đoán argmax. Tất cả chín run và Joint Oracle cùng đạt chính xác $\bar{A} = 63.10\%$ (với lưu ý đây là số liệu $n=1$ tại $d=1024$, không đem thay R1 $d=8000$). Sang Slide 27, ta xem batch size ảnh hưởng thế nào đến tốc độ và RAM.

---

### Slide 27 — Kết quả R3 (2/2): Đánh đổi Thời gian & Bộ nhớ Peak RSS theo Batch Size
- **Tiêu đề trên slide:** Đánh đổi R3: Batch 512 nhanh gấp **$3.47\times$** Batch 32 nhưng Peak RSS tăng $+18.45\text{ MiB}$ so với Batch 128
- **Bố cục trực quan:** Trái (58%): Chèn hình tự vẽ [`reports/figures/r3_cifar100_k5_batch_analysis.png`](../reports/figures/r3_cifar100_k5_batch_analysis.png); Phải (42%): Bảng tách riêng *Base vs 5 Incremental Phases*, *Tổng thời gian*, *Peak RSS* & *Nhận định phát hiện bất ngờ*.
- **Nội dung trên slide:**
  - **Hình minh họa:** Chèn [`reports/figures/r3_cifar100_k5_batch_analysis.png`](../reports/figures/r3_cifar100_k5_batch_analysis.png) (*Trái: Analytic learning time; Phải: Observed learning peak RSS cho 3 timing repeats*).
  - **Bảng phân rã Thời gian & Peak RSS (`d=1024`, 4 threads CPU, $n=3$ timing repeats):**
| Batch ($b$) | Base (s) | 5 Phases Inc (s) | Tổng Analytic (s) | Tỷ lệ vs $b=128$ | Peak RSS (MiB) |
|---:|---:|---:|---:|---:|---:|
| **32** | 10.71 ± 0.07 | 6.62 ± 0.10 | **17.33 ± 0.08** | $2.503\times$ | 151.35 ± 0.97 |
| **128** | 3.19 ± 0.07 | 3.73 ± 0.04 | **6.92 ± 0.11** | $1.000\times$ | **146.10 ± 0.92** *(thấp nhất)* |
| **512** | 1.28 ± 0.07 | 3.71 ± 0.05 | **5.00 ± 0.11** | **$0.722\times$** | 164.55 ± 4.38 |
  - **Hai phát hiện định lượng quan trọng:**
    1. Khi tăng $b: 128 \to 512$, tổng thời gian giảm $1.93\text{ s}$ nhưng **Base giảm $1.91\text{ s}$ ($10.71 \to 3.19 \to 1.28\text{ s}$)**, còn **5 phases incremental gần như đi ngang ($3.73 \to 3.71\text{ s}$, chỉ $-0.02\text{ s}$)** do ma trận lõi Woodbury $b\times b$ tăng kích thước lên $512 \times 512$ bù trừ lợi ích giảm số vòng lặp!
    2. State cuối ($R/W/\mathrm{FE}$) cố định **$9.28\text{ MiB}$** ở mọi batch, nhưng bộ đệm tạm & allocator làm Peak RSS của $b=512$ cao hơn $b=128$ trung bình **$18.45\text{ MiB}$** $\implies$ $b=128$ cân bằng tốt giữa tốc độ incremental và Peak RAM.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. (W&B: luf7uv40, yqxph1bb, 59vlfxzv, o54hbdfp, 4481bnn1, 3gv0moj7, lnk0grbq, 5ehmjvh3, xesh2kjl).`
- **Speaker Notes (~210 từ):**
  > Slide 27 phân tích khía cạnh hiệu năng của R3 qua biểu đồ tự vẽ từ `scripts/plot_r3.py`. Trước tiên lưu ý phạm vi đo rất sạch: đồng hồ bấm giờ chỉ bao gồm 6 cửa sổ học analytic (biến đổi FE, tích lũy/giải base và 5 pha RLS) trên tiến trình CPU 4 luồng; việc tải dữ liệu, đánh giá test hay chạy Joint Oracle đều nằm hoàn toàn ngoài timer. Nhìn cột tổng thời gian, batch 512 hoàn thành trong $5.00\text{ s}$, nhanh hơn $1.386\times$ so với batch 128 ($6.92\text{ s}$) và $3.47\times$ so với batch 32 ($17.33\text{ s}$). Tuy nhiên, bảng phân rã bên phải chỉ ra một phát hiện sâu sắc hơn: hầu hết lợi ích tăng tốc từ 128 lên 512 đến từ bước Base accumulation (giảm từ $3.19\text{ s}$ xuống $1.28\text{ s}$ nhờ phép nhân ma trận BLAS cấp 3 lớn hơn), còn tổng thời gian của 5 pha cập nhật Woodbury gần như đứng yên ở $3.73\text{ s}$ và $3.71\text{ s}$ vì chi phí $\mathcal{O}(b^3 + b^2 d)$ giải hệ $512\times 512$ bắt đầu lộ rõ. Đồng thời Peak RSS ở $b=512$ nhảy thêm $18.45\text{ MiB}$ dù kích thước state cuối cùng bằng nhau ($9.28\text{ MiB}$). Như vậy cấu hình $b=128$ là điểm cân bằng hợp lý cho triển khai thực tế.

---

### Slide 28 — Đối chiếu Implementation Tác giả (1/2): Thiết kế C1 / C2 / C1R & Phát hiện Tầng Dữ liệu–Backbone
- **Tiêu đề trên slide:** Chạy trực tiếp source tác giả (`0afd6bf`): Ba phát hiện ở tầng Mapping, Normalization và Global Pooling
- **Bố cục trực quan:** Trái: Thiết kế 3 kịch bản **C1**, **C2**, **C1R** (trên venv tách biệt, giữ source tác giả nguyên bản); Phải: 3 thẻ phát hiện kỹ thuật ở tầng Dữ liệu & Backbone (kèm bằng chứng số đo thực tế).
- **Nội dung trên slide:**
  - **Thiết lập đối chiếu (`scripts/compare_author_acil.py`, repo `ZHUANGHP/Analytic-continual-learning` @ `0afd6bf`):**
    - Tách môi trường riêng (không sửa hay chép code tác giả vào `src/`); dùng chung checkpoint 160 seed 42, $d=1024, \gamma=0.1$.
    - **C1 (Controlled):** Gọi class `ACIL` tác giả với cùng FE $\mathcal{N}(0,1)$ & thứ tự mẫu như R3.
    - **C2 (Native FE / `main.main`):** Nạp checkpoint sang ResNet-32 tác giả, dùng `RandomBuffer` gốc và DataLoader shuffle gốc.
    - **C1R (Diagnostic hậu nghiệm):** Nạp state $(R_0, W_0)$ của nhóm rồi gọi 5 bước cập nhật tác giả.
  - **Ba khác biệt định vị được trước tầng Analytic:**
    1. **Mapping có điều kiện (`DatasetWrapper`):** Bật `shuffle_seed=42` trong wrapper tác giả gây `IndexError: Target 91 is out of bounds` do shuffle lớp nhưng **chưa remap target về `0..49`** *(pipeline nhóm luôn áp `original_to_global` cho train/test)*.
    2. **Preprocessing Normalization:** Nhóm tính trên 25,000 ảnh base seed 42 vs tác giả dùng hằng số CIFAR cố định $\implies$ max diff tensor chuẩn hóa $= 0.0252 \implies$ max diff feature CNN $= 0.3558$ (trên 1,000 ảnh test).
    3. **Backbone Global Pooling (`FP32` vs `FP64`):** Trọng số trước pooling khớp **bitwise (`0.0`)**; sau pooling (`adaptive_avg_pool` vs `avg_pool`) lệch max **$3.81\times 10^{-6}$ ở FP32** (và **giảm về $3.55\times 10^{-15}$ ở FP64**), prediction vẫn khớp `1000/1000`.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022 (Đối chiếu với ZHUANGHP/Analytic-continual-learning commit 0afd6bf trên W&B 0tlcy6id, 8muywodv, yeb0jqfb).`
- **Speaker Notes (~215 từ):**
  > Theo đúng yêu cầu môn học, thay vì chỉ đọc mã nguồn gốc trên GitHub, nhóm tạo một môi trường ảo độc lập, checkout repository của tác giả tại commit `0afd6bf`, và chạy đối chiếu trực tiếp qua ba thí nghiệm: C1 kiểm soát cùng ma trận mở rộng FE, C2 chạy qua chính hàm `main.main` với khởi tạo gốc `RandomBuffer` (lưu ý `reset_parameters` của PyTorch Linear sinh phân phối khác hẳn $\mathcal{N}(0,1)$ của nhóm), và C1R dùng chung trạng thái nền $(R_0, W_0)$. Khi đối chiếu từng tầng từ đầu vào tới đầu ra trên $1{,}000$ ảnh test, nhóm phát hiện ba chi tiết rất thực tế. Thứ nhất, wrapper dữ liệu của repo tác giả khi bật `shuffle_seed` làm nổ lỗi `IndexError: Target 91 is out of bounds` trước khi học vì nhãn chưa được ánh xạ lại về đoạn `0..49`. Thứ hai, sự khác nhau giữa thống kê chuẩn hóa tính riêng từ 25,000 ảnh base với bộ hằng số CIFAR mặc định tạo ra chênh lệch đặc trưng lên tới $0.3558$. Thứ ba, ngay cả khi nạp cùng trọng số ResNet-32, tensor trước pooling giống hệt nhau nhưng khác nhau nhỏ ở phép cộng dấu phẩy động FP32 giữa `adaptive_avg_pool` và `avg_pool` ($3.81\times 10^{-6}$, biến mất khi chuyển sang FP64).

---

### Slide 29 — Đối chiếu Implementation Tác giả (2/2): Kết quả C1 / C2 / C1R & Định vị Sai số Số học
- **Tiêu đề trên slide:** Kết quả so sánh số học: C1 lệch 2/3.55M phần tử score ở phase 3–4; C1R thu hẹp sai lệch về $10^{-13}$
- **Bố cục trực quan:** Trái (58%): Chèn hình tự vẽ [`reports/figures/author_acil_comparison.png`](../reports/figures/author_acil_comparison.png); Phải (42%): Bảng 3 lượt đối chiếu C1 / C2 / C1R (trung thực giữ `C1 equivalence=false`) + chẩn đoán cơ chế hội tụ ở bước Base.
- **Nội dung trên slide:**
  - **Hình minh họa:** Chèn [`reports/figures/author_acil_comparison.png`](../reports/figures/author_acil_comparison.png) (*Trái: Accuracy C1 vs C2 trên `d=1024`; Phải: Relative Frobenius W error ở thang log từ $10^{-8}$ xuống $10^{-13}$*).
  - **Bảng đối chiếu kết quả (`rtol=1e-6, atol=1e-8`, seed 42, `d=1024`, $\gamma=0.1$):**
| Lượt | $\bar{A}$ (%) | $A_{\mathrm{final}}$ (%) | Max rel $W$ err | Max abs score err | Đạt `allclose` (R/W/Score) | Argmax khác | W&B Run |
|---|---:|---:|---:|---:|---|---|---|
| **C1 (Controlled N(0,1))** | 63.10 | 51.73 | `3.025e-8` | `2.084e-8` | R `6/6`, W `6/6`, **Score `4/6`** *(Fail)* | **0** | [0tlcy6id](https://wandb.ai/hsangg-2006/acil-reproduction/runs/0tlcy6id) |
| **C2 (Native FE / main)** | 63.15 | 52.24 | `6.911e-11` | `3.349e-11` | **Cả ba `6/6`** | **0** | [8muywodv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/8muywodv) |
| **C1R (Matched Base)** | — | 51.73 | **`5.607e-13`** | **`6.645e-13`** | **Cả ba `5/5`** | **0** | [yeb0jqfb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/yeb0jqfb) |
  - **Giải thích trung thực:**
    - Giữ nguyên cờ **C1 `analytic_equivalence_passed = false`** (dù chỉ **2 trong 3,550,000 phần tử score** ở phase 3–4 nhỉnh hơn ngưỡng `allclose` và `0` nhãn dự đoán lệch).
    - **Chẩn đoán C1R định vị nguyên nhân:** Khi nạp đúng $(R_0, W_0)$ của nhóm rồi chạy 5 phase incremental bằng code tác giả, sai số giảm $50{,}000\times$ xuống cỡ **$10^{-13}$** $\implies$ chênh lệch ở C1 nảy sinh chủ yếu từ **cách khởi tạo Base** (tác giả RLS từ $R = I/\gamma$ + `torch.inverse` vs nhóm dùng tổng Gram + Cholesky).
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022 (Đối chiếu với ZHUANGHP/Analytic-continual-learning @ 0afd6bf).`
- **Speaker Notes (~215 từ):**
  > Slide 29 báo cáo đầy đủ kết quả đối chiếu implementation với tinh thần khoa học trung thực cao nhất. Trên đường đồ thị bên phải, khi giữ cùng FE $\mathcal{N}(0,1)$ ở lượt **C1**, sai số tương đối của $W$ nằm ở mức $3\times 10^{-8}$; dù toàn bộ $6/6$ mốc của $R$, $W$ và $100\%$ dự đoán nhãn argmax hoàn toàn trùng khớp với hệ nhóm, có tổng cộng $2$ trên $3.55\text{ triệu}$ phần tử ma trận score ở Phase 3 và 4 vượt nhẹ ngưỡng `allclose(1e-6, 1e-8)`. Nhóm tuyệt đối không nới lỏng ngưỡng hay sửa trạng thái mà giữ nguyên cờ `C1 = false` trên cả raw JSON lẫn W&B. Thay vào đó, nhóm thiết kế bước chẩn đoán **C1R** — đường xanh lá đứt nét nằm ở đáy biểu đồ: khi nạp chung trạng thái $(R_0, W_0)$ sau Phase 0 rồi mới gọi phần cập nhật tăng dần của tác giả, sai số lập tức rớt xuống cỡ $10^{-13}$ và đạt trọn $5/5$ pha. Điều này chứng minh phần mềm của nhóm và tác giả đồng nhất thuật toán incremental, còn chênh lệch nhỏ ban đầu nằm ở phương pháp tính Phase 0: code tác giả khởi đầu từ $I/\gamma$ rồi RLS từng batch bằng `torch.inverse`, còn nhóm dùng tích lũy Gram và Cholesky ổn định hơn khi ma trận $\mathcal{N}(0,1)$ chưa chuẩn hóa biên độ.

---

### Slide 30 — Phân tích Tổng hợp Sai lệch ($\bar{A}_{\mathrm{R1}} - \bar{A}_{\mathrm{paper}} = -1.18\text{ pp}$): Đã kiểm chứng vs. Giả thuyết
- **Tiêu đề trên slide:** Vì sao R1 đạt $65.12\%$ thay vì $66.30\%$ ($-1.18\text{ pp}$)? Phân tách rạch ròi điều đã đo và giả thuyết
- **Bố cục trực quan:** Bảng hai vùng rõ rệt: Vùng 1 — *Các sai khác cấu hình/số học ĐÃ XÁC NHẬN qua kiểm chứng* (4 yếu tố); Vùng 2 — *Giới hạn phân tích — CHƯA CHẠY ABLATION tách từng nhân tố* (khẳng định không quy kết võ đoán).
- **Nội dung trên slide:**
  - **1. Các nguồn sai khác ĐÃ ĐƯỢC XÁC NHẬN qua kiểm chứng giao thức & đối chiếu tác giả:**
    1. **Thứ tự chia lớp (`class_order`):** Nhóm hoán vị theo seed `42/43/44` (SD $= 0.61\%$ chứa cả độ khó của 50 lớp nền), trong khi PODNet [4] / AANets [5] giữ cố định một hoặc ba thứ tự dựng sẵn và bài báo không công bố danh sách order của Table 1.
    2. **Phân phối Feature Expansion $W_{\mathrm{fe}}$:** Nhóm dùng $\mathcal{N}(0,1)$ theo mục 3.1 bài báo [1]; repo tác giả `0afd6bf` dùng `Linear.reset_parameters()` (độ lệch chuẩn thực đo $\approx 0.0724$, có hiệu ứng nhỏ trên 1 seed ở C2 là $+0.045\text{ pp}$ $\bar{A}$ / $+0.51\text{ pp}$ final).
    3. **Thống kê chuẩn hóa ảnh (Normalization):** Nhóm tính từ base train ($25{,}000$ ảnh) để tránh rò rỉ dữ liệu vs tác giả dùng hằng số CIFAR-100 $\implies \Delta F_{\max} = 0.3558$.
    4. **Chi tiết số học & môi trường:** Phép pooling FP32 (`adaptive_avg_pool` vs `avg_pool`), Python 3.13 / Torch 2.11 so với năm 2022, và hướng dẫn training trong README mới của tác giả (`300 epochs`, cosine, `d=8192`) khác với thông số `160 epochs / d=8000` trong paper Table 1.
  - **2. Điều CHƯA được phép khẳng định (cần Ablation riêng ở bước sau):**
    - Nhóm **chưa chạy 3-seed ablation riêng** để định lượng xem trong $-1.18\text{ pp}$, bao nhiêu phần trăm đến từ `class_order` hay từ khởi tạo $W_{\mathrm{fe}}$ / normalization. Không lấy suy đoán thay bằng chứng.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021. [6] K. He et al., CVPR, 2016.`
- **Speaker Notes (~210 từ):**
  > Slide 30 tổng kết bài học về tính liêm chính khoa học khi giải thích khoảng cách $-1.18$ điểm phần trăm giữa $\bar{A}$ của nhóm ($65.12\%$) và con số $66.30\%$ của bài báo. Nhóm tách rạch ròi giữa hai cột: những yếu tố *đã kiểm chứng thực tế là khác nhau* và những gì *chưa được chạy thí nghiệm tách riêng*. Ở nhóm đã xác nhận, thứ nhất là thứ tự chia lớp: kiểm tra mã nguồn PODNet và AANets cho thấy các repo nền giữ thứ tự lớp cố định, còn nhóm sinh hoán vị độc lập cho cả ba seed 42, 43, 44. Thứ hai, đối chiếu code `0afd6bf` chứng minh tác giả khởi tạo tầng Buffer theo `Linear.reset_parameters` chứ không phải $\mathcal{N}(0,1)$ chuẩn tắc như bài báo mô tả, và phép chuẩn hóa CIFAR cố định khác với chuẩn hóa base-only của nhóm. Thứ ba, chính README mới của tác giả đã đổi sang lịch 300 epochs và $d=8192$, khác bản gốc NeurIPS 2022. Tuy vậy, nhóm không được phép võ đoán rằng "tại khác X nên mới lệch $-1.18\text{ pp}$", bởi muốn quy kết nguyên nhân định lượng thì cần chạy ablation từng yếu tố trên đủ 3 seeds.

---

### Slide 31 — Các Thay đổi Kỹ thuật Trọng yếu & Kỷ luật Tái lập
- **Tiêu đề trên slide:** Kỹ nghệ Triển khai: 5 Quyết định kiến trúc giúp kết quả sạch lỗi, chống tràn RAM và truy nguyên 100%
- **Bố cục trực quan:** Danh sách cấu trúc 5 dòng *Vấn đề phát hiện $\to$ Giải pháp đã cài đặt (`src/`) $\to$ Giá trị bảo chứng khoa học*.
- **Nội dung trên slide:**
  1. **Nhãn toàn cục (`original_to_global`) & Global Residual Update:** Phát hiện lỗi tiềm ẩn từ công thức ghép khối Eq. (9) nếu gọi theo minibatch làm phình cột $W$; chuyển sang ID toàn cục & đệm $W_{\mathrm{pad}}$ giữ số cột bằng đúng $C_k$ (`src/analytic.py`, `src/cifar_data.py`).
  2. **Streaming Base & Khóa bộ nhớ (`initialize_batches`):** Tích lũy Gram/Cross theo batch + Cholesky giúp chạy $d=8000$ không phải giữ ma trận $H_0$ ($25{,}000\times 8000 \approx 1.6\text{ GB}$), tách hẳn Joint Oracle ra khỏi đường benchmark (`src/streaming.py`).
  3. **Cô lập lượt chạy (`attempt_id` + `filelock`):** Mọi lần chạy tạo thư mục riêng; trạng thái `running / complete / failed` cập nhật an toàn chống race condition, tránh lấy nhầm JSON thành công của lần chạy trước (`src/run_attempt.py`).
  4. **Checkpoint v2 cho Kaggle Timeout:** Lưu đầy đủ `model`, `optimizer`, `scheduler`, `completed_epoch`, RNG của Python/NumPy/Torch CPU+CUDA và `DataLoader` generator để resume chính xác ở biên epoch (`src/base_training.py`, `src/verify_resume.py`).
  5. **Truy nguyên & Đồng bộ W&B R3 (`reconcile_r3_tracking.py`):** Giữ nguyên JSON gốc chụp offline; vá lỗi ghi đè summary R3 trên W&B bằng alias `r3/learning_seconds_total` & `r3/max_abs_score_error_all_phases` kèm SHA-256 trên bytes UTF-8/LF.
- **Footer IEEE:** `Nội bộ nhóm (Chi tiết triển khai trong src/ và reports/BAO_CAO_THUC_NGHIEM_GIUA_KY_20261107.md).`
- **Speaker Notes (~195 từ):**
  > Trước khi kết thúc phần nội dung chính, Slide 31 tóm tắt năm bài học kỹ thuật xương sống mà nhóm đúc kết qua hàng loạt test và sự cố thực tế trong quá trình tự xây dựng codebase. Nổi bật nhất là việc chuyển công thức ghép khối sang cập nhật residual trên ID nhãn toàn cục để hỗ trợ minibatch đúng toán học; thứ hai là kiến trúc streaming Gram cộng dồn theo batch ở Phase 0 nhằm hạ peak RAM trên Kaggle; thứ ba là cơ chế cô lập theo `attempt_id` kết hợp khóa tệp `filelock`, đảm bảo một lượt chạy lỗi không bao giờ để sót file kết quả cũ gây nhầm lẫn; thứ tư là hệ thống checkpoint v2 bảo toàn trọn vẹn bộ sinh số ngẫu nhiên cho kịch bản timeout hai giờ; và thứ năm là kỷ luật provenance: khi phát hiện W&B mặc định lấy giá trị pha cuối ghi đè lên tổng thời gian của R3 lúc `wandb sync`, nhóm giữ nguyên file numeric gốc không sửa số, viết script đối chiếu bổ sung trường alias và gắn 4 regression tests kiểm tra tự động sau khi chạy.

---

### Slide 32 — Tổng kết Giữa kỳ & Định hướng Cuối kỳ (`Critical Analysis` & `Empirical Demonstration`)
- **Tiêu đề trên slide:** Tổng kết Giữa kỳ (Core Scope hoàn tất 100%) & Lộ trình Phản biện – Thực nghiệm Cuối kỳ
- **Bố cục trực quan:** Trái: 4 gạch thành tựu đã chốt kèm phạm vi giới hạn trung thực; Phải: Sơ đồ chuyển tiếp sang 2 hướng điều tra Cuối kỳ (P1 & P2 theo `docs/PHAN_CONG_NHOM.md`).
- **Nội dung trên slide:**
  - **Thành tựu đã hoàn tất cho Giữa kỳ:**
    - Đọc hiểu trọn vẹn ACIL (5 lớp phương pháp, chứng minh & điều kiện giới hạn Định lý 3.1, so sánh 3 tiền nhiệm, tự vẽ 2 sơ đồ + 5 biểu đồ).
    - Tái lập đầy đủ **Core Scope (`CIFAR-100 / K=5`)**: **R1** ($\bar{A}=65.12 \pm 0.61\%$), **R2** (FE $+11.86\text{ pp}$, $\gamma$ ổn định), **R3** (54/54 mốc khớp Joint $\le 1.16\times 10^{-11}$), **Author Compare C1/C2/C1R** — với 24+ runs public trên W&B.
  - **Kết luận chưa được phép rút ra ở mốc hiện tại:** Chưa tái lập ImageNet / $K=25, 50$ / hàng $\gamma=0$; chưa chạy replay baselines cạnh tranh; *exemplar-free* chưa phải *Differential Privacy*.
  - **Định hướng Cuối kỳ (`statement/analysis/` & `statement/empirics/`):**
    - Hoàn thiện phân tích 4 trục lý thuyết, tối thiểu 3 Điểm mạnh (có bằng chứng), 2 Hạn chế lý thuyết & 2 Hướng phát triển tương ứng:
      1. **Hạn chế P1 (Phụ thuộc tập Base đóng băng):** Kiểm chứng độ nhạy của ACIL khi thành phần/số lượng lớp Phase 0 thay đổi hoặc lệch phân phối với các phase sau.
      2. **Hạn chế P2 (Chi phí bộ nhớ $\mathcal{O}(d^2)$ của $R_k$ vs Replay):** Đo đạc công bằng ngân sách bộ nhớ thực (`~522.5 MB` ở $d=8000$ `float64`) so với lưu exemplar hoặc giảm chiều $d$/nén $R_k$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~200 từ):**
  > Slide 32 khép lại phần trình bày chính của báo cáo giữa kỳ. Tính đến mốc nộp giữa kỳ, toàn bộ các hạng mục thuộc **Reading the Paper** và **Experimental Reproduction** trong Core Scope trên CIFAR-100/$K=5$ đã được thực thi hoàn chỉnh, kiểm thử tự động và công khai trên GitHub cùng Weights & Biases. Nhóm giữ thái độ học thuật thận trọng: chỉ kết luận đúng trong phạm vi dữ liệu đã đo và tránh phóng đại sang ImageNet hay chế độ nhiều pha $K=25, 50$ khi chưa có thí nghiệm tương ứng. Bước sang giai đoạn Cuối kỳ, nhóm đã xác định hai mũi nhọn phản biện lý thuyết đi kèm thiết kế thực nghiệm chứng minh (`Empirical Demonstration`). Hướng P1 kiểm tra giả định ẩn quan trọng nhất của ACIL: chất lượng toàn chuỗi phụ thuộc nặng nề vào tập 50 lớp nền ban đầu khi backbone bị đóng băng vĩnh viễn sau Phase 0. Hướng P2 đối chiếu công bằng đánh đổi bộ nhớ $\mathcal{O}(d^2)$ của ma trận $R_k$ (hơn $522\text{ MB}$ ở $d=8000$ `float64`) dưới cùng một ngân sách lưu trữ cụ thể. Phần tiếp theo (Slides 33–36) là Phụ lục kỹ thuật bổ trợ cho việc kiểm định chi tiết.

---

## PHẦN 4: APPENDIX KỸ THUẬT & TRUY NGUYÊN (SLIDES 33–36 — KHÔNG PHẢI REFERENCES SLIDE)

### Slide 33 — Appendix A1: Chứng minh Quy nạp Định lý 3.1 & Tương đương Công thức Residual
- **Tiêu đề trên slide:** Appendix A1: Biến đổi đại số đầy đủ của Định lý 3.1 và Công thức Residual theo Minibatch
- **Bố cục trực quan:** Trái: 5 bước chứng minh quy nạp ma trận khối $k-1 \to k$; Phải: Chứng minh 2 dòng tính đồng nhất giữa ghép khối Eq. (9) và dạng Residual toàn cục trong `src/analytic.py`.
- **Nội dung trên slide:**
  - **1. Bước quy nạp $k-1 \to k$ cho Định lý 3.1 [1]:**
    - Giả sử $R_{k-1} = (\tilde{H}_{k-1}^T \tilde{H}_{k-1} + \gamma I_d)^{-1}$ và $W_{k-1} = R_{k-1} \tilde{H}_{k-1}^T \tilde{Y}_{k-1}$.
    - Vì $\tilde{H}_k = [\tilde{H}_{k-1}^T, H_k^T]^T \implies R_k^{-1} = R_{k-1}^{-1} + H_k^T I_{N_k} H_k$. Áp dụng bổ đề Woodbury cho $(A + U C V)^{-1}$ ra trực tiếp Eq. (10) cho $R_k$.
    - Do $\tilde{Y}_k = \operatorname{diag}(\tilde{Y}_{k-1}, Y_k)$, tách cột $\tilde{W}_k = R_k \tilde{H}_k^T \tilde{Y}_k = \big[ R_k \tilde{H}_{k-1}^T \tilde{Y}_{k-1} \;\; R_k H_k^T Y_k \big]$.
    - Từ $R_{k-1}^{-1} = R_k^{-1} - H_k^T H_k \implies I_d = R_k R_{k-1}^{-1} + R_k H_k^T H_k \implies R_k = R_{k-1} - R_k H_k^T H_k R_{k-1}$.
    - Nhân vào $\tilde{H}_{k-1}^T \tilde{Y}_{k-1}$ được khối trái: $W_{k-1} - R_k H_k^T H_k W_{k-1}$ $\implies$ khớp chính xác Eq. (9) $\blacksquare$.
  - **2. Chứng minh dạng Global Residual ($W_{\text{new}} = W_{\text{pad}} + R_k H_b^T (Y_{b,\text{global}} - H_b W_{\text{pad}})$):**
    - Khi mở rộng cột cho $c_k$ lớp mới ($W_{\text{pad}} = [W_{\text{old}}, \mathbf{0}]$) và dữ liệu batch chỉ thuộc các lớp mới ($Y_{b,\text{global}} = [\mathbf{0}, Y_b]$):
      - **Khối cột cũ:** $W_{\text{old}} + R_k H_b^T(\mathbf{0} - H_b W_{\text{old}}) = W_{\text{old}} - R_k H_b^T H_b W_{\text{old}}$.
      - **Khối cột mới (batch đầu):** $\mathbf{0} + R_k H_b^T(Y_b - \mathbf{0}) = R_k H_b^T Y_b$.
    - Khi xử lý minibatch thứ 2 trong cùng phase $k$, các cột lớp mới giữ nguyên chỉ số nên tự động cộng dồn đúng chuẩn RLS.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Slide Phụ lục A1 cung cấp chuỗi biến đổi đại số hoàn chỉnh mà Slide 14 và Slide 13 chỉ tóm lược ý chính. Ở nửa trái là phép quy nạp cho Định lý 3.1: bước mấu chốt nhất là từ định nghĩa $R_{k-1}^{-1} = R_k^{-1} - H_k^T H_k$, khi nhân hai vế bên trái với $R_k$ và bên phải với $R_{k-1}$, ta thu được đồng nhất thức $R_k = R_{k-1} - R_k H_k^T H_k R_{k-1}$. Thay đồng nhất thức này vào khối trái của tích $R_k \tilde{H}_k^T \tilde{Y}_k$, cụm $R_{k-1} \tilde{H}_{k-1}^T \tilde{Y}_{k-1}$ tự động thu gọn về chính $W_{k-1}$ đã có từ pha trước — chứng tỏ không hề cần dùng lại mẫu đặc trưng lịch sử $\tilde{H}_{k-1}$ mà vẫn cập nhật chính xác toàn bộ trọng số cũ. Ở nửa phải là chứng minh vì sao đoạn code trong `src/analytic.py` dùng công thức phần dư toàn cục thay cho ghép khối: khi chia riêng hai nhóm cột lớp cũ và lớp mới, biểu thức phần dư tự động tách thành hai nhánh khớp từng ký tự với Phương trình (9), đồng thời xử lý mượt mà tình huống một lớp xuất hiện rải rác qua nhiều minibatch liên tiếp mà không phát sinh cột dư.

---

### Slide 34 — Appendix A2: Ma trận Truy nguyên Toàn bộ Thực nghiệm trên W&B & GitHub
- **Tiêu đề trên slide:** Appendix A2: Bảng ánh xạ truy nguyên toàn bộ các lượt chạy (31 Runs W&B Public)
- **Bố cục trực quan:** Bảng tổng hợp toàn cảnh ánh xạ `Thí nghiệm ↔ Notebook / Script ↔ W&B Run IDs ↔ Kết quả chính` (trích đồng bộ từ bảng trong `README.md`).
- **Nội dung trên slide:**
| Nhóm thí nghiệm | Notebook / Script | W&B Runs Public (`hsangg-2006/acil-reproduction`) | Tóm tắt kết quả |
|---|---|---|---|
| **VERIFY-01 / 02** | `notebooks/01_verify_analytic_update.ipynb` | `7aeaupq7`, `wi4wau2d` | Khớp Joint tổng hợp (`~1.84e-14` / `~3.43e-13`), 11/11 tests |
| **CIFAR Smoke 01 / 02** | `02_cifar100_smoke.ipynb`, `03_cifar100_safe_attempts.ipynb` | `o2r63pj2`, `blhw74ff` | Smoke 2 epochs/d256 đủ 50→100 lớp (`22.05%`), khớp Joint |
| **Profile d8000 / BP5** | `04_resume_benchmark_profile.ipynb` (V1, V3) | `gey0yop4`, `pv82dy1f` | Profile d8000 `34.07s`; BP5 `76.11s`, CUDA resume `53/54` |
| **R1 (3 seeds, $d=8000$)** | `06`, `07`, `08_benchmark_seed{42,43,44}_ready.ipynb` | `zt1014ei`, `v51m2dan`, `vqcs1ql4` | $\bar{A}=65.12\pm 0.61\%$, $A_{\mathrm{final}}=56.13\%$, $F_{\mathrm{base}}=8.31\text{ pp}$ |
| **R2 (9 runs mới, $d=8000/64$)** | `09`, `10`, `11_r2_ablation_seed{42,43,44}.ipynb` | `xdrm0d27, qcgdklxh, 3yp99ewh` (42) · `7wgme4so, xfleip26, aler6nj2` (43) · `81aci8h1, dub202jk, l9c50ezb` (44) | 4/5 hàng Table 2: No-FE `53.25%`; FE $\gamma=0.01/0.001$ `65.12%` |
| **R3 (9 runs, $d=1024$)** | `12_r3_batch_analysis.ipynb` / `scripts/r3_runner.py` | `luf7uv40, yqxph1bb, 59vlfxzv` (b32) · `o54hbdfp, 4481bnn1, 3gv0moj7` (b128) · `lnk0grbq, 5ehmjvh3, xesh2kjl` (b512) | 54/54 mốc khớp Joint ($\le 1.16\times 10^{-11}$); $b=512$ nhanh $3.47\times$ $b=32$ |
| **Author C1 / C2 / C1R** | `scripts/compare_author_acil.py`, `diagnose_author_base_precision.py` | `0tlcy6id` (`equiv=false`), `8muywodv`, `yeb0jqfb` | C1 prediction 6/6, score 4/6; C2 6/6; C1R 5/5 ($10^{-13}$) |
- **Footer IEEE:** `Nội bộ nhóm (Mọi URL chi tiết dạng clickable xem tại bảng Experiment Mapping trong README.md và reports/README.md).`
- **Speaker Notes (~165 từ):**
  > Slide Phụ lục A2 gom toàn bộ 31 lượt chạy đã công bố trên dự án Weights & Biases công khai `hsangg-2006/acil-reproduction` vào một ma trận ánh xạ duy nhất để thầy cô thuận tiện kiểm tra chéo từ slide sang W&B và GitHub. Nhìn vào cột thứ hai, mỗi con số thực nghiệm báo cáo ở các slide trước đều truy vết trực tiếp về một file notebook hoặc script trong repository, từ những lượt kiểm thử nền tảng ban đầu (`VERIFY-01/02`, `CIFAR-KAGGLE-01/02`, `gey0yop4`, `pv82dy1f`) cho tới bộ 21 runs chính thức của R1, R2, R3 và 3 runs đối chiếu mã nguồn tác giả (`C1, C2, C1R`). Ngoại trừ `0tlcy6id` giữ cờ kiểm tra ngưỡng `equivalence = false` do 2 phần tử score lệch vượt ngưỡng như đã phân tích trung thực ở Slide 29, toàn bộ các runs còn lại đều đạt trạng thái Finished, công khai hoàn toàn và có đối chiếu checksum JSON kèm manifest provenance tương ứng trên GitHub.

---

### Slide 35 — Appendix A3: Kiến trúc Mã nguồn, Unit Tests & Lý giải Thư viện Ngoài Stack
- **Tiêu đề trên slide:** Appendix A3: Cấu trúc 14 Module tự viết (`src/`), 65 Unit Tests & Giải thích Thư viện ngoài Standard Stack
- **Bố cục trực quan:** Chia 2 cột: Trái là sơ đồ nhóm chức năng 14 file `src/` + thống kê kiểm thử; Phải là bảng ghi nhận thành phần dùng lại & giải thích các thư viện ngoài `numpy, scipy, torch, scikit-learn`.
- **Nội dung trên slide:**
  - **Cấu trúc `src/` (100% có docstring 3 phần — *mục đích, tham số, trả về* — cho đủ 81 hàm/method):**
    - *Core Toán học & Kiểm chứng:* `analytic.py`, `verification.py`, `verify_analytic.py`.
    - *Dữ liệu & Backbone:* `cifar_data.py`, `resnet.py` (tự viết từ paper [6]).
    - *Training, Resume & Pipeline:* `base_training.py`, `verify_resume.py`, `cifar_smoke.py`, `streaming.py`, `profile_analytic.py`.
    - *An toàn, Metric & Đóng gói:* `metrics.py`, `run_attempt.py`, `reporting.py`, `notebook_bundle.py`.
    - **Kiểm thử tự động tại local (CPU):** `tests/`: **54 tests** (53 passed, 1 skipped CUDA test — chạy pass trên Kaggle GPU); `scripts/tests/`: **11/11 passed**.
  - **Thành phần dùng lại & Thư viện ngoài Standard Stack (trích `README.md`):**
    - *Thành phần học thuật tái sử dụng (không phải đóng góp nhóm):* ResNet-32 Option A [6], CIFAR-100 (`uoft-cs/cifar100`) [7], Protocol 50%+K [4], [5], Repo tác giả `0afd6bf` (chỉ để đối chiếu).
    - *Thư viện ngoài stack:* `wandb` (tracking công khai bắt buộc), `huggingface_hub` + `pyarrow` + `Pillow` (tải/giải mã Parquet CIFAR-100), `filelock` (chống race condition file trạng thái), `psutil` + `threadpoolctl` (đo Peak RSS & khóa 4 threads CPU), `matplotlib` (tự vẽ 7 hình `reports/figures/`), `nbformat/nbclient/ipykernel` (kiểm schema & chạy notebook), `tqdm/PyYAML/torchvision` (chỉ cài ở venv riêng chạy code tác giả).
- **Footer IEEE:** `[4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021. [6] K. He et al., CVPR, 2016. [7] A. Krizhevsky, Tech. Rep., 2009.`
- **Speaker Notes (~195 từ):**
  > Slide Phụ lục A3 trình bày kiến trúc phần mềm sạch của repository `20-ML-Paper` đáp ứng chuẩn đánh giá 5% chất lượng repo và khả năng bảo trì. Toàn bộ 14 module trong `src/` được nhóm tự tổ chức rành mạch từ mô tả bài báo, không hề sao chép code của tác giả; mã nguồn tác giả được giữ ở checkout ngoài trong một môi trường ảo biệt lập (`requirements-author-comparison.txt`). Tất cả 81 hàm và phương thức trong `src/` đều có docstring ba phần ghi rõ mục đích, tham số và giá trị trả về; đi kèm là bộ kiểm thử gồm 54 core tests và 11 script tests tự động. Về mặt phụ thuộc thư viện, ngoài bộ lõi tính toán khoa học chuẩn (`numpy`, `scipy`, `torch`, `scikit-learn`), mọi gói mở rộng đều có vai trò hạ tầng rõ ràng được phân tách trong các file `requirements-*.txt` riêng: như `huggingface_hub`/`pyarrow` để tải dữ liệu tự động không chép file tay, `psutil`/`threadpoolctl` cho phép đo RSS và cố định 4 luồng BLAS ở R3, hay `matplotlib` tách riêng chỉ dùng sinh đồ thị. Đồng thời nhóm ghi nhận minh bạch các thành phần dùng lại như kiến trúc ResNet-32 và giao thức chia lớp PODNet/AANets.

---

### Slide 36 — Appendix A4: Quy trình Tái tạo Kết quả & Checklist Nghiệm thu Trước khi Nộp
- **Tiêu đề trên slide:** Appendix A4: Hướng dẫn tái lập độc lập bằng CLI & Checklist Kiểm định File Nộp Giữa kỳ
- **Bố cục trực quan:** Trái: Các lệnh tái kiểm thử & tổng hợp lại bảng/hình từ repo (`README.md`); Phải: Bảng Checklist kiểm định cuối cùng đối chiếu trực tiếp với quy định Lab Book HCMUS.
- **Nội dung trên slide:**
  - **Các bước tái lập & kiểm chứng nhanh khi clone `https://github.com/24122050S/20-ML-Paper`:**
    1. Kiểm thử logic giải tích & toàn bộ suite: `pip install -r requirements-verification.txt` (hoặc `requirements-cifar.txt`) $\to$ `python -m unittest discover -s tests -v` và `scripts/tests`.
    2. Tái tính toàn bộ bảng tổng hợp `mean ± SD` từ JSON gốc: `python scripts/aggregate_r1.py`, `aggregate_r2.py`, `aggregate_r3.py`.
    3. Tái vẽ lại đủ 7 hình PNG (`requirements-plots.txt`): `python scripts/plot_r1.py`, `plot_r2.py`, `plot_r3.py`, `plot_author_comparison.py`, `plot_acil_diagrams.py`.
    4. Toàn bộ manifest `reports/*-provenance.json` lưu SHA-256 trên bytes chuẩn hóa `UTF-8/LF`.
  - **Checklist Nghiệm thu Sản phẩm Giữa kỳ (HCMUS Lab Book):**
    - `[x]` **Số slide & dung lượng:** **36 slides** (trong chuẩn 20–40 slides); dự kiến **$< 5\text{ MB}$** ($< 20\text{ MB}$).
    - `[x]` **Speaker Notes:** Đầy đủ trên **36/36 slides**, mỗi slide $\approx 160\text{–}220$ từ ($\le 500$ từ/slide).
    - `[x]` **Hình ảnh & Code:** 7 hình tự vẽ của nhóm; 0 hình cắt từ PDF; 0 code dump.
    - `[x]` **Trích dẫn IEEE:** Đánh số tại footer từng slide trích nguồn; **không tạo slide References riêng**.
    - `[x]` **Nộp riêng 2 mục trên Moodle & Quyền Public:** File `.pptx` + Link Public Repo (và kiểm lại các notebook Kaggle 01–12 ở chế độ Public).
- **Footer IEEE:** `Nội bộ nhóm (Xem chi tiết hướng dẫn chạy lại tại README.md và kiểm định tại docs/KICH_BAN_36_SLIDE_GIUA_KY.md).`
- **Speaker Notes (~175 từ):**
  > Slide Phụ lục cuối cùng cung cấp chỉ dẫn tái lập cho bất kỳ ai muốn kiểm chứng lại toàn bộ sản phẩm từ repository công khai của nhóm. Sau khi clone `20-ML-Paper`, người kiểm tra có thể chạy lại trọn vẹn 65 unit tests, chạy lại các script tổng hợp `aggregate_r1/r2/r3.py` để xác nhận các bảng trung bình và độ lệch chuẩn khớp từng chữ số với JSON công bố, cũng như sinh lại cả 7 biểu đồ và sơ đồ bằng các script trong `scripts/` mà không cần cấp lại GPU chạy lại 160 epochs. Mọi file chứng cứ lịch sử đều gắn mã kiểm tra SHA-256 trên chuẩn mã hóa `UTF-8` và xuống dòng `LF`. Ở cột bên phải là bảng kiểm nghiệm thu cuối cùng trước khi xuất file `.pptx` nộp lên Moodle: deck dài đúng 36 slides với Speaker Notes chi tiết từ Slide 1 tới Slide 36, tất cả đều nằm dưới giới hạn 500 từ/slide, hình ảnh tự vẽ hoàn toàn, trích dẫn IEEE nằm đúng vị trí chân trang của từng slide và không dùng slide References rời ở cuối. Cảm ơn thầy cô và anh chị trợ giảng đã theo dõi.
