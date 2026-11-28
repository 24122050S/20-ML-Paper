# Kịch bản Chi tiết 36 Slides — Báo cáo Giữa kỳ ACIL (NeurIPS 2022)

- **Mục đích tài liệu:** Bản thiết kế nội dung cho **01 file Slide Deck `.pptx` (36 slides: 32 slides chính + 4 slides Appendix)** gồm hai phần **Reading the Paper** và **Experimental Reproduction** của môn Nhập môn Máy học (HCMUS).
- **Quy chuẩn hình thức áp dụng:**
  1. **Tổng số slide:** `36` slides (nằm trong khoảng `20–40 slides` tính cả Appendix).
  2. **Dung lượng mục tiêu:** `< 20 MB` (sử dụng 7 hình PNG tự vẽ tại `reports/figures/` tổng dung lượng `< 1.7 MB`).
  3. **Speaker Notes:** Đầy đủ từ slide `1` đến `36` (`~150–220 từ/slide`, đảm bảo `<= 500 từ/slide`), giải thích rõ ký hiệu công thức và số liệu trên slide.
  4. **Trích dẫn IEEE:** Đánh số `[1]`–`[8]` và đặt **ở footer của từng slide có sử dụng nguồn**. **Không tạo slide `References` riêng.**
  5. **Hình ảnh & Code:** Sử dụng 7 hình tự vẽ của nhóm cùng bảng/sơ đồ PowerPoint; **không cắt hình từ bài báo gốc**, **không chép nguyên đoạn code dài lên slide**. Các con số thực nghiệm của nhóm đều gắn link **W&B Public**.

---

## Mục lục phân bổ 36 Slides

| Nhóm Slide | Chủ đề chính | Số slides | Trọng tâm & Hình sử dụng |
|---|---|---:|---|
| **Slides 1–2** | Trang bìa, Bố cục & Core Scope | 2 | Tiêu đề, nhóm thực hiện, GitHub/W&B public links, tóm tắt kết quả |
| **Slides 3–7** | Bài toán, Ví dụ thực tế, Tiền nhiệm, Đóng góp & Điểm mới | 5 | Bài toán CIL, Ví dụ da liễu liên viện, LwF / iCaRL / PODNet & AANets + `acil_problem_and_predecessors.png`, Đóng góp 3 nhóm & Điểm mới |
| **Slides 8–16** | Phương pháp ACIL (5 lớp nội dung) | 9 | Ký hiệu, Pipeline `acil_pipeline_end_to_end.png`, Phân tích từng thành phần (CNN+FE, ARaBT, Woodbury $R_k$, Residual $W_k$), Định lý 3.1, Absolute Memorization & Privacy + Giới hạn, Độ phức tạp |
| **Slides 17–20** | Thiết lập Thực nghiệm, Protocol, Metrics & Kiểm tra Bước đầu | 4 | Core Scope R1–R3, CIFAR-100 & ResNet-32 Protocol v1.0, Metrics v1.0 & dấu $F$, Kiểm tra `VERIFY-01/02`, Smoke, Profile/BP5 |
| **Slides 21–23** | Kết quả R1 — Benchmark chính (3 seeds, $d=8000$) | 3 | Training curve (`r1_cifar100_k5_training.png`), Accuracy curve (`r1_cifar100_k5_accuracy.png`), Đối chiếu Table 1 & bối cảnh |
| **Slides 24–25** | Kết quả R2 — Ablation FE & Độ nhạy $\gamma$ | 2 | Biểu đồ & bảng 4/5 hàng Table 2 (`r2_cifar100_k5_ablation.png`), Lưu ý khi đọc $F_{\mathrm{base}}$ & giới hạn |
| **Slides 26–27** | Kết quả R3 — Phân tích Minibatch (tự thiết kế) | 2 | Tương đương đại số 54/54 mốc, Thời gian & Peak RSS (`r3_cifar100_k5_batch_analysis.png`) |
| **Slides 28–30** | Đối chiếu Implementation Tác giả & Phân tích Sai lệch | 3 | C1/C2/C1R tầng mapping–normalization–pooling–FE, Biểu đồ (`author_acil_comparison.png`), Tổng hợp sai lệch $-1.18\text{ pp}$ (Đã kiểm chứng vs Giả thuyết) |
| **Slides 31–32** | Kinh nghiệm Triển khai, Tổng kết & Định hướng Cuối kỳ | 2 | Các quyết định kỹ thuật (`attempt_id`, streaming, checkpoint v2), Kết luận trong phạm vi đã đo & Kế hoạch Poster |
| **Slides 33–36** | Appendix kỹ thuật & Truy nguyên (Không có References slide) | 4 | A1: Chứng minh quy nạp Theorem 3.1 & Residual; A2: Bảng toàn bộ W&B runs; A3: Cấu trúc `src/`, Unit Tests & Thư viện; A4: Lệnh tái tạo & Checklist nộp bài |

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
- **Bố cục trực quan:** Nền trắng đơn giản, thanh màu xanh dương đậm bên trái, khung thông tin nhóm & 2 liên kết Public GitHub / Public W&B ở nửa dưới.
- **Nội dung trên slide:**
  - **Bài báo gốc:** *ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection* (Zhuang et al., NeurIPS 2022) [1]
  - **Môn học:** Nhập môn Máy học (Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM) — Báo cáo Giữa kỳ
  - **Thành viên thực hiện (Nhóm 3):** Nguyễn Hoàng Sang (`24122050` — Phương pháp, Analytic Head, R3), Trần Hoàng Minh Vĩ (`24122026` — Dataset, Protocol, ResNet-32, R1), Trần Đình Tấn Phát (`24122046` — Metrics, Ablation R2, W&B, Tổng hợp)
  - **Mã nguồn công khai:** `https://github.com/24122050S/20-ML-Paper`
  - **Theo dõi thực nghiệm công khai (W&B):** `https://wandb.ai/hsangg-2006/acil-reproduction`
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 35, pp. 11602–11614, 2022.`
- **Speaker Notes (~155 từ):**
  > Kính chào Thầy/Cô và các bạn. Hôm nay nhóm em xin trình bày báo cáo giữa kỳ môn Nhập môn Máy học với bài báo ACIL, công bố tại NeurIPS 2022. Bài báo cáo gồm 32 slides nội dung chính và 4 slides phụ lục kỹ thuật, tập trung vào hai nội dung: thứ nhất là đọc hiểu và trình bày lại cơ sở toán học của phương pháp học tăng dần giải tích ACIL; thứ hai là tự cài đặt lại thuật toán từ mô tả trong bài báo, chạy thực nghiệm trên tập dữ liệu CIFAR-100, kiểm tra thêm ảnh hưởng của kích thước minibatch và đối chiếu trực tiếp với mã nguồn của tác giả. Toàn bộ code, file kết quả JSON, bộ unit test và các hình vẽ đều được nhóm công khai trên repository GitHub `24122050S/20-ML-Paper`, đi kèm 31 lượt chạy công khai trên Weights & Biases tại project `hsangg-2006/acil-reproduction`.

---

### Slide 2 — Tổng quan Bố cục & Phạm vi Đồ án (Core Scope)
- **Tiêu đề trên slide:** Bố cục trình bày & Các thực nghiệm đã hoàn tất trong Core Scope
- **Bố cục trực quan:** Sơ đồ 4 phần ngang (Phần I $\to$ IV) ở nửa trên; 4 khung tóm tắt số liệu ở nửa dưới (R1, R2, R3, Đối chiếu tác giả).
- **Nội dung trên slide:**
  - **Cấu trúc 4 phần (32 slides chính + 4 Appendix):**
    1. **Bối cảnh & Tiền nhiệm (Slides 3–7):** Bài toán CIL, ví dụ y tế, LwF / iCaRL / PODNet & AANets, đóng góp & điểm mới.
    2. **Phương pháp ACIL (Slides 8–16):** Ký hiệu, Pipeline tổng thể, phân tích từng khối, Định lý 3.1 & Giới hạn, Độ phức tạp.
    3. **Thực nghiệm Tái lập (Slides 17–30):** Protocol/Metrics, kiểm tra trước khi chạy dài, **R1** (3 seeds), **R2** (ablation), **R3** (batch), **Author Comparison**.
    4. **Tổng kết & Định hướng Cuối kỳ (Slides 31–32):** Kinh nghiệm triển khai, giới hạn phạm vi, kế hoạch phân tích phản biện cuối kỳ.
  - **Tóm tắt các kết quả đã chạy xong (Core Scope — CIFAR-100, $K=5$):**
    - **R1 (3 seeds, $d=8000$):** $\bar{A} = 65.12 \pm 0.61\%$ (so với paper $66.30\%$, lệch $-1.18\text{ pp}$).
    - **R2 (4/5 hàng Table 2 × 3 seeds):** FE giúp tăng $+11.86\text{ pp}$; $\gamma \in [0.001, 0.1]$ thay đổi chỉ $0.0032\text{ pp}$.
    - **R3 (tự thiết kế, $d=1024$):** 54/54 mốc khớp Joint Oracle (sai số tương đối $W \le 1.16\times 10^{-11}$, 0 argmax khác); batch 512 nhanh gấp $3.47\times$ batch 32.
    - **Author Compare (`0afd6bf`):** C1/C2/C1R xác định khác biệt ở chuẩn hóa ảnh, pooling FP32 và cách khởi tạo base.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022.`
- **Speaker Notes (~175 từ):**
  > Slide này tóm tắt bố cục bài báo cáo và phạm vi thực nghiệm nhóm đã thực hiện. Do giới hạn thời gian GPU trên Kaggle và nhân lực hiện tại, nhóm chọn phạm vi **Core Scope**, tập trung kiểm tra kỹ trên bộ dữ liệu CIFAR-100 với thiết lập $50$ lớp base cộng $5$ phase tăng dần, mỗi phase thêm $10$ lớp ($K=5$). Nhóm chưa chạy lại các phương pháp baseline khác trong Bảng 1 và chưa mở rộng sang ImageNet hay các cấu hình $K=25, 50$; vì vậy những kết quả ở số phase lớn của bài báo chỉ được trích dẫn để đối chiếu và nêu rõ nguồn. Bốn nhóm thực nghiệm ở nửa dưới slide — gồm R1 ba seed độc lập tại $d=8000$, R2 ablation bốn cấu hình, R3 khảo sát kích thước minibatch $32/128/512$, và phần chạy đối chiếu mã nguồn tác giả tại commit `0afd6bf` — đều đã có đầy đủ file JSON, mã kiểm tra SHA-256 và link W&B công khai trong repo.

---

### Slide 3 — Bài toán Class-Incremental Learning (CIL) & 3 Thách thức Chính
- **Tiêu đề trên slide:** Học tăng dần theo lớp (CIL): Phân loại tất cả lớp đã gặp mà không lưu lại dữ liệu cũ
- **Bố cục trực quan:** Thanh timeline tuần tự `Phase 0` $\to$ `Phase 1` $\to \dots \to$ `Phase K` $\to$ `Single-Head Eval` ở trên; 3 cột thách thức chính ở dưới.
- **Nội dung trên slide:**
  - **Thiết lập bài toán CIL [1], [2]:**
    - Chuỗi phase $k = 0, 1, \dots, K$: mỗi phase $k$ chỉ cung cấp tập dữ liệu $\mathcal{D}_k^{\text{train}} = \{(X_k^{(i)}, y_k^{(i)})\}_{i=1}^{N_k}$ của $c_k$ lớp mới; các tập lớp không giao nhau ($\mathcal{C}_i \cap \mathcal{C}_j = \emptyset$).
    - Sau phase $k$, mô hình phải dự đoán trên **toàn bộ $C_k = \sum_{j=0}^{k} c_j$ lớp đã thấy** (*single-head, task-agnostic* — **không có Task ID** lúc kiểm tra).
  - **Ba thách thức chính của bài toán:**
    1. **Không giữ dữ liệu cũ (Exemplar-Free):** $\mathcal{D}_0, \dots, \mathcal{D}_{k-1}$ không còn truy cập được do giới hạn lưu trữ hoặc yêu cầu bảo mật dữ liệu gốc [1].
    2. **Đánh đổi Ổn định – Dẻo (*Stability–Plasticity Dilemma*):** Huấn luyện tiếp bằng SGD trên dữ liệu mới dễ ghi đè trọng số và làm lệch ranh giới của lớp cũ (*Catastrophic Forgetting*) [2], [3].
    3. **Không gian phân loại mở rộng dần:** Số lớp cần chọn trong $\operatorname{argmax}$ tăng từ $50 \to 100$ làm tăng khả năng nhầm lẫn chéo ngay cả khi đặc trưng giữ nguyên.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., "iCaRL," CVPR, 2017. [3] Z. Li and D. Hoiem, "Learning without forgetting," IEEE TPAMI, 2018.`
- **Speaker Notes (~185 từ):**
  > Slide 3 phát biểu bài toán Class-Incremental Learning (CIL) mà bài báo giải quyết. Ký hiệu $k$ là chỉ số giai đoạn huấn luyện: Phase $0$ là tập nền ban đầu gồm $c_0$ lớp, còn các Phase $k \ge 1$ lần lượt bổ sung thêm $c_k$ lớp mới không trùng với các phase trước. Điểm khó của CIL nằm ở cách đánh giá single-head không cung cấp Task ID khi kiểm tra: một ảnh test bất kỳ được lấy $\operatorname{argmax}$ trực tiếp trên toàn bộ $C_k$ lớp đã từng xuất hiện. Nếu lấy mô hình từ phase $k-1$ rồi tiếp tục huấn luyện SGD chỉ trên dữ liệu lớp mới $\mathcal{D}_k^{\text{train}}$, gradient của hàm Cross-Entropy sẽ kéo đầu ra nghiêng hẳn về $c_k$ lớp mới và gây ra hiện tượng quên thảm khốc (*Catastrophic Forgetting*). Bài toán càng chặt hơn khi đặt thêm điều kiện *exemplar-free*, tức không được lưu lại ảnh gốc của các phase trước. Bên cạnh đó, cần phân biệt việc mô hình quên kiến thức cũ với việc độ chính xác giảm tự nhiên khi số lớp cạnh tranh trong $\operatorname{argmax}$ mở rộng từ 50 lên 100 lớp.

---

### Slide 4 — Ví dụ Thực tế Tự đặt: Sàng lọc Tổn thương Da liễu Liên viện
- **Tiêu đề trên slide:** Ví dụ minh họa: Hệ thống hỗ trợ sàng lọc da liễu cập nhật theo quý không lưu ảnh bệnh nhân cũ
- **Bố cục trực quan:** Sơ đồ luồng kịch bản bệnh viện (Trái: Quý 0 — 50 bệnh phổ biến $\to$ Quý $k$ — +10 biến thể mới $\to$ Chẩn đoán chung); Phải: 3 lý do các cách cập nhật thông thường khó áp dụng.
- **Nội dung trên slide:**
  - **Bối cảnh minh họa (Nhóm tự xây dựng):** Hệ thống hỗ trợ sàng lọc tổn thương da liễu triển khai qua nhiều đợt tại bệnh viện tuyến tỉnh:
    - **Phase 0 (Khởi tạo):** Mô hình học **50 bệnh da liễu phổ biến** (viêm da cơ địa, vảy nến, nấm da...).
    - **Phases $k = 1 \dots K$ (Mỗi quý):** Bổ sung **10 nhóm tổn thương mới / ít gặp** (biến chứng da hậu nhiễm virus mới, phản ứng thuốc, ký sinh trùng địa phương).
    - **Khi khám lâm sàng:** Mô hình cần phân loại chung cả $50 + 10k$ nhóm bệnh (*không biết trước bệnh nhân thuộc đợt dữ liệu quý nào*).
  - **Vì sao các hướng thông thường gặp vướng mắc?**
    - **Không được giữ kho ảnh bệnh nhân cũ (Quyền riêng tư):** Ảnh tổn thương da là dữ liệu y tế nhạy cảm; quy định bảo mật hồ sơ yêu cầu xóa ảnh gốc sau đợt huấn luyện, không thể đóng gói exemplar buffer gửi liên viện.
    - **Chỉ fine-tune trên 10 bệnh mới:** Mô hình dễ dự đoán nhầm các bệnh phổ biến cũ sang 10 bệnh mới.
    - **Huấn luyện lại từ đầu (Joint Retrain):** Vừa vướng quy định lưu trữ ảnh lịch sử, vừa tốn nhiều giờ GPU mỗi lần cập nhật ở tuyến dưới.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~175 từ):**
  > Để làm rõ động lực thực tế của bài toán, nhóm xây dựng một ví dụ giả định trong y tế là hệ thống hỗ trợ sàng lọc tổn thương da liễu tại bệnh viện. Ở giai đoạn đầu (Phase 0), mô hình được huấn luyện trên 50 nhóm bệnh da phổ biến. Qua mỗi quý tiếp theo, cơ sở y tế thu thập thêm dữ liệu của 10 nhóm tổn thương mới phát sinh. Khi bác sĩ sử dụng thực tế, ảnh chụp của bệnh nhân mới có thể thuộc bất kỳ nhóm nào trong tổng số $50 + 10k$ nhóm bệnh đã biết. Tình huống này gặp hai ràng buộc thực tế lớn: thứ nhất, quy định bảo vệ dữ liệu cá nhân trong y tế không cho phép giữ lại hay chuyển giao kho ảnh gốc của bệnh nhân các quý trước để chạy huấn luyện gộp hoặc phát lại mẫu (*exemplar replay*); thứ hai, máy trạm tại bệnh viện tuyến tỉnh hạn chế về GPU nên khó chạy lại hàng trăm epoch backpropagation mỗi đợt. Đây là bối cảnh phù hợp cho một phương pháp cập nhật nhanh và không cần lưu ảnh gốc.

---

### Slide 5 — Tiền nhiệm 1 & 2: Regularization (LwF) và Exemplar Replay (iCaRL)
- **Tiêu đề trên slide:** Hai hướng tiếp cận tiền nhiệm tiêu biểu: LwF (không lưu mẫu) và iCaRL (lưu mẫu đại diện)
- **Bố cục trực quan:** Chia 2 cột dọc cho **LwF [3]** và **iCaRL [2]**, mỗi cột trình bày rõ 3 ý: *Ý tưởng chính* $\to$ *Hạn chế* $\to$ *Hệ quả thực tế*.
- **Nội dung trên slide:**
  - **Tiền nhiệm 1 — Learning without Forgetting (LwF, Li & Hoiem 2018) [3]** *(Regularization / Knowledge Distillation, không lưu mẫu)*
    - **Ý tưởng chính:** Đóng băng bản sao mạng phase $k-1$ làm *Teacher*. Khi học ảnh lớp mới $X_k$, tối ưu tổng *Cross-Entropy trên lớp mới + Knowledge Distillation* giữ cho logit lớp cũ của Student gần với logit của Teacher tính trên chính ảnh $X_k$.
    - **Hạn chế:** Mạng Teacher chỉ được truy vấn bằng **ảnh của lớp mới $X_k$** (ngoài phân phối của lớp cũ); sai số chưng cất cộng dồn qua nhiều phase và thiếu dữ liệu để căn chỉnh trực tiếp ranh giới giữa lớp cũ và lớp mới [1], [2].
    - **Hệ quả thực tế:** Độ chính xác giảm nhanh khi chuỗi phase dài; dự đoán bị lệch mạnh về các lớp mới học sau cùng.
  - **Tiền nhiệm 2 — iCaRL (Rebuffi et al. 2017) [2]** *(Exemplar Replay + Nearest-Mean Classifier)*
    - **Ý tưởng chính:** Giữ bộ nhớ đệm cố định $\mathcal{M}$ ảnh gốc cũ (ví dụ $M = 2000$ ảnh CIFAR-100 chọn bằng *herding*); gộp $\mathcal{M} \cup \mathcal{D}_k^{\text{train}}$ chạy SGD + distillation, phân loại bằng tâm lớp gần nhất trên đặc trưng chuẩn hóa $L_2$.
    - **Hạn chế:** Phải **lưu giữ ảnh gốc của các phase trước**; khi tổng số lớp $C_k$ tăng mà $|\mathcal{M}|$ cố định, số mẫu/lớp cũ ($m = |\mathcal{M}|/C_k$) giảm dần gây **mất cân bằng lớp** với dữ liệu lớp mới [1], [4].
    - **Hệ quả thực tế:** Không dùng được khi hệ thống cấm lưu dữ liệu thô; mỗi phase vẫn phải chạy lại nhiều epoch SGD.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., "iCaRL," CVPR, 2017. [3] Z. Li and D. Hoiem, "Learning without forgetting," IEEE TPAMI, 2018. [4] A. Douillard et al., "PODNet," ECCV, 2020.`
- **Speaker Notes (~195 từ):**
  > Trước ACIL, các nghiên cứu về CIL thường đi theo hai hướng chính. Hướng thứ nhất là dùng điều chuẩn và chưng cất tri thức để không phải lưu mẫu cũ, tiêu biểu là **LwF** [3]. LwF giữ bản sao mạng ở phase $k-1$ làm mạng giáo viên để ràng buộc các đầu ra lớp cũ của mạng mới. Điểm hạn chế là vì không còn ảnh cũ, mạng giáo viên chỉ sinh nhãn mềm trên chính ảnh của lớp mới $X_k$. Qua nhiều phase liên tiếp, sai số xấp xỉ này tích tụ dần và ranh giới giữa lớp cũ với lớp mới không được cập nhật trực tiếp, làm độ chính xác trên các lớp đầu giảm nhanh. Hướng thứ hai là lưu lại một tập mẫu đại diện như **iCaRL** [2], chẳng hạn giữ cố định 2000 ảnh gốc để huấn luyện kèm với dữ liệu mới. Cách này giữ độ chính xác tốt hơn LwF nhưng không áp dụng được cho bài toán cấm lưu ảnh gốc; ngoài ra khi số lớp tăng lên 80–100 lớp với bộ nhớ cố định, mỗi lớp cũ chỉ còn 20 ảnh so với 500 ảnh/lớp mới, gây lệch dữ liệu và vẫn tốn chi phí chạy SGD ở từng phase.

---

### Slide 6 — Tiền nhiệm 3 (PODNet & AANets) & Đối chiếu Tổng quan với ACIL
- **Tiêu đề trên slide:** Các phương pháp Distillation + Replay gần hơn (PODNet, AANets) vẫn giảm hiệu năng khi số phase $K$ lớn
- **Bố cục trực quan:** Nhúng sơ đồ tự vẽ `reports/figures/acil_problem_and_predecessors.png` (~65% diện tích bên phải/dưới); cột tóm tắt **PODNet [4] & AANets [5]** ở bên trái/trên.
- **Nội dung trên slide:**
  - **Tiền nhiệm 3 — PODNet (Douillard et al. 2020) [4] & AANets (Liu et al. 2021) [5]:**
    - **Ý tưởng chính:** Cải tiến hướng replay bằng chưng cất đặc trưng không gian ở các tầng trung gian (*Pooled Outputs Distillation* — PODNet) hoặc kết hợp hai nhánh residual *plastic–stable* có trọng số thích nghi (AANets), đi cùng kho mẫu đại diện (thường 20 ảnh/lớp) [4], [5].
    - **Hạn chế:** Vẫn cần lưu ảnh cũ và chạy tối ưu SGD nhiều epoch ở mỗi phase; không gian đặc trưng vẫn bị dịch chuyển dần (*representation drift*) khi chia nhỏ thành nhiều đợt ($K = 25, 50$) [1].
    - **Hệ quả thực tế:** Chi phí GPU tăng theo $K \times E$ epochs; theo Table 1 bài báo [1] trên CIFAR-100, khi tăng từ $K=5 \to K=25$, $\bar{A}$ của PODNet giảm $64.83\% \to 60.72\%$ ($-4.11\text{ pp}$), AANets giảm $66.31\% \to 62.31\%$ ($-4.00\text{ pp}$).
  - **Hình minh họa tự vẽ:** Chèn [`reports/figures/acil_problem_and_predecessors.png`](../reports/figures/acil_problem_and_predecessors.png) (Panel A: Bối cảnh CIL & 3 thách thức; Panel B: Đối chiếu *Regularization vs. Exemplar Replay vs. ACIL*).
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., CVPR, 2017. [3] Z. Li and D. Hoiem, IEEE TPAMI, 2018. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021.`
- **Speaker Notes (~190 từ):**
  > Slide 6 xét hai phương pháp mạnh hơn trong nhóm lưu mẫu là **PODNet** và **AANets**, đồng thời cũng là hai công trình định hình giao thức đánh giá 50% lớp base mà bài báo ACIL sử dụng. PODNet giữ cấu trúc biểu diễn bằng cách chưng cất đặc trưng không gian qua các tầng trung gian của ResNet, trong khi AANets ghép song song nhánh ổn định và nhánh dẻo để cân bằng giữa lớp cũ và lớp mới. Ở thiết lập ít phase ($K=5$), cả hai đạt độ chính xác cao trên CIFAR-100 ($64.83\%$ và $66.31\%$ theo Table 1 [1]). Tuy nhiên, cả hai vẫn phải lưu mẫu ảnh cũ (thường 20 ảnh mỗi lớp) và huấn luyện lại mạng bằng SGD qua từng phase. Khi số phase tăng lên $K=25$, việc cập nhật lặp lại nhiều đợt làm biểu diễn lệch dần, khiến độ chính xác trung bình của cả hai giảm khoảng $4$ điểm phần trăm. Hình sơ đồ tự vẽ bên phải tóm tắt sự khác nhau giữa ba hướng: Regularization, Exemplar Replay và hướng cập nhật giải tích trạng thái $(R_k, W_k)$ của ACIL.

---

### Slide 7 — Phân loại Đóng góp & Điểm mới của ACIL
- **Tiêu đề trên slide:** Đóng góp và vị trí của ACIL: Chuyển bước cập nhật tăng dần sang hồi quy tuyến tính nghiệm đóng
- **Bố cục trực quan:** Bảng 3 hàng phân loại đóng góp (*Lý thuyết / Kỹ thuật / Thực nghiệm*) bên trái; Khung đối chiếu *Các hướng dùng SGD vs. ACIL* bên phải.
- **Nội dung trên slide:**
  - **1. Phân loại đóng góp của bài báo [1]:**
    - **Lý thuyết (Theoretical):** Phát biểu và chứng minh **Định lý 3.1**: cập nhật đệ quy không lưu mẫu cũ ($W_k$) cho nghiệm **trùng khớp về mặt đại số** với hồi quy Ridge học gộp ($\tilde{W}_k$) trên đặc trưng cố định.
    - **Kỹ thuật (Technical):** Kết hợp chuỗi **Freeze CNN + Random Feature Expansion ($\operatorname{ReLU}(F W_{\mathrm{fe}})$) + ARaBT ($R_0, W_0$) + Cập nhật Woodbury RLS ($R_k, W_k$)** chỉ duyệt dữ liệu mới một lượt (*single-pass*).
    - **Thực nghiệm (Empirical):** Trên CIFAR-100 & ImageNet (trích bài báo [1]), phương pháp không lưu mẫu này giữ kết quả ổn định theo $K$ và đạt $\bar{A}$ cao hơn PODNet/AANets ở $K=25, 50$ ($65.95\%$ so với $60.72\% / 62.31\%$ trên CIFAR-100 $K=25$).
  - **2. Điểm mới cốt lõi so với tiền nhiệm:**
    - *Các hướng dùng SGD:* Cập nhật gradient cục bộ thiếu dữ liệu cũ nên nghiệm lệch dần khỏi học gộp và nhạy với số phase $K$.
    - *ACIL (Closed-form Analytic Update):* Tách việc học biểu diễn (chỉ ở Phase 0) khỏi việc giải bộ phân loại lồi có **thống kê đủ bậc hai cộng tính** $\sum H_j^T H_j$ $\to$ lưu thông tin lịch sử gọn trong ma trận $R_k \in \mathbb{R}^{d\times d}$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021.`
- **Speaker Notes (~180 từ):**
  > Slide 7 tách riêng các đóng góp của ACIL thành ba khía cạnh để tiện đánh giá đúng phạm vi của từng phần. Về **lý thuyết**, Định lý 3.1 chứng minh rằng trọng số phân loại tính đệ quy qua từng phase bằng đúng nghiệm của bài toán hồi quy Ridge khi giải gộp trên toàn bộ đặc trưng đã gặp, với điều kiện hàm trích đặc trưng cố định. Về **kỹ thuật**, dù hồi quy bình phương tối thiểu đệ quy (RLS) và công thức Woodbury đã có sẵn trong đại số tuyến tính, đóng góp của tác giả là cách ghép nối chúng vào bài toán CIL thông qua việc đóng băng CNN sau Phase 0, thêm tầng chiếu ngẫu nhiên phi tuyến FE để tăng khả năng tách tuyến tính và dùng bước căn chỉnh ARaBT thay cho lớp FC của SGD. Về **thực nghiệm**, bài báo cho thấy cách làm exemplar-free này cạnh tranh tốt với các phương pháp lưu mẫu trên CIFAR-100 và ImageNet khi số phase $K$ lớn.

---

## PHẦN 2: PHƯƠNG PHÁP ACIL — PHÂN TÍCH 5 LỚP NỘI DUNG (SLIDES 8–16)

### Slide 8 — Lớp 1: Hệ thống Ký hiệu & Hàm Mục tiêu Học Gộp
- **Tiêu đề trên slide:** Lớp 1 (Ký hiệu & Bài toán): Mô hình hóa tầng phân loại CIL thành bài toán hồi quy Ridge
- **Bố cục trực quan:** Bảng ký hiệu (kèm kích thước & ý nghĩa) bên trái (60%); khung công thức hàm mục tiêu Joint Analytic Ridge & nghiệm đóng bên phải (40%).
- **Nội dung trên slide:**
  - **Hệ thống ký hiệu (đồng bộ giữa bài báo [1] & [`docs/Dac_ta_ACIL.md`](Dac_ta_ACIL.md)):**
    - $k \in \{0, \dots, K\}$: chỉ số phase ($k=0$ là Base phase); $c_k$: số lớp mới ở phase $k$; $C_k = \sum_{j=0}^k c_j$: tổng số lớp đã gặp.
    - $F_k = f_{\text{cnn}}(X_k; \theta_0) \in \mathbb{R}^{N_k \times d_{\text{cnn}}}$: đặc trưng sau Global Avg Pool của CNN đóng băng ($d_{\text{cnn}} = 64$ với ResNet-32).
    - $W_{\mathrm{fe}} \in \mathbb{R}^{d_{\text{cnn}} \times d}$: ma trận chiếu ngẫu nhiên cố định ($\mathcal{N}(0,1)$, $d=8000$).
    - $H_k = \operatorname{ReLU}(F_k W_{\mathrm{fe}}) \in \mathbb{R}^{N_k \times d}$: đặc trưng mở rộng ở phase $k$; $\tilde{H}_k = [H_0^T, \dots, H_k^T]^T$: ghép dọc từ phase $0$ đến $k$.
    - $Y_k \in \mathbb{R}^{N_k \times c_k}$: nhãn one-hot cục bộ phase $k$; $\tilde{Y}_k = \operatorname{diag}(Y_0, \dots, Y_k)$: ma trận nhãn khối chéo tích lũy.
    - $R_k \in \mathbb{R}^{d \times d}$, $W_k \in \mathbb{R}^{d \times C_k}$, $\gamma > 0$: ma trận nghịch đảo tương quan (RFAM), trọng số phân loại và hệ số điều chuẩn Ridge.
  - **Hàm mục tiêu học gộp tại phase $k$ [1]:**
    $$
    \min_{W \in \mathbb{R}^{d \times C_k}} \left\| \tilde{Y}_k - \tilde{H}_k W \right\|_F^2 + \gamma \| W \|_F^2 \implies \tilde{W}_k = \left( \tilde{H}_k^T \tilde{H}_k + \gamma I_d \right)^{-1} \tilde{H}_k^T \tilde{Y}_k
    $$
    *Bài toán đặt ra:* Tính $\tilde{W}_k$ chỉ từ $(H_k, Y_k)$ và $(R_{k-1}, W_{k-1})$, **không dùng lại $\tilde{H}_{k-1}, \tilde{Y}_{k-1}$**.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Slide 8 thống nhất các ký hiệu toán học dùng chung giữa bài báo [1] và mã nguồn tái lập của nhóm. Ở mỗi phase $k$, $N_k$ là số ảnh huấn luyện mới — với CIFAR-100 $K=5$ thì Phase $0$ có $25{,}000$ ảnh của 50 lớp và mỗi phase tiếp theo có $5{,}000$ ảnh của 10 lớp. Mỗi ảnh đi qua mạng ResNet-32 đã đóng băng cho vector đặc trưng $64$ chiều $F_k$, sau đó nhân với ma trận ngẫu nhiên cố định $W_{\mathrm{fe}}$ và qua hàm $\operatorname{ReLU}$ để tạo ma trận đặc trưng mở rộng $H_k$ kích thước $N_k \times d$ ($d = 8000$). Nếu giả sử ta giữ lại được toàn bộ dữ liệu từ đầu đến phase $k$ — viết gộp thành $\tilde{H}_k$ và ma trận nhãn đường chéo khối $\tilde{Y}_k$ — ACIL thay hàm Cross-Entropy bằng hàm bình phương tối thiểu có điều chuẩn $L_2$ với chuẩn Frobenius $\|\cdot\|_F^2$ và hệ số $\gamma > 0$. Vì hàm mục tiêu này lồi chặt theo $W$, đạo hàm bậc nhất bằng $0$ cho nghiệm đóng duy nhất $\tilde{W}_k$ ở khung bên phải. Vấn đề cần giải quyết là tính đúng $\tilde{W}_k$ mà chỉ được giữ trạng thái $(R_{k-1}, W_{k-1})$.

---

### Slide 9 — Lớp 2: Kiến trúc & Luồng Pipeline End-to-End của ACIL
- **Tiêu đề trên slide:** Lớp 2 (Pipeline tổng thể): Backpropagation chỉ chạy ở Phase 0; các phase sau cập nhật bằng đại số tuyến tính
- **Bố cục trực quan:** Nhúng hình tự vẽ [`reports/figures/acil_pipeline_end_to_end.png`](../reports/figures/acil_pipeline_end_to_end.png) ở trung tâm (~85% chiều rộng), phía dưới tóm tắt 3 giai đoạn nối tiếp.
- **Nội dung trên slide:**
  - **Sơ đồ tự vẽ:** Chèn [`reports/figures/acil_pipeline_end_to_end.png`](../reports/figures/acil_pipeline_end_to_end.png) (13 khối xử lý chia thành 3 giai đoạn khớp với `src/resnet.py`, `src/base_training.py`, `src/analytic.py`, `src/streaming.py`).
  - **Ba giai đoạn xử lý:**
    - **Stage 1 (Phase 0 — Base BP & ARaBT):** Huấn luyện ResNet-32 + `FC50` (160 epochs SGD) $\to$ đóng băng (`eval()`) và bỏ `FC50` $\to$ trích $F_0 \to H_0$ $\to$ khởi tạo nghiệm đóng $(R_0, W_0)$.
    - **Stage 2 (Phases $k = 1 \dots K$ — Analytic Incremental):** Trích $F_k \to H_k$ bằng cùng backbone/FE đóng băng (0 gradient, 0 mẫu cũ) $\to$ mở rộng cột $W_{\mathrm{pad}}$ $\to$ cập nhật $(R_k, W_k)$ theo Phương trình (9)–(10).
    - **Stage 3 (All-Seen Inference):** Ảnh kiểm tra đi qua cùng backbone/FE $\to$ tính điểm $S = H_{\mathrm{test}} W_k \to \hat{y} = \operatorname{argmax}_c S_{:,c}$ trên toàn bộ $C_k$ lớp.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [6] K. He et al., "Deep residual learning for image recognition," CVPR, 2016.`
- **Speaker Notes (~190 từ):**
  > Slide 9 trình bày sơ đồ pipeline tổng thể do nhóm tự vẽ, thể hiện luồng xử lý từ ảnh đầu vào đến nhãn dự đoán khớp với cấu trúc code trong thư mục `src/`. Ở **Stage 1**, chỉ riêng tại Phase 0, tập ảnh 50 lớp đầu có áp dụng augmentation (padding, random crop, lật ngang) mới được đưa vào huấn luyện ResNet-32 bằng backpropagation trong 160 epochs. Sau khi kết thúc bước này, lớp `FC50` tạm thời được tháo bỏ, trọng số và thống kê BatchNorm của ResNet-32 được khóa cố định ở chế độ `eval()`, rồi nhóm trích đặc trưng sạch (không crop/flip) qua tầng mở rộng $W_{\mathrm{fe}}$ để tính cặp ma trận khởi tạo $(R_0, W_0)$. Sang **Stage 2**, ở mỗi phase tăng dần $k=1\dots K$, ảnh của 10 lớp mới chỉ chạy lan truyền tiến một lượt qua backbone đóng băng để cập nhật trực tiếp $(R_k, W_k)$ mà không lưu lại ảnh cũ. Ở **Stage 3**, khi kiểm tra, mọi ảnh test đều được nhân với $W_k$ và lấy $\operatorname{argmax}$ trên tất cả $C_k$ lớp đã thấy.

---

### Slide 10 — Lớp 3a: Đóng băng Backbone CNN & Mở rộng Đặc trưng Ngẫu nhiên (FE)
- **Tiêu đề trên slide:** Lớp 3a: Vì sao cần đóng băng CNN và mở rộng đặc trưng từ 64 lên 8000 chiều?
- **Bố cục trực quan:** Chia 2 khung ngang: Khung trên trình bày **Backbone CNN đóng băng $\theta_0$**; Khung dưới trình bày **Random Feature Expansion $H_k = \operatorname{ReLU}(F_k W_{\mathrm{fe}})$** kèm số liệu kiểm chứng từ R2.
- **Nội dung trên slide:**
  - **Thành phần 1 — Backbone CNN đóng băng ($\operatorname{CNN}_{\theta_0}$):**
    - **Cấu hình:** ResNet-32 CIFAR ($d_{\text{cnn}} = 64$ sau Global Avg Pool) huấn luyện bằng SGD + Cross-Entropy trên 50 lớp base, sau đó bỏ `FC50`, đặt `requires_grad = False` và `eval()`.
    - **Lý do cần thiết:** (i) Tránh việc gradient trên lớp mới làm lệch biểu diễn của lớp cũ (*representation drift*); (ii) Giữ cho đặc trưng ở mọi phase nằm trên **cùng một hàm ánh xạ cố định** để công thức đệ quy khớp với học gộp [1].
    - **Cách cài đặt:** Chuẩn hóa ảnh bằng `mean/std` tính từ base train; chạy CNN ở `float32` rồi chuyển sang `float64` ở tầng analytic; kiểm tra fingerprint tham số & BatchNorm không đổi qua các phase.
  - **Thành phần 2 — Random Feature Expansion (FE):**
    - **Cấu hình:** Phép chiếu ngẫu nhiên cố định đi kèm $\operatorname{ReLU}$: $H_k = \operatorname{ReLU}(F_k W_{\mathrm{fe}})$, $W_{\mathrm{fe}} \in \mathbb{R}^{64 \times 8000}$ ($\mathcal{N}(0,1)$, không bias) [1].
    - **Động lực toán học:** Đặc trưng $64$ chiều của ResNet-32 nhỏ hơn tổng số $100$ lớp (*under-parameterized*); theo **Định lý Cover**, phép chiếu phi tuyến lên chiều cao $d=8000 \gg 64$ giúp các lớp dễ tách tuyến tính hơn. *(Thực nghiệm R2 của nhóm: bật FE giúp $\bar{A}$ tăng **$+11.86\text{ pp}$**, từ $53.25\%$ lên $65.12\%$)*.
    - **Cách cài đặt:** Sinh $W_{\mathrm{fe}}$ một lần bằng RNG riêng (`seed + 1000`), không cập nhật gradient và giữ nguyên từ Phase 0 đến $K$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [6] K. He et al., "Deep residual learning for image recognition," CVPR, 2016.`
- **Speaker Notes (~200 từ):**
  > Slide 10 đi vào hai thành phần đầu tiên của pipeline. Thành phần thứ nhất là việc đóng băng hoàn toàn mạng ResNet-32 sau Phase 0. Về mặt toán học, các công thức cập nhật đệ quy ở những slide sau dựa trên điều kiện $H_0$ của phase đầu và $H_k$ của các phase sau được trích ra từ cùng một hàm đặc trưng cố định; nếu tiếp tục fine-tune trọng số tích chập hoặc để `running_mean` của BatchNorm trôi theo dữ liệu mới, ma trận $R_{k-1}$ lưu từ trước sẽ không còn khớp với hệ tọa độ đặc trưng mới. Thành phần thứ hai là tầng mở rộng đặc trưng ngẫu nhiên FE. Trên CIFAR-100, ResNet-32 chỉ xuất ra vector $64$ chiều — nhỏ hơn cả số lượng $100$ lớp cần phân loại ở phase cuối, nên đặt thẳng một bộ phân loại tuyến tính lên vector $64$ chiều sẽ rất hạn chế về khả năng phân tách. Việc nhân với ma trận ngẫu nhiên cố định $W_{\mathrm{fe}}$ cỡ $64 \times 8000$ rồi qua hàm phi tuyến $\operatorname{ReLU}$ đưa biểu diễn lên không gian $8000$ chiều, giúp bộ phân loại Ridge tách tuyến tính hiệu quả hơn; trong thí nghiệm R2 của nhóm, bỏ tầng này làm $\bar{A}$ giảm từ $65.12\%$ xuống $53.25\%$.

---

### Slide 11 — Lớp 3b: Khởi tạo Giải tích Base (ARaBT) & Cộng dồn theo Batch
- **Tiêu đề trên slide:** Lớp 3b: Căn chỉnh giải tích ARaBT thay cho lớp FC của Backpropagation ở Phase 0
- **Bố cục trực quan:** Trái: Công thức & lý do cần bước ARaBT (Phương trình 3–4 [1]); Phải: Công thức cộng dồn Gram/Cross theo batch & mức tiết kiệm bộ nhớ RAM khi $N_0 = 25{,}000, d=8000$.
- **Nội dung trên slide:**
  - **Công thức khởi tạo (Phương trình (4) trong bài báo [1]):**
    - Tính cặp trạng thái xuất phát trên dữ liệu Phase 0 ($H_0 \in \mathbb{R}^{N_0 \times d}, Y_0 \in \mathbb{R}^{N_0 \times c_0}$):
      $$
      R_0 = \left( H_0^T H_0 + \gamma I_d \right)^{-1}, \qquad W_0 = R_0 H_0^T Y_0
      $$
  - **Vì sao không giữ lại trọng số `FC50` của SGD?**
    - Lớp `FC50` được học với hàm mất mát Cross-Entropy trên đặc trưng 64 chiều nên không phải nghiệm cực tiểu của bài toán Ridge trên $H_0$, đồng thời chưa có ma trận $R_0$. Để các bước đệ quy ở phase $k \ge 1$ nối tiếp đúng về nghiệm học gộp $\tilde{W}_k$, trạng thái ở $k=0$ cần được **tính lại bằng nghiệm đóng Ridge (ARaBT)** [1].
  - **Giải pháp cộng dồn tiết kiệm RAM trong [`initialize_batches`](../src/analytic.py):**
    - Nếu ghép nguyên ma trận $H_0$ ($25{,}000 \times 8000$ `float64`), riêng $H_0$ đã tốn **$1.60\text{ GB}$ RAM** trước khi tạo bản sao giải hệ.
    - Trong code của nhóm, hai tổng ma trận được cộng dồn theo từng batch $b$:
      $$
      A_0 = \gamma I_d + \sum_{b} H_{0,b}^T H_{0,b}, \qquad B_0 = \sum_{b} H_{0,b}^T Y_{0,b}
      $$
      sau đó giải hệ đối xứng xác định dương bằng phân rã **Cholesky** ($A_0 = L L^T$) để thu $R_0 = A_0^{-1}$ và $W_0 = A_0^{-1} B_0$.
    - **Lưu ý cài đặt:** $\gamma I_d$ chỉ cộng **một lần khi lập $A_0$**; không cộng lặp lại ở các batch hoặc các phase sau.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Slide 11 giải thích bước ARaBT (*Analytic Re-alignment on Base Training*), tương ứng với Phương trình (4) của bài báo. Ở Phase 0, lớp `FC50` đi kèm hàm Cross-Entropy chỉ dùng trong lúc huấn luyện backpropagation để giúp ResNet-32 học bộ trích đặc trưng, sau đó được bỏ đi. Nhóm cho 25,000 ảnh base (không bật random crop/flip) chạy qua backbone đã khóa và tầng FE để tính nghiệm Ridge khởi đầu $R_0$ cỡ $8000 \times 8000$ và $W_0$ cỡ $8000 \times 50$. Khi cài đặt thực tế trên Kaggle, có hai chi tiết kỹ thuật cần xử lý cẩn thận: thứ nhất, thay vì nối toàn bộ 25,000 hàng của $H_0$ thành một mảng lớn chiếm 1.6 GB RAM (và tốn thêm bộ nhớ đệm khi nhân ma trận), hàm `initialize_batches` cộng dồn trực tiếp tích $H_{0,b}^T H_{0,b}$ theo từng minibatch rồi giải bằng phân rã Cholesky mà vẫn cho cùng kết quả toán học; thứ hai, hạng tử điều chuẩn $\gamma I_d$ chỉ được cộng một lần duy nhất vào $A_0$, nên việc chia batch nhỏ hay lớn không làm thay đổi độ mạnh của $\gamma$.

---

### Slide 12 — Lớp 3c: Cập nhật Đệ quy $R_k$ qua Bổ đề Woodbury
- **Tiêu đề trên slide:** Lớp 3c: Bổ đề Woodbury giảm bậc nghịch đảo mỗi batch từ $\mathcal{O}(d^3)$ xuống $\mathcal{O}(b^3 + b d^2)$
- **Bố cục trực quan:** Khung công thức Phương trình (10) ở trên; So sánh kích thước ma trận cần giải hệ giữa nghịch đảo trực tiếp ($8000 \times 8000$) và Woodbury theo batch ($128 \times 128$) ở giữa; 4 bước tính số học ở dưới.
- **Nội dung trên slide:**
  - **Công thức đệ quy (Phương trình (10) trong bài báo [1]):**
    - Khi nhận một batch đặc trưng mới $H_b \in \mathbb{R}^{b \times d}$ tại phase $k \ge 1$, ma trận nghịch đảo tương quan được cập nhật:
      $$
      R_{\text{new}} = R_{\text{old}} - R_{\text{old}} H_b^T \left( I_b + H_b R_{\text{old}} H_b^T \right)^{-1} H_b R_{\text{old}}
      $$
  - **Lợi ích về chi phí tính toán:**
    - Do $R_{\text{new}}^{-1} = R_{\text{old}}^{-1} + H_b^T H_b$, nếu cộng $H_b^T H_b$ rồi nghịch đảo trực tiếp ma trận $d \times d$ ($8000 \times 8000$) ở mỗi batch thì chi phí là $\mathcal{O}(d^3)$, tốn nhiều thời gian tính toán.
    - Nhờ **bổ đề nghịch đảo Sherman–Morrison–Woodbury**, phép nghịch đảo cỡ $d \times d$ được chuyển về giải hệ trên ma trận lõi $S_b = I_b + H_b R_{\text{old}} H_b^T$ chỉ cỡ **$b \times b$** (với minibatch $b = 128 \ll d = 8000$).
  - **Quy trình tính 4 bước theo từng minibatch $b$ trong `src/analytic.py`:**
    1. Tính $P = R_{\text{old}} H_b^T \in \mathbb{R}^{d \times b}$ ($\mathcal{O}(bd^2)$).
    2. Lập $S_b = I_b + H_b P \in \mathbb{R}^{b \times b}$ — đối xứng xác định dương vì $R_{\text{old}} \succ 0$ và cộng $I_b$.
    3. Giải hệ tuyến tính $S_b K^T = P^T$ bằng Cholesky thay vì gọi `inv()` tường minh.
    4. Cập nhật $R_{\text{new}} = R_{\text{old}} - K^T P^T$ và đối xứng hóa $\frac{1}{2}(R_{\text{new}} + R_{\text{new}}^T)$ để hạn chế sai số làm tròn.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Slide 12 trình bày cách cập nhật ma trận $R_k = (\tilde{H}_k^T \tilde{H}_k + \gamma I_d)^{-1}$ ở các phase tăng dần mà không cần lưu đặc trưng cũ hay nghịch đảo lại ma trận $8000 \times 8000$ sau mỗi batch. Khi có thêm $b$ mẫu mới biểu diễn bởi $H_b$, ma trận tự tương quan nghịch đảo $R^{-1}$ được cộng thêm tích hạng thấp $H_b^T H_b$. Áp dụng bổ đề Sherman–Morrison–Woodbury, thay vì nghịch đảo trong không gian $d = 8000$ chiều, ta chỉ cần giải hệ tuyến tính với ma trận trung gian $S_b = I_b + H_b R_{\text{old}} H_b^T$ có kích thước bằng đúng cỡ minibatch $b \times b$, tức $128 \times 128$ trong cấu hình chuẩn của nhóm. Khi tự cài đặt trong `src/analytic.py`, nhóm giữ tính toán ở kiểu `float64`, dùng hàm giải hệ tuyến tính Cholesky thay cho nghịch đảo ma trận tường minh, đồng thời lấy trung bình đối xứng $\frac{1}{2}(R_{\text{new}} + R_{\text{new}}^T)$ sau mỗi minibatch để tránh sai số dấu phẩy động làm lệch tính đối xứng khi lặp qua hàng chục batch.

---

### Slide 13 — Lớp 3d: Cập nhật $W_k$ — Dạng Ghép khối và Dạng Phần dư Toàn cục
- **Tiêu đề trên slide:** Lớp 3d: Cập nhật đồng thời trọng số lớp cũ và trọng số lớp mới theo từng minibatch
- **Bố cục trực quan:** Trên: Công thức ghép khối Phương trình (9) [1] bên trái $\iff$ Công thức Global Residual của nhóm bên phải; Dưới: Giải thích ý nghĩa số hạng $-R_k H_k^T H_k W_{k-1}$ và cách xử lý minibatch.
- **Nội dung trên slide:**
  - **Hai cách biểu diễn tương đương đại số:**
    - **Dạng ghép khối cho cả phase (Eq. 9 [1]):**
      $$
      W_k = \left[\, \underbrace{W_{k-1} - R_k H_k^T H_k W_{k-1}}_{\text{Điều chỉnh các lớp cũ } (d \times C_{k-1})} \;\;\;\;\;\; \underbrace{R_k H_k^T Y_k}_{\text{Lớp mới } (d \times c_k)} \,\right]
      $$
    - **Dạng phần dư toàn cục theo minibatch ([`incremental_update`](../src/analytic.py)):**
      $$
      W_{\text{pad}} = \left[\, W_{\text{old}} \;\; \mathbf{0}_{d \times (C_k - C_{\text{old}})} \,\right], \qquad W_{\text{new}} = W_{\text{pad}} + R_{\text{new}} H_b^T \left( Y_{b,\text{global}} - H_b W_{\text{pad}} \right)
      $$
  - **Ý nghĩa của số hạng hiệu chỉnh $-R_k H_k^T H_k W_{k-1}$:**
    - Các mẫu ảnh mới $H_k$ **không thuộc các lớp cũ**, nên nhãn đích của chúng trên các cột lớp cũ bằng $\mathbf{0}$. Số hạng $-R_k H_k^T (H_k W_{k-1})$ điều chỉnh các cột trọng số lớp cũ giảm phản hồi trên vùng đặc trưng lớp mới, **giảm việc lớp cũ dự đoán nhầm trên ảnh lớp mới mà không cần xem lại ảnh cũ**.
  - **Lý do dùng dạng Residual khi chia Minibatch:**
    - Với nhãn $Y_{b,\text{global}}$ đánh ID toàn cục ($0\dots C_k-1$), trên các cột lớp cũ ($Y_{\text{old}} = 0$) công thức cho đúng $W_{\text{old}} - R_k H_b^T H_b W_{\text{old}}$; khi cùng một lớp mới xuất hiện lại ở các minibatch sau trong cùng phase, trọng số được **cập nhật tiếp vào đúng cột đó** thay vì nối thêm cột mới.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~200 từ):**
  > Slide 13 trình bày bước cập nhật ma trận trọng số phân loại $W_k$. Trong Phương trình (9) của bài báo, $W_k$ gồm hai phần ghép ngang: khối bên trái cỡ $d \times C_{k-1}$ ứng với các lớp đã học từ trước, và khối bên phải $R_k H_k^T Y_k$ ứng với $c_k$ lớp mới. Điểm đáng lưu ý là khối trọng số lớp cũ không giữ nguyên $W_{k-1}$ mà được trừ đi lượng $R_k H_k^T H_k W_{k-1}$. Lý do là trên tập mẫu mới $H_k$, nhãn one-hot tại các cột lớp cũ đều bằng 0; nếu $H_k W_{k-1}$ khác 0 thì số hạng trừ này sẽ điều chỉnh trọng số cũ dựa trên ma trận $R_k$ để giảm kích hoạt nhầm trên lớp mới. Một chi tiết thực tế khi cài đặt là Phương trình (9) trong bài báo viết cho toàn bộ một phase; nếu áp dụng nguyên dạng ghép cột cho từng minibatch trong 40 batch của một phase thì số cột của $W$ sẽ tăng sai theo số batch. Vì vậy, trong `src/analytic.py`, nhóm viết lại dưới dạng phần dư toàn cục $(Y_{b,\text{global}} - H_b W_{\mathrm{pad}})$, vừa tương đương đại số với Phương trình (9), vừa giữ đúng kích thước $d \times C_k$ ở mọi cỡ minibatch.

---

### Slide 14 — Lớp 4a: Định lý 3.1 — Tương đương Học Gộp & 4 Điều kiện Giới hạn
- **Tiêu đề trên slide:** Lớp 4a (Lý thuyết): Định lý 3.1 chứng minh tương đương đại số với Joint Ridge — kèm 4 điều kiện áp dụng
- **Bố cục trực quan:** Trái: Phát biểu Theorem 3.1 [1] + ý chính chứng minh quy nạp + kết quả kiểm tra R3 (54/54 mốc khớp); Phải: 4 điều kiện biên và giới hạn của định lý.
- **Nội dung trên slide:**
  - **Định lý 3.1 (Equivalence of Incremental & Joint Analytic Learning) [1]:**
    - Nếu khởi tạo bằng Eq. (4) và cập nhật đệ quy mọi phase $k \ge 1$ bằng Eq. (9)–(10) trên cùng đặc trưng cố định, thì:
      $$
      \forall k \ge 0:\quad W_k = \tilde{W}_k = \left(\tilde{H}_k^T \tilde{H}_k + \gamma I_d\right)^{-1} \tilde{H}_k^T \tilde{Y}_k
      $$
  - **Ý nghĩa trực quan & Kiểm chứng thực nghiệm:**
    - Nghiệm Ridge chỉ phụ thuộc vào hai **thống kê đủ bậc hai cộng tính**: $\sum H_j^T H_j$ và $\sum H_j^T Y_j$; cách chia batch hay phase không làm đổi tổng.
    - *Kiểm chứng R3 (`d=1024`, [luf7uv40..3gv0moj7]):* Khớp Joint Oracle tại **54/54 mốc** (max rel $W$ error $\le 1.159\times 10^{-11}$, `0` argmax khác).
  - **4 Điều kiện biên & Giới hạn cần lưu ý:**
    1. **Hàm đặc trưng $\phi(x)$ không đổi:** Backbone CNN, thống kê BatchNorm, chuẩn hóa ảnh và $W_{\mathrm{fe}}$ phải giữ nguyên qua mọi phase.
    2. **$\gamma > 0$ cố định & cộng 1 lần:** Đảm bảo $H_0^T H_0 + \gamma I_d$ khả nghịch; cộng lặp $\gamma$ sẽ làm đổi hàm mục tiêu.
    3. **Tương đương *Joint Ridge trên feature cố định* $\neq$ *Joint CNN End-to-End*:** Chất lượng biểu diễn vẫn phụ thuộc vào 50 lớp học ở Phase 0.
    4. **Sai số dấu phẩy động:** Thực tế cần tính ở `float64`; chạy nhiều bước trên `float32` dễ tích lũy sai số làm tròn trên $R_k$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~205 từ):**
  > Slide 14 trình bày Định lý 3.1 — kết quả lý thuyết trung tâm của bài báo ACIL [1]. Định lý phát biểu rằng ma trận trọng số $W_k$ tính đệ quy qua từng phase mà không lưu dữ liệu cũ trùng với nghiệm $\tilde{W}_k$ khi gom toàn bộ dữ liệu từ Phase 0 đến Phase $k$ lại để giải Ridge một lần. Ý tưởng trực quan ở đây là nghiệm bình phương tối thiểu không cần từng mẫu riêng lẻ mà chỉ phụ thuộc vào hai tổng ma trận $\sum H_j^T H_j$ và $\sum H_j^T Y_j$, vốn có tính kết hợp và giao hoán. Nhóm cũng kiểm tra trực tiếp tính chất này trong thực nghiệm R3 qua 54 mốc đối chiếu và ghi nhận sai lệch tương đối của $W$ chỉ ở mức $10^{-11}$ trên `float64`. Bên cạnh đó, cột bên phải nêu rõ bốn điều kiện biên để tránh hiểu rộng quá phạm vi định lý: thứ nhất, backbone CNN và BatchNorm phải đóng băng hoàn toàn; thứ hai, $\gamma > 0$ chỉ cộng một lần ở Phase 0; thứ ba, sự tương đương này chỉ xét ở tầng phân loại tuyến tính trên đặc trưng cố định chứ không tương đương với việc huấn luyện lại toàn bộ mạng ResNet-32 trên 100 lớp; và thứ tư, tính toán máy tính cần độ chính xác `float64`.

---

### Slide 15 — Lớp 4b: Phạm vi của "Absolute Memorization" & "Privacy Protection"
- **Tiêu đề trên slide:** Lớp 4b (Lý thuyết): Phân biệt rõ *Khớp nghiệm đại số* với *Độ quên $F_{\mathrm{base}}$* và phạm vi *Privacy*
- **Bố cục trực quan:** Chia 2 cột song song đối chiếu *Thuật ngữ trong tiêu đề bài báo [1]* với *Nội dung toán học thực tế & Giới hạn*.
- **Nội dung trên slide:**
  - **1. "Absolute Memorization" (Ghi nhớ tuyệt đối) [1]:**
    - **Nội dung đúng:** Giữ chính xác **nghiệm đại số học gộp** ($\|W_k - \tilde{W}_k\|_F = 0$), nhờ đó nghiệm cuối $A_K$ **không phụ thuộc vào cách chia số phase $K$** hay thứ tự minibatch (khi tập Phase 0 cố định).
    - **Điểm dễ hiểu nhầm:** **Không đồng nghĩa $F_{\mathrm{base}} = 0$ hay độ chính xác lớp cũ giữ nguyên không giảm.** Khi số lớp tăng từ $50 \to 100$ trên cùng đặc trưng cố định, bài toán $\operatorname{argmax}$ có nhiều lớp cạnh tranh hơn nên một số mẫu lớp cũ vẫn bị rơi sang lớp mới.
    - *Số liệu R1 của nhóm:* Độ chính xác tập 50 lớp gốc giảm từ $B_0 = 76.64\%$ xuống $B_5 = 68.33\%$ ($F_{\mathrm{base}} = 8.31 \pm 0.62\text{ pp}$, gần với mức $F = 9.00$ trong Table 1 [1]).
  - **2. "Privacy Protection" (Bảo vệ quyền riêng tư) [1]:**
    - **Nội dung đúng:** Không lưu ảnh gốc $X_k$ hay ma trận từng mẫu $H_k$; không thể giải ngược duy nhất $H_k$ từ $R_k^{-1} - R_{k-1}^{-1} = H_k^T H_k$ vì mọi ma trận trực giao $Q$ đều cho $(QH_k)^T(QH_k) = H_k^T H_k$ (cộng thêm phần thông tin mất qua GAP và $\operatorname{ReLU}$).
    - **Giới hạn lý thuyết:** Đây là tính chất **không lưu mẫu thô & bất định tuyến tính từ tích Gram**, **không phải** chuẩn **Differential Privacy ($(\varepsilon, \delta)$-DP)**; ma trận $R_k, W_k$ vẫn chứa thống kê bậc hai chính xác của dữ liệu đã học.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Tiêu đề bài báo [1] có hai cụm từ dễ gây hiểu rộng nếu không đọc kỹ định nghĩa là "Absolute Memorization" và "Privacy Protection", nên Slide 15 làm rõ phạm vi thực tế của cả hai. Thứ nhất, *Absolute Memorization* trong bài báo có nghĩa là thuật toán đệ quy giữ đúng nghiệm đại số $\tilde{W}_k$ như khi giải gộp trên toàn bộ đặc trưng từ đầu đến hiện tại, không làm mất thông tin thống kê từ Phase 0. Tuy nhiên, điều đó không dẫn tới chỉ số quên $F_{\mathrm{base}} = 0$: khi số lớp cần phân biệt trong $\operatorname{argmax}$ tăng từ 50 lên 100 trên cùng một backbone cố định, bài toán phân loại khó hơn nên độ chính xác trên 50 lớp nền vẫn giảm $8.31$ điểm phần trăm trong R1 của nhóm (tương tự mức $9.00\%$ ở Table 1 bài báo). Thứ hai, về *Privacy Protection*, tác giả lập luận rằng từ hiệu $H_k^T H_k$ không thể khôi phục duy nhất ma trận mẫu $H_k$ do bất định phép nhân trực giao $Q H_k$. Dù vậy, xét theo chuẩn bảo mật hiện nay, đây là cơ chế không lưu ảnh thô chứ không phải Differential Privacy có ngưỡng $(\varepsilon, \delta)$.

---

### Slide 16 — Lớp 5: Độ phức tạp Thời gian, Bộ nhớ & Đánh đổi với Tiền nhiệm
- **Tiêu đề trên slide:** Lớp 5 (Độ phức tạp): Đổi vòng lặp SGD nhiều epoch lấy cập nhật một lượt $\mathcal{O}(N_k d^2)$ và bộ nhớ $\mathcal{O}(d^2)$
- **Bố cục trực quan:** Bảng đối chiếu độ phức tạp (5 cột: Tiêu chí, Joint CNN, LwF/EWC, iCaRL/PODNet/AANets, ACIL) ở nửa trên; Hai khung số liệu đo thực tế về thời gian và bộ nhớ (`522.5 MB` ở $d=8000$) ở nửa dưới.
- **Nội dung trên slide:**
  - **Bảng đối chiếu Độ phức tạp tại một phase $k \ge 1$ ($N_k$ mẫu, batch $b$, $d=8000$, $E$ epochs SGD, $M$ mẫu buffer):**
| Tiêu chí | Joint CNN | LwF / EWC [3] | iCaRL / PODNet / AANets [2], [4], [5] | **ACIL [1]** |
|---|---|---|---|---|
| **Buffer ảnh cũ** | $\mathcal{O}(N_{\text{total}})$ | **0** | $M$ ảnh ($\mathcal{O}(M D_{\text{img}})$) | **0** |
| **State giữ giữa phase** | $\mathcal{O}(\|\theta\| + N_{\text{total}} D_{\text{img}})$ | $\mathcal{O}(2\|\theta\|)$ | $\mathcal{O}(2\|\theta\| + M D_{\text{img}})$ | $\mathcal{O}(\|\theta_0\| + d^2 + d C_k)$ |
| **Lượt duyệt CNN** | $E$ epochs (Fwd+Bwd) | $E$ epochs (2 Fwd+Bwd) | $E$ epochs trên $(N_k+M)$ | **1 lượt ($E=1$)**, 0 Bwd |
| **Thời gian phase $k$** | $\mathcal{O}(E \sum N_j (C_{\text{fwd}}+C_{\text{bwd}}))$ | $\mathcal{O}(E N_k (2C_{\text{fwd}}+C_{\text{bwd}}))$ | $\mathcal{O}(E (N_k+M) (2C_{\text{fwd}}+C_{\text{bwd}}))$ | $\mathcal{O}(N_k C_{\text{fwd}} + N_k d^2)$ |
| **Khớp nghiệm học gộp** | Chuẩn CNN | Không | Không (giảm ở $K$ lớn) | **Có (Joint Analytic)** |
  - **Đánh đổi định lượng từ số đo R1/R3 của nhóm:**
    - **Thời gian cập nhật:** Trong R1 ($d=8000$, [`zt1014ei`](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei)), BP160 mất **$40.4\text{–}42.3\text{ phút}$** trên T4, còn 1 phase incremental (5,000 ảnh) mất **$41.8\text{–}44.7\text{ s}$** trên CPU 4 luồng; toàn bộ streaming 6 mốc mất **$11.1\text{–}11.4\text{ phút}$**. Ở $d=1024$ (R3, $b=512$), toàn bộ analytic 6 phase mất **$5.00\text{ s}$**.
    - **Đánh đổi $\mathcal{O}(d^2)$ của $R_k$:** Với `float64` ở $d=8000$, $R_k = 512.00\text{ MB}$, $W_5 = 6.40\text{ MB}$, $W_{\mathrm{fe}} = 4.10\text{ MB}$ $\implies$ tổng **$522.50\text{ MB}$** (lớn hơn $2000$ ảnh CIFAR-100 thô chỉ $\approx 6.14\text{ MB}$, nhưng giữ cố định khi số lượng mẫu tăng lớn như trên ImageNet).
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., CVPR, 2017. [3] Z. Li and D. Hoiem, IEEE TPAMI, 2018. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021.`
- **Speaker Notes (~210 từ):**
  > Slide 16 tổng kết phần phương pháp bằng bảng so sánh độ phức tạp thời gian và bộ nhớ giữa ACIL với các hướng tiền nhiệm. Với các phương pháp như iCaRL, PODNet hay AANets, mỗi phase mới đều cần huấn luyện lại mạng qua $E$ epochs (thường từ 70 đến 160 epochs) gồm cả lan truyền tiến và ngược. Trong khi đó, ACIL chỉ cho $N_k$ ảnh mới chạy lan truyền tiến đúng một lượt qua mạng đóng băng ($E=1$), sau đó cập nhật ma trận với chi phí xấp xỉ $\mathcal{O}(N_k d^2)$ nhờ chia minibatch $b \ll d$. Thực tế trên cấu hình R1 ($d=8000$), bước cập nhật giải tích cho một phase 5,000 ảnh chỉ mất khoảng $42\text{–}45\text{ giây}$ trên CPU 4 luồng, và ở cấu hình R3 ($d=1024, b=512$) cả 6 phase hoàn thành trong $5.00\text{ giây}$. Đổi lại, điểm hạn chế cần lưu ý là bộ nhớ trạng thái tỷ lệ với $\mathcal{O}(d^2)$: ở $d=8000$ kiểu `float64`, riêng ma trận $R_k$ chiếm $512\text{ MB}$ (tổng trạng thái analytic là $522.50\text{ MB}$), lớn hơn nhiều so với 2000 ảnh CIFAR-100 kích thước $32 \times 32$ ($\approx 6.14\text{ MB}$), và chỉ tiết kiệm hơn kho ảnh khi làm việc với ảnh độ phân giải lớn hoặc tập dữ liệu rất nhiều mẫu.

---

## PHẦN 3 (NỬA ĐẦU): THIẾT LẬP THỰC NGHIỆM, PROTOCOL, METRICS & KIỂM TRA BƯỚC ĐẦU (SLIDES 17–20)

### Slide 17 — Phạm vi Tái lập (Core Scope) & Bố trí Thực nghiệm R1–R3
- **Tiêu đề trên slide:** Thiết kế thực nghiệm trong Core Scope: Bốn bộ kiểm tra từ Benchmark chính đến đối chiếu code tác giả
- **Bố cục trực quan:** Bảng 4 nhóm thí nghiệm (**R1**, **R2**, **R3**, **Author C1/C2/C1R**) gồm: *Câu hỏi kiểm tra*, *Cấu hình chốt*, *Khối lượng chạy*, và ghi chú *Ranh giới phạm vi* ở bên dưới.
- **Nội dung trên slide:**
  - **Lý do chọn Core Scope (`docs/protocol_chot.md`):** Tập trung nguồn lực Kaggle T4/CPU để chạy đủ số seed độc lập có kiểm chứng thay vì chạy dàn trải nhiều bộ dữ liệu.
  - **Bốn bộ thực nghiệm đã thực hiện:**
    1. **R1 — So sánh chính (`CIFAR-100 / K=5`):** $d=8000, \gamma=0.1$, 3 seeds (`42, 43, 44`), mỗi seed huấn luyện mới 160 epochs $\to$ Đường cong 6 mốc, $\bar{A}, F_{\mathrm{base}}$ đối chiếu Table 1 [1].
    2. **R2 — Ablation thành phần (`Table 2`):** Dùng lại đúng checkpoint 160 mỗi seed; 4/5 hàng (*No-FE $\gamma=0.1$*, *FE8000 $\gamma \in \{0.1, 0.01, 0.001\}$*) $\times$ 3 seeds (9 runs mới + 3 R1).
    3. **R3 — Phân tích Batch Size (Nhóm tự thiết kế):** Checkpoint seed 42, $d=1024, \gamma=0.1$; quét $b \in \{32, 128, 512\} \times 3$ timing repeats (9 runs, 54 so sánh Joint Oracle).
    4. **Author Comparison (`0afd6bf`):** Chạy nguyên bản code tác giả (`C1`, `C2`, `C1R`) trong môi trường riêng trên checkpoint seed 42, $d=1024$.
  - **Phạm vi chưa thực hiện:** Chưa tự chạy các baseline khác trong Table 1; chưa chạy $K=10/25/50$, ImageNet hoặc hàng $\gamma=0$ của Table 2 (do solver hiện yêu cầu $\gamma > 0$).
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021.`
- **Speaker Notes (~185 từ):**
  > Bước sang phần thực nghiệm, Slide 17 trình bày tổng thể bốn bộ thí nghiệm mà nhóm đã chốt cấu hình trước trong file `docs/protocol_chot.md`, không chỉnh sửa siêu tham số theo điểm test. Trong phạm vi Core Scope, nhóm thực hiện phép so sánh chính R1 với ba seed độc lập ở chiều mở rộng chuẩn $d=8000$; thực hiện phần ablation R2 cho 4 trên 5 hàng của Table 2 (bỏ hàng $\gamma = 0$ vì công thức Woodbury hiện tại yêu cầu $\gamma > 0$ để ma trận xác định dương, và nhóm không thay bằng một số $\varepsilon$ nhỏ rồi gọi là $\gamma=0$); thiết kế thêm thí nghiệm R3 để kiểm tra sự tương đương nghiệm qua các cỡ batch $32, 128, 512$ kèm đo thời gian và RAM; đồng thời chạy đối chiếu trực tiếp mã nguồn của tác giả. Các nhận xét ở phần sau đều giới hạn trong phạm vi CIFAR-100 với $K=5$ và phân biệt rõ giữa kết quả nhóm tự chạy với con số trích từ bài báo.

---

### Slide 18 — Dữ liệu, Tiền xử lý & Giao thức Huấn luyện (`Protocol v1.0`)
- **Tiêu đề trên slide:** Dữ liệu & Protocol (`ACIL-C100-v1.0`): Chia lớp theo seed, chuẩn hóa từ tập Base và cấu hình ResNet-32
- **Bố cục trực quan:** Chia 2 cột: Trái là bảng *Dữ liệu & Chia lớp theo Phase*; Phải là bảng *Cấu hình Backbone ResNet-32 & Training Phase 0* kèm ghi chú khác biệt thứ tự lớp so với PODNet/AANets.
- **Nội dung trên slide:**
  - **Dữ liệu & Ánh xạ nhãn toàn cục:**
    - Nguồn dữ liệu: Hugging Face `uoft-cs/cifar100` [7] (pin revision `aadb3af...`, kiểm SHA-256 từng file Parquet): 50,000 train / 10,000 test (`32x32` RGB).
    - **Chia phase ($K=5$):** Phase 0: 50 lớp (`25,000` train / `5,000` test); Phases $1\dots 5$: mỗi phase $+10$ lớp (`5,000` train / `+1,000` test/phase; tổng `10,000` test ở Phase 5).
    - **Thứ tự lớp & Normalization:** `class_order = default_rng(seed).permutation(100)` (`42, 43, 44`); `mean/std` chỉ tính từ **25,000 ảnh base train** rồi giữ nguyên cho mọi phase/test. *(Khác biệt ghi nhận: PODNet/AANets dùng thứ tự lớp cố định giữa các lần chạy [4], [5])*.
  - **Backbone & Lịch huấn luyện Phase 0 ([1], [6]):**
    - **Backbone:** ResNet-32 cho CIFAR nhóm tự viết ($n=5$, kênh `16/32/64`, shortcut option A, feature $64$ chiều) [6] + lớp tạm `FC50`.
    - **Tối ưu BP (160 epochs, lấy checkpoint epoch cuối):** SGD (`lr=0.1`, momentum `0.9`, weight decay `0.0005`, batch `128`), `MultiStepLR` milestones `[80, 120]` ($\times 0.1$ tại epoch 80 và 120).
    - **Augmentation:** Chỉ bật lúc huấn luyện BP (`pad 4 -> random crop 32 -> flip p=0.5`); **không áp dụng crop/flip khi chạy analytic & test**.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021. [6] K. He et al., CVPR, 2016. [7] A. Krizhevsky, Tech. Rep., 2009.`
- **Speaker Notes (~200 từ):**
  > Slide 18 tóm tắt giao thức `ACIL-C100-v1.0` dùng xuyên suốt cho R1, R2 và R3. Bộ dữ liệu CIFAR-100 được tải tự động từ Hugging Face ở đúng mã phiên bản đã ghim và kiểm tra SHA-256 trên từng file Parquet trước khi chạy. Với mỗi seed `42, 43, 44`, 100 lớp được hoán vị theo `class_order` riêng, lấy 50 lớp đầu làm Phase 0 và 50 lớp còn lại chia thành 5 phase, mỗi phase 10 lớp. Nhóm cũng ghi nhận rõ một điểm khác biệt khi đối chiếu với code tham chiếu của PODNet [4] và AANets [5]: các repo đó giữ cố định một thứ tự lớp qua nhiều lần chạy, còn giao thức của nhóm hoán vị lại lớp theo từng seed nên độ lệch chuẩn giữa 3 seed bao gồm cả biến thiên do cách chia tập 50 lớp nền. Thống kê chuẩn hóa `mean` và `std` chỉ được tính trên $25{,}000$ ảnh train của Phase 0 để tránh rò rỉ thông tin từ tập test hay các lớp tương lai. Mạng ResNet-32 dùng cấu hình CIFAR chuẩn với shortcut Option A, huấn luyện đủ 160 epochs trên GPU T4 và lấy checkpoint ở epoch cuối.

---

### Slide 19 — Định nghĩa Hệ thống Metrics & Quy ước Dấu (`Metrics v1.0`)
- **Tiêu đề trên slide:** Định nghĩa Metric & Thống kê (`ACIL-metrics-v1.0`): Làm rõ quy ước dấu của $F$ và cách gộp 3 seeds
- **Bố cục trực quan:** Bên trái là 4 công thức metric chính ($A_k, \bar{A}, A_{\mathrm{final}}, B_k / F_{\mathrm{base}}$); Bên phải là ghi chú về dấu của chỉ số $F$ trong bài báo [1] & quy tắc tính `mean ± SD` ($n=3$, `ddof=1`).
- **Nội dung trên slide:**
  - **Các độ đo đánh giá (`docs/metrics.md`, single-head trên $100 \times C_k$ ảnh test):**
    - **$A_k$ (All-seen Accuracy):** Độ chính xác (%) trên toàn bộ $C_k$ lớp đã thấy sau phase $k$; **lấy $\operatorname{argmax}$ trên tất cả lớp đã gặp, không cung cấp Task ID**.
    - **$\bar{A}$ (Average Incremental Accuracy):** $\bar{A} = \frac{1}{K+1}\sum_{k=0}^{K} A_k$ ($K=5 \implies$ trung bình đều trên **6 mốc**, tính cả $A_0$ **sau ARaBT**, không nhân trọng số theo số ảnh).
    - **$A_{\mathrm{final}} = A_5$:** Độ chính xác ở phase cuối trên đủ $100$ lớp ($10{,}000$ ảnh test).
    - **$B_k$ (Base Accuracy under All-Seen Competition):** Độ chính xác riêng trên $5{,}000$ ảnh test của 50 lớp nền khi vẫn dự đoán trên đủ $C_k$ cột.
  - **Quy ước dấu của chỉ số quên $F$ khi đối chiếu bài báo [1]:**
    - Mục 4.2 bài báo in công thức $F = B_K - B_0$, nhưng Table 1 lại ghi số dương ($F = 9.00\%$). Nhóm thống nhất báo cáo **$F_{\mathrm{base}} = B_0 - B_K\text{ (pp)}$** (dương nghĩa là giảm nhiều hơn) để tiện so với Table 1, đồng thời lưu kèm `base_accuracy_delta_pp` ($B_K - B_0$).
  - **Cách tính thống kê:** Tính từng metric hoàn chỉnh theo mỗi seed trước, sau đó lấy `mean ± sample SD (n=3, ddof=1)`; không gộp 6 phases làm $n=6$, không lấy các lần đo thời gian R3 làm SD của accuracy.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Slide 19 làm rõ cách tính các chỉ số đánh giá trước khi đọc bảng kết quả. Trong thực nghiệm CIL, có ba điểm cần thống nhất ngay từ đầu: thứ nhất, độ chính xác trung bình $\bar{A}$ được tính trung bình cộng không trọng số trên cả 6 mốc từ $k=0$ đến $k=5$, trong đó $A_0$ là điểm đo sau bước căn chỉnh giải tích ARaBT chứ không lấy từ đầu ra `FC50` của backpropagation; thứ hai, ở mọi mốc đánh giá, mô hình đều phải chọn trên toàn bộ $C_k$ lớp đã gặp; thứ ba là quy ước dấu của độ quên $F$. Trong bài báo gốc [1], mục 4.2 viết $F = B_K - B_0$ (cho giá trị âm khi độ chính xác giảm) nhưng ở Bảng 1 lại trình bày giá trị dương $9.00$. Để tránh nhầm lẫn, nhóm đặt `base_forgetting_pp` $= B_0 - B_K$ khi so sánh với Bảng 1 và vẫn giữ song song giá trị có dấu `base_accuracy_delta_pp` trong file JSON. Ngoài ra, đơn vị lấy mẫu thống kê ở R1 và R2 là 3 seed độc lập (`ddof=1`), còn 3 lượt lặp ở R3 chỉ dùng để tính độ lệch chuẩn của thời gian và RAM.

---

### Slide 20 — Các Bước Kiểm tra Trước Khi Chạy Benchmark Dài
- **Tiêu đề trên slide:** Kiểm tra từng chặng trước khi chạy 160 epochs: Công thức số học, Smoke 100 lớp & Đo tài nguyên T4
- **Bố cục trực quan:** Sơ đồ 4 bước kiểm tra tuần tự (*Dữ liệu tổng hợp $\to$ CIFAR Smoke 100 lớp $\to$ Resume CPU/CUDA $\to$ Profile d8000 & BP5*) kèm mã Run ID W&B công khai của từng bước.
- **Nội dung trên slide:**
| Bước | Nội dung kiểm tra | Kết quả đạt được | W&B Run (Public) |
|---|---|---|---|
| **G1 — Dữ liệu tổng hợp** | `VERIFY-01/02`: 3 phase $\times 2$ lớp, $d=24$, minibatch $b=7$, lớp lặp | **11/11 tests đạt**; minibatch khớp Joint Ridge (max rel $W \approx 3.43\times 10^{-13}$, `0` argmax lệch). | [7aeaupq7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7aeaupq7) · [wi4wau2d](https://wandb.ai/hsangg-2006/acil-reproduction/runs/wi4wau2d) |
| **G2 — CIFAR Smoke** | `CIFAR-KAGGLE-01/02`: 2 epochs, $d=256$ trên GPU T4 (50+10 và đủ 50→100 lớp) | Xác nhận freeze/eval, tải HF tự động, $W_{256\times 100}$, khớp Joint 6/6 mốc ($A_{\mathrm{final}}=22.05\%$). | [o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2) · [blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) |
| **G3 — Resume v2 & Streaming** | Checkpoint v2 (RNG/optimizer/scheduler) CPU + CUDA & streaming d256 | CPU resume khớp trạng thái; CUDA TinyCNN đạt (**53/54 tests**, 1 snapshot skip); streaming khớp oracle 6/6. | Đính kèm trong [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) |
| **G4 — Profile d8000 & BP5** | `PROFILE-KAGGLE-01` (d8000 subset) & `BP5-KAGGLE-01` (5 epochs 25k ảnh) | Profile d8000: `34.07 s`, peak RSS `~3.20 GB`; BP5: `76.11 s` $\implies$ ước tính R1 `~59 phút` (< 120 phút T4). | [gey0yop4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/gey0yop4) · [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) |
- **Footer IEEE:** `Nội bộ nhóm (Không trích nguồn ngoài; chi tiết manifest tại docs/protocol_chot.md và reports/README.md).`
- **Speaker Notes (~190 từ):**
  > Trước khi gửi các lượt huấn luyện dài 160 epochs ở $d=8000$ lên Kaggle, nhóm thực hiện bốn bước kiểm tra ngắn để tránh sai công thức hoặc tràn RAM giữa chừng. Ở bước G1 trên dữ liệu tổng hợp (`VERIFY-01` và `VERIFY-02`), module `src/analytic.py` vượt qua 11 unit test kiểm tra trường hợp lớp lặp lại qua nhiều minibatch, tính bất biến của mảng đầu vào và độ khớp với nghiệm học gộp (sai lệch tương đối của $W$ khoảng $3.43\times 10^{-13}$). Ở bước G2, nhóm chạy thử toàn bộ luồng CIFAR-100 với 2 epochs và $d=256$ trên GPU T4 (lượt `o2r63pj2` cho 60 lớp và `blhw74ff` cho đủ 100 lớp), xác nhận việc đóng băng ResNet-32 và tính streaming cho dự đoán trùng khớp với Joint Oracle ở cả 6 mốc. Cuối cùng ở G3 và G4, nhóm kiểm tra cơ chế lưu/khôi phục checkpoint v2 trên CUDA, đo RAM đỉnh của bước analytic $d=8000$ ($\approx 3.20\text{ GB}$ ở lượt `gey0yop4`) và bấm giờ 5 epochs thật (`pv82dy1f`), từ đó ước lượng một seed R1 mất khoảng 59 phút, nằm an toàn trong khung thời gian của phiên chạy.

---

## PHẦN 3 (TIẾP THEO): KẾT QUẢ R1, R2, R3 & ĐỐI CHIẾU TÁC GIẢ (SLIDES 21–32)

### Slide 21 — Kết quả R1 (1/3): Huấn luyện Base 160 Epochs & Tổng hợp 3 Seeds Độc lập
- **Tiêu đề trên slide:** Kết quả R1: Huấn luyện 160 epochs hội tụ ổn định trên cả 3 seeds — $\bar{A} = 65.12 \pm 0.61\%$
- **Bố cục trực quan:** Trái (60%): Chèn hình tự vẽ [`reports/figures/r1_cifar100_k5_training.png`](../reports/figures/r1_cifar100_k5_training.png); Phải (40%): Bảng tổng hợp 3 seeds (`42, 43, 44`) kèm thời gian chạy trên T4 và link W&B từng run.
- **Nội dung trên slide:**
  - **Hình minh họa:** Chèn [`reports/figures/r1_cifar100_k5_training.png`](../reports/figures/r1_cifar100_k5_training.png) (*Loss Cross-Entropy Base 25,000 ảnh/epoch & lịch LR giảm bậc tại epoch 80, 120 cho seeds 42, 43, 44*).
  - **Bảng kết quả R1 theo từng seed (`d=8000`, $\gamma=0.1$, $K=5$, $n=3$, $\mathrm{ddof}=1$):**
| Seed | $\bar{A}$ (%) | $A_{\mathrm{final}}$ (%) | $F_{\mathrm{base}}$ (pp) | Numeric Runtime (phút) | W&B Run (Finished / Public) |
|---:|---:|---:|---:|---:|---|
| **42** | 65.16 | 55.87 | 8.32 | 53.78 *(BP 41.60m)* | [zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) |
| **43** | 65.71 | 56.77 | 8.92 | 52.25 *(BP 40.39m)* | [v51m2dan](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan) |
| **44** | 64.49 | 55.75 | 7.68 | 54.13 *(BP 42.29m)* | [vqcs1ql4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4) |
| **Mean ± SD** | **65.12 ± 0.61** | **56.13 ± 0.56** | **8.31 ± 0.62** | **~53.39 phút/seed** | `r1_cifar100_k5_summary.json` |
  - **Tài nguyên ghi nhận ở seed 42:** Peak CPU RSS $\approx 3.55\text{ GB}$; Peak CUDA allocated $\approx 378.42\text{ MiB}$; kích thước state R/W/FE $= 522.50\text{ MB}$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [6] K. He et al., "Deep residual learning for image recognition," CVPR, 2016.`
- **Speaker Notes (~195 từ):**
  > Slide 21 trình bày kết quả tổng hợp của thí nghiệm chính **R1** trên ba seed độc lập `42, 43, 44`. Trên biểu đồ bên trái, đường loss Cross-Entropy của cả ba lần huấn luyện ResNet-32 ở Phase 0 bám rất sát nhau và giảm rõ tại hai mốc giảm learning rate ở epoch $80$ ($0.1 \to 0.01$) và epoch $120$ ($0.01 \to 0.001$). Mỗi lượt chạy mất khoảng 40.4 đến 42.3 phút cho 160 epochs backpropagation trên GPU T4, và tổng cộng 52.3 đến 54.1 phút cho cả quy trình tính luôn 6 mốc streaming giải tích ở $d=8000$. Ở bảng bên phải, trung bình qua ba seed đạt $\bar{A} = 65.12 \pm 0.61\%$ (con số đầy đủ trong file tổng hợp là $65.1196 \pm 0.6104\%$), độ chính xác ở phase cuối trên 100 lớp đạt $56.13 \pm 0.56\%$, và mức giảm trên 50 lớp nền là $8.31 \pm 0.62$ điểm phần trăm. Cả ba lượt chạy (gồm cả `seed 44` có kết quả thấp hơn một chút ở mức `64.49%`) đều được giữ nguyên và công khai trên W&B.

---

### Slide 22 — Kết quả R1 (2/3): Diễn biến Accuracy qua 6 Phases ($50 \to 100$ Lớp)
- **Tiêu đề trên slide:** Đường cong R1 qua 6 phases: Độ chính xác tập Base giữ ở mức $68.33\%$ khi số lớp tăng lên 100
- **Bố cục trực quan:** Trái (58%): Chèn hình tự vẽ [`reports/figures/r1_cifar100_k5_accuracy.png`](../reports/figures/r1_cifar100_k5_accuracy.png); Phải (42%): Bảng 6 mốc ($k=0\dots 5$) gồm $A_k$ (All-seen), $B_k$ (Base 50), và $N_k$ (Lớp mới từng phase).
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
  - **Nhận xét từ bảng số liệu:** $B_k$ giảm chậm ($\approx 0.9\text{–}3.1\text{ pp}$/phase) với SD nhỏ ($\le 0.50\%$), trong khi $N_k$ dao động mạnh giữa các nhóm 10 lớp ($42.00\text{–}54.40\%$) do backbone ResNet-32 được cố định từ 50 lớp base.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Slide 22 đi sâu vào diễn biến độ chính xác qua từng mốc từ 50 đến 100 lớp. Đường màu đen trên hình là giá trị trung bình kèm dải độ lệch chuẩn mẫu qua ba seed. Tại mốc khởi đầu $k=0$ sau căn chỉnh giải tích ARaBT, mô hình đạt $76.64 \pm 0.12\%$ trên 5,000 ảnh test của 50 lớp đầu. Khi thêm dần mỗi đợt 10 lớp mới, độ chính xác tổng $A_k$ giảm đều xuống $56.13\%$ ở mốc 100 lớp. Khi tách riêng hai thành phần ở bảng bên phải, có thể thấy rõ nguyên nhân: trên 50 lớp nền ban đầu, độ chính xác $B_k$ chỉ giảm nhẹ từng bước và vẫn giữ được $68.33\%$ khi phải cạnh tranh trên toàn bộ 100 lớp; ngược lại, độ chính xác trên 10 lớp mới của từng phase ($N_k$) thấp hơn đáng kể, nằm trong khoảng $42\text{–}54\%$ với phương sai giữa các seed khá lớn (đến $7.07\%$ ở Phase 2). Kết quả này phản ánh đúng đặc điểm của việc đóng băng CNN từ Phase 0: khả năng phân biệt các lớp mới phụ thuộc nhiều vào mức độ tương đồng đặc trưng giữa chúng với 50 lớp nền.

---

### Slide 23 — Kết quả R1 (3/3): Đối chiếu Paper Table 1 & Bối cảnh Thực nghiệm
- **Tiêu đề trên slide:** Đối chiếu với Table 1 bài báo: R1 đạt $\bar{A} = 65.12\%$ so với $66.30\%$ (chênh $-1.18\text{ pp}$, tương ứng $-1.78\%$)
- **Bố cục trực quan:** Bảng so sánh trực tiếp giữa **ACIL Nhóm (R1)** và **ACIL Paper Table 1** ở trên; Khung tham khảo **các phương pháp khác trong Table 1 (trích từ paper)** ở dưới.
- **Nội dung trên slide:**
  - **1. Đối chiếu trực tiếp dòng ACIL (CIFAR-100, $K=5$, $d=8000$, $\gamma=0.1$):**
| Metric | Nhóm R1 (Mean ± SD, $n=3$) | Paper Table 1 [1] | Ours − Paper (pp) | Sai lệch tương đối (%) | Nguồn W&B |
|---|---:|---:|---:|---:|---|
| **$\bar{A}$ (%)** | **65.12 ± 0.61** *(65.1196)* | **66.30** | **-1.18** *(-1.1804)* | **-1.78%** *(-1.7804%)* | [zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) · [v51m2dan](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan) · [vqcs1ql4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4) |
| **$A_{\mathrm{final}}$ (%)** | **56.13 ± 0.56** *(56.1300)* | *Không ghi trong Table 1* | — | — | Như trên |
| **$F_{\mathrm{base}}$ (pp)** | **8.31 ± 0.62** *(8.3067)* | **9.00** *(mức giảm dương)* | **-0.69** *(-0.6933)* | **-7.70%** *(-7.7037%)* | Như trên |
  - **2. Tham chiếu bối cảnh Table 1 (CIFAR-100, $K=5$ — *Trích từ bài báo [1], nhóm không tự chạy lại các baseline này*):**
    - *Không lưu mẫu (Paper):* EWC ($16.04\%$) $\cdot$ LwF ($24.03\%$) $\cdot$ **ACIL Paper ($66.30\%$)** vs. **ACIL Nhóm R1 ($65.12\%$)**.
    - *Có lưu mẫu 20 ảnh/lớp (Paper):* iCaRL [2] ($57.12\%$) $\cdot$ LUCIR [8] ($63.17\%$) $\cdot$ PODNet [4] ($64.83\%$) $\cdot$ AANets [5] ($66.31\%$).
  - **Nhận xét phạm vi:** Ở $K=5$, kết quả tái lập $65.12\%$ của nhóm nằm gần mức PODNet ($64.83\%$) trong bài báo, tuy nhiên nhóm chưa tự chạy lại các phương pháp lưu mẫu hay cấu hình $K=25$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [2] S.-A. Rebuffi et al., CVPR, 2017. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021. [8] S. Hou et al., CVPR, 2019.`
- **Speaker Notes (~200 từ):**
  > Slide 23 đặt kết quả tái lập R1 cạnh số liệu công bố ở Bảng 1 của bài báo NeurIPS 2022. Trên cùng cấu hình CIFAR-100 năm phase với $d=8000$ và $\gamma=0.1$, độ chính xác trung bình $\bar{A}$ của nhóm đạt $65.12\%$, thấp hơn mức $66.30\%$ trong bài báo $1.18$ điểm phần trăm (tương ứng mức lệch tương đối $-1.78\%$). Ở chỉ số giảm độ chính xác trên tập nền $F_{\mathrm{base}}$, kết quả của nhóm là $8.31\text{ pp}$ so với $9.00$ trong Bảng 1 (thấp hơn $0.69\text{ pp}$). Bài báo không báo cáo riêng cột $A_{\mathrm{final}}$ trong Bảng 1 nên nhóm để trống ô đối chiếu này. Phần dưới của slide trích lại các con số của LwF ($24.03\%$), iCaRL ($57.12\%$), PODNet ($64.83\%$) và AANets ($66.31\%$) từ Bảng 1 để thấy vị trí tương đối của kết quả tái lập, đồng thời ghi chú rõ đây là số liệu trích dẫn từ bài báo gốc chứ không phải kết quả tự chạy của nhóm. Các yếu tố kỹ thuật có thể dẫn tới khoảng lệch $-1.18\text{ pp}$ sẽ được tổng hợp ở Slide 30 sau khi xem xét R2 và code tác giả.

---

### Slide 24 — Kết quả R2 (1/2): Vai trò của Feature Expansion & Độ nhạy theo $\gamma$
- **Tiêu đề trên slide:** Kết quả R2 (4/5 hàng Table 2): Bật FE giúp tăng **$+11.86\text{ pp}$**, còn $\gamma \in [0.001, 0.1]$ gần như không đổi
- **Bố cục trực quan:** Trái (55%): Chèn hình tự vẽ [`reports/figures/r2_cifar100_k5_ablation.png`](../reports/figures/r2_cifar100_k5_ablation.png); Phải (45%): Bảng kết quả 4 hàng × 3 seeds kèm cột Paper Table 2, độ lệch và hiệu ghép cặp $\Delta\bar{A}$ trên cùng checkpoint.
- **Nội dung trên slide:**
  - **Hình minh họa:** Chèn [`reports/figures/r2_cifar100_k5_ablation.png`](../reports/figures/r2_cifar100_k5_ablation.png) (Trái: cột $\bar{A}$ đối chiếu hình thoi Paper Table 2; Phải: hiệu ghép cặp $\Delta\bar{A}$ theo từng seed so với $\gamma=0.1$).
  - **Bảng đối chiếu 4/5 hàng Table 2 (Dùng lại checkpoint R1 cùng seed — 9 runs mới + 3 R1):**
| Cấu hình | Ours $\bar{A}$ ± SD (%) | Paper Table 2 [1] | Ours − Paper (pp) | Paired $\Delta\bar{A}$ vs R1 (pp) | W&B Runs (Seeds 42 / 43 / 44) |
|---|---:|---:|---:|---:|---|
| **No-FE ($d=64$), $\gamma=0.1$** | **53.25 ± 0.28** | 52.99 | **+0.26** | **-11.86 ± 0.37** | [xdrm0d27](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xdrm0d27) · [7wgme4so](https://wandb.ai/hsangg-2006/acil-reproduction/runs/7wgme4so) · [81aci8h1](https://wandb.ai/hsangg-2006/acil-reproduction/runs/81aci8h1) |
| **FE 8000, $\gamma=0.1$ (R1)** | **65.12 ± 0.61** | 66.30 | -1.18 | 0.00 | [zt1014ei](https://wandb.ai/hsangg-2006/acil-reproduction/runs/zt1014ei) · [v51m2dan](https://wandb.ai/hsangg-2006/acil-reproduction/runs/v51m2dan) · [vqcs1ql4](https://wandb.ai/hsangg-2006/acil-reproduction/runs/vqcs1ql4) |
| **FE 8000, $\gamma=0.01$** | **65.12 ± 0.61** *(65.1173)* | 66.25 | -1.13 | -0.0022 ± 0.0020 | [qcgdklxh](https://wandb.ai/hsangg-2006/acil-reproduction/runs/qcgdklxh) · [xfleip26](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xfleip26) · [dub202jk](https://wandb.ai/hsangg-2006/acil-reproduction/runs/dub202jk) |
| **FE 8000, $\gamma=0.001$** | **65.12 ± 0.61** *(65.1163)* | 66.23 | -1.11 | -0.0032 ± 0.0035 | [3yp99ewh](https://wandb.ai/hsangg-2006/acil-reproduction/runs/3yp99ewh) · [aler6nj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/aler6nj2) · [l9c50ezb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/l9c50ezb) |
  - **Kết quả chính:** (1) Bỏ FE làm $\bar{A}$ giảm **$-11.86\text{ pp}$**, cho thấy vai trò quan trọng của tầng mở rộng đặc trưng; (2) Trong dải $\gamma \in [0.001, 0.1]$, $\bar{A}$ chỉ thay đổi **$0.0032\text{ pp}$**.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Slide 24 trình bày kết quả bộ thí nghiệm ablation **R2**, tái lập 4 trên 5 hàng trong Table 2 của bài báo [1]. Để phép so sánh giữa các cấu hình được công bằng, tất cả các lượt chạy của cùng một seed đều nạp lại đúng một file checkpoint ResNet-32 từ R1 (hàm train backbone bị khóa trong runner R2 để không thể vô tình huấn luyện lại), và ba cấu hình đổi $\gamma$ dùng chung cùng một ma trận mở rộng $W_{\mathrm{fe}}$. Số liệu thu được cho thấy hai xu hướng rất rõ: thứ nhất, khi bỏ tầng Feature Expansion và đưa thẳng đặc trưng 64 chiều $F$ vào bộ phân loại giải tích, $\bar{A}$ giảm từ $65.12\%$ xuống $53.25\%$, tức mất $11.86$ điểm phần trăm (khá sát mức $52.99\%$ trong Table 2 của bài báo, chênh $+0.26\text{ pp}$). Điều này phù hợp với phân tích lý thuyết rằng không gian 64 chiều không đủ tách tuyến tính cho 100 lớp. Thứ hai, trên biểu đồ bên phải, khi giảm $\gamma$ từ $0.1$ xuống $0.01$ và $0.001$, hiệu ghép cặp theo từng seed chỉ dao động $-0.0022$ và $-0.0032\text{ pp}$, cho thấy mô hình rất ổn định với $\gamma$ trong khoảng này.

---

### Slide 25 — Kết quả R2 (2/2): Lưu ý Khi Đọc Chỉ số "$F_{\mathrm{base}} = 1.56\text{ pp}$" của No-FE & Giới hạn
- **Tiêu đề trên slide:** Lưu ý khi đọc kết quả R2: Vì sao No-FE có $F_{\mathrm{base}}$ nhỏ ($1.56\text{ pp}$) nhưng chất lượng tổng thể lại thấp?
- **Bố cục trực quan:** Bảng đặt cạnh nhau $\bar{A}$, $A_{\mathrm{final}}$ và $F_{\mathrm{base}}$ giữa No-FE và FE 8000 ở trên; Hai khung giải thích nguyên nhân & 2 giới hạn của R2 ở dưới.
- **Nội dung trên slide:**
  - **Bảng đối chiếu $\bar{A}$, $A_{\mathrm{final}}$ & $F_{\mathrm{base}}$ (3 seeds, `ddof=1`):**
| Cấu hình | $\bar{A}$ (%) | $A_{\mathrm{final}}$ (100 lớp, %) | $F_{\mathrm{base}}$ (pp) | Nhận xét kết hợp |
|---|---:|---:|---:|---|
| **No-FE ($d=64$), $\gamma=0.1$** | 53.25 ± 0.28 | **38.33 ± 0.37** | **1.56 ± 0.64** | Biểu diễn lớp mới quá yếu $\to$ $A_{\mathrm{final}}$ giảm mạnh |
| **FE 8000, $\gamma \in \{0.1, 0.01, 0.001\}$** | 65.12 ± 0.61 | **56.13 ± 0.56** | **8.31 ± 0.62** | Cao hơn No-FE **+17.80 pp** ở $A_{\mathrm{final}}$ |
  - **Giải thích nguyên nhân:** Nếu chỉ nhìn riêng cột $F_{\mathrm{base}}$, có thể hiểu nhầm No-FE "ít quên hơn" ($1.56 < 8.31\text{ pp}$). Thực tế, trên không gian $64$ chiều không có FE, bộ phân loại tuyến tính học các lớp mới rất kém nên điểm logit của các lớp mới ít khi vượt qua được 50 lớp cũ trong $\operatorname{argmax}$, kéo theo $A_{\mathrm{final}}$ chỉ đạt **$38.33\%$**. Vì vậy cần **đọc $F_{\mathrm{base}}$ đi kèm với $\bar{A}$ và $A_{\mathrm{final}}$**.
  - **Hai giới hạn của thiết kế R2:**
    1. Cấu hình No-FE bỏ đồng thời phép chiếu $64\to 8000$ và hàm $\operatorname{ReLU}$, nên chưa tách riêng đóng góp của số chiều $d$ với tính phi tuyến $\operatorname{ReLU}$.
    2. Chưa kiểm tra hàng $\gamma=0$ (báo cáo **4/5 hàng Table 2**) do công thức RLS + Woodbury hiện tại cần $\gamma > 0$ để đảm bảo $A_0 \succ 0$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~190 từ):**
  > Slide 25 nêu một điểm cần lưu ý khi đọc các chỉ số của thí nghiệm R2. Ở cấu hình bỏ mở rộng đặc trưng (No-FE), mức giảm độ chính xác trên 50 lớp nền $F_{\mathrm{base}}$ chỉ là $1.56\text{ pp}$, nhỏ hơn nhiều so với mức $8.31\text{ pp}$ khi dùng FE 8000. Tuy nhiên, không thể từ đó kết luận rằng bỏ FE giúp mô hình hoạt động tốt hơn: ở mốc 100 lớp cuối cùng, độ chính xác tổng $A_{\mathrm{final}}$ của No-FE chỉ đạt $38.33\%$, thấp hơn cấu hình FE 8000 tới $17.80$ điểm phần trăm ($56.13\%$). Nguyên nhân là không gian 64 chiều không đủ khả năng biểu diễn thêm 50 lớp mới sau khi CNN đã đóng băng, làm cho điểm số của các lớp mới thường thấp và ít khi đạt giá trị lớn nhất trong hàm $\operatorname{argmax}$, dẫn tới các lớp cũ ít bị cạnh tranh hơn. Do đó, khi đánh giá CIL, chỉ số $F_{\mathrm{base}}$ luôn phải đặt cạnh $\bar{A}$ và $A_{\mathrm{final}}$. Ở phần cuối slide, nhóm cũng ghi rõ hai giới hạn của R2 là chưa tách riêng vai trò của chiều $d$ với hàm ReLU và chưa chạy trường hợp $\gamma = 0$.

---

### Slide 26 — Kết quả R3 (1/2): Kiểm chứng Tương đương Đại số Theo Kích thước Minibatch
- **Tiêu đề trên slide:** Thí nghiệm tự thiết kế R3: Thay đổi cỡ batch ($32 / 128 / 512$) vẫn khớp Joint Oracle ở cả **54/54 mốc**
- **Bố cục trực quan:** Khung tóm tắt thiết lập kiểm tra (*checkpoint 160 seed 42, $d=1024, \gamma=0.1$, 9 tiến trình CPU độc lập, lịch chạy xoay vòng*) ở trên; Bảng sai số đối chiếu Joint Oracle ở dưới.
- **Nội dung trên slide:**
  - **Thiết kế thực nghiệm (`docs/R3_KE_HOACH_DO.md`):**
    - Trên cùng checkpoint seed 42 ($d=1024, \gamma=0.1$, cố định thứ tự mẫu): khảo sát 3 cỡ batch $b \in \{32, 128, 512\} \times 3$ lần đo thời gian = **9 tiến trình CPU riêng biệt (4 luồng BLAS)** theo thứ tự xoay `32→128→512`, `128→512→32`, `512→32→128`.
    - **Joint Oracle** chạy tách rời ngoài cửa sổ bấm giờ, ghép đủ $H_{0:k}, Y_{0:k}$ để giải nghiệm tham chiếu tại từng phase ($9 \times 6 = \mathbf{54\text{ mốc so sánh}}$).
  - **Bảng đối chiếu sai số với Joint Oracle (54/54 đạt ngưỡng `rtol=1e-6, atol=1e-8`):**
| Batch ($b$) | Số mốc | Max rel $W$ error | Max rel $R$ error | Max abs score error | Argmax khác Joint | W&B Runs (3 timing repeats) |
|---:|---:|---:|---:|---:|---:|---|
| **32** | 18 | `1.159e-11` | `3.823e-15` | `2.475e-12` | **0 / 10,000** | [luf7uv40](https://wandb.ai/hsangg-2006/acil-reproduction/runs/luf7uv40) · [yqxph1bb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/yqxph1bb) · [59vlfxzv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/59vlfxzv) |
| **128** | 18 | `9.679e-12` | `3.399e-15` | `2.302e-12` | **0 / 10,000** | [o54hbdfp](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o54hbdfp) · [4481bnn1](https://wandb.ai/hsangg-2006/acil-reproduction/runs/4481bnn1) · [3gv0moj7](https://wandb.ai/hsangg-2006/acil-reproduction/runs/3gv0moj7) |
| **512** | 18 | `9.782e-12` | `3.272e-15` | `2.178e-12` | **0 / 10,000** | [lnk0grbq](https://wandb.ai/hsangg-2006/acil-reproduction/runs/lnk0grbq) · [5ehmjvh3](https://wandb.ai/hsangg-2006/acil-reproduction/runs/5ehmjvh3) · [xesh2kjl](https://wandb.ai/hsangg-2006/acil-reproduction/runs/xesh2kjl) |
  - **Độ chính xác trùng nhau trên cả 9 runs & Joint (`d=1024`, seed 42):** $\bar{A} = 63.10\%$, $A_{\mathrm{final}} = 51.73\%$, $F_{\mathrm{base}} = 5.96\text{ pp}$. *(Ghi chú: $d=1024$ là cấu hình phân tích riêng trên 1 seed, không dùng thay cho R1 $d=8000$).*
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~195 từ):**
  > Slide 26 trình bày **R3** — thí nghiệm bổ sung do nhóm tự thiết kế để kiểm tra thực tế trên CIFAR-100 xem việc chia dữ liệu thành các minibatch cỡ $32$, $128$ hay $512$ có làm lệch nghiệm so với giải gộp một lần hay không. Để việc tính nghiệm tham chiếu Joint Oracle trên toàn bộ $50{,}000$ ảnh nằm gọn trong giới hạn RAM của Kaggle, nhóm chốt trước chiều mở rộng $d = 1024$ trên checkpoint epoch 160 của seed 42, giữ cố định thứ tự mẫu và khởi chạy 9 tiến trình CPU độc lập theo lịch xoay vòng nhằm tránh nhiễu hệ thống. Kết quả trên cả $54$ mốc kiểm tra (9 lượt chạy × 6 mốc phase) đều khớp sát với Joint Oracle: sai số tương đối lớn nhất của trọng số $W$ không vượt quá $1.159\times 10^{-11}$, sai số của $R$ ở bậc $10^{-15}$, và toàn bộ $10{,}000$ ảnh kiểm tra đều cho nhãn dự đoán giống hệt nhau. Cả 9 lượt chạy và Joint Oracle đều cho cùng $\bar{A} = 63.10\%$ tại cấu hình $d=1024$.

---

### Slide 27 — Kết quả R3 (2/2): Đánh đổi Thời gian & Bộ nhớ Peak RSS theo Batch Size
- **Tiêu đề trên slide:** Đánh đổi trong R3: Batch 512 nhanh gấp **$3.47\times$** Batch 32 nhưng Peak RSS cao hơn $+18.45\text{ MiB}$ so với Batch 128
- **Bố cục trực quan:** Trái (58%): Chèn hình tự vẽ [`reports/figures/r3_cifar100_k5_batch_analysis.png`](../reports/figures/r3_cifar100_k5_batch_analysis.png); Phải (42%): Bảng tách riêng thời gian *Base vs 5 Incremental Phases*, *Tổng thời gian*, *Peak RSS* & nhận xét.
- **Nội dung trên slide:**
  - **Hình minh họa:** Chèn [`reports/figures/r3_cifar100_k5_batch_analysis.png`](../reports/figures/r3_cifar100_k5_batch_analysis.png) (*Trái: Analytic learning time; Phải: Observed learning peak RSS qua 3 lần đo*).
  - **Bảng tách chi tiết Thời gian & Peak RSS (`d=1024`, 4 threads CPU, $n=3$ timing repeats):**
| Batch ($b$) | Base (s) | 5 Phases Inc (s) | Tổng Analytic (s) | Tỷ lệ vs $b=128$ | Peak RSS (MiB) |
|---:|---:|---:|---:|---:|---:|
| **32** | 10.71 ± 0.07 | 6.62 ± 0.10 | **17.33 ± 0.08** | $2.503\times$ | 151.35 ± 0.97 |
| **128** | 3.19 ± 0.07 | 3.73 ± 0.04 | **6.92 ± 0.11** | $1.000\times$ | **146.10 ± 0.92** *(thấp nhất)* |
| **512** | 1.28 ± 0.07 | 3.71 ± 0.05 | **5.00 ± 0.11** | **$0.722\times$** | 164.55 ± 4.38 |
  - **Hai nhận xét định lượng từ số đo:**
    1. Khi tăng $b$ từ $128 \to 512$, tổng thời gian giảm $1.93\text{ s}$ nhưng **phần giảm chủ yếu nằm ở bước Base ($3.19 \to 1.28\text{ s}$, giảm $1.91\text{ s}$)**, còn **tổng 5 phases incremental gần như giữ nguyên ($3.73 \to 3.71\text{ s}$)** vì ma trận trung gian Woodbury $b\times b$ tăng lên $512 \times 512$ bù lại số vòng lặp ít hơn.
    2. Kích thước state cuối ($R/W/\mathrm{FE}$) luôn là **$9.28\text{ MiB}$** ở mọi batch, nhưng bộ nhớ đệm tạm làm Peak RSS của $b=512$ cao hơn $b=128$ trung bình **$18.45\text{ MiB}$** $\implies$ $b=128$ là mức cân bằng tốt giữa thời gian cập nhật và RAM.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. (W&B: luf7uv40, yqxph1bb, 59vlfxzv, o54hbdfp, 4481bnn1, 3gv0moj7, lnk0grbq, 5ehmjvh3, xesh2kjl).`
- **Speaker Notes (~200 từ):**
  > Slide 27 phân tích thời gian tính toán và mức RAM đỉnh đo được trong thí nghiệm R3. Ở đây, đồng hồ đo thời gian chỉ tính đúng 6 cửa sổ học giải tích (gồm bước chiếu FE, tích lũy/giải Base và 5 pha cập nhật đệ quy) trên tiến trình CPU cố định 4 luồng; thời gian đọc dữ liệu, chấm điểm tập test và chạy Joint Oracle đều được tách ra ngoài. Về tổng thời gian, batch 512 mất trung bình $5.00\text{ s}$, nhanh hơn $1.386$ lần so với batch 128 ($6.92\text{ s}$) và $3.47$ lần so với batch 32 ($17.33\text{ s}$). Tuy nhiên, khi tách riêng bước Base và 5 pha tăng dần ở bảng bên phải, có thể thấy phần tiết kiệm thời gian khi tăng batch từ 128 lên 512 hầu hết đến từ phép cộng dồn ma trận Gram ở bước Base ($3.19\text{ s}$ xuống $1.28\text{ s}$), còn 5 pha cập nhật Woodbury hầu như không nhanh thêm ($3.73\text{ s}$ so với $3.71\text{ s}$) do phải giải hệ $512\times 512$ lớn hơn ở mỗi batch. Đồng thời, Peak RSS của batch 512 tăng thêm $18.45\text{ MiB}$ so với batch 128, nên lựa chọn $b=128$ mặc định của nhóm là hợp lý.

---

### Slide 28 — Đối chiếu Implementation Tác giả (1/2): Thiết kế C1 / C2 / C1R & Khác biệt ở Tầng Tiền xử lý–Backbone
- **Tiêu đề trên slide:** Chạy trực tiếp code tác giả (`0afd6bf`): Ba điểm khác biệt ở ánh xạ nhãn, chuẩn hóa ảnh và Global Pooling
- **Bố cục trực quan:** Trái: Thiết kế 3 kịch bản **C1**, **C2**, **C1R** (chạy trong môi trường riêng, giữ nguyên code tác giả); Phải: 3 điểm khác biệt ghi nhận được ở tầng dữ liệu & backbone kèm số đo.
- **Nội dung trên slide:**
  - **Thiết lập đối chiếu (`scripts/compare_author_acil.py`, repo `ZHUANGHP/Analytic-continual-learning` @ `0afd6bf`):**
    - Cài môi trường riêng (không sửa hay chép code tác giả vào `src/`); dùng chung checkpoint 160 seed 42, $d=1024, \gamma=0.1$.
    - **C1 (Controlled):** Gọi class `ACIL` của tác giả với cùng FE $\mathcal{N}(0,1)$ & thứ tự mẫu như R3.
    - **C2 (Native FE / `main.main`):** Nạp checkpoint sang ResNet-32 tác giả, dùng `RandomBuffer` gốc và DataLoader shuffle gốc.
    - **C1R (Diagnostic bổ sung):** Nạp state $(R_0, W_0)$ của nhóm rồi chạy 5 bước cập nhật của tác giả.
  - **Ba điểm khác biệt xác định được trước tầng Analytic:**
    1. **Ánh xạ nhãn khi xáo lớp (`DatasetWrapper`):** Bật `shuffle_seed=42` trong wrapper của tác giả phát sinh `IndexError: Target 91 is out of bounds` do xáo thứ tự lớp nhưng **chưa ánh xạ lại nhãn về `0..49`** *(pipeline của nhóm luôn ánh xạ qua `original_to_global`)*.
    2. **Thống kê chuẩn hóa ảnh:** Nhóm tính trên 25,000 ảnh base seed 42, còn code tác giả dùng hằng số CIFAR cố định $\implies$ độ lệch lớn nhất trên tensor ảnh $= 0.0252$ và trên feature CNN $= 0.3558$ (kiểm trên 1,000 ảnh test).
    3. **Global Pooling (`FP32` vs `FP64`):** Trước pooling hai bên khớp **từng bit (`0.0`)**; sau pooling (`adaptive_avg_pool` vs `avg_pool`) lệch tối đa **$3.81\times 10^{-6}$ ở FP32** (giảm về **$3.55\times 10^{-15}$ ở FP64**), nhãn dự đoán vẫn trùng `1000/1000`.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022 (Đối chiếu với ZHUANGHP/Analytic-continual-learning commit 0afd6bf trên W&B 0tlcy6id, 8muywodv, yeb0jqfb).`
- **Speaker Notes (~205 từ):**
  > Bên cạnh việc đọc mã nguồn trên GitHub, nhóm tạo một môi trường ảo riêng, lấy đúng commit `0afd6bf` từ repository của tác giả và chạy đối chiếu trực tiếp qua ba kịch bản: C1 dùng chung ma trận mở rộng $\mathcal{N}(0,1)$ của nhóm, C2 gọi thẳng hàm `main.main` với lớp `RandomBuffer` gốc của tác giả, và C1R dùng chung trạng thái nền $(R_0, W_0)$ để tách riêng phần cập nhật tăng dần. Khi kiểm tra từng bước từ dữ liệu đầu vào tới đặc trưng ResNet-32 trên $1{,}000$ ảnh test, nhóm ghi nhận ba chi tiết cụ thể. Thứ nhất, khi truyền `shuffle_seed=42` vào `DatasetWrapper` của code tác giả, chương trình dừng ngay với lỗi `IndexError: Target 91 is out of bounds` vì danh sách lớp được xáo lại nhưng giá trị nhãn chưa đổi về đoạn `0..49`. Thứ hai, việc nhóm tính `mean/std` riêng trên 25,000 ảnh base còn code tác giả dùng hằng số CIFAR toàn cục tạo ra chênh lệch đặc trưng tối đa $0.3558$. Thứ ba, với cùng trọng số mạng, đầu ra trước tầng pooling giống nhau tuyệt đối nhưng phép lấy trung bình không gian ở `float32` giữa `adaptive_avg_pool` và `avg_pool` lệch nhẹ $3.81\times 10^{-6}$.

---

### Slide 29 — Đối chiếu Implementation Tác giả (2/2): Kết quả C1 / C2 / C1R & Nguyên nhân Lệch Số học
- **Tiêu đề trên slide:** Kết quả so sánh số học: C1 lệch 2/3.55M phần tử score ở phase 3–4; C1R giảm sai lệch về mức $10^{-13}$
- **Bố cục trực quan:** Trái (58%): Chèn hình tự vẽ [`reports/figures/author_acil_comparison.png`](../reports/figures/author_acil_comparison.png); Phải (42%): Bảng kết quả 3 lượt C1 / C2 / C1R (giữ nguyên `C1 equivalence=false`) + chẩn đoán nguyên nhân từ bước khởi tạo Base.
- **Nội dung trên slide:**
  - **Hình minh họa:** Chèn [`reports/figures/author_acil_comparison.png`](../reports/figures/author_acil_comparison.png) (*Trái: Accuracy C1 vs C2 trên `d=1024`; Phải: Sai số tương đối Frobenius của $W$ ở thang log từ $10^{-8}$ xuống $10^{-13}$*).
  - **Bảng đối chiếu kết quả (`rtol=1e-6, atol=1e-8`, seed 42, `d=1024`, $\gamma=0.1$):**
| Lượt | $\bar{A}$ (%) | $A_{\mathrm{final}}$ (%) | Max rel $W$ err | Max abs score err | Đạt `allclose` (R/W/Score) | Argmax khác | W&B Run |
|---|---:|---:|---:|---:|---|---|---|
| **C1 (Controlled N(0,1))** | 63.10 | 51.73 | `3.025e-8` | `2.084e-8` | R `6/6`, W `6/6`, **Score `4/6`** *(Fail)* | **0** | [0tlcy6id](https://wandb.ai/hsangg-2006/acil-reproduction/runs/0tlcy6id) |
| **C2 (Native FE / main)** | 63.15 | 52.24 | `6.911e-11` | `3.349e-11` | **Cả ba `6/6`** | **0** | [8muywodv](https://wandb.ai/hsangg-2006/acil-reproduction/runs/8muywodv) |
| **C1R (Matched Base)** | — | 51.73 | **`5.607e-13`** | **`6.645e-13`** | **Cả ba `5/5`** | **0** | [yeb0jqfb](https://wandb.ai/hsangg-2006/acil-reproduction/runs/yeb0jqfb) |
  - **Diễn giải kết quả:**
    - Giữ nguyên trạng thái **C1 `analytic_equivalence_passed = false`** theo ngưỡng đã đặt trước (dù chỉ có **2 trên 3,550,000 phần tử score** ở phase 3–4 vượt nhẹ ngưỡng `allclose` và `0` nhãn dự đoán khác nhau).
    - **Kiểm tra C1R xác định nguồn lệch:** Khi truyền cùng $(R_0, W_0)$ của nhóm rồi chạy 5 phase tăng dần bằng code tác giả, sai số giảm khoảng $50{,}000$ lần xuống cỡ **$10^{-13}$** $\implies$ chênh lệch nhỏ ở C1 đến từ **cách tính trạng thái Base** (code tác giả khởi tạo $R = I/\gamma$ rồi cập nhật RLS từng batch bằng `torch.inverse`, còn nhóm cộng dồn Gram + giải Cholesky).
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022 (Đối chiếu với ZHUANGHP/Analytic-continual-learning @ 0afd6bf).`
- **Speaker Notes (~205 từ):**
  > Slide 29 trình bày kết quả đối chiếu tầng giải tích giữa hai bản cài đặt. Ở kịch bản **C1** khi dùng chung ma trận mở rộng $\mathcal{N}(0,1)$, sai số tương đối của trọng số $W$ ở mức $3\times 10^{-8}$; cả $6/6$ pha của $R$, $W$ và toàn bộ nhãn dự đoán argmax đều khớp nhau, nhưng có $2$ trong tổng số $3.55\text{ triệu}$ phần tử điểm số ở Phase 3 và Phase 4 vượt nhẹ ngưỡng kiểm tra `allclose(1e-6, 1e-8)`. Nhóm không nới ngưỡng kiểm tra sau khi chạy mà giữ nguyên cờ `equivalence = false` trong file kết quả và trên W&B. Để tìm hiểu vì sao có mức lệch $10^{-8}$ này, nhóm chạy thêm phép chẩn đoán **C1R** (đường nét đứt màu xanh lá ở dưới cùng đồ thị bên phải): khi nạp cùng cặp $(R_0, W_0)$ của nhóm ở Phase 0 rồi để code tác giả chạy tiếp 5 pha tăng dần, sai số giảm ngay xuống mức $10^{-13}$ và đạt đủ $5/5$ pha. Như vậy, phần cập nhật tăng dần của hai bên là tương đương, còn sai khác nhỏ ở C1 phát sinh từ cách tính Phase 0 khi code tác giả cập nhật RLS từ $I/\gamma$ bằng `torch.inverse` thay vì giải Cholesky một lần từ tổng Gram.

---

### Slide 30 — Tổng hợp Nguyên nhân Chênh lệch ($\bar{A}_{\mathrm{R1}} - \bar{A}_{\mathrm{paper}} = -1.18\text{ pp}$): Đã Xác nhận vs. Chưa Tách riêng
- **Tiêu đề trên slide:** Phân tích khoảng lệch $-1.18\text{ pp}$ giữa R1 ($65.12\%$) và bài báo ($66.30\%$): Điều đã xác nhận và giới hạn kết luận
- **Bố cục trực quan:** Chia thành 2 phần rõ ràng: Phần 1 — *Các điểm khác biệt ĐÃ XÁC NHẬN giữa hai thiết lập* (4 yếu tố); Phần 2 — *Giới hạn hiện tại — CHƯA CHẠY ABLATION tách riêng từng yếu tố*.
- **Nội dung trên slide:**
  - **1. Các khác biệt thực tế ĐÃ ĐƯỢC XÁC NHẬN qua kiểm tra giao thức & code tác giả:**
    1. **Thứ tự chia lớp (`class_order`):** Nhóm sinh hoán vị theo từng seed `42/43/44` (SD $= 0.61\%$ gồm cả độ khó của 50 lớp nền), trong khi PODNet [4] / AANets [5] dùng thứ tự lớp cố định và bài báo không công bố danh sách lớp của Table 1.
    2. **Phân phối khởi tạo $W_{\mathrm{fe}}$:** Nhóm cài $\mathcal{N}(0,1)$ đúng theo mục 3.1 bài báo [1]; repo tác giả `0afd6bf` dùng `Linear.reset_parameters()` (độ lệch chuẩn thực đo $\approx 0.0724$; trên 1 seed ở C2 cho mức chênh nhỏ $+0.045\text{ pp}$ $\bar{A}$ và $+0.51\text{ pp}$ final).
    3. **Thống kê chuẩn hóa ảnh (Normalization):** Nhóm tính từ tập base train ($25{,}000$ ảnh) để tránh rò rỉ thống kê, còn code tác giả dùng hằng số CIFAR-100 $\implies \Delta F_{\max} = 0.3558$.
    4. **Môi trường & cấu hình repo mới:** Khác biệt nhỏ ở pooling FP32 (`adaptive_avg_pool` vs `avg_pool`), phiên bản Python 3.13 / Torch 2.11 so với năm 2022, và README hiện tại của repo tác giả dùng lịch `300 epochs`, cosine, `d=8192` thay vì `160 epochs / d=8000` như Table 1 bài báo.
  - **2. Giới hạn kết luận (chưa suy đoán định lượng khi chưa chạy tách biến):**
    - Nhóm **chưa chạy thêm các lượt 3-seed tách riêng từng yếu tố** để biết chính xác mỗi khác biệt trên đóng góp bao nhiêu phần trăm vào mức $-1.18\text{ pp}$.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL...," NeurIPS, 2022. [4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021. [6] K. He et al., CVPR, 2016.`
- **Speaker Notes (~200 từ):**
  > Slide 30 tổng hợp lại các nguyên nhân có thể giải thích mức chênh lệch $-1.18$ điểm phần trăm giữa kết quả R1 của nhóm ($65.12\%$) và con số $66.30\%$ trong Bảng 1 của bài báo. Ở đây nhóm tách riêng những khác biệt đã kiểm tra được bằng thực nghiệm với những phần chưa đủ dữ liệu để kết luận định lượng. Về các khác biệt đã xác nhận: thứ nhất là cách chọn thứ tự 50 lớp nền và 5 pha tăng dần, khi nhóm xáo lại theo từng seed 42, 43, 44 còn các repo tham chiếu PODNet/AANets giữ thứ tự lớp cố định; thứ hai là repo `0afd6bf` của tác giả khởi tạo tầng mở rộng bằng `Linear.reset_parameters` thay vì phân phối chuẩn tắc $\mathcal{N}(0,1)$ viết trong bài báo, đồng thời dùng hằng số chuẩn hóa CIFAR chung thay vì tính riêng trên tập base; thứ ba là bản thân repo hiện tại của tác giả cũng đã chuyển sang cấu hình 300 epochs và $d=8192$. Tuy nhiên, vì nhóm chưa chạy ablation 3 seeds cho từng yếu tố riêng lẻ trên, nhóm chỉ dừng ở việc liệt kê các nguồn sai khác đã định vị được chứ không gán cứng tỷ trọng đóng góp cho từng yếu tố.

---

### Slide 31 — Kinh nghiệm Triển khai Kỹ thuật & Quản lý Thực nghiệm
- **Tiêu đề trên slide:** Kinh nghiệm triển khai: 5 quyết định kỹ thuật giúp ổn định bộ nhớ, chống ghi đè kết quả và đảm bảo truy nguyên
- **Bố cục trực quan:** Danh sách 5 mục theo cấu trúc *Vấn đề thực tế gặp phải $\to$ Cách xử lý trong `src/` $\to$ Tác dụng*.
- **Nội dung trên slide:**
  1. **Nhãn toàn cục (`original_to_global`) & Global Residual Update:** Tránh lỗi nối thừa cột $W$ nếu áp dụng trực tiếp công thức ghép khối Eq. (9) cho từng minibatch; giữ số cột luôn bằng đúng $C_k$ (`src/analytic.py`, `src/cifar_data.py`).
  2. **Cộng dồn Base theo batch (`initialize_batches`):** Tích lũy Gram/Cross từng batch + giải Cholesky giúp chạy $d=8000$ mà không cần cấp phát mảng $H_0$ cỡ $25{,}000\times 8000$ ($\approx 1.6\text{ GB}$), tách riêng Joint Oracle khỏi luồng benchmark (`src/streaming.py`).
  3. **Tách biệt từng lượt chạy (`attempt_id` + `filelock`):** Mỗi lần chạy ghi vào thư mục riêng; trạng thái `running / complete / failed` cập nhật nguyên tử, tránh đọc nhầm file JSON của lượt chạy trước khi lượt mới bị lỗi (`src/run_attempt.py`).
  4. **Checkpoint v2 hỗ trợ Resume trên Kaggle:** Lưu đủ `model`, `optimizer`, `scheduler`, `completed_epoch`, trạng thái RNG của Python/NumPy/Torch CPU+CUDA và `DataLoader` generator để nối tiếp chính xác giữa các phiên (`src/base_training.py`, `src/verify_resume.py`).
  5. **Đối chiếu Summary W&B cho R3 (`reconcile_r3_tracking.py`):** Giữ nguyên JSON gốc chụp offline; bổ sung trường `r3/learning_seconds_total` & `r3/max_abs_score_error_all_phases` để không nhầm giá trị pha cuối với tổng cả chuỗi, kèm kiểm tra SHA-256 (`UTF-8/LF`).
- **Footer IEEE:** `Nội bộ nhóm (Chi tiết triển khai trong src/ và reports/BAO_CAO_THUC_NGHIEM_GIUA_KY_20261107.md).`
- **Speaker Notes (~190 từ):**
  > Slide 31 điểm lại năm quyết định kỹ thuật quan trọng mà nhóm đã thực hiện khi tự xây dựng pipeline thực nghiệm. Thứ nhất là việc chuẩn hóa ánh xạ nhãn về ID toàn cục và dùng công thức cập nhật phần dư để chạy đúng với mọi kích thước minibatch. Thứ hai là thiết kế cộng dồn ma trận Gram theo từng lô ở Phase 0 để kiểm soát mức RAM dưới ngưỡng cho phép của Kaggle khi chiều mở rộng lên tới $d=8000$. Thứ ba là cơ chế quản lý lượt chạy bằng `attempt_id` đi kèm `filelock`, giúp trường hợp một lượt chạy giữa chừng bị lỗi sẽ ghi nhận rõ trạng thái `failed` thay vì để lại kết quả cũ dễ gây nhầm lẫn. Thứ tư là định dạng checkpoint v2 lưu toàn bộ trạng thái bộ sinh số ngẫu nhiên phục vụ việc chạy nối tiếp qua giới hạn thời gian phiên Kaggle. Thứ năm là bước kiểm tra hậu kỳ trên W&B: khi nhận thấy `wandb sync` mặc định lấy giá trị log ở bước cuối làm summary cho R3, nhóm giữ nguyên file JSON gốc và viết script bổ sung các khóa tổng hợp tường minh kèm unit test kiểm tra tự động.

---

### Slide 32 — Tổng kết Giữa kỳ & Kế hoạch Giai đoạn Cuối kỳ
- **Tiêu đề trên slide:** Tổng kết kết quả Giữa kỳ trong Core Scope & Hai hướng phân tích tiếp theo cho Cuối kỳ
- **Bố cục trực quan:** Trái: Tóm tắt các kết quả đã có bằng chứng kèm phạm vi giới hạn; Phải: Hai hướng phản biện & thực nghiệm cho giai đoạn Poster Cuối kỳ.
- **Nội dung trên slide:**
  - **Các nội dung đã hoàn thành ở mốc Giữa kỳ:**
    - Đọc hiểu bài báo ACIL (phân tích phương pháp 5 lớp, chứng minh & điều kiện biên Định lý 3.1, đối chiếu 3 hướng tiền nhiệm, tự vẽ 2 sơ đồ + 5 biểu đồ kết quả).
    - Thực thi đầy đủ **Core Scope (`CIFAR-100 / K=5`)**: **R1** ($\bar{A}=65.12 \pm 0.61\%$), **R2** (FE $+11.86\text{ pp}$, ổn định theo $\gamma$), **R3** (54/54 mốc khớp Joint $\le 1.16\times 10^{-11}$), **Author Compare C1/C2/C1R** — công khai trên GitHub & W&B.
  - **Các phạm vi chưa kiểm chứng thực nghiệm:** Chưa chạy trên ImageNet, $K=25, 50$ hay hàng $\gamma=0$; chưa tự chạy lại các baseline lưu mẫu; tính chất *exemplar-free* không đồng nhất với *Differential Privacy*.
  - **Kế hoạch cho giai đoạn Cuối kỳ (Critical Analysis & Poster):**
    - Hoàn thiện đánh giá điểm mạnh / hạn chế lý thuyết và kiểm tra thực nghiệm cho 2 hướng:
      1. **Hướng P1 (Sự phụ thuộc vào tập lớp Base đóng băng):** Khảo sát mức biến động chất lượng của ACIL khi thay đổi quy mô hoặc phân phối của tập lớp ở Phase 0.
      2. **Hướng P2 (Chi phí bộ nhớ $\mathcal{O}(d^2)$ của ma trận $R_k$):** Phân tích đánh đổi dung lượng lưu trữ thực tế (`~522.5 MB` ở $d=8000$ `float64`) khi giảm chiều mở rộng $d$ hoặc đặt cùng ngân sách bộ nhớ.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~190 từ):**
  > Slide 32 tổng kết lại phần trình bày chính của nhóm ở giai đoạn giữa kỳ. Trong phạm vi Core Scope trên CIFAR-100 với $K=5$, nhóm đã hoàn thành cả phần đọc hiểu lý thuyết lẫn bốn bộ thực nghiệm R1, R2, R3 và chạy đối chiếu mã nguồn tác giả, với đầy đủ dữ liệu kiểm chứng trên GitHub và Weights & Biases. Nhóm cũng nêu rõ những giới hạn chưa thực hiện — như chưa chạy ở kịch bản nhiều pha $K=25, 50$, chưa thử trên ImageNet và chưa chạy lại các phương pháp replay — để giữ mọi nhận định nằm đúng trong phạm vi bằng chứng hiện có. Ở chặng cuối kỳ sắp tới, nhóm dự kiến tập trung vào phần phân tích phản biện và thực nghiệm bổ sung theo hai hướng đã lên kế hoạch: thứ nhất (P1) là kiểm tra mức độ phụ thuộc của toàn bộ chuỗi tăng dần vào chất lượng biểu diễn học từ tập lớp Base ban đầu khi CNN bị đóng băng; thứ hai (P2) là khảo sát đánh đổi giữa chiều mở rộng $d$, chi phí bộ nhớ $\mathcal{O}(d^2)$ của ma trận $R_k$ và độ chính xác phân loại. Bốn slide tiếp theo là phần Phụ lục kỹ thuật.

---

## PHẦN 4: APPENDIX KỸ THUẬT & TRUY NGUYÊN (SLIDES 33–36 — KHÔNG PHẢI REFERENCES SLIDE)

### Slide 33 — Appendix A1: Chứng minh Quy nạp Định lý 3.1 & Tương đương Công thức Residual
- **Tiêu đề trên slide:** Appendix A1: Biến đổi đại số của Định lý 3.1 và Công thức Residual theo Minibatch
- **Bố cục trực quan:** Trái: Các bước biến đổi quy nạp ma trận khối $k-1 \to k$; Phải: Kiểm tra tính tương đương giữa dạng ghép khối Eq. (9) và dạng Residual toàn cục trong `src/analytic.py`.
- **Nội dung trên slide:**
  - **1. Bước quy nạp $k-1 \to k$ cho Định lý 3.1 [1]:**
    - Giả sử $R_{k-1} = (\tilde{H}_{k-1}^T \tilde{H}_{k-1} + \gamma I_d)^{-1}$ và $W_{k-1} = R_{k-1} \tilde{H}_{k-1}^T \tilde{Y}_{k-1}$.
    - Vì $\tilde{H}_k = [\tilde{H}_{k-1}^T, H_k^T]^T \implies R_k^{-1} = R_{k-1}^{-1} + H_k^T I_{N_k} H_k$. Áp dụng bổ đề Woodbury cho $(A + U C V)^{-1}$ thu được Eq. (10) cho $R_k$.
    - Do $\tilde{Y}_k = \operatorname{diag}(\tilde{Y}_{k-1}, Y_k)$, tách cột $\tilde{W}_k = R_k \tilde{H}_k^T \tilde{Y}_k = \big[ R_k \tilde{H}_{k-1}^T \tilde{Y}_{k-1} \;\; R_k H_k^T Y_k \big]$.
    - Từ $R_{k-1}^{-1} = R_k^{-1} - H_k^T H_k \implies I_d = R_k R_{k-1}^{-1} + R_k H_k^T H_k \implies R_k = R_{k-1} - R_k H_k^T H_k R_{k-1}$.
    - Nhân vào $\tilde{H}_{k-1}^T \tilde{Y}_{k-1}$ được khối trái: $W_{k-1} - R_k H_k^T H_k W_{k-1}$ $\implies$ khớp Eq. (9) $\blacksquare$.
  - **2. Đối chiếu dạng Global Residual ($W_{\text{new}} = W_{\text{pad}} + R_k H_b^T (Y_{b,\text{global}} - H_b W_{\text{pad}})$):**
    - Khi đệm cột cho $c_k$ lớp mới ($W_{\text{pad}} = [W_{\text{old}}, \mathbf{0}]$) và dữ liệu batch thuộc các lớp mới ($Y_{b,\text{global}} = [\mathbf{0}, Y_b]$):
      - **Khối cột cũ:** $W_{\text{old}} + R_k H_b^T(\mathbf{0} - H_b W_{\text{old}}) = W_{\text{old}} - R_k H_b^T H_b W_{\text{old}}$.
      - **Khối cột mới (batch đầu):** $\mathbf{0} + R_k H_b^T(Y_b - \mathbf{0}) = R_k H_b^T Y_b$.
    - Ở các minibatch kế tiếp trong cùng phase $k$, các cột lớp mới giữ nguyên chỉ số nên tiếp tục cập nhật đúng theo RLS.
- **Footer IEEE:** `[1] H. Zhuang et al., "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection," NeurIPS, 2022.`
- **Speaker Notes (~185 từ):**
  > Slide Phụ lục A1 ghi lại đầy đủ các dòng biến đổi đại số cho Định lý 3.1 và công thức cập nhật phần dư đã tóm tắt ở Slide 13 và Slide 14, phòng trường hợp cần trao đổi sâu hơn ở phần hỏi đáp. Ở cột bên trái, bước biến đổi quan trọng nhất trong chứng minh quy nạp là xuất phát từ hệ thức $R_{k-1}^{-1} = R_k^{-1} - H_k^T H_k$: khi nhân bên trái hai vế với $R_k$ và nhân bên phải với $R_{k-1}$, ta rút ra được $R_k = R_{k-1} - R_k H_k^T H_k R_{k-1}$. Thay biểu thức này vào khối cột của các lớp cũ trong $\tilde{W}_k$, tích $R_{k-1} \tilde{H}_{k-1}^T \tilde{Y}_{k-1}$ thu gọn về đúng $W_{k-1}$, cho thấy không cần lưu lại $\tilde{H}_{k-1}$ hay $\tilde{Y}_{k-1}$ mà vẫn tính được trọng số mới. Ở cột bên phải là phép tách khối xác nhận công thức phần dư toàn cục trong `src/analytic.py` cho kết quả trùng với Phương trình (9) ở batch đầu và tự động cộng dồn đúng cho các batch tiếp theo trong cùng một phase.

---

### Slide 34 — Appendix A2: Bảng Truy nguyên Toàn bộ Các Lượt Chạy trên W&B & GitHub
- **Tiêu đề trên slide:** Appendix A2: Bảng ánh xạ toàn bộ các lượt thực nghiệm (31 Runs W&B Public)
- **Bố cục trực quan:** Bảng tổng hợp ánh xạ `Nhóm thí nghiệm ↔ Notebook / Script ↔ W&B Run IDs ↔ Kết quả chính` (đồng bộ với bảng trong `README.md`).
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
- **Footer IEEE:** `Nội bộ nhóm (Các đường dẫn clickable chi tiết xem tại bảng Experiment Mapping trong README.md và reports/README.md).`
- **Speaker Notes (~160 từ):**
  > Slide Phụ lục A2 tổng hợp toàn bộ 31 lượt chạy đã lưu trên project Weights & Biases công khai `hsangg-2006/acil-reproduction` vào chung một bảng để tiện tra cứu chéo giữa slide, W&B và GitHub. Mỗi nhóm số liệu trình bày trong báo cáo đều gắn với file notebook hoặc script tạo ra kết quả tương ứng, đi từ các bước kiểm tra ban đầu (`VERIFY-01/02`, `CIFAR-KAGGLE-01/02`, `gey0yop4`, `pv82dy1f`) đến 21 lượt chạy của ba bộ R1, R2, R3 và 3 lượt chạy đối chiếu code tác giả (`C1, C2, C1R`). Ngoại trừ lượt `0tlcy6id` của C1 giữ cờ `equivalence = false` do 2 phần tử điểm số vượt nhẹ ngưỡng `allclose` như đã giải thích ở Slide 29, tất cả các lượt chạy đều hoàn tất ở trạng thái Finished, truy cập công khai không cần đăng nhập và có file manifest lưu mã băm SHA-256 trong thư mục `reports/`.

---

### Slide 35 — Appendix A3: Tổ chức Mã nguồn, Unit Tests & Giải thích Thư viện Sử dụng
- **Tiêu đề trên slide:** Appendix A3: Cấu trúc 14 module tự viết (`src/`), 65 Unit Tests & các thư viện ngoài Standard Stack
- **Bố cục trực quan:** Chia 2 cột: Trái là nhóm chức năng của 14 file trong `src/` + số lượng unit test; Phải là bảng ghi nhận thành phần dùng lại & lý do dùng từng thư viện ngoài `numpy, scipy, torch, scikit-learn`.
- **Nội dung trên slide:**
  - **Cấu trúc `src/` (có đầy đủ docstring mô tả mục đích, tham số và giá trị trả về cho các hàm/lớp):**
    - *Toán học giải tích & Kiểm chứng:* `analytic.py`, `verification.py`, `verify_analytic.py`.
    - *Dữ liệu & Backbone:* `cifar_data.py`, `resnet.py` (tự viết theo kiến trúc bài báo [6]).
    - *Training, Resume & Pipeline:* `base_training.py`, `verify_resume.py`, `cifar_smoke.py`, `streaming.py`, `profile_analytic.py`.
    - *Quản lý lượt chạy, Metric & Notebook:* `metrics.py`, `run_attempt.py`, `reporting.py`, `notebook_bundle.py`.
    - **Bộ kiểm thử tự động trên máy (CPU):** `tests/`: **54 tests** (53 passed, 1 skipped CUDA test — đã chạy pass trên GPU Kaggle); `scripts/tests/`: **11/11 passed**.
  - **Thành phần dùng lại & Thư viện ngoài Standard Stack (`README.md`):**
    - *Thành phần học thuật kế thừa:* Kiến trúc ResNet-32 Option A [6], tập dữ liệu CIFAR-100 (`uoft-cs/cifar100`) [7], Protocol chia lớp 50%+K [4], [5], Repo tác giả `0afd6bf` (chỉ chạy để đối chiếu).
    - *Thư viện bổ trợ:* `wandb` (ghi log thực nghiệm công khai), `huggingface_hub` + `pyarrow` + `Pillow` (tải/đọc Parquet CIFAR-100), `filelock` (khóa file trạng thái giữa các lần chạy), `psutil` + `threadpoolctl` (đo Peak RSS & giới hạn 4 luồng CPU), `matplotlib` (vẽ 7 hình trong `reports/figures/`), `nbformat/nbclient/ipykernel` (kiểm tra & chạy notebook), `tqdm/PyYAML/torchvision` (chỉ cài ở venv riêng khi chạy code tác giả).
- **Footer IEEE:** `[4] A. Douillard et al., ECCV, 2020. [5] Y. Liu et al., CVPR, 2021. [6] K. He et al., CVPR, 2016. [7] A. Krizhevsky, Tech. Rep., 2009.`
- **Speaker Notes (~185 từ):**
  > Slide Phụ lục A3 tóm tắt cấu trúc mã nguồn và bộ kiểm thử tự động trong repository `20-ML-Paper`. Cả 14 module trong thư mục `src/` đều do nhóm tự viết từ mô tả thuật toán trong bài báo mà không sao chép mã nguồn của tác giả; phần code của tác giả chỉ được checkout ở một thư mục ngoài và chạy trong môi trường ảo tách biệt (`requirements-author-comparison.txt`) để phục vụ đối chiếu. Các hàm và phương thức trong `src/` đều được viết docstring ghi rõ chức năng, tham số đầu vào và đầu ra, đi kèm 54 unit test cho phần core và 11 unit test cho các script tổng hợp. Về thư viện phụ thuộc, bên cạnh các thư viện tính toán quen thuộc (`numpy`, `scipy`, `torch`, `scikit-learn`), những gói cài thêm đều được tách riêng theo mục đích trong từng file `requirements-*.txt`: ví dụ `huggingface_hub` và `pyarrow` để tải dữ liệu tự động có kiểm tra mã băm, `psutil` và `threadpoolctl` để đo bộ nhớ RSS và khóa số luồng BLAS trong R3, hay `matplotlib` chỉ dùng cho bước xuất biểu đồ.

---

### Slide 36 — Appendix A4: Các Lệnh Tái tạo Kết quả & Danh mục Kiểm tra Trước khi Nộp
- **Tiêu đề trên slide:** Appendix A4: Hướng dẫn chạy kiểm tra lại bằng CLI & Checklist rà soát sản phẩm Giữa kỳ
- **Bố cục trực quan:** Trái: Các lệnh chạy unit test, tính lại bảng tổng hợp và vẽ lại hình từ repo (`README.md`); Phải: Bảng rà soát các yêu cầu định dạng trước khi xuất file nộp.
- **Nội dung trên slide:**
  - **Các bước kiểm tra & tái tạo nhanh sau khi clone `https://github.com/24122050S/20-ML-Paper`:**
    1. Chạy bộ unit test kiểm tra công thức & pipeline: `pip install -r requirements-verification.txt` (hoặc `requirements-cifar.txt`) $\to$ `python -m unittest discover -s tests -v` và `scripts/tests`.
    2. Tính lại các bảng tổng hợp `mean ± SD` từ JSON gốc: `python scripts/aggregate_r1.py`, `aggregate_r2.py`, `aggregate_r3.py`.
    3. Vẽ lại đủ 7 hình PNG (`requirements-plots.txt`): `python scripts/plot_r1.py`, `plot_r2.py`, `plot_r3.py`, `plot_author_comparison.py`, `plot_acil_diagrams.py`.
    4. Các file manifest `reports/*-provenance.json` lưu mã kiểm tra SHA-256 trên chuẩn `UTF-8/LF`.
  - **Checklist rà soát trước khi nộp bài Giữa kỳ:**
    - `[x]` **Số lượng slide & dung lượng:** **36 slides** (nằm trong khoảng 20–40 slides); dung lượng dự kiến **$< 5\text{ MB}$** (giới hạn $< 20\text{ MB}$).
    - `[x]` **Speaker Notes:** Có trên đủ **36/36 slides**, độ dài khoảng $155\text{–}210$ từ/slide ($\le 500$ từ/slide).
    - `[x]` **Hình ảnh & Code:** 7 hình do nhóm tự vẽ; không cắt hình từ PDF bài báo; không chép code dài lên slide.
    - `[x]` **Trích dẫn IEEE:** Đặt tại footer của từng slide có dùng nguồn; **không tạo slide References riêng**.
    - `[x]` **Quyền truy cập công khai:** Kiểm tra file `.pptx`, link GitHub Repo Public, project W&B Public và chế độ Public của các notebook Kaggle `01`–`12`.
- **Footer IEEE:** `Nội bộ nhóm (Hướng dẫn chi tiết xem tại README.md và docs/KICH_BAN_36_SLIDE_GIUA_KY.md).`
- **Speaker Notes (~170 từ):**
  > Slide Phụ lục cuối cùng tổng hợp các câu lệnh dòng lệnh để người đọc có thể tự kiểm tra lại kết quả từ repository công khai của nhóm. Sau khi tải repo `20-ML-Paper` về máy, có thể chạy ngay toàn bộ 65 unit test, chạy lại các script `aggregate_r1/r2/r3.py` để kiểm tra các con số trung bình và độ lệch chuẩn từ file JSON gốc, cũng như vẽ lại cả 7 hình minh họa và biểu đồ bằng các script trong `scripts/` mà không cần chạy lại bước huấn luyện 160 epochs trên GPU. Khung bên phải là danh mục nhóm dùng để tự rà soát lại file thuyết trình trước khi nộp lên hệ thống: đảm bảo tổng độ dài 36 slides, mỗi slide đều có phần ghi chú trình bày dưới 500 từ, toàn bộ hình ảnh tự vẽ, và các trích dẫn IEEE được đặt ở phần chân của từng slide sử dụng tài liệu thay vì dồn vào một slide References ở cuối. Nhóm em xin cảm ơn Thầy/Cô và các bạn đã lắng nghe.
