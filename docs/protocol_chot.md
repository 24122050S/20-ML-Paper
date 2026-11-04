# Protocol benchmark ACIL v1.0

Chốt ngày **04/11/2026** theo yêu cầu của Sang. Người thực hiện hiện tại: **Sang**. Đây là quyết định về thiết kế thí nghiệm; bảng nghiệm thu ở mục 8 ghi phần code còn phải hoàn thành trước khi chạy dài. Mọi thay đổi sau mốc này phải ghi phiên bản/lý do trước khi chạy, không chọn cấu hình theo accuracy test.

## 1. Phạm vi và câu hỏi

Chọn **core scope** theo [yêu cầu reproduction của môn](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction/): R1 là phép so sánh chính, R2 là thí nghiệm thành phần quan trọng, R3 là phân tích bổ sung do nhóm thiết kế. Thu hẹp về CIFAR-100 vì nguồn lực Kaggle và một người đang thực hiện; không tuyên bố đã tái lập toàn bộ các bộ dữ liệu/K của paper.

| ID | Thiết kế bắt buộc | Câu hỏi / đầu ra |
|---|---|---|
| R1 | CIFAR-100, 50 base + 5×10 lớp, ACIL d=8000, gamma=0.1; 3 seed | Đường cong 6 phase, A-bar/F-base đối chiếu Table 1; phân tích sai lệch |
| R2 | Cùng checkpoint R1 mỗi seed; 4 hàng của Table 2: no-FE/gamma0.1; FE8000/gamma0.1,0.01,0.001 | Tác động của FE và độ nhạy gamma; paired differences theo seed |
| R3 | Checkpoint benchmark seed42, K=5, d=1024, gamma0.1; batch32/128/512; 3 lượt đo thời gian mỗi batch | Minibatch có đổi nghiệm không, và đánh đổi thời gian/bộ nhớ ra sao; joint oracle chạy ngoài vùng đo |

R3 dùng d=1024 để giữ chi phí oracle và sweep trong ngân sách; báo rõ đây là cấu hình phân tích riêng. Không dùng số R3 như benchmark d=8000. K=25 là mở rộng ưu tiên sau core, cần runner hỗ trợ; K=10/50 và ImageNet nằm ngoài core đã chốt. Vì paper nhấn mạnh large-K, mọi kết luận nhóm đã tái lập ưu thế large-K phải chờ run large-K thật.

R2 **không bao gồm hàng gamma=0** trong phạm vi bắt buộc: module hiện yêu cầu gamma>0; gamma=0 cần thiết kế solver/điều kiện hạng riêng. Không thay bằng epsilon hay gọi pinv là cùng thuật toán RLS hiện tại. Khi báo cáo phải ghi “4/5 hàng Table 2”, không ghi đã tái lập cả bảng.

## 2. Nguồn quyết định và số tham chiếu

- [Paper NeurIPS 2022, mục 4.1–4.2 và Table 1, trang PDF 7](https://proceedings.neurips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf#page=7): backbone/base training, protocol 50%, K và metric. Caption quy định 1k=1000, nên 8k được chốt là **8000**.
- [Mục 4.4/Table 2, trang PDF 9](https://proceedings.neurips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf#page=9): các hàng ablation. Mục 3.1/phương trình (2) cho FE dùng ReLU và ma trận ngẫu nhiên phân phối chuẩn.
- [Repo tác giả tại commit 0afd6bf](https://github.com/ZHUANGHP/Analytic-continual-learning/tree/0afd6bf87622e808ce7f730f8556e78229b9e758) chỉ dùng để đối chiếu, không chép implementation cốt lõi vào repo nhóm. [README ở commit này](https://github.com/ZHUANGHP/Analytic-continual-learning/blob/0afd6bf87622e808ce7f730f8556e78229b9e758/README.md) đã dùng recipe mới: 300 epochs, cosine, LR0.5/batch256, buffer8192 và augmentation bổ sung. Không lấy bảng benchmark mới của README thay Table 1 năm 2022.
- [Smoke full5 đã có](../reports/cifar100_kaggle_full5_results.json), [manifest](../reports/CIFAR-KAGGLE-02-provenance.json): kiểm đường đi dữ liệu, checkpoint/freeze, analytic và tracking. 22,05% là accuracy cuối của 2 epochs/d256; không phải A-bar benchmark.

| Mục tiêu paper | A-bar (%) | F trong bảng (%) | Phạm vi đối chiếu |
|---|---:|---:|---|
| Table 1 ACIL, CIFAR-100, K=5 | 66.30 | 9.00 | R1; phải ghi rõ quy ước dấu theo [metrics](metrics.md) |
| Table 1 ACIL, CIFAR-100, K=25 | 65.95 | 9.28 | Chỉ khi mở rộng và có run thật |
| Table 2 no-FE, gamma0.1 | 52.99 | Không báo trong Table 2 | R2 |
| Table 2 FE8000, gamma0.1 / 0.01 / 0.001 | 66.30 / 66.25 / 66.23 | Không báo trong Table 2 | R2 |

Không đặt mục tiêu “đạt 66,30% mới được giữ run”. Số tham chiếu phục vụ đo sai lệch; mọi run đã đăng ký phải được lưu, kể cả thấp hơn paper.

Bảng R1 phải đặt ACIL của nhóm cạnh ACIL của paper và có deviation cho cùng metric/setting. Nếu thêm các baseline khác của Table 1 để thể hiện bối cảnh so sánh chính, ghi rõ đó là số trích paper; chưa có run baseline của nhóm thì không gắn nhãn đã tái lập baseline.

## 3. Dữ liệu, nhãn và seed

| Thành phần | Giá trị chốt | Cơ sở |
|---|---|---|
| Dataset | CIFAR-100 fine labels, RGB 32×32; train50000/test10000; mỗi lớp500/100 ảnh | Paper và loader hiện có |
| Nguồn | HF `uoft-cs/cifar100`, revision `aadb3af77e9048adbea6b47c21a81e47dd092ae5` | Đã chạy smoke; tải/cache tự động |
| Chia lớp | `class_order = np.random.default_rng(seed).permutation(100)` | Quy ước nhóm đã kiểm; lưu permutation thực tế |
| Nhãn toàn cục | Vị trí trong class_order, cùng mapping cho train/test | Base0..49; phase1..5 lần lượt50..59,...,90..99 |
| Seed độc lập của R1/R2 | 42,43,44 | 3 runs; paper cũng báo trung bình 3 runs nhưng không nêu cùng bộ seed này |
| Chính sách RNG | Cùng seed cho Python/NumPy/Torch và class order; FE dùng `seed+1000` | Tương thích code hiện tại; không giả vờ các seed chỉ khác khởi tạo backbone |
| Thứ tự trong analytic | Cố định theo chỉ số mẫu của từng phase, không shuffle/augment | R3 chỉ đổi cách chia batch, không đổi thứ tự mẫu |
| Normalization | Mean/std tính từ **base train của chính seed**, rồi freeze | Quyết định nhóm; không dùng test/future train để fit thống kê |

Với một seed, R1/R2/R3 có liên quan phải dùng đúng checkpoint, class_order, normalization, revision và FE matrix đã pin. Với seed khác, class order và representation có thể khác; SD giữa 3 seed bao gồm các nguồn biến thiên đó. Không lấy backbone seed42 dùng cho mapping seed43.

Giữ test split gốc chỉ để đánh giá/report. Không chọn epoch, gamma, d hay seed tốt nhất theo test. Chọn checkpoint **epoch cuối 160**; nếu cần tuning về sau, tạo protocol validation từ train riêng và nâng phiên bản trước khi chạy.

## 4. Base training chốt theo paper

Schedule/SGD/batch chính lấy từ paper; shortcut, seed, thống kê normalization, dtype và threads là các quy ước nhóm được ghi riêng ở mục 3 và 6.

| Tham số | Giá trị |
|---|---|
| Backbone | ResNet-32 CIFAR của nhóm: n=5, kênh16/32/64, shortcut option A, feature64 |
| Lớp train / head BP | 50 lớp base / FC50 logits, CrossEntropyLoss, không label smoothing |
| Epochs | **160** |
| Optimizer | SGD, learning_rate0.1, momentum0.9, weight_decay0.0005, Nesterov=False |
| Scheduler | MultiStepLR milestones[80,120], gamma0.1; step sau mỗi epoch |
| LR theo cách gọi hiện tại | Epoch1..80:0.1; 81..120:0.01; 121..160:0.001; log LR thực tế |
| Batch / workers / threads | Train128; eval256; num_workers0; torch/BLAS threads4 |
| Augmentation BP | Zero padding4 → random crop32; horizontal flip p=0.5; rồi normalize |
| ARaBT/incremental/test | Không crop/flip ngẫu nhiên; cùng normalize đã freeze |
| Dtype / thiết bị | CNN float32 trên một GPU; analytic NumPy float64 trên CPU |
| Trạng thái sau BP | Reload checkpoint, kiểm logits, freeze toàn bộ tham số và `eval()` mọi BatchNorm |

Classifier BP chỉ phục vụ huấn luyện representation. A0 trong bảng kết quả được đo **sau analytic re-alignment**, không dùng accuracy của FC BP để thay A0. Không train lại CNN ở incremental phase.

Lưu ý nguồn: paper mục 4.1 ghi crop/flip/normalize cho BP. README tác giả hiện tại có mô tả augmentation khác và có câu về thí nghiệm paper không dùng augmentation; khi có mâu thuẫn này, protocol nhóm ưu tiên mô tả trực tiếp ở paper trang 7 và ghi sai khác, không ghép recipe mới vào recipe 160 epochs.

## 5. Analytic head và ablation

- R1/R2 có FE: `H = ReLU(F @ W_fe)`, W_fe shape64×8000, không bias, giữ cố định; nhóm chọn **N(0,1)** như code smoke. Paper nêu normal distribution nhưng không chốt đầy đủ mean/std trong đoạn đó; N(0,1) là quyết định nhóm được ghi nhận, không suy từ default của repo mới.
- Không thêm L2 normalization của feature, softmax hay head theo task. Dự đoán là argmax trên **tất cả lớp đã thấy**.
- Objective: `||Y-HW||_F^2 + gamma*||W||_F^2`, dạng **sum**, gamma0.1 của R1 chỉ cộng khi khởi tạo. Không scale gamma theo số ảnh/minibatch hay cộng lại ở từng batch.
- Base analytic: xử lý đủ25000 ảnh base sạch. Benchmark cần tích lũy thống kê theo batch hoặc cách giải tương đương đã kiểm, tránh giữ H25000×8000 chỉ để khởi tạo.
- Mỗi phase mới:5000 ảnh, batch analytic128, batch cuối8 ảnh;40 minibatch. Cột W tăng theo lớp, không tăng theo số batch. R luôn8000×8000, W cuối8000×100.
- No-FE của R2: dùng **F64 trực tiếp làm H**, không random projection64→64 để gọi là bỏ FE. Dùng cùng backbone/normalization/gamma0.1 và quy tắc đánh giá.
- Ba hàng gamma của R2 phải dùng **chính một FE matrix** trong cùng seed. Giữ dataset/order/checkpoint/dtype cố định; chỉ đổi gamma. No-FE giữ nguyên F64 của checkpoint đó.
- Benchmark chính không giữ feature lịch sử để giải joint mỗi phase. Joint oracle dùng ở verification/R3 riêng; field không được kiểm phải ghi `not_run`/null, không điền0 như thể đã so khớp.

## 6. Sai khác đã biết và đối chiếu code tác giả

| Điểm | Quyết định nhóm | Ý nghĩa khi phân tích sai lệch |
|---|---|---|
| Class order/seed | NumPy default_rng theo seed42/43/44 | Chưa khẳng định trùng permutation của paper |
| Normalize/shortcut | Base-train statistics, option A của nhóm | Chưa xác nhận trùng nguyên bản paper; ghi trong bảng sai khác |
| Kích thước FE | 8000 theo caption paper | Repo tác giả mới dùng8192; phải gắn đúng source cho số so sánh |
| Khởi tạo FE | N(0,1) không bias | [RandomBuffer mới](https://github.com/ZHUANGHP/Analytic-continual-learning/blob/0afd6bf87622e808ce7f730f8556e78229b9e758/analytic/Buffer.py) gọi reset_parameters của Linear; không suy rằng giống normal initialization của nhóm |
| Training | Recipe160/milestone từ paper | Repo mới300/cosine/augmentation khác; score README mới không phải score Table 1 |
| Precision | BP32, analytic64 | Ghi dtype và tolerance; không bảo đảm bitwise giữa môi trường |

Phần đối chiếu code gốc là đầu ra bắt buộc của reproduction: chạy source tác giả trong môi trường riêng, pin commit/config/checkpoint, lưu output và W&B. Không ghi “đã chạy code gốc” chỉ vì đã đọc. So sánh theo tầng: mapping → preprocessing → feature → FE → analytic → metric; chỉ quy nguyên nhân cho tầng có bằng chứng. Hiện chưa có artifact chạy code gốc trong repo nhóm.

## 7. Ngân sách và thứ tự thực hiện

R1/R2 tái dùng **mỗi checkpoint160 epochs cho cùng seed**; không train thêm backbone cho từng gamma hay batch size. Thực hiện seed42 trước để kiểm tài nguyên, rồi43/44; mọi seed đã chạy đều có link và trạng thái.

Với float64 và d=8000: R=512.000.000 bytes; W100=6.400.000; FE64×8000=4.096.000. Tổng ba ma trận=**522.496.000 bytes**, chưa gồm CNN, solver workspace, dữ liệu/cache. Giữ toàn bộ H50000×8000 còn tốn3.200.000.000 bytes; vstack và joint reference có thể tạo nhiều bản sao. Các số này là kích thước ma trận, không phải peak RAM đã đo.

Hai epoch của run blhw74ff mất tổng34,3874 giây cho BP; ngoại suy tuyến tính160 epochs ≈45,85 phút/seed **chỉ cho BP trên môi trường lượt cũ**. Đây không phải dự báo tổng runtime d8000. Trước chạy dài, profile BP5 epochs và analytic d8000 theo batch trên máy/phiên Kaggle thực tế; ghi RAM/GPU/time/quota còn lại. Chỉ mở lượt dài khi ước lượng có dự phòng ít nhất30% tài nguyên phiên/thời gian còn lại; vượt ngưỡng thì tối ưu implementation hoặc sửa scope có phiên bản, không âm thầm đổi d/gamma.

## 8. Nghiệm thu trước khi mở benchmark

| Gate | Điều kiện đạt | Hiện trạng ở main f8bad6e |
|---|---|---|
| G1: dữ liệu/numerical smoke | Audit, mapping, reload/freeze, K5 đủ100 lớp và tracking | Đã có ở d256/2 epochs, blhw74ff |
| G2: resume BP | Lưu model, optimizer, scheduler, completed_epoch, target_epochs, config/protocol/norm/revision, Python/NumPy/Torch CPU+CUDA RNG, DataLoader generator và history; resume đúng ở biên epoch, giữ mục tiêu160 epochs | Chưa có; reuse checkpoint hoàn tất không phải resume |
| G3: kiểm resume | Run ngắn liên tục2 epochs so với1+resume1: cùng sample order, LR, loss/state trên môi trường kiểm tra; thử checkpoint thiếu/khác config | Cần triển khai và kiểm trước160 epochs |
| G4: benchmark analytic | Base theo batch; không giữ lịch sử/joint oracle trong đường benchmark; no-FE đúng; metadata mode rõ | Runner smoke hiện giữ pooled features/joint; cần mode riêng |
| G5: metric export | A0..AK/B0..BK đầy đủ, số mẫu đúng; tính A-bar/F theo metrics; không lấy final thay A-bar | Công thức đã chốt ở tài liệu, script export còn cần triển khai |
| G6: profile/provenance | Source commit/hashes, config, checkpoint hashes, package/device, peak memory/time, version Kaggle và public W&B | Cần run profile thật; smoke chưa đo d8000 |

**Việc làm ngay sau khi chốt tài liệu:** triển khai G2/G3, sau đó G4/G5 và profile G6. Chưa mở run160 epochs bằng cách chỉ sửa epochs/d trong notebook03. Notebook03 hiện phân phối smoke runner; notebook benchmark mới phải có source bundle riêng, giữ nguồn đã công bố của các run cũ.

Mỗi kết quả benchmark phải lưu config, protocol/class_order, normalization, checkpoint/state, per-phase JSON và summary metrics; README ghép đúng source/JSON/version/W&B cùng lượt. Xuất UTF-8/LF, tính SHA-256 trên bytes công bố. [metrics.md](metrics.md) là định nghĩa duy nhất của các metric dùng trong bảng/deck.
