# Phân công ACIL: Sang, Vĩ, Phát

> Sang đang thực hiện một mình; hiện Sang thực hiện cả ba nhóm việc trong kế hoạch. Smoke100 đã có JSON/W&B. [Protocol v1.0](protocol_chot.md) và [metrics v1.0](metrics.md) đã chốt ngày 04/11; resume CPU/benchmark streaming/metric export và profile d8000 CPU đã có; analytic Kaggle d8000 đã có gey0yop4 finished; BP5/CUDA cũng đã chạy đạt qua API ngày05/11; việc tiếp theo là tracking BP5 và R1 bằng [notebook06](../notebooks/06_benchmark_seed42_ready.ipynb). Chưa chạy benchmark160 epochs.

Cập nhật **04/11/2026**, giờ Việt Nam, sau khi nhận JSON full5 và xác minh run W&B blhw74ff. Nền tảng chạy chính: **Kaggle**.
Hạn chính thức: PPTX **21/11/2026 lúc 23:59**, poster **05/12/2026 lúc 23:59**.
Mốc nộp nội bộ: **20/11** và **04/12**. Mốc trung gian dưới đây là kế hoạch nhóm đề xuất; cần điều chỉnh theo thời gian thực tế của cả ba.

Chưa có thông tin ai mạnh Python/toán hơn, nên đây là phân công mặc định để bắt đầu. Có thể đổi người phụ trách nếu cần; vẫn giữ một người chịu trách nhiệm chính cho mỗi đầu ra. Tên trong bảng là phân công đề xuất, không khẳng định các bạn đã nhận hoặc hoàn thành việc.

## 1. Chốt vai trò

| Thành viên    | Chịu trách nhiệm chính                                                                   | Sản phẩm khoa học                                                                                        | Người kiểm tra                                                                  |
| --------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Sang**  | Hiểu phương pháp, đặc tả, analytic head, kiểm tra tương đương joint/incremental | Công thức, điều kiện định lý, pipeline phương pháp, phân tích bổ sung R3                      | Vĩ kiểm implementation; Phát kiểm cách diễn giải kết quả                  |
| **Vĩ**   | Dataset/protocol, Kaggle, ResNet-32, checkpoint, chạy kết quả chính R1                   | Protocol tái lập, checkpoint có nguồn gốc, bảng/đường cong theo phase, đối chiếu code tác giả | Phát kiểm metric/run; Sang kiểm tương thích phương pháp                   |
| **Phát** | Đánh giá, ablation R2, tổng hợp bằng chứng, W&B, ghép sản phẩm                     | Ablation, sai lệch với paper, biểu đồ/bảng có run links, kiểm soát chất lượng deck/poster       | Sang kiểm tính đúng của ablation; Vĩ kiểm truy nguyên dữ liệu/checkpoint |

Mỗi người đều có phần đọc, code/thực nghiệm, phân tích và viết. Phát phụ trách ghép file sau khi Sang/Vĩ tự hoàn thành nội dung phần mình. Mỗi người phải giải thích được toàn bộ sản phẩm ở workshop.

## 2. Những gì đã có và những gì còn phải làm

| Hiện trạng                                            | Bằng chứng                                                                                                                                                                                                                                                                                       | Việc tiếp tục                                                                                                           |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Đã có analytic head dùng chung, hỗ trợ minibatch  | `src/analytic.py`, `src/verification.py`, notebook 01                                                                                                                                                                                                                                          | Sang hiểu và kiểm tra; tiếp tục thiết kế R3                                                                         |
| Pipeline CIFAR và ResNet-32 đã lên main qua PR #5   | [Loader/protocol](../src/cifar_data.py), [backbone](../src/resnet.py), [runner](../src/cifar_smoke.py), [config](../configs/cifar100_smoke.json)                                                                                                                                                       | Chốt protocol benchmark; runner hiện giới hạn K=5                                                                      |
| Lượt CPU smoke 50+10 đã có JSON trong repo         | [JSON CPU](../reports/cifar100_smoke_results.json); source dirty và W&B tắt tại thời điểm chạy                                                                                                                                                                                               | Giữ provenance lịch sử; checkpoint/state vẫn local, bị Git bỏ qua                                                    |
| Lượt Kaggle GPU 50+10 đã có tracking               | [Version 2 / 364942764](https://www.kaggle.com/code/hsangg206/notebook7e7946d4ee?scriptVersionId=364942764), [JSON](../reports/cifar100_kaggle_results.json), [W&amp;B o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2), [manifest](../reports/CIFAR-KAGGLE-01-provenance.json) | Mốc 2 epochs/d=256: base 32,58%, 60 lớp 29,70%; giữ mốc lịch sử riêng với lượt 100 lớp                       |
| Smoke CUDA đủ 100 lớp đã có tracking | [JSON](../reports/cifar100_kaggle_full5_results.json), [manifest](../reports/CIFAR-KAGGLE-02-provenance.json), [scriptVersionId 365084714](https://www.kaggle.com/code/hsangg206/03-cifar100-safe-attempts/output?scriptVersionId=365084714), [W&B blhw74ff](https://wandb.ai/hsangg-2006/acil-reproduction/runs/blhw74ff) finished | Đã đạt mốc kỹ thuật: W256×100, test 10.000, accuracy 22,05%; chốt protocol benchmark |
| Suite54 tests:53 đạt,1 skip CUDA trên CPU    | `tests/`: analytic, reporting gồm UTF-8/LF, CIFAR, run attempts và notebook distribution                                                                                                                                                                                                       | Chạy unittest khi đổi code; 11 tests của VERIFY-02 và 20 tests của lượt CPU là số lịch sử                      |
| Đã có đặc tả và bản giải thích minibatch      | `docs/Dac_ta_ACIL.md`, `docs/THAY_DOI_VERIFY02.md`                                                                                                                                                                                                                                             | Sang tự giải thích, Vĩ/Phát kiểm chéo                                                                               |
| Project W&B đã Public                                 | `hsangg-2006/acil-reproduction`, VERIFY-01 run `7aeaupq7`; VERIFY-02 run `wi4wau2d` Finished                                                                                                                                                                                                 | Phát kiểm ID/URL từng run; VERIFY-02 lấy URL tự động sau logging                                                    |
| Đã có VERIFY-01 và VERIFY-02 chạy trên Kaggle CPU | Version 8/source 23af76f; Version 9/source 217684f, 11 tests đạt                                                                                                                                                                                                                                 | Giữ riêng hai lượt tổng hợp với CIFAR-KAGGLE-01                                                                     |
| Notebook 03 đã lên main qua PR #7                    | [Notebook 03](../notebooks/03_cifar100_safe_attempts.ipynb), [giải thích sửa trạng thái](SUA_LOI_TRANG_THAI_RUN.md): artifact riêng, running/failed/complete, kích thước cuối theo config                                                                                                  | Lượt 03 full5 đã có JSON/source hashes và W&B finished; manifest ghi rõ giới hạn xác minh giao diện Kaggle |
| Repository đã Public                                  | https://github.com/24122050S/20-ML-Paper                                                                                                                                                                                                                                                           | Cả ba dùng PR và cập nhật bằng chứng của phần mình                                                               |

Kaggle được ghi nhận Private trong manifest lần đối chiếu Version 2; cần kiểm quyền truy cập khi nộp. JSON/manifest của lượt đó đã nằm trên GitHub. Bảng này mô tả bằng chứng đã có, còn các mục S/V/P phía dưới là kế hoạch công việc.

## 3. Sang: phương pháp và analytic head

### S01 — Rà soát đặc tả và hiểu code

- **Hạn:** 03/11, 21:00. Dự kiến 2–3 giờ tập trung.
- **Đọc:** `docs/Dac_ta_ACIL.md`; `src/analytic.py`; notebook 01; mục 3 và phụ lục chứng minh của paper.
- **Làm:** ghi bảng ký hiệu và kích thước; giải thích khởi tạo R/W, Woodbury, thêm cột cho lớp mới; phân biệt nhãn one-hot cục bộ trong công thức khối với one-hot toàn cục trong code.
- **Bàn giao:** cập nhật đặc tả bằng lời của mình; ghi nguồn công thức và một ví dụ base 2 lớp, phase mới 2 lớp.
- **Đạt khi:** tự giải thích 10 phút không nhìn PDF; chạy toy và chỉ được dòng code tương ứng từng công thức; Vĩ đọc và kiểm được kích thước.
- **Đã hỗ trợ khởi động:** có ghi chú rà soát riêng, chưa tự sửa nội dung người học viết.

### S02 — Chốt điều kiện của các tuyên bố và chi phí

- **Hạn:** 04/11, 21:00. Dự kiến 2–3 giờ.
- **Làm:** viết phạm vi Theorem 3.1: cùng feature map, cùng objective/gamma, phép cập nhật hợp lệ, sai số hữu hạn; phân biệt joint analytic và joint CNN training. Diễn giải privacy theo phạm vi paper và các giả định liên quan.
- **Bàn giao:** `docs/phuong_phap_va_dieu_kien.md` (file cần tạo), pipeline tự vẽ, bảng chi phí R/W/expansion theo d và số lớp.
- **Đạt khi:** không suy diễn “absolute memorization” thành accuracy cũ bất biến hoặc F=0; mọi biến đều có định nghĩa; chỉ rõ chi phí bộ nhớ theo byte.
- **Kiểm tra:** Phát đọc để chuyển thành câu diễn giải dễ hiểu.

### S03 — Thiết kế và chạy phân tích bổ sung R3

- **Thiết kế xong:** 06/11; kết quả đầu 10/11; chốt 13/11.
- **Làm trước:** sweep batch size 32/128/512, giữ cùng checkpoint, expansion, class order, gamma, precision và thứ tự mẫu. Đo sai lệch W, score, số nhãn khác nhau, thời gian; giải thích vì sao batching lý tưởng không đổi nghiệm.
- **Bàn giao:** script/config R3, JSON/CSV kết quả, W&B runs, một hình và phần diễn giải.
- **Đạt khi:** mỗi điểm số có config/run; không dùng các backbone khác nhau để kết luận về batch size. Nhánh giữ lịch sử tính joint phải được tách khỏi thuật toán incremental.
- **Điểm hiện còn thiếu:** `src/analytic.py` ép float64; chưa thể chỉ đổi tên config thành float32 để coi là precision sweep. Muốn thử precision phải hỗ trợ dtype thực sự và kiểm tra lại.
- **Kiểm tra:** Vĩ chạy lại một cấu hình; Phát kiểm bảng thống kê.

### S04 — Phần nội dung giữa kỳ

- **Hạn nội dung/hình:** 14/11; **notes hoàn chỉnh:** 16/11.
- **Sở hữu slides:** 7–16 và 26 theo dàn ý ở mục 7, tổng 11 slides.
- **Bàn giao:** hình tự làm, công thức, notes, nguồn IEEE cho từng slide. Nội dung R3 chỉ đưa vào khi có kết quả; nếu chưa có, đổi bố cục theo thực tế.

### S05 — Phần cuối kỳ

- **Hạn lập luận:** 24/11; rà soát bằng chứng đến 30/11.
- Viết cơ sở lý thuyết cho hai hướng phản biện, ít nhất 3 điểm mạnh có bằng chứng, và hai hướng phát triển gắn hạn chế.
- Cùng Vĩ thiết kế P1 về phụ thuộc tập base; cùng Phát làm rõ P2 về bộ nhớ thực và giới hạn tài nguyên. Các hướng này là giả thuyết cần kiểm tra, chưa là kết luận.
- Rà lại mọi claim trên poster trước 03/12; soạn câu hỏi về định lý và giới hạn cho cả nhóm luyện.

## 4. Vĩ: dữ liệu, Kaggle, backbone và R1

### V01 — Mở rộng run Kaggle từ 60 lên 100 lớp — đã có bằng chứng

- **Hạn:** 03/11, 21:00. Dự kiến 2–4 giờ gồm setup và chạy thử.
- **Đã có:** pipeline/source trên main và notebook 02 Version 2 chạy CUDA 50+10, W&B o2r63pj2 Finished. Không cần tạo lại pipeline hoặc chạy lại chỉ để chứng minh mốc này.
- **Kết quả xác nhận 04/11:** đã nhận JSON chuỗi 6 điểm 50→100 lớp, status complete/verification true; W&B blhw74ff finished, config/source/summary khớp. Xem [manifest](../reports/CIFAR-KAGGLE-02-provenance.json). Việc tiếp theo là V02/P02/S02 và lưu checkpoint/state của lượt này.
- **Bàn giao:** URL notebook/version, tên GPU, Python/Torch/CUDA, config, log, checkpoint và `results.json` của run; giữ output trong `/kaggle/working` và lưu version.
- **Đạt khi:** thật sự nhận `torch.cuda.is_available() == True`; tests đạt; root/latest và results cùng attempt ID, status complete, verification_passed true; smoke K=5 có base + đủ 5 phase, W cuối 256×100 và test 10.000 ảnh. Ghép version/source/JSON/W&B của chính lượt mới. Nếu lỗi, giữ failed status và log.
- **Kiểm tra:** Phát đọc output/config; Sang kiểm R/W và test class-incremental.
- **Lưu ý:** code hiện dùng một GPU; hai GPU có sẵn không tự đồng nghĩa training dùng cả hai.

### V02 — Chốt protocol và ngân sách trước khi chạy dài

- **Hạn:** 04/11, 21:00.
- **Làm:** xác nhận fine labels, 50 lớp base, class order, normalization, augmentation và ResNet shortcut. Ghi các điểm đã khớp paper/code và điểm còn khác. Chạy đo thời gian trên Kaggle, không suy thời gian GPU từ benchmark CPU.
- **Bàn giao:** [protocol_chot.md](protocol_chot.md) đã chốt v1.0, có danh sách sai khác, cấu hình và gate trước lượt dài; còn cần artifact profile và checkpoint benchmark thật. Tài liệu hoàn tất không đồng nghĩa runner đã đáp ứng các gate.
- **Đạt khi:** dùng protocol v1.0 và ghi mỗi checkpoint cùng mapping/normalization/config; Sang kiểm bằng evidence của các gate trong tài liệu. Hiện Sang làm một mình; khi Vĩ/Phát tham gia lại sẽ review chéo cùng phiên bản.

### V03 — Huấn luyện base chuẩn và bảo toàn checkpoint

- **Hạn checkpoint seed đầu:** 07/11; các seed tiếp theo theo ngân sách, mục tiêu 11/11.
- **Làm:** sau khi V02 đạt, huấn luyện đủ cấu hình đã chốt, mục tiêu 160 epochs với mốc LR 80/120. Lưu run và checkpoint cuối; checkpoint tạm cho phiên dài cần hỗ trợ và kiểm tra resume đúng optimizer/scheduler/RNG.
- **Bàn giao:** checkpoint cho từng seed, hash, history, run URL, config, thời gian. Chia checkpoint cho Sang/Phát bằng input/artifact có version.
- **Đạt khi:** tải lại cho dự đoán tương thích; freeze đúng; checkpoint không bị dùng nhầm cho class order/normalization khác. `--reuse-checkpoint` dùng lại base hoàn chỉnh; `--resume-checkpoint` v2 tiếp tục BP tại biên epoch. CPU đã so optimizer/scheduler/RNG/history chính xác; CUDA TinyCNN resume test passed trong BP5 Version3 và checkpoint GPU5 epochs đã kiểm.

### V04 — Kết quả chính R1

- **Hạn run đầu K=5:** 08/11; **K=25:** 10/11; chốt các run chính 13/11.
- **Làm:** chạy ACIL từ checkpoint chuẩn; ưu tiên K=5. K=25 là mục tiêu mở rộng và cần sửa protocol/runner trước vì smoke hiện chỉ hỗ trợ K=5; K=10/50 cũng chưa được hỗ trợ. Nếu thu hẹp phạm vi, ghi rõ trong scope/deck.
- **Bàn giao:** accuracy từng phase, toàn bộ lớp đã thấy, accuracy base, config và run links; phân biệt số trích từ paper và số nhóm chạy.
- **Đạt khi:** chỉ gọi average incremental accuracy đủ K khi đã có K+1 điểm tính cả base. Phát tính lại metric độc lập từ dữ liệu đầu ra.
- **Kiểm tra:** Sang xác nhận mọi run cùng feature map trong một chuỗi incremental.

### V05 — Đối chiếu implementation tác giả

- **Hạn:** 12/11.
- Chạy code gốc trong môi trường riêng, lưu commit/config; chỉ dùng để kiểm chứng và so sánh. Không chép code cốt lõi vào repo nhóm.
- Bàn giao bảng khác biệt về class order, backbone, normalization, expansion 8000/8192, metric và precision; các điểm phương pháp/metric nhờ Sang/Phát kiểm cùng.

### V06 — Nội dung deck và cuối kỳ

- **Deck:** slides 17–23 và 27, 8 slides; hình/nội dung 14/11, notes 16/11.
- **Poster:** phụ trách chạy P1 về ảnh hưởng thành phần lớp base theo thiết kế đã thống nhất; bàn giao số liệu và biểu đồ trước 30/11.
- Khi đổi thành phần lớp base để kiểm tra P1, giữ số lớp và ngân sách huấn luyện có thể so sánh; không trộn ảnh hưởng đổi tập base với đổi thứ tự phase trong cùng kết luận.

## 5. Phát: metric, ablation và bằng chứng

### P01 — Repository, W&B và bảng run

- **Hạn tạo:** 03/11; cập nhật sau mỗi lượt chạy.
- **Làm:** repo và project W&B đã được tạo; tiếp tục kiểm public access và bảng run/config/source, không tạo trùng project. Cả ba dùng tài khoản của mình có quyền phù hợp, không chia sẻ API key trong chat/source.
- **Bàn giao:** hai URL; bảng `experiment -> config/script -> checkpoint -> run -> kết luận`; hướng dẫn chạy và bổ sung nguồn tái sử dụng vào README.
- **Đạt khi:** kiểm tra link bằng cửa sổ không đăng nhập; ít nhất một run đã xem được công khai. Các run offline đã có được giữ lại và sync có kiểm tra.

### P02 — Chuẩn hóa metrics và bảng kết quả

- **Hạn:** 05/11.
- **Làm:** định nghĩa accuracy từng phase trên tất cả lớp đã thấy; A-bar có tính base; quy ước forgetting và dấu của phép trừ. Đối chiếu công thức paper với bảng/code trước khi gọi metric là F của paper.
- **Bàn giao:** [metrics.md](metrics.md) đã chốt v1.0: A-bar có base, F-base/delta có quy ước dấu và ví dụ smoke thật. `src/metrics.py` đã xuất summary vào JSON/W&B; bảng benchmark vẫn cần run160 epochs thật. Báo deviation theo pp và tương đối khi áp dụng được.
- **Đạt khi:** dùng một ví dụ số nhỏ để kiểm phép tính; tính lại được số từ run của Vĩ, không chép số bằng tay. Một seed không có độ lệch chuẩn giữa seeds.
- **Kiểm tra:** Sang rà định nghĩa; Vĩ đối chiếu dữ liệu đầu ra.

### P03 — Ablation R2

- **Thiết kế:** 06/11; run thử 09/11; chốt 13/11.
- **Làm:** tái lập các hàng Table 2 về feature expansion và regularization trên K=5, dùng checkpoint/mapping của Vĩ. Giữ các yếu tố khác cố định khi đổi một yếu tố.
- **Bàn giao:** ma trận config, script, run links, bảng/hình và giải thích cơ chế cùng Sang.
- **Điểm phải xử lý:** module analytic hiện nhận H trực tiếp và yêu cầu gamma>0. Với no-FE, dùng đúng feature backbone làm H, không thay bằng một random expansion cùng chiều. Hàng gamma=0 cần Sang kiểm tra điều kiện khả nghịch/phương pháp giải trước; không thay bằng gamma rất nhỏ rồi gắn nhãn gamma=0. Nếu chưa hỗ trợ được một hàng, ghi rõ hàng chưa tái lập và lý do.
- **Đạt khi:** mô tả được mỗi cấu hình đã bỏ/thay thành phần nào; dùng cùng backbone và dữ liệu. Không chọn hyperparameter dựa trên test để làm đẹp kết quả.

### P04 — Phân tích sai lệch, hình và nguồn

- **Hạn:** 14/11.
- Tổng hợp vì sao số nhóm khác paper: protocol, checkpoint, expansion, thư viện, seed và phần chưa biết; chỉ coi nguyên nhân là đã kiểm chứng khi có đối chiếu phù hợp.
- Tự dựng bảng/đồ thị từ dữ liệu xuất; chuẩn hóa tên metric, đơn vị và chú thích. Số nhóm gắn run; số paper gắn nguồn.
- Đọc ít nhất hai phương pháp tiền nhiệm để viết bối cảnh cùng cả nhóm, ví dụ iCaRL và LwF; không nhận xét chỉ từ tên phương pháp.

### P05 — Ghép deck và kiểm định dạng

- **Nội dung sở hữu:** slides 1–6, 24–25, 28–30, tổng 11 slides.
- **Hạn nhận nội dung các thành viên:** 16/11. **Deck bản hoàn chỉnh:** 17/11.
- Chuẩn hóa template, font, trích IEEE chân slide, speaker notes và run links. Sang/Vĩ tự sửa nội dung phần mình khi Phát trả nhận xét.
- **Đạt khi:** 20–40 slides kể cả appendix; mỗi slide có notes không quá 500 từ; dưới 20 MB; hình tự làm; không có slide references riêng; repo link nộp riêng trên Moodle.

### P06 — Poster và P2

- **Thiết kế P2:** 24/11; kết quả 30/11; poster nháp 02/12; chốt 03/12.
- P2 đo bộ nhớ theo byte, thời gian và chất lượng dưới ngân sách rõ ràng; tính cả R/W/expansion và phân biệt trạng thái, peak memory, cache chẩn đoán joint.
- Đọc phần strict-memory của phụ lục trước khi phản biện; với cùng ngân sách phải ghi dtype và cách lưu exemplar nếu có baseline replay.
- Ghép một slide PPTX A2 dọc, đặt lý thuyết cạnh đúng bằng chứng. Cả ba chọn vấn đề đưa lên poster dựa trên kết quả.

## 6. Các điểm bàn giao phụ thuộc nhau

1. **Sang -> Vĩ/Phát:** đặc tả và điều kiện đúng; điểm cần sửa trong analytic head.
2. **Vĩ -> Sang/Phát:** protocol version, checkpoint hash, class order, normalization và run base. Chưa có gói này thì run ablation chỉ được gắn nhãn thử nghiệm sơ bộ.
3. **Phát -> Vĩ:** định nghĩa metric/schema kết quả trước khi batch run R1.
4. **Sang -> Phát:** đường no-FE/gamma=0 đã kiểm tra hoặc giới hạn đã mô tả trước khi báo cáo R2.
5. **Cả ba -> Phát:** nội dung slide đã có hình, notes và nguồn, không chỉ một file văn bản thô.

Mỗi task chỉ hoàn tất khi có file/commit/run và người kiểm tra chéo đã đọc. “Đang chạy” hoặc “đã đọc paper” không thay thế cho đầu ra.

## 7. Dàn ý deck mục tiêu 30 slides

| Slides | Nội dung                                                                    | Chủ trì                          |
| ------ | ---------------------------------------------------------------------------- | ---------------------------------- |
| 1–6   | Bài toán, ví dụ, tiền nhiệm, khoảng trống và đóng góp            | Phát, Sang/Vĩ kiểm              |
| 7–16  | Ký hiệu, pipeline, base/FE/ridge/recursive update, định lý và chi phí | Sang; Vĩ cung cấp hình backbone |
| 17–20 | Dataset, protocol, metrics, scope                                            | Vĩ; Phát kiểm metric            |
| 21–23 | Bảng kết quả chính và đường cong                                     | Vĩ                                |
| 24–25 | Ablation                                                                     | Phát; Sang kiểm                  |
| 26     | Phân tích bổ sung R3                                                      | Sang                               |
| 27     | Đối chiếu implementation gốc                                             | Vĩ, cả ba cung cấp phát hiện  |
| 28–30 | Sai lệch, kết luận có giới hạn, hướng dẫn tái lập                 | Phát                              |

Đây là khung phân trách nhiệm; điều chỉnh theo kết quả nhưng giữ tổng số slide trong giới hạn. Không thêm slide references riêng. Mốc deck chấm từ file và notes; luyện giải thích để kiểm hiểu, không giả định có buổi trình chiếu giữa kỳ.

## 8. Lịch chung đến hai hạn nộp

| Ngày     | Sang                  | Vĩ                                  | Phát                          | Mốc chung                                                                                                     |
| --------- | --------------------- | ------------------------------------ | ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| 02–03/11 | S01                   | V01                                  | P01                            | Repo và run 60 lớp đã có; mốc 100 lớp được xác nhận thêm ngày 04/11                         |
| 04–06/11 | S02, thiết kế R3    | V02, bắt đầu base chuẩn          | P02, thiết kế R2             | Chốt protocol + bảng run, đo ngân sách Kaggle                                                             |
| 07–10/11 | R3 và hỗ trợ R2    | Checkpoint + R1 K=5/25               | R2, kiểm metric               | Có kết quả đầu và phát hiện sai lệch sớm                                                             |
| 11–13/11 | Chốt R3              | Hoàn tất run chính + đối chiếu | Chốt R2 + bảng sai lệch     | Khóa tập kết quả đưa vào deck                                                                           |
| 14–16/11 | 11 slides + notes     | 8 slides + notes                     | 11 slides + notes              | Mọi claim có nguồn, mọi số nhóm có run                                                                  |
| 17/11     | Đọc chéo           | Đọc chéo                          | Ghép deck                     | Deck hoàn chỉnh bản 1                                                                                       |
| 18–19/11 | Sửa theo nhận xét  | Sửa theo nhận xét                 | Kiểm định dạng và links   | Mở file ở máy khác, đọc notes, kiểm public access                                                       |
| 20/11     | Cùng nghiệm thu     | Cùng nghiệm thu                    | Nộp và ghi bằng chứng nộp | Hạn nội bộ; deck và repo link ở mục riêng                                                               |
| 21/11     | Dự phòng            | Dự phòng                           | Dự phòng                     | Hạn chính thức 23:59                                                                                        |
| 22–25/11 | Lý thuyết P1/P2     | Thiết kế P1                        | Thiết kế P2                  | Ít nhất 3 strengths, 2 limitations, 2 future directions                                                      |
| 26–30/11 | Kiểm lập luận      | Chạy P1                             | Chạy P2                       | Chọn bằng chứng đáng tin cậy; giả thuyết không được ủng hộ phải được diễn giải trung thực |
| 01–03/12 | Nội dung lý thuyết | Nội dung thực nghiệm              | Ghép poster                   | PPTX một slide A2 dọc, luyện hỏi chéo                                                                     |
| 04/12     | Cùng nghiệm thu     | Cùng nghiệm thu                    | Nộp poster                    | Hạn nội bộ                                                                                                  |
| 05/12     | Dự phòng            | Dự phòng                           | Dự phòng                     | Hạn chính thức 23:59                                                                                        |

## 9. Cách phối hợp để không làm trùng

- Một repository chuẩn, một project W&B chuẩn. Có thể dùng nhánh `sang/analytic`, `vi/kaggle-backbone`, `phat/evaluation`; đây là tên đề xuất, chưa khẳng định nhánh đã tồn tại.
- Theo kế hoạch nhóm: Sang chủ trì `src/analytic.py` và verification; Vĩ chủ trì các module dữ liệu/backbone đã có; Phát chủ trì metric/reporting và hình. Hiện Sang thực hiện cả ba nhóm việc; thống nhất API và review thay đổi ở file chung.
- Chia quota bằng **run ID/config đã giao**. Một checkpoint/seed/protocol chuẩn được Vĩ tạo rồi dùng lại cho R2/R3; chỉ train lại khi seed/representation/protocol cần đổi.
- Mỗi ngày đồng bộ 10–15 phút: đã tạo artifact nào, còn vướng gì, đầu ra ngày mai là gì. Chốt vào bảng task, không chỉ báo bằng lời.
- Mỗi run ghi seed, checkpoint hash, class order hash, normalization, d, gamma, K, dtype, batch size, nguồn số liệu và URL/version Kaggle/W&B.
- Người chủ trì nộp file kiểm tra lại trên Moodle; cả nhóm cùng chịu trách nhiệm. ACIL đã được giảng viên duyệt theo xác nhận của Sang; giữ bằng chứng đăng ký khi bàn giao.

## 10. Phiên làm việc kế tiếp

**Sang đang làm một mình:** trợ lý đã chạy BP5/CUDA qua API, không cần làm05 lại. [Notebook06](../notebooks/06_benchmark_seed42_ready.ipynb) đã cấu hình160/d8000/seed42. Xem [trạng thái ngắn](BUOC_TIEP_THEO_SANG.md), giữ protocol/metrics v1.0.

**Đã có:** checkpoint v2/resume CPU+CUDA test, streaming/no-FE/metrics, profile d8000 T4 và BP5 Version3; [BP5 JSON](../reports/profile_base_kaggle_ready.json)/checkpoint đã kiểm. Budget~59 phút, cửa sổ2 giờ, quota26h từ Sang. Tracking BP5 [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) đã Finished và được API công khai đối chiếu. **Tiếp theo:** run R1 seed42; chưa khởi chạy160 epochs.

**Sau đó:** khi G2/G3 CUDA và G6 Kaggle/quota đạt, chạy R1 seed42 trước, rồi43/44; reuse backbone tương ứng cho R2. R3 theo cấu hình d1024 đã chốt. Nếu Vĩ/Phát tham gia lại, chia việc theo bảng vai trò.

**Quy ước hiện trạng:** README và bảng run phản ánh artifact thật trong repo. Source hoặc kết quả chỉ có tại workspace hỗ trợ được ghi rõ chưa công bố; run Kaggle cần xác nhận bằng version và output thực tế.
