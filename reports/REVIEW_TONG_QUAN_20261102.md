# Review tổng quan tiến độ ACIL — 02/11/2026

## Kết luận về vị trí hiện tại

Đã hoàn thành một implementation analytic được kiểm chứng và một pipeline ảnh chạy thử đầu-cuối. Chưa hoàn thành bộ thí nghiệm tái lập theo phạm vi nộp, slide deck giữa kỳ hoặc poster cuối kỳ.

Sang hiện tự thực hiện công việc; không cần chờ bàn giao từ Vỹ/Phát. Đã xác nhận với Sang rằng bài ACIL được giảng viên duyệt. Các phân công ba người trước đây là kế hoạch lịch sử, không phải danh sách việc đã hoàn tất.

Không quy đổi số file hoặc số test thành phần trăm hoàn thành đồ án. Các bài kiểm tra số học chứng minh tính nhất quán trong những trường hợp đã kiểm, còn các thí nghiệm tái lập và phân tích trả lời câu hỏi khác.

## Căn cứ của review

- Đọc lại [trang tổng quan chính thức](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/) và đủ năm trang selection/reading/reproduction/analysis/empirics ngày 02/11/2026.
- Fetch GitHub; `origin/main` vẫn là `8e2a14368d1b5acb3948e9c5529ae1e753872e28`. Kiểm tra cây file của main và working tree tại `D:\ML_Paper\20-ML-Paper`.
- Đọc code, notebook, config, JSON và manifest; đối chiếu checksum VERIFY-02, CIFAR-SMOKE-01, source chạy CIFAR, checkpoint và analytic state: đều khớp.
- Kết quả 20 tests và notebook bootstrap dựa trên lượt thực thi gần nhất đã lưu/quan sát; lần review này không train lại hoặc lặp lại benchmark.
- Không thấy deck/poster `.pptx` hoàn chỉnh trong workspace đã kiểm. Chưa có bằng chứng version Kaggle mới cho notebook 02 trong các artifact hiện có.

## 1. Những gì đã làm được

| Hạng mục | Trạng thái xác nhận | Bằng chứng/phạm vi |
|---|---|---|
| Chọn và đăng ký paper | Hoàn tất theo xác nhận của Sang | ACIL đã được giảng viên duyệt |
| Đọc paper | Sang xác nhận đã đọc hết | Có đặc tả phương pháp; khả năng giải thích toàn bộ cần tiếp tục luyện qua deck/notes |
| Đặc tả/API | Có bản thống nhất | H/y/R/W, class order, owner của mapping, backbone/BatchNorm/FE cố định trong [đặc tả](../docs/Dac_ta_ACIL.md) |
| Analytic head | Đã chạy kiểm chứng | Joint ridge, khởi tạo, Woodbury/residual update, nhãn toàn cục, minibatch và lớp lặp; không tạo cột lớp trùng |
| Kiểm chứng tổng hợp | Đã có bằng chứng public trong repo/W&B | VERIFY-02: Kaggle Version 9, source 217684f, run wi4wau2d, W 24×2 → 24×4 → 24×6 |
| CIFAR-100 loader/protocol | Đã chạy local | HF revision cố định, kiểm 50.000 train/10.000 test, mapping dùng chung, chia base 50 và 5 nhóm mới x 10 |
| ResNet-32/base training | Đã train mới local 2 epochs | 25.000 ảnh base; loss 3.4773 → 3.0187; trọng số đổi, reload checkpoint khớp, freeze đúng |
| Ghép analytic trên ảnh thật | Đã chạy local 50+10 | W 256×50 → 256×60, test 5.000 → 6.000 ảnh; dự đoán khớp joint; mọi check đạt |
| Tests | 20 tests đạt trong môi trường đầy đủ | 11 test core/tracking + 9 test pipeline ảnh; tests ảnh được skip có thông báo nếu thiếu dependency |
| Notebook 02 | Đã chuẩn bị và kiểm phần khởi động | Bundle source/checksum, dependency, tests chạy trong kernel mới; chưa gọi đây là một full run Kaggle |
| Repo/tracking nền tảng | Đã có | GitHub và project W&B đã public; repo local có Git độc lập, requirements, hướng dẫn và manifest |

Các con số CIFAR mới trong [JSON](cifar100_smoke_results.json):

| Mốc | Lớp đã thấy | Accuracy toàn bộ lớp đã thấy | Relative W error so joint | Dự đoán khác joint |
|---|---|---|---|---|
| Base | 50 | 32,68% | 2,047e-11 | 0 |
| Sau phase mới đầu tiên | 60 | 29,3667% | 1,771e-11 | 0 |

Đây là kết quả smoke với 2 epochs, expansion 256, seed 42. Số 29,37% chưa thể đem so trực tiếp với một dòng benchmark paper dùng cấu hình khác. Sai lệch joint nhỏ cũng không chứng minh accuracy lớp cũ luôn bất biến: ở phase 1, accuracy riêng base là 29,92%, riêng lớp mới là 26,60%.

## 2. Phân biệt máy cá nhân, GitHub, Kaggle và W&B

| Mốc | Máy cá nhân | GitHub main | Kaggle | W&B |
|---|---|---|---|---|
| VERIFY-02 tổng hợp | Có | Có source/output/JSON | Version 9 đã chạy | wi4wau2d, ghi nhận Finished |
| CIFAR-SMOKE-01 mới | Đã chạy, có checkpoint/state/JSON | Chưa có source mới | Notebook 02 đã tạo; chưa có version full run được lưu trong bằng chứng | `not_logged`, ID/URL null |
| Dọn cấu trúc/đặc tả mới | Đã sửa working tree | Chưa commit/push | Không áp dụng | Không áp dụng |

Vì vậy nói “đã làm pipeline CIFAR” đúng ở mức local, nhưng nói “GitHub main đã có pipeline CIFAR” hiện chưa đúng. `source_dirty=true` của CIFAR được giữ nguyên; commit cha 8e2a143 không đại diện trọn source mới.

Notebook Kaggle 01 được ghi nhận Private ở lần kiểm tra trước. Repo/W&B công khai đã có, nhưng cần kiểm lại quyền xem khi dùng link làm bằng chứng nộp.

## 3. Những gì còn thiếu để tái lập thí nghiệm

[Yêu cầu reproduction](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction) cho phép full scope hoặc core scope. Với core scope, phải có tối thiểu hai thí nghiệm quan trọng, gồm phép so sánh chính, cộng một phân tích bổ sung tự thiết kế. R1/R2/R3 là nhãn kế hoạch của chúng ta, không phải tên bắt buộc trong đề.

| Phần thiếu | Hiện trạng | Tiêu chí hoàn thành |
|---|---|---|
| Scope/protocol tái lập | Có kế hoạch, chưa có bản chốt đủ cho benchmark | Nêu rõ scope; dataset, thứ tự lớp, preprocessing, backbone, FE, gamma, metric và các sai khác với paper |
| Kết quả chính R1 | Chưa có benchmark hoàn chỉnh | Training đủ cấu hình đã chốt; chuỗi incremental hoàn chỉnh; bảng/đường cong so với paper và sai lệch |
| Đủ 100 lớp | Mới chạy base + 1/5 phase mới | Có base + đủ 5 phase, kết thúc ở 100 lớp và đánh giá 10.000 ảnh test |
| Ablation R2 | Chưa có bộ kết quả | Thí nghiệm thành phần đã chọn, đối chứng cùng checkpoint/protocol và run links |
| Phân tích bổ sung R3 | Có ý tưởng batch size | Có driver sweep, đo kết quả/sai số/thời gian và diễn giải từ run thật |
| Đối chiếu code tác giả | Chưa có bằng chứng chạy | Môi trường riêng, commit/config pin, so output và định vị khác biệt; không chép core implementation |
| Metrics tổng hợp | Có accuracy theo phase/base/new | Chốt và kiểm phép tính average incremental accuracy/forgetting, chênh lệch điểm phần trăm/relative deviation theo paper |
| Seeds benchmark | Ảnh thật mới một seed | Chọn số seed theo scope/tài nguyên; chỉ báo mean/std khi có đủ runs. Ba seed trong unit test không phải ba benchmark CIFAR |
| CIFAR tracking/public evidence | Chưa có W&B run mới | Run thật hoàn tất có config/metrics; README ghép đúng source, artifact và W&B URL |

Một checkpoint 2 epochs không phải checkpoint benchmark hoàn chỉnh. Chạy thêm đủ phase với chính checkpoint đó trước hết vẫn là kiểm tra pipeline đầy đủ.

## 4. Những giới hạn code cần xử lý trước lượt chạy lớn

1. `src/cifar_smoke.py` hiện giới hạn K=5; K=25 trong roadmap không thể hoàn thành chỉ bằng đổi một giá trị JSON.
2. `--reuse-checkpoint` dùng lại base đã hoàn thành. Runner chưa lưu đầy đủ optimizer/scheduler/RNG theo epoch để resume một phiên training dài bị ngắt.
3. Analytic vẫn là NumPy CPU. GPU giúp base training/trích feature; cần đo riêng chi phí analytic trước khi tăng expansion.
4. Runner giữ feature lịch sử để tính joint oracle mỗi phase. Ở expansion lớn cần tách mode kiểm chứng và mode benchmark; không nhầm bộ nhớ oracle với trạng thái thuật toán.
5. Chưa có đường no-FE đúng nghĩa; đặt expansion_dim=64 nhưng vẫn random projection không phải bỏ FE. Gamma=0 đang bị từ chối; không thay epsilon rồi gọi là zero.

Đây là các việc mở rộng từ một smoke runner đã đạt sang experiment runner. Chúng không làm mất giá trị những kiểm chứng hiện có.

## 5. Sản phẩm nộp còn thiếu

### Giữa kỳ — 21/11/2026, 23:59 theo hạn Sang cung cấp

Chưa thấy file deck hoàn chỉnh. [Yêu cầu reading](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reading): PPTX 20–40 slides kể cả appendix, dưới 20 MB, notes mỗi slide tối đa 500 từ, hình tự làm, trích IEEE trên từng slide có nguồn, không có references slide riêng. File deck và repo link nộp riêng qua Moodle.

Cần bổ sung bối cảnh và ít nhất hai hướng tiền nhiệm; phân loại đóng góp; hình pipeline; ý nghĩa và giới hạn định lý; độ phức tạp; kết quả tái lập đối chiếu paper với W&B run links. Đặc tả code mới chỉ bao phủ một phần các nội dung này.

### Cuối kỳ — 05/12/2026, 23:59 theo hạn Sang cung cấp

P1 về phụ thuộc tập base và P2 về bộ nhớ hiện là ý tưởng. Chưa có bộ lập luận và thực nghiệm chứng minh hạn chế để đưa lên poster.

[Yêu cầu analysis](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis) cần ít nhất ba điểm mạnh có căn cứ, hai hạn chế lý thuyết có thể kiểm nghiệm và hai hướng phát triển gắn với hạn chế. [Yêu cầu empirics](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics) ghép lập luận với bằng chứng thực nghiệm, poster một slide PPTX A2 dọc, số liệu có W&B run links; bảo vệ trực tiếp tại workshop.

## 6. Thứ tự làm cho một người

1. **Chốt mốc hiện tại:** review/commit/push phần dọn cấu trúc + CIFAR mới, không commit data, checkpoint hoặc credential. Giữ chính xác JSON của run local hiện có; không sửa `not_logged` thành logged bằng tay.
2. **Có một run CIFAR được track:** chạy thật với W&B bật. Có thể dùng checkpoint đã kiểm tương thích và ghi rõ reused; vẫn thực thi feature/analytic/evaluation. Local hoặc Kaggle đều hợp lệ, ghép đúng source/config/JSON/run.
3. **Kiểm tra đủ chuỗi 100 lớp:** tiếp tục cấu hình nhỏ 2 epochs/d=256, chạy đủ 5 phase và kiểm metrics. Sau đó chốt protocol benchmark, đo tài nguyên và xử lý resume/reference lớn.
4. **Hoàn thành core reproduction:** kết quả chính, ablation đã chọn và một phân tích bổ sung; đối chiếu paper/code tác giả, giải thích sai lệch. Viết nội dung deck song song khi scope và kết quả đã rõ.
5. **Hoàn thiện deck trước hạn; sau đó tập trung poster:** phát triển P1/P2 thành giả thuyết, lý thuyết, đối chứng và phép đo; chỉ đưa kết luận được bằng chứng ủng hộ.

[Đề chính thức](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction) chấp nhận cả máy cá nhân và hosted notebook. Kaggle/GPU là lựa chọn tài nguyên của kế hoạch; chưa có run Kaggle không phủ nhận một run local đã hoàn tất. Điều bắt buộc là source truy nguyên được, run thật và tracking công khai.

Việc ưu tiên ngay là đóng gói mốc hiện tại và có run CIFAR gắn W&B. Không cần bắt đầu lại từ đọc paper, viết lại analytic hoặc train lại chỉ vì muốn chuyển sang Kaggle.

## 7. Cách dùng tài liệu để đỡ rối

- Dùng bản review này để xem trạng thái toàn đồ án.
- Dùng [HUONG_DAN_CIFAR_SMOKE.md](../docs/HUONG_DAN_CIFAR_SMOKE.md) khi chạy pipeline; [Dac_ta_ACIL.md](../docs/Dac_ta_ACIL.md) khi giải thích API/công thức.
- Các report review cũ mô tả snapshot trước khi sửa, còn `planning/` ở workspace ngoài repo có nhiều kế hoạch cũ. Không dùng chúng để suy rằng lỗi minibatch vẫn còn hoặc số 13 tests là hiện trạng suite.
- Phân công ba người, các hạn nội bộ và các run R1/R2/R3 chưa có artifact là kế hoạch, không phải kết quả. Trước khi nộp cần cập nhật tài liệu kế hoạch cho đúng cách làm hiện tại của Sang.
