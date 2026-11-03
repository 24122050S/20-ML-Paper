# Review tổng quan tiến độ ACIL — 02/11/2026

> Lập ngày 02/11, cập nhật hiện trạng ngày **03/11/2026** sau PR #5–#7. Pipeline CIFAR đã lên main; notebook 02 Version 2 (`364942764`) đã có JSON CUDA và W&B `o2r63pj2` Finished. Notebook 03 và bản sửa trạng thái/kích thước đã có trên main; chưa xác nhận lượt training Kaggle mới của bản 03. Nội dung dưới đây đã được cập nhật, không còn dùng trạng thái working tree ngày 02/11 để mô tả main hiện tại. Lịch sử bản ban đầu vẫn có trong Git.

## Kết luận về vị trí hiện tại

Đã hoàn thành một implementation analytic được kiểm chứng và một pipeline ảnh chạy thử đầu-cuối. Chưa hoàn thành bộ thí nghiệm tái lập theo phạm vi nộp, slide deck giữa kỳ hoặc poster cuối kỳ.

Sang hiện tự thực hiện công việc; không cần chờ bàn giao từ Vĩ/Phát. Đã xác nhận với Sang rằng bài ACIL được giảng viên duyệt. Các phân công ba người trước đây là kế hoạch lịch sử, không phải danh sách việc đã hoàn tất.

Không quy đổi số file hoặc số test thành phần trăm hoàn thành đồ án. Các bài kiểm tra số học chứng minh tính nhất quán trong những trường hợp đã kiểm, còn các thí nghiệm tái lập và phân tích trả lời câu hỏi khác.

## Căn cứ của review

- Đọc lại [trang tổng quan chính thức](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/) và đủ năm trang selection/reading/reproduction/analysis/empirics ngày 02/11/2026.
- Mốc GitHub đối chiếu cho bản cập nhật này: [main sau PR #7, f6d1459](https://github.com/24122050S/20-ML-Paper/commit/f6d145986314f21ddc7aa4bb6565a6699ccb6a94). Pipeline đã được đưa lên qua [PR #5](https://github.com/24122050S/20-ML-Paper/pull/5); README ghép bằng chứng Kaggle qua [PR #6](https://github.com/24122050S/20-ML-Paper/pull/6).
- Đối chiếu code/config, JSON CPU và [JSON Kaggle](cifar100_kaggle_results.json), [manifest Kaggle](CIFAR-KAGGLE-01-provenance.json), source bundle và W&B URL. VERIFY-02, manifest Kaggle và manifest CPU hiện khớp file được công bố. [Bản sửa checksum CPU](../docs/SUA_LOI_CHECKSUM_JSON.md) dùng Git blob LF, giữ checksum raw CRLF cũ riêng; số liệu JSON không đổi và không train lại.
- Kiểm tra bản sửa PR #7 ngày 03/11: 37 tests đạt trên local; bootstrap bundle trong kernel mới có 36 tests đạt và 1 test snapshot 02 skip vì bundle không chứa notebook lịch sử. Cell cuối nhận đúng W256×60, W256×100 và W128×100; config 100 lớp từ chối W256×60. Đây là kiểm tra code/notebook, không phải một lượt training Kaggle mới.
- Sau bản sửa UTF-8/LF: suite đầy đủ có 38 tests đạt trên local, gồm test tái hiện CRLF trên Windows; notebook 03 được đóng gói lại với bộ ghi report mới. Số 37 tests phía trên thuộc lần kiểm PR #7.
- Chưa có artifact deck/poster `.pptx` hoàn chỉnh được ghi nhận. Kaggle Version 2 của notebook 02 đã có bằng chứng; bản notebook 03 chưa có version training được xác nhận.

## 1. Những gì đã làm được

| Hạng mục | Trạng thái xác nhận | Bằng chứng/phạm vi |
|---|---|---|
| Chọn và đăng ký paper | Hoàn tất theo xác nhận của Sang | ACIL đã được giảng viên duyệt |
| Đọc paper | Sang xác nhận đã đọc hết | Có đặc tả phương pháp; khả năng giải thích toàn bộ cần tiếp tục luyện qua deck/notes |
| Đặc tả/API | Có bản thống nhất | H/y/R/W, class order, owner của mapping, backbone/BatchNorm/FE cố định trong [đặc tả](../docs/Dac_ta_ACIL.md) |
| Analytic head | Đã chạy kiểm chứng | Joint ridge, khởi tạo, Woodbury/residual update, nhãn toàn cục, minibatch và lớp lặp; không tạo cột lớp trùng |
| Kiểm chứng tổng hợp | Đã có bằng chứng public trong repo/W&B | VERIFY-02: Kaggle Version 9, source 217684f, run wi4wau2d, W 24×2 → 24×4 → 24×6 |
| CIFAR-100 loader/protocol | Đã lên main và chạy local/Kaggle | HF revision cố định, kiểm 50.000 train/10.000 test, mapping dùng chung, chia base 50 và 5 nhóm mới x 10 |
| ResNet-32/base training | Có lượt train mới 2 epochs trên CPU và CUDA | 25.000 ảnh base; trọng số đổi, reload checkpoint khớp, freeze đúng; JSON của hai môi trường giữ riêng |
| Ghép analytic trên ảnh thật | Kaggle 50+10 đã hoàn tất và có tracking | W 256×50 → 256×60, test 5.000 → 6.000 ảnh; mọi check đạt, W&B o2r63pj2 Finished |
| Tests | Suite đầy đủ đã kiểm tra 38 tests trên local | Analytic, reporting gồm UTF-8/LF, CIFAR, attempt và notebook distribution; 11/20 tests thuộc snapshot VERIFY-02/CPU cũ |
| Notebook 02 | Có run Kaggle Version 2 đã đối chiếu | JSON CUDA, source hashes trong bundle, manifest và W&B; giữ bản 02 làm snapshot lịch sử |
| Notebook 03 | Source mới đã lên main qua PR #7 | Artifact riêng mỗi attempt; running/failed/complete; cell cuối theo config; bootstrap đã kiểm, chưa có run training Kaggle mới |
| Repo/tracking nền tảng | Đã có | GitHub và project W&B đã public; repo local có Git độc lập, requirements, hướng dẫn và manifest |

Các con số của lượt CPU ban đầu trong [JSON CPU](cifar100_smoke_results.json):

| Mốc | Lớp đã thấy | Accuracy toàn bộ lớp đã thấy | Relative W error so joint | Dự đoán khác joint |
|---|---|---|---|---|
| Base | 50 | 32,68% | 2,047e-11 | 0 |
| Sau phase mới đầu tiên | 60 | 29,3667% | 1,771e-11 | 0 |

Đây là kết quả smoke với 2 epochs, expansion 256, seed 42. Số 29,37% chưa thể đem so trực tiếp với một dòng benchmark paper dùng cấu hình khác. Sai lệch joint nhỏ cũng không chứng minh accuracy lớp cũ luôn bất biến: ở phase 1, accuracy riêng base là 29,92%, riêng lớp mới là 26,60%.

Lượt CUDA được lưu riêng trong [JSON Kaggle](cifar100_kaggle_results.json): base 50 lớp đạt **32,58%**, sau phase đầu 60 lớp đạt **29,70%**, không có dự đoán khác joint, mọi check đạt. [Version 2](https://www.kaggle.com/code/hsangg206/notebook7e7946d4ee?scriptVersionId=364942764) ghép với [W&B o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2) và [manifest](CIFAR-KAGGLE-01-provenance.json). Không gán metric CPU sang lượt CUDA.

## 2. Phân biệt máy cá nhân, GitHub, Kaggle và W&B

| Mốc | Máy cá nhân | GitHub main | Kaggle | W&B |
|---|---|---|---|---|
| VERIFY-02 tổng hợp | Có | Có source/output/JSON | Version 9 đã chạy | wi4wau2d, ghi nhận Finished |
| CIFAR-SMOKE-01 CPU | Đã chạy, checkpoint/state lưu local | Có source pipeline, JSON CPU và manifest | Không áp dụng cho lượt CPU này | `not_logged`, ID/URL null |
| CIFAR-KAGGLE-01 CUDA | JSON đã tải về | Có JSON/manifest và source bundle trong notebook 02 | Version 2 / 364942764, 50+10 lớp | o2r63pj2, ghi nhận Finished; JSON `logged` |
| Dọn cấu trúc/đặc tả và pipeline | Đã có trong checkout | Đã lên main qua PR #5 | Source bundle của Version 2 truy nguyên bằng hashes | Không phải run riêng |
| Sửa attempt và cell kích thước | Đã kiểm tra local/kernel mới | Đã lên main qua PR #7; notebook 03 đã tạo | Chưa có version training mới của bản 03 được xác nhận | Chưa có run mới tương ứng bản 03 |

Pipeline CIFAR hiện đã có trên GitHub main. `source_dirty=true` và commit cha 8e2a143 trong JSON CPU vẫn mô tả **thời điểm lượt CPU được thực thi**, không phải trạng thái source hôm nay. Lượt Kaggle có git_commit null vì chạy source giải nén từ bundle; manifest và hashes giữ liên kết tới source thực thi. Không chỉnh metadata lịch sử để giả lập một run trên commit mới.

Kaggle được ghi nhận Private trong lần đối chiếu version; repo/W&B đã được ghi nhận công khai. Cần kiểm quyền xem khi bàn giao; JSON/manifest đã có trên GitHub.

## 3. Những gì còn thiếu để tái lập thí nghiệm

[Yêu cầu reproduction](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction) cho phép full scope hoặc core scope. Với core scope, phải có tối thiểu hai thí nghiệm quan trọng, gồm phép so sánh chính, cộng một phân tích bổ sung tự thiết kế. R1/R2/R3 là nhãn kế hoạch của chúng ta, không phải tên bắt buộc trong đề.

| Phần thiếu | Hiện trạng | Tiêu chí hoàn thành |
|---|---|---|
| Scope/protocol tái lập | Có kế hoạch, chưa có bản chốt đủ cho benchmark | Nêu rõ scope; dataset, thứ tự lớp, preprocessing, backbone, FE, gamma, metric và các sai khác với paper |
| Kết quả chính R1 | Chưa có benchmark hoàn chỉnh | Training đủ cấu hình đã chốt; chuỗi incremental hoàn chỉnh; bảng/đường cong so với paper và sai lệch |
| Bộ bằng chứng 100 lớp có tracking | Bằng chứng Kaggle/W&B công bố hiện là base + 1/5 phase mới; kiểm tra local không thay thế một version Kaggle mới | Lưu base + đủ 5 phase, kết thúc ở 100 lớp/10.000 ảnh test; ghép source/config/JSON/version/W&B của cùng lượt |
| Ablation R2 | Chưa có bộ kết quả | Thí nghiệm thành phần đã chọn, đối chứng cùng checkpoint/protocol và run links |
| Phân tích bổ sung R3 | Có ý tưởng batch size | Có driver sweep, đo kết quả/sai số/thời gian và diễn giải từ run thật |
| Đối chiếu code tác giả | Chưa có bằng chứng chạy | Môi trường riêng, commit/config pin, so output và định vị khác biệt; không chép core implementation |
| Metrics tổng hợp | Có accuracy theo phase/base/new | Chốt và kiểm phép tính average incremental accuracy/forgetting, chênh lệch điểm phần trăm/relative deviation theo paper |
| Seeds benchmark | Ảnh thật mới một seed | Chọn số seed theo scope/tài nguyên; chỉ báo mean/std khi có đủ runs. Ba seed trong unit test không phải ba benchmark CIFAR |
| Công bố kết quả tiếp theo | Mốc 60 lớp đã có o2r63pj2 và manifest; bản 03 chưa có run training mới được xác nhận | Lượt 100 lớp/benchmark có config/metrics và W&B riêng; kiểm quyền xem Kaggle, cập nhật README bằng cùng lượt |

Một checkpoint 2 epochs không phải checkpoint benchmark hoàn chỉnh. Chạy thêm đủ phase với chính checkpoint đó trước hết vẫn là kiểm tra pipeline đầy đủ.

## 4. Những giới hạn code cần xử lý trước lượt chạy lớn

1. `src/cifar_smoke.py` hiện giới hạn K=5; K=25 trong roadmap không thể hoàn thành chỉ bằng đổi một giá trị JSON.
2. `--reuse-checkpoint` dùng lại base đã hoàn thành. Runner chưa lưu đầy đủ optimizer/scheduler/RNG theo epoch để resume một phiên training dài bị ngắt.
3. Analytic vẫn là NumPy CPU. GPU giúp base training/trích feature; cần đo riêng chi phí analytic trước khi tăng expansion.
4. Runner giữ feature lịch sử để tính joint oracle mỗi phase. Ở expansion lớn cần tách mode kiểm chứng và mode benchmark; không nhầm bộ nhớ oracle với trạng thái thuật toán.
5. Chưa có đường no-FE đúng nghĩa; đặt expansion_dim=64 nhưng vẫn random projection không phải bỏ FE. Gamma=0 đang bị từ chối; không thay epsilon rồi gọi là zero.

Hai lỗi của snapshot trước đã được xử lý trong PR #7: attempt mới thất bại không dùng lại pass cũ; cell cuối notebook 03 tính số cột W theo config. Khi chạy mới dùng bản 03; bản 02 giữ source snapshot gắn Version 2. Xem [báo cáo sửa trạng thái](../docs/SUA_LOI_TRANG_THAI_RUN.md).

Đây là các việc mở rộng từ một smoke runner đã đạt sang experiment runner. Chúng không làm mất giá trị những kiểm chứng hiện có.

## 5. Sản phẩm nộp còn thiếu

### Giữa kỳ — 21/11/2026, 23:59 theo hạn Sang cung cấp

Chưa thấy file deck hoàn chỉnh. [Yêu cầu reading](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reading): PPTX 20–40 slides kể cả appendix, dưới 20 MB, notes mỗi slide tối đa 500 từ, hình tự làm, trích IEEE trên từng slide có nguồn, không có references slide riêng. File deck và repo link nộp riêng qua Moodle.

Cần bổ sung bối cảnh và ít nhất hai hướng tiền nhiệm; phân loại đóng góp; hình pipeline; ý nghĩa và giới hạn định lý; độ phức tạp; kết quả tái lập đối chiếu paper với W&B run links. Đặc tả code mới chỉ bao phủ một phần các nội dung này.

### Cuối kỳ — 05/12/2026, 23:59 theo hạn Sang cung cấp

P1 về phụ thuộc tập base và P2 về bộ nhớ hiện là ý tưởng. Chưa có bộ lập luận và thực nghiệm chứng minh hạn chế để đưa lên poster.

[Yêu cầu analysis](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis) cần ít nhất ba điểm mạnh có căn cứ, hai hạn chế lý thuyết có thể kiểm nghiệm và hai hướng phát triển gắn với hạn chế. [Yêu cầu empirics](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics) ghép lập luận với bằng chứng thực nghiệm, poster một slide PPTX A2 dọc, số liệu có W&B run links; bảo vệ trực tiếp tại workshop.

## 6. Thứ tự làm cho một người

1. **Dùng source mới:** import [notebook 03](../notebooks/03_cifar100_safe_attempts.ipynb) vào Kaggle. Pipeline và mốc 60 lớp đã công bố, không cần dựng lại hoặc chạy lại chỉ để có W&B đầu tiên.
2. **Lưu đủ chuỗi 100 lớp:** đặt `RUN_INCREMENTAL_PHASES=5`, bật tracking, tiếp tục cấu hình nhỏ 2 epochs/d=256. Có thể reuse checkpoint base đã kiểm tương thích và ghi rõ reused. Kiểm đủ 6 điểm tính cả base, W256×100, 10.000 ảnh test và status complete/verification true cùng attempt ID.
3. **Ghép bằng chứng và chốt benchmark:** lưu version Kaggle, lấy JSON/checkpoint/state từ output_dir trong report, ghép đúng W&B URL và source hashes. Sau đó chốt protocol/metric, đo tài nguyên và xử lý resume/reference lớn. Bộ ghi report đã chuẩn UTF-8/LF; tính checksum theo đúng bytes của file được công bố.
4. **Hoàn thành core reproduction:** kết quả chính, ablation đã chọn và một phân tích bổ sung; đối chiếu paper/code tác giả, giải thích sai lệch. Viết nội dung deck song song khi scope và kết quả đã rõ.
5. **Hoàn thiện deck trước hạn; sau đó tập trung poster:** phát triển P1/P2 thành giả thuyết, lý thuyết, đối chứng và phép đo; chỉ đưa kết luận được bằng chứng ủng hộ.

[Đề chính thức](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction) chấp nhận cả máy cá nhân và hosted notebook. Kaggle/GPU là lựa chọn tài nguyên của kế hoạch. Mỗi lượt local/Kaggle được ghi đúng môi trường, source và trạng thái tracking của chính nó.

Việc ưu tiên tiếp theo là lưu mốc **100 lớp bằng notebook 03 với tracking**, rồi chuyển sang cấu hình benchmark đã chốt. Mốc smoke 60 lớp trên Kaggle/W&B đã có.

## 7. Cách dùng tài liệu để đỡ rối

- Dùng bản review này để xem trạng thái toàn đồ án.
- Dùng [HUONG_DAN_CIFAR_SMOKE.md](../docs/HUONG_DAN_CIFAR_SMOKE.md) khi chạy pipeline; [Dac_ta_ACIL.md](../docs/Dac_ta_ACIL.md) khi giải thích API/công thức.
- Các report review cũ mô tả snapshot trước khi sửa, còn `planning/` ở workspace ngoài repo có nhiều kế hoạch cũ. Không dùng chúng để suy rằng lỗi minibatch vẫn còn hoặc số 13 tests là hiện trạng suite.
- [Phân công](../docs/PHAN_CONG_NHOM.md) đã cập nhật cùng hiện trạng: Sang đang làm một mình; vai trò ba người và các hạn nội bộ vẫn là kế hoạch. R1/R2/R3 chỉ được coi là kết quả khi có artifact tương ứng.
