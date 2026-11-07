# Review tổng quan tiến độ ACIL — 02/11/2026

> Lập ngày02/11; cập nhật trạng thái mới nhất **07/11/2026**: R1 đủ3 seeds, R2 đủ4 hàng/3 seeds (9 analytic runs mới +3 baseline R1), tracking công khai đã kiểm. [Báo cáo R1](R1_CIFAR100_K5_3SEEDS_20261106.md) · [Báo cáo R2](R2_CIFAR100_K5_20261107.md). R3 cũng hoàn tất9 phép đo/54 so sánh joint ([báo cáo](R3_CIFAR100_K5_BATCH_20261107.md)); còn đối chiếu code tác giả và sản phẩm PPTX/poster. Chi tiết smoke/ngày cũ bên dưới giữ vai trò lịch sử; không dùng chúng làm trạng thái benchmark hiện tại.

## Kết luận về vị trí hiện tại

Đã có implementation analytic được kiểm chứng, benchmark R1 đủ 3 seeds và ablation R2 đủ 4 hàng/3 seeds. R3 cũng đã hoàn tất theo kế hoạch. Chưa có artifact đối chiếu source tác giả và sản phẩm slide deck giữa kỳ/poster cuối kỳ hoàn chỉnh.

Sang hiện tự thực hiện công việc; không cần chờ bàn giao từ Vĩ/Phát. Đã xác nhận với Sang rằng bài ACIL được giảng viên duyệt. Các phân công ba người trước đây là kế hoạch lịch sử, không phải danh sách việc đã hoàn tất.

Không quy đổi số file hoặc số test thành phần trăm hoàn thành đồ án. Các bài kiểm tra số học chứng minh tính nhất quán trong những trường hợp đã kiểm, còn các thí nghiệm tái lập và phân tích trả lời câu hỏi khác.

## Căn cứ của review

- Đọc lại [trang tổng quan chính thức](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/) và đủ năm trang selection/reading/reproduction/analysis/empirics ngày 02/11/2026.
- Mốc GitHub đối chiếu cho bản cập nhật này: [main sau PR #7, f6d1459](https://github.com/24122050S/20-ML-Paper/commit/f6d145986314f21ddc7aa4bb6565a6699ccb6a94). Pipeline đã được đưa lên qua [PR #5](https://github.com/24122050S/20-ML-Paper/pull/5); README ghép bằng chứng Kaggle qua [PR #6](https://github.com/24122050S/20-ML-Paper/pull/6).
- Đối chiếu code/config, JSON CPU và [JSON Kaggle](cifar100_kaggle_results.json), [manifest Kaggle](CIFAR-KAGGLE-01-provenance.json), source bundle và W&B URL. VERIFY-02, manifest Kaggle và manifest CPU hiện khớp file được công bố. [Bản sửa checksum CPU](../docs/SUA_LOI_CHECKSUM_JSON.md) dùng Git blob LF, giữ checksum raw CRLF cũ riêng; số liệu JSON không đổi và không train lại.
- Kiểm tra bản sửa PR #7 ngày 03/11: 37 tests đạt trên local; bootstrap bundle trong kernel mới có 36 tests đạt và 1 test snapshot 02 skip vì bundle không chứa notebook lịch sử. Cell cuối nhận đúng W256×60, W256×100 và W128×100; config 100 lớp từ chối W256×60. Đây là kiểm tra code/notebook, không phải một lượt training Kaggle mới.
- Sau bản sửa UTF-8/LF: suite đầy đủ có 38 tests đạt trên local, gồm test tái hiện CRLF trên Windows; notebook 03 được đóng gói lại với bộ ghi report mới. Số 37 tests phía trên thuộc lần kiểm PR #7.
- Cập nhật 04/11: nhận JSON smoke 100, kiểm đủ 6 phase, source/input/effective config hashes và public W&B API finished/config/source/summary. Xem [review lượt mới](REVIEW_CIFAR_KAGGLE_FULL5_20261104.md). Chưa có artifact deck/poster hoàn chỉnh được ghi nhận; không tạo run hay train lại khi nhập bằng chứng.

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
| Ghép analytic trên ảnh thật | Chuỗi đủ 5 phase, 100 lớp có tracking | [JSON full5](cifar100_kaggle_full5_results.json), W cuối 256×100, test 10.000 ảnh, accuracy 22,05%, mọi check đạt; W&B blhw74ff finished |
| Tests | Suite54 tests:53 đạt,1 skip CUDA trên local CPU | Analytic, reporting gồm UTF-8/LF, CIFAR, attempt và notebook distribution; 11/20 tests thuộc snapshot VERIFY-02/CPU cũ |
| Notebook 02 | Có run Kaggle Version 2 đã đối chiếu | JSON CUDA, source hashes trong bundle, manifest và W&B; giữ bản 02 làm snapshot lịch sử |
| Notebook 03 | Full5 có JSON/source hashes và W&B xác nhận | [scriptVersionId 365084714](https://www.kaggle.com/code/hsangg206/03-cifar100-safe-attempts/output?scriptVersionId=365084714); giao diện/số version hiển thị Kaggle chưa được đọc độc lập, ghi rõ trong manifest |
| Repo/tracking nền tảng | Đã có | GitHub và project W&B đã public; repo local có Git độc lập, requirements, hướng dẫn và manifest |

Các con số của lượt CPU ban đầu trong [JSON CPU](cifar100_smoke_results.json):

| Mốc | Lớp đã thấy | Accuracy toàn bộ lớp đã thấy | Relative W error so joint | Dự đoán khác joint |
|---|---|---|---|---|
| Base | 50 | 32,68% | 2,047e-11 | 0 |
| Sau phase mới đầu tiên | 60 | 29,3667% | 1,771e-11 | 0 |

Đây là kết quả smoke với 2 epochs, expansion 256, seed 42. Số 29,37% chưa thể đem so trực tiếp với một dòng benchmark paper dùng cấu hình khác. Sai lệch joint nhỏ cũng không chứng minh accuracy lớp cũ luôn bất biến: ở phase 1, accuracy riêng base là 29,92%, riêng lớp mới là 26,60%.

Lượt CUDA được lưu riêng trong [JSON Kaggle](cifar100_kaggle_results.json): base 50 lớp đạt **32,58%**, sau phase đầu 60 lớp đạt **29,70%**, không có dự đoán khác joint, mọi check đạt. [Version 2](https://www.kaggle.com/code/hsangg206/notebook7e7946d4ee?scriptVersionId=364942764) ghép với [W&B o2r63pj2](https://wandb.ai/hsangg-2006/acil-reproduction/runs/o2r63pj2) và [manifest](CIFAR-KAGGLE-01-provenance.json). Không gán metric CPU sang lượt CUDA.

Lượt mới CIFAR-KAGGLE-02 chạy đủ 100 lớp: accuracy lần lượt 33,46% /29,9167% /27,50% /25,2250% /23,3111% /22,05% ở 50/60/70/80/90/100 lớp. Tất cả dự đoán khớp joint; [bảng và kiểm chứng](REVIEW_CIFAR_KAGGLE_FULL5_20261104.md). Đây vẫn là smoke 2 epochs/d256.

## 2. Phân biệt máy cá nhân, GitHub, Kaggle và W&B

| Mốc | Máy cá nhân | GitHub main | Kaggle | W&B |
|---|---|---|---|---|
| VERIFY-02 tổng hợp | Có | Có source/output/JSON | Version 9 đã chạy | wi4wau2d, ghi nhận Finished |
| CIFAR-SMOKE-01 CPU | Đã chạy, checkpoint/state lưu local | Có source pipeline, JSON CPU và manifest | Không áp dụng cho lượt CPU này | `not_logged`, ID/URL null |
| CIFAR-KAGGLE-01 CUDA | JSON đã tải về | Có JSON/manifest và source bundle trong notebook 02 | Version 2 / 364942764, 50+10 lớp | o2r63pj2, ghi nhận Finished; JSON `logged` |
| CIFAR-KAGGLE-02 CUDA | JSON do Sang tải và cung cấp | Có JSON/manifest/source bundle tham chiếu immutable | Link scriptVersionId365084714 do Sang cung cấp; giao diện chưa đọc độc lập | blhw74ff finished, public API khớp JSON/config/source |
| Dọn cấu trúc/đặc tả và pipeline | Đã có trong checkout | Đã lên main qua PR #5 | Source bundle của Version 2 truy nguyên bằng hashes | Không phải run riêng |
| Sửa attempt và cell kích thước | Đã kiểm tra local/kernel mới | Đã lên main qua PR #7; notebook 03 đã tạo | JSON full5 của bản 03 status complete, 100 lớp | blhw74ff finished |

Pipeline CIFAR hiện đã có trên GitHub main. `source_dirty=true` và commit cha 8e2a143 trong JSON CPU vẫn mô tả **thời điểm lượt CPU được thực thi**, không phải trạng thái source hôm nay. Lượt Kaggle có git_commit null vì chạy source giải nén từ bundle; manifest và hashes giữ liên kết tới source thực thi. Không chỉnh metadata lịch sử để giả lập một run trên commit mới.

Kaggle được ghi nhận Private trong lần đối chiếu version; repo/W&B đã được ghi nhận công khai. Cần kiểm quyền xem khi bàn giao; JSON/manifest đã có trên GitHub.

## 3. Những gì còn thiếu để tái lập thí nghiệm

[Yêu cầu reproduction](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction) cho phép full scope hoặc core scope. Với core scope, phải có tối thiểu hai thí nghiệm quan trọng, gồm phép so sánh chính, cộng một phân tích bổ sung tự thiết kế. R1/R2/R3 là nhãn kế hoạch của chúng ta, không phải tên bắt buộc trong đề.

| Hạng mục | Hiện trạng | Bằng chứng / việc còn lại |
|---|---|---|
| Scope/protocol tái lập | Đã chốt [v1.0](../docs/protocol_chot.md), core CIFAR100/K5 | R1/R2 base160/d8000/gamma0.1; R3 dùng checkpoint160/d1024/gamma0.1; sai khác và readiness gates có trong tài liệu; runner streaming/CUDA/BP5 đã đạt; budget2 giờ đã lập; tracking BP5 [pv82dy1f](https://wandb.ai/hsangg-2006/acil-reproduction/runs/pv82dy1f) Finished, đã đối chiếu API công khai |
| Kết quả chính R1 | Hoàn tất42/43/44, n=3 | Ā65,1196 ± 0,6104%, final56,1300 ± 0,5575%, F_base8,3067 ± 0,6201pp (sample SD, ddof=1); [báo cáo](R1_CIFAR100_K5_3SEEDS_20261106.md)/[summary](r1_cifar100_k5_summary.json), cả3 public W&B Finished |
| Mốc smoke 100 lớp | Đã có JSON/manifest; W&B blhw74ff finished và config/source/summary khớp | Giữ Output checkpoint/state; quyền xem và số version hiển thị Kaggle còn cần kiểm trước nộp. [Manifest](CIFAR-KAGGLE-02-provenance.json) phân biệt phần đã kiểm và phần user cung cấp |
| Ablation R2 | Hoàn tất4 hàng/3 seeds | [Báo cáo/hình](R2_CIFAR100_K5_20261107.md), [summary](r2_cifar100_k5_summary.json), [manifest](R2-C100-K5-provenance.json);9 run mới và3 baseline R1, không train thêm backbone; phạm vi4/5 Table2, gamma0 ngoài scope |
| Phân tích bổ sung R3 | Hoàn tất9 timing runs/54 so sánh joint | Batch32/128/512,d1024,gamma0.1,seed42;3timing repeats/batch, oracle riêng,0 argmax khác; [báo cáo](R3_CIFAR100_K5_BATCH_20261107.md)/[summary](r3_cifar100_k5_summary.json),9W&B Finished/công khai |
| Đối chiếu code tác giả | Chưa có bằng chứng chạy | Môi trường riêng, commit/config pin, so output và định vị khác biệt; không chép core implementation |
| Metrics tổng hợp | Đã chốt [metrics v1.0](../docs/metrics.md), ví dụ smoke đã tính lại | A-bar có base; F-base/delta có dấu rõ; summary R1/R2/R3 đã có; R1/R2 thống kê3 seeds, R3 thống kê3 timing repeats trên1 seed, đều tính lại từ JSON thật |
| Seeds benchmark | R1 và R2 đủ seeds42/43/44 | R1 train base mới từng seed; R2 dùng checkpoint cùng seed. Mean ± sample SD với n=3, ddof=1; không lấy phase hoặc unit test làm seed benchmark. R3 chỉ dùng1 seed với3 timing repeats |
| Công bố kết quả tiếp theo | R1 đủ3 seeds | [Manifest tổng hợp](R1-C100-K5-3SEEDS-provenance.json) nối3 manifest/version/source/checkpoint/W&B; R2/R3 đã đủ theo kế hoạch; tiếp theo đối chiếu tác giả và sản phẩm nộp |

Một checkpoint 2 epochs không phải checkpoint benchmark hoàn chỉnh. Chạy thêm đủ phase với chính checkpoint đó trước hết vẫn là kiểm tra pipeline đầy đủ.

## 4. Những giới hạn code cần xử lý trước lượt chạy lớn

1. `src/cifar_smoke.py` hiện giới hạn K=5; K=25 trong roadmap không thể hoàn thành chỉ bằng đổi một giá trị JSON.
2. `--reuse-checkpoint` dùng lại base đã hoàn thành. Đã có checkpoint v2 mỗi epoch và resume đủ optimizer/scheduler/RNG/loader. CPU continuous so với resume và CUDA TinyCNN test đều đạt; BP5 GPU/checkpoint đã kiểm. Xem review BP5 để phân biệt phạm vi từng phép kiểm.
3. Analytic vẫn là NumPy CPU. GPU giúp base training/trích feature; cần đo riêng chi phí analytic trước khi tăng expansion.
4. Mode smoke giữ oracle để kiểm nhỏ; mode benchmark mới đọc batch, không giữ train history và không gọi oracle. Đã kiểm đủ100 lớp CPU; phân biệt peak process RAM với bytes R/W/FE.
5. Đã có no-FE đúng nghĩa: dùng F64 trực tiếp, không ReLU/projection; có test. Gamma=0 đang bị từ chối; không thay epsilon rồi gọi là zero.

Hai lỗi của snapshot trước đã được xử lý trong PR #7: attempt mới thất bại không dùng lại pass cũ; cell cuối notebook 03 tính số cột W theo config. Khi chạy source mới dùng bản04; bản02/03 giữ source snapshot của các bằng chứng Kaggle. Xem [báo cáo sửa trạng thái](../docs/SUA_LOI_TRANG_THAI_RUN.md).

Đây là các việc mở rộng từ một smoke runner đã đạt sang experiment runner. Chúng không làm mất giá trị những kiểm chứng hiện có.

## 5. Sản phẩm nộp còn thiếu

### Giữa kỳ — 21/11/2026, 23:59 theo hạn Sang cung cấp

Chưa thấy file deck hoàn chỉnh. [Yêu cầu reading](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reading): PPTX 20–40 slides kể cả appendix, dưới 20 MB, notes mỗi slide tối đa 500 từ, hình tự làm, trích IEEE trên từng slide có nguồn, không có references slide riêng. File deck và repo link nộp riêng qua Moodle.

Cần bổ sung bối cảnh và ít nhất hai hướng tiền nhiệm; phân loại đóng góp; hình pipeline; ý nghĩa và giới hạn định lý; độ phức tạp; kết quả tái lập đối chiếu paper với W&B run links. Đặc tả code mới chỉ bao phủ một phần các nội dung này.

### Cuối kỳ — 05/12/2026, 23:59 theo hạn Sang cung cấp

P1 về phụ thuộc tập base và P2 về bộ nhớ hiện là ý tưởng. Chưa có bộ lập luận và thực nghiệm chứng minh hạn chế để đưa lên poster.

[Yêu cầu analysis](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis) cần ít nhất ba điểm mạnh có căn cứ, hai hạn chế lý thuyết có thể kiểm nghiệm và hai hướng phát triển gắn với hạn chế. [Yêu cầu empirics](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics) ghép lập luận với bằng chứng thực nghiệm, poster một slide PPTX A2 dọc, số liệu có W&B run links; bảo vệ trực tiếp tại workshop.

## 6. Thứ tự làm cho một người

1. **Đối chiếu code tác giả:** chạy trong môi trường riêng theo protocol, pin commit/config/checkpoint và lưu output; xác định khác biệt theo từng tầng. Chỉ đọc source chưa được ghi là đã chạy.
2. **Ghép cả R1/R2/R3 vào deck:** dùng bảng/hình tự làm và run links; [R1](R1_CIFAR100_K5_3SEEDS_20261106.md), [R2](R2_CIFAR100_K5_20261107.md), [R3](R3_CIFAR100_K5_BATCH_20261107.md). R1/R2 dùng3 seeds; R3 dùng3 timing repeats trên1seed. R2 là4/5 Table2, không có gamma0.
3. **Hoàn thiện đọc hiểu và phân tích sai lệch:** nối công thức với code, điều kiện tương đương joint, khác biệt protocol và giới hạn. Nêu giả thuyết giải thích riêng với nguyên nhân đã kiểm.
4. **Nghiệm thu PPTX rồi poster:** kiểm nội dung/notes/nguồn/định dạng/hạn chính thức. Hoàn tất ba bộ thực nghiệm không đồng nghĩa toàn bộ yêu cầu môn đã hoàn tất.

[Đề chính thức](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction) chấp nhận cả máy cá nhân và hosted notebook. Kaggle/GPU là lựa chọn tài nguyên của kế hoạch. Mỗi lượt local/Kaggle được ghi đúng môi trường, source và trạng thái tracking của chính nó.

[Protocol](../docs/protocol_chot.md) và [metrics](../docs/metrics.md) giữ v1.0. R1 đã đủ42/43/44 với mean±sample_SD(ddof=1), [báo cáo tổng hợp](R1_CIFAR100_K5_3SEEDS_20261106.md)/[summary](r1_cifar100_k5_summary.json). R2/R3 đã hoàn tất theo các báo cáo. Việc ưu tiên tiếp theo là **đối chiếu source tác giả và hoàn thiện PPTX**; còn deck/poster và phân tích theo yêu cầu môn. Các báo cáo nghiệm thu từng seed giữ vai trò snapshot lịch sử, không thay bằng số tổng hợp.

## 7. Cách dùng tài liệu để đỡ rối

- Dùng bản review này để xem trạng thái toàn đồ án.
- Dùng [HUONG_DAN_CIFAR_SMOKE.md](../docs/HUONG_DAN_CIFAR_SMOKE.md) khi chạy pipeline; [Dac_ta_ACIL.md](../docs/Dac_ta_ACIL.md) khi giải thích API/công thức.
- Các report review cũ mô tả snapshot trước khi sửa, còn `planning/` ở workspace ngoài repo có nhiều kế hoạch cũ. Không dùng chúng để suy rằng lỗi minibatch vẫn còn hoặc số 13 tests là hiện trạng suite.
- [Phân công](../docs/PHAN_CONG_NHOM.md) đã cập nhật cùng hiện trạng: Sang đang làm một mình; vai trò ba người và các hạn nội bộ vẫn là kế hoạch. R1/R2/R3 chỉ được coi là kết quả khi có artifact tương ứng.
