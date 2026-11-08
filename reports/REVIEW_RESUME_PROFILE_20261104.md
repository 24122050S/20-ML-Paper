# Nghiệm thu resume, benchmark streaming và profile d8000

Ngày 04/11/2026; triển khai theo [protocol v1.0](../docs/protocol_chot.md). [Manifest](RESUME-PROFILE-01-provenance.json) lưu checksum, phạm vi và source bundle. Các lượt dưới đây chạy **local CPU**, tracking tắt; không gắn chúng với version Kaggle hay W&B của smoke cũ.

## Kết quả đã kiểm

| Phép kiểm | Kết quả | Phạm vi |
|---|---|---|
| Unit suite |54 tests:53 đạt,1 skip CUDA|CPU; gồm resume/RNG/shuffle, atomic save, checkpoint sai điều kiện, streaming/no-FE/metrics và lỗi trạng thái/tracking|
| Bootstrap notebook 04|Kernel mới giải nén bundle 30 file,54 tests:52 đạt,2 skip|Chạy cells 0..5; skip CUDA và test snapshot 02/03 không nằm trong bundle. Chưa chạy action Kaggle.|
| ResNet32/CIFAR resume|2 epochs liên tục =1+resume 1, chính xác từng tensor model/BatchNorm, optimizer/scheduler/RNG/loader state; history ngoài thời gian giống|200 ảnh base thật/epoch, crop/flip, CPU. Checkpoint đầu vào không bị thay đổi. [JSON](resume_cpu_verification.json)|
| Streaming full 100|Accuracy khớp oracle CPU lịch sử ở cả 6 mốc; W cuối 256×100, test 10.000 ảnh; train history cache=0|Reuse backbone 2 epochs, d256. Chặn `joint_ridge` và `extract_features` trong lượt kiểm. [JSON mới](cifar100_streaming_cpu_verification.json), [oracle lịch sử](cifar100_joint_cpu_reference.json)|
| Profile d8000|R8000×8000, W8000×100, FE64×8000; state hữu hạn, backbone giữ nguyên|1000 ảnh base +200 ảnh từ 50 lớp mới +600 test, analytic batch 128, float64,4 threads. [JSON](profile_d8000_cpu_results.json)|

Accuracy CPU streaming lần này là 32,68→29,3667→26,9714→24,3875→22,9556→21,49%; A-bar 26,308525%, base forgetting 7,62pp. Đây là kiểm runner với backbone CPU lịch sử, **khác** run Kaggle blhw74ff có final 22,05%. Không thay hoặc ghép số giữa hai lượt; cả hai đều chưa phải benchmark 160 epochs/d8000.

## Profile tài nguyên thực đo

Lượt công bố: attempt `20261104T122103104596Z-23b7ebfbad12`; Python 3.12.7, Torch 2.11.0+cpu, NumPy 2.1.3, SciPy 1.17.0. Dữ liệu/checkpoint có checksum trong JSON.

| Stage | Giây |
|---|---:|
| Trích feature subset |4,3453|
| Tích lũy thống kê base 1000 ảnh |22,7857|
| Cholesky và giải R/W |12,3791|
| Incremental 200 ảnh |4,2576|
| Chấm 600 ảnh |0,0760|
| Toàn cửa sổ profile, gồm kiểm tra |44,7485|

Peak CPU RSS sampled 20ms là **2.111.639.552 bytes** (~2,11GB,1,97GiB). Ba ma trận R/W/FE chiếm **522.496.000 bytes** (~522,50MB,498,29MiB). RSS bao gồm bộ nhớ process và các workspace/mảng tạm; không dùng kích thước R làm peak RAM. CUDA memory=null trên CPU. Có thể bỏ lỡ đỉnh ngắn hơn 20ms; không coi phép đo là cam kết tài nguyên Kaggle.

Ngoại suy tuyến tính trong JSON chỉ giúp nhận diện quy mô chi phí. Nó chưa gồm full evaluation, training BP160, artifact I/O và biến động host. Analytic vẫn chạy NumPy/SciPy **CPU**; GPU giúp CNN. Profile subset/cache F64 riêng để tách timing CNN khỏi analytic, không lưu R lớn ra NPZ. Runner chính dùng iterator CNN xen kẽ nên các timing bao hàm đọc iterator không được cộng trùng với feature extraction.

## Vì sao triển khai như vậy

**Checkpoint ở biên epoch:** dễ xác định phần đã hoàn thành và phát lại chính xác epoch kế tiếp với `num_workers=0`. Mỗi checkpoint chứa trạng thái SGD momentum/scheduler cùng tất cả RNG và generator của loader. Chỉ lưu model sẽ làm mất LR hoặc thứ tự crop/shuffle. Ghi atomic giữ checkpoint cũ khi ghi file lỗi; exception giữa epoch giữ epoch trước. `STOP_AFTER_EPOCH` chủ động làm attempt failed vì cả pipeline còn dở, đồng thời để lại checkpoint có thể resume.

**Tách reuse và resume:** checkpoint base hoàn tất được dùng lại cho analytic; checkpoint đang dở dùng resume và phải giữ nguyên target/config/runtime/source. Checkpoint cũ thiếu optimizer/RNG không thể được suy diễn thành resume. Checkpoint subset 200 ảnh chỉ phục vụ kiểm cơ chế, bị chặn khi reuse cho full CIFAR benchmark.

**Streaming base:** cộng thống kê Gram/cross theo batch cho đúng objective ridge, cộng gamma một lần. SciPy Cholesky giải hệ đối xứng xác định dương và tạo R cần cho Woodbury; không thay công thức ACIL bằng phương pháp học khác. Test so R/W với nghiệm direct, kiểm weak references để xác nhận không giữ đầu vào. Joint oracle tiếp tục ở smoke/test, không chạy trong đường benchmark.

**No-FE và metric:** no-FE lấy feature backbone 64 chiều trực tiếp. Export phân biệt A-bar toàn chuỗi và trung bình partial; field oracle chưa đo=null. Các quyết định này giữ ý nghĩa thí nghiệm và tránh báo một phép kiểm chưa chạy là đạt.

## Sang làm tiếp

Theo đúng [hướng dẫn notebook 04](../docs/RESUME_BENCHMARK_PROFILE.md): attach Output 03 chứa base checkpoint, chạy action mặc định `profile_analytic`, lưu version/JSON/W&B. Test CUDA phải đạt trên Kaggle. Tiếp đó action `profile_base` đo 5 epochs; đối chiếu quota và dự phòng 30% rồi mới chạy benchmark seed 42. Chưa có lượt 160 epochs hoặc nghiệm thu G6 Kaggle trong báo cáo này.

Notebook 02/03 và các JSON/manifest đã công bố trước giữ nguyên bytes. Source mới có hash khớp bundle 04; JSON giữ `source_dirty=true` và commit gốc tại lúc chạy, không thay bằng commit được tạo sau đó.
