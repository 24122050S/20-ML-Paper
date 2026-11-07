# Kế hoạch đối chiếu implementation tác giả — 07/11/2026

Chốt trước khi chạy số. Mục tiêu là thực thi source tác giả và định vị điểm khác với implementation nhóm; không train lại R1/R2/R3 hoặc chép code lõi tác giả vào `src/`.

## Source và môi trường

- Repo ngoài project: `ZHUANGHP/Analytic-continual-learning`, commit **0afd6bf87622e808ce7f730f8556e78229b9e758** như protocol v1.0. Checkout riêng, không sửa file tracked; hash source và kiểm Git trước/sau.
- Đây là bản repo cập nhật năm2024 có ACIL/G-ACIL hỗ trợ minibatch, không gọi nó là snapshot source năm2022.
- Venv riêng: Python3.12 của máy, PyTorch2.5.1 CPU/torchvision0.20.1, NumPy2.1.3, SciPy1.14.1. Author YAML yêu cầu Python3.11 và torch>=2.2; khác Python được công bố, không gọi là môi trường gốc bitwise. Môi trường R1/R2/R3 giữ nguyên.
- Analytic float64, CPU4threads; không dùng GPU/quota Kaggle cho lượt đối chiếu này. Dùng checkpoint160 seed42 và fixture full CIFAR100 đã pin của R3 (50.000train/10.000test).

## Hai lượt thực thi chính

| ID | Code thực thi | Điều kiện kiểm soát / câu hỏi |
|---|---|---|
| C1 — controlled | `analytic.ACIL.ACIL.fit/forward/update` từ source tác giả | F64/mapping/normalization/FE1024 N(0,1)/gamma0.1 và thứ tự ảnh giống R3/batch128. Nạp FE qua buffer ở ranh giới, không sửa thuật toán. So R/W/score/prediction tại6mốc với state R3 và joint đã kiểm |
| C2 — native entry point | Hàm `main.main(args)` và `ACILLearner` nguyên bản | Full cache F64 từ checkpoint nhóm; d1024/gamma0.1; FE khởi tạo mặc định của tác giả, seed42; shuffle minibatch nguyên bản. Adapter dataset trả `Features` cho nguồn cache pin, observer lưu state/metadata. Không đổi source core. So với analytic nhóm và joint độc lập dùng chính FE lấy từ tác giả |

`main.main` là entry point thật trong file `main.py`, gọi bằng Python với dictionary args được lưu. Các boundary adapter phải công bố: nguồn dataset/cache thay cho việc tải lại ảnh qua torchvision; backbone bridge từ checkpoint nhóm; factory/observer ghi state; guard cấm BP mới. Không gọi C2 là benchmark fresh training300epochs hay score nguyên bản của tác giả. Không lấy Ād1024 thay R1d8000.

## Chẩn đoán từng tầng

1. **Mapping:** chạy `DatasetWrapper/Features` với shuffle_seed42 trên fixture có đủ100nhãn; kiểm lớp được chọn và nhãn trả về. Nếu phát hiện nhãn không được remap về base0..49, tái hiện lỗi bằng `ACILLearner.base_training` trên fixture nhỏ, giữ exception/log và gọi đây là lỗi có điều kiện. Không sửa source tác giả để che lỗi.
2. **Preprocessing:** so transform CIFAR100 nguyên bản với normalization base-only của nhóm trên1.000ảnh test cân bằng10ảnh/lớp; chỉ đo diagnostic, không tuning hoặc thay recipe R1.
3. **Backbone:** chuyển tên tensor checkpoint (`stage`→`layer`, `classifier`→`fc`) và strict-load vào ResNet32 tác giả; cùng tensor ảnh chuẩn hóa thì so feature/logits trên1.000ảnh. Kiểm state không đổi; chưa train backbone theo recipe tác giả.
4. **FE:** ghi hash/phân phối/shape của FE N(0,1) và FE native; so phép biến đổi NumPy với forward thực của buffer tác giả. Không quy chênh lệch C1/C2 chỉ cho scale vì direction/phân phối và batch order cũng khác.
5. **Analytic:** C1 đối chiếu R3; C2 so implementation nhóm và joint trên cùng FE. Cache/oracle giữ lịch sử chỉ phục vụ kiểm chứng, không gọi là learner replay.
6. **Metric:** accuracy all-seen50..100lớp, base accuracy, Ā6mốc, F_base; kiểm CSV `IL.csv` của main và quy đổi0..1→0..100.

## Tiêu chí, bằng chứng và failure policy

- R/W/score: **rtol1e-6, atol1e-8**, ngoài ra lưu relative Frobenius error/max absolute error và số argmax khác; yêu cầu0argmax khác trong phép đối chiếu cùng FE.
- FE float64: rtol1e-12, atol1e-10. Backbone FP32: rtol1e-5, atol1e-6. Không nới tolerance sau khi xem kết quả; thất bại được giữ để phân tích.
- Mỗi lượt lưu6state NPZ, per-phase JSON, config, runtime, source/fixture/checkpoint hashes và log. Suite dùng attempt riêng và running/failed/complete; không dùng kết quả cũ khi lượt mới lỗi.
- Dữ liệu ảnh chẩn đoán được chọn theo nhãn/index trước kết quả, không chọn ảnh theo độ khớp. Trọng số và cache không commit Git.
- Hai lượt chính có W&B riêng trong project hiện tại, log từ kết quả đã kiểm; giữ source numeric và metadata tracking riêng, xác minh Finished/public/config/history sau logging. Không chạy lại số chỉ để sync.
- Báo cáo phải phân biệt: thực thi code tác giả đã hoàn tất; tương đương trong điều kiện kiểm soát có đạt hay không; benchmark native fresh training chưa chạy. Kết quả chỉ n=1/d1024.

## Bổ sung sau phép chẩn đoán đầu tiên, trước khi chạy C1/C2

Attempt đầu dừng do assertion ở kiểm feature/logits backbone FP32; chưa chạy C1/C2. Giữ traceback/source hash/failed attempt. Không đổi ngưỡng FP32 đã chốt. Sửa observer để **ghi lại kết quả không khớp thay vì chặn hai phép analytic độc lập**, thêm hook đo tensor ngay trước global pooling trên1.000ảnh, và follow-up FP64 trên100ảnh (1ảnh/lớp) với rtol/atol1e-12. Các phép mới dùng để định vị sai khác, không thay thế điều kiện FP32 ban đầu. Suite `verification_passed` chỉ có phạm vi tương đương analytic C1/C2; trạng thái backbone và các failure/difference phải báo riêng. Đây là bổ sung chẩn đoán có phiên bản, không thay FE/gamma/d hoặc điều kiện chấp nhận analytic sau khi thấy score.

## Follow-up sau C1/C2, trước phép nạp state base

C1 đã chạy đủ6phase: R/W đều đạt, prediction giống nhau nhưng score pointwise tại phase3/4 không đạt tolerance; C2 native FE đạt6phase. Giữ C1=false/suite failed theo gate ban đầu, không nới tolerance hoặc đổi score. Thực hiện **diagnostic hậu nghiệm C1R**: đếm số phần tử score vượt ngưỡng của C1; nạp R/W sau base của nhóm rồi gọi nguyên bản tác giả trên5phase incremental, cùng FE/order/batch. Nếu sai khác giảm mạnh sau khi căn chỉnh base state thì bằng chứng định vị phần lớn sai khác đã có ở base accumulation. Phép này không chứng minh nguyên nhân riêng là inverse, dot product hoặc thư viện và không thay thế run cold-start C1. Chỉ báo5cập nhật được kiểm, phase0 là state tiêm vào, không đếm thành một chứng minh mới. C1R có tracking riêng và nhãn posthoc diagnostic; không tính thêm seed hoặc benchmark độc lập.
