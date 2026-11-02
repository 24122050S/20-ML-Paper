# Sửa lỗi lượt mới thất bại nhưng kết quả cũ vẫn báo thành công

Ngày 03/11/2026. Phạm vi: vòng đời và lưu artifact của một lần gọi CIFAR pipeline.

## Lỗi cũ

Lượt seed42 đã complete. Gọi lượt seed43 vào cùng output, nhưng checkpoint không tồn tại. Runner ghi config/protocol mới rồi phát sinh FileNotFoundError trước khi thay results/progress. Người đọc chỉ hai file kết quả thấy seed42 verification_passed=true và stage complete của lượt trước.

## Thiết kế mới

`src/run_attempt.py` quản lý một context cho mỗi lần gọi `run()`, trước cả bước đọc/validate config, tải dữ liệu và chuẩn bị tracking.

```text
cifar100_smoke/
├── latest.json              # ID và trạng thái của lượt mới nhất
├── progress.json            # status/stage/checks_passed của cùng ID
├── results.json             # summary của cùng ID; running/failed luôn false
└── attempts/
    ├── <id-1>/              # config, protocol, audit, model, state, result riêng
    ├── <id-2>/
    └── legacy-<id>/         # giữ nguyên JSON flat của phiên bản cũ nếu có
```

- ID gồm thời gian UTC và UUID; artifact của các lượt không ghi đè nhau.
- Ngay khi vào context, root/results bị thay bằng marker running với verification_passed=false, không có metric hay W&B URL cũ.
- Progress ghi stage hiện tại: config_load, validating, tracking_setup, download_and_audit, checkpoint_load, base_training, checkpoint_reload, analytic, tracking.
- Chỉ sau khi numeric checks và logging/finish hoàn tất mới complete và công bố report đầy đủ.
- Exception hoặc Ctrl+C ghi failed và error_type/failed_stage, verification_passed=false, rồi ném lại exception gốc. Không nuốt lỗi để notebook hiện thành công.
- Report sau lỗi W&B có thể giữ numerical_verification_passed để phân biệt kiểm số học đã đạt với cả lượt đã thất bại. Không tái sử dụng URL của lượt cũ.
- Mỗi attempt luôn cập nhật thư mục riêng. Khóa FileLock bảo vệ cập nhật metadata root: một lượt cũ kết thúc muộn không được thay kết quả của lượt mới hơn.
- Metadata/report được ghi UTF-8/LF qua file tạm cùng thư mục và replace để hạn chế JSON ghi dở.

Root/latest, progress và results được kiểm bằng cùng attempt ID. Các file được thay từng file, không phải transaction chung ba file; nếu đọc đúng lúc cập nhật, ID/status có thể chưa đồng bộ trong một khoảng ngắn. Khi đó chờ đọc lại, không coi là complete.

Nếu kernel/process bị tắt cứng, code không thể ghi failed; trạng thái có thể dừng ở running. Nó không được coi là một lượt đã đạt. Lần gọi mới vẫn nhận ID riêng.

## Tương thích và bằng chứng cũ

JSON flat cũ được copy nguyên bytes sang thư mục legacy trước khi root summary thay đổi. Checkpoint/state cũ vẫn ở root để đường dẫn reuse đã có hoạt động. Lượt mới chỉ ghi config/model/state vào thư mục riêng; dùng `output_dir` trong report để tìm đúng artifact.

Notebook 02 và JSON/manifest của những run đã công bố được giữ nguyên. Generator tạo notebook 03 mới và chặn ghi đè notebook 02, vì manifest Kaggle Version 2 dẫn tới source bundle của bản 02.

Notebook 03 gọi login W&B qua callback bên trong attempt; nếu secret/login lỗi thì vẫn có failed status. Cell cuối kiểm kích thước và số phase theo config, hỗ trợ cả mốc 60 và 100 lớp. Mặc định vẫn smoke một phase mới, 2 epochs, expansion256.

## Kiểm tra đã thực hiện

- 37 tests đầy đủ trên máy, gồm test seed42 complete → seed43 thiếu checkpoint; sai seed checkpoint; lỗi config/JSON/download; lỗi giữa training; W&B finish; Ctrl+C; hai attempt ghi metadata theo thứ tự kết thúc khác nhau; config overrides; source bundle và snapshot cũ.
- Một lượt ảnh thật dùng checkpoint CPU 2 epochs có sẵn chạy base + một phase mới, đạt W256×60, test6.000 ảnh, accuracy29,3667%, dự đoán khớp joint.
- Gọi tiếp seed43 với checkpoint thiếu vào cùng output: root ghi failed, seed43, verification_passed=false; metric và URL cũ không bị gắn sang lượt mới. Report của attempt đã complete, JSON legacy và checkpoint cũ giữ nguyên bytes.

Các test của phần đóng gói cần nbformat từ requirements-dev. Trong source bundle Kaggle không kèm notebook 02, test bất biến của snapshot 02 được skip có lý do; các test runtime của attempt vẫn chạy. Notebook mới chưa tự được gọi là một run Kaggle đã hoàn tất.

## Người dùng làm gì?

CLI vẫn dùng `python -m src.cifar_smoke`. Sau khi chạy, mở root/latest và results; nhận kết quả khi cùng attempt ID, status complete và verification_passed true. Lấy file từ output_dir ghi trong report.

Trên Kaggle, import `notebooks/03_cifar100_safe_attempts.ipynb` để dùng source mới. Bản notebook đã save trên Kaggle không tự thay đổi chỉ vì GitHub được cập nhật. Chi tiết path và checkpoint: [hướng dẫn CIFAR](HUONG_DAN_CIFAR_SMOKE.md).

Nguồn hạ tầng khóa file: [filelock](https://py-filelock.readthedocs.io/en/latest/). Công thức analytic, dữ liệu và hyperparameter smoke không thay đổi bởi bản sửa trạng thái.
