# Chạy kiểm chứng analytic trên Kaggle

1. Từ gốc repo, import `notebooks/01_verify_analytic_update.ipynb`, chọn Python và Accelerator None.
2. Bật Internet để fetch source; notebook dùng source snapshot từ repo, ghi commit/hash vào JSON.
3. Bật secret `WANDB_API_KEY` cho notebook khi LOG_WANDB=True; project là `hsangg-2006/acil-reproduction`.
4. Chạy từ đầu; các test phải đạt. Save & Run All tạo một version có output của chính lần chạy đó.
5. Tải `verify_analytic_update_v2_results.json`; kiểm tra experiment, source commit, tracking status, run ID và URL.
6. Ghép đúng file này với notebook version/source snapshot trong README. Nếu chạy lại, dùng URL mới do SDK trả, không sửa URL cố định trong code.

SDK khóa `wandb==0.28.1`. `logged` chỉ ghi sau khi run context kết thúc; lỗi logging/finish ghi `failed` rồi dừng notebook. Nếu tắt tracking, report có `not_logged` và URL null.

Notebook 01 chỉ kiểm chứng analytic tổng hợp và không cần GPU. Pipeline CIFAR-100/ResNet-32 đã có trên main; notebook 02 Version 2 đã chạy CUDA 50+10 và có W&B o2r63pj2. Để chạy ảnh thật bằng source mới, dùng [notebook 03](../notebooks/03_cifar100_safe_attempts.ipynb) và [hướng dẫn CIFAR](HUONG_DAN_CIFAR_SMOKE.md). Bản 02 giữ snapshot của lượt đã công bố; bản 03 có trạng thái từng attempt và kiểm kích thước theo config.
