# Đặc tả phần học tăng dần của ACIL

## 1. Mục tiêu

Ở mỗi phase, mô hình học thêm các lớp mới bằng dữ liệu của phase hiện tại. Sau cập nhật, mô hình phân loại được **tất cả các lớp đã xuất hiện**, không cần đọc lại dữ liệu huấn luyện của các phase trước.

## 2. Đầu vào mỗi phase $k$

- $H_k\in\mathbb{R}^{N_k\times d}$: đặc trưng mở rộng của $N_k$ mẫu mới.
- $Y_k\in\mathbb{R}^{N_k\times c_k}$: nhãn one-hot cho $c_k$ lớp mới.
- $R_{k-1}$ và $W_{k-1}$: trạng thái từ phase trước.

Đặc trưng được tính bằng:

$$
H_k=\operatorname{ReLU}\left(\operatorname{Flatten}(\operatorname{CNN}(X_k))W_{\mathrm{fe}}\right).
$$

Các lớp của phase mới không trùng với các lớp đã học.

## 3. Trạng thái giữ lại

| Thành phần | Kích thước | Ý nghĩa |
| --- | --- | --- |
| $R_{k-1}$ | $d\times d$ | Ma trận nghịch đảo tương quan đặc trưng có regularization |
| $W_{k-1}$ | $d\times C_{k-1}$ | Trọng số phân loại toàn bộ lớp cũ |
| Danh sách lớp | $C_{k-1}$ nhãn | Ánh xạ cột dự đoán về tên hoặc mã lớp |

Trong đó $C_{k-1}$ là tổng số lớp đã học. Không giữ ảnh hay đặc trưng của từng mẫu cũ để phát lại.

## 4. Thành phần cố định

- Backbone CNN đã huấn luyện ở phase 0.
- Ma trận mở rộng đặc trưng $W_{\mathrm{fe}}$.
- Hàm kích hoạt ReLU.
- Hệ số regularization $\gamma>0$.
- Cách tiền xử lý và trích xuất đặc trưng.

Backbone chạy ở chế độ đánh giá khi trích xuất đặc trưng để tránh thay đổi trạng thái như thống kê BatchNorm.

## 5. Khởi tạo từ phase 0

$$
R_0=(H_0^TH_0+\gamma I)^{-1},
$$

$$
W_0=R_0H_0^TY_0.
$$

Lưu $R_0,W_0$ và thứ tự các lớp ban đầu.

## 6. Xử lý khi có phase mới

Đầu tiên, cập nhật $R$:

$$
R_k=R_{k-1}-R_{k-1}H_k^T(I+H_kR_{k-1}H_k^T)^{-1}H_kR_{k-1}.
$$

Sau đó cập nhật classifier:

$$
W_k=\left[W_{k-1}-R_kH_k^TH_kW_{k-1}\quad R_kH_k^TY_k\right].
$$

Hai khối được **ghép theo chiều ngang**: khối trái cho các lớp cũ, khối phải cho các lớp mới.

## 7. Đầu ra và dự đoán

Trạng thái mới:

$$
R_k\in\mathbb{R}^{d\times d},\qquad W_k\in\mathbb{R}^{d\times(C_{k-1}+c_k)}.
$$

Với đặc trưng kiểm tra $H_{\mathrm{test}}$, tính điểm:

$$
S=H_{\mathrm{test}}W_k.
$$

Mỗi mẫu được gán vào lớp có điểm cao nhất:

$$
\hat y_j=\operatorname*{argmax}_{c}S_{j,c}.
$$

Dùng danh sách lớp đã lưu để chuyển chỉ số cột thành nhãn thực tế. Các điểm này không mặc nhiên là xác suất.

## 8. Tiêu chí kiểm tra đúng

- Kích thước $R$ giữ nguyên; số cột $W$ tăng đúng bằng số lớp mới.
- Sau mỗi phase, đánh giá trên **toàn bộ lớp đã gặp**.
- Với cùng đặc trưng, thứ tự lớp và $\gamma$, nghiệm cập nhật phải gần nghiệm ridge học gộp trong sai số số học.
- Quá trình cập nhật không truy cập dữ liệu của các phase trước.
