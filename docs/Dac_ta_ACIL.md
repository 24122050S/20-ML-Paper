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

## 9. Hợp đồng giữa pipeline CNN và analytic head

### 9.1. Đặc trưng H

ResNet-32 trả `F` kích thước `(n,64)` sau global average pooling. Backbone tính bằng float32; pipeline chuyển F sang float64 trước bước analytic:

$$
H=\operatorname{ReLU}(F W_{\mathrm{fe}}),\qquad H\in\mathbb{R}^{n\times d}.
$$

Smoke test dùng `d=256`, nên `W_fe` có shape `(64,256)`. Ma trận này sinh một lần bằng RNG riêng `seed+1000`; giữ nguyên qua mọi phase. H phải hữu hạn, không rỗng. Trích feature không crop/flip ngẫu nhiên; normalization được tính từ base train và dùng chung train/test.

### 9.2. Nhãn y và class_order

Pipeline sinh một hoán vị cố định `class_order` của 100 nhãn CIFAR gốc, rồi xây ánh xạ nghịch đảo:

```python
original_to_global[class_order] = np.arange(100)
y_global = original_to_global[y_original]
```

Train và test dùng **cùng** mapping. Với seed 42: nhãn CIFAR 59 ánh xạ thành 0; 21 thành 1; 83 thành 50. Base dùng ID `0..49`, phase mới đầu tiên dùng `50..59`. Không đặt lại nhãn phase mới thành `0..9`.

Cấu hình chia 50 lớp còn lại thành 5 phase × 10 lớp; smoke hiện chỉ thực thi phase mới đầu tiên. `protocol.json` lưu đủ permutation, mapping và nhóm lớp để chạy lại.

### 9.3. Ai quản lý class_ids?

**API analytic chỉ trả hai mảng `(R,W)`.** `class_ids` không phải giá trị trả về hoặc trường trạng thái bên trong `src/analytic.py`. Pipeline quản lý metadata lớp bằng `class_order` và `original_to_global`, lưu riêng trong protocol/checkpoint.

Với C lớp đã thấy, `class_order[:C]` là nhãn CIFAR gốc tương ứng với C cột của W. Dự đoán cột `j` được đổi lại nhãn CIFAR bằng `class_order[j]`.

Module không tự nén nhãn rời rạc: truyền trực tiếp nhãn gốc `[3,97]` có thể làm W cần tới 98 cột. Vì vậy pipeline phải remap trước khi gọi analytic. Module suy số cột từ nhãn lớn nhất hoặc tham số `n_classes`, không suy từ số nhãn khác nhau trong từng minibatch.

### 9.4. API cho base và minibatch

```python
r, w = initialize(h_base, y_base_global, gamma=0.1, n_classes=50)
# Phase đầu: dù batch chỉ có một số lớp, giữ nguyên 60 cột cho toàn phase.
r, w = incremental_update(r, w, h_batch, y_batch_global, n_classes=60)
```

| Mốc | R | W | Nhãn train hiện tại | Nhãn test |
|---|---|---|---|---|
| Analytic base | `(d,d)` | `(d,50)` | `0..49` | `0..49` |
| Sau phase mới đầu tiên | `(d,d)` | `(d,60)` | `50..59` | `0..59` |

Một lớp xuất hiện lại trong nhiều minibatch tiếp tục dùng đúng cột cũ. Không nối thêm một nhóm cột sau mỗi lần gọi. Gamma chỉ được cộng ở khởi tạo, không cộng lại theo batch.

### 9.5. Công thức dùng nhãn toàn cục

Mục 6 dùng one-hot cục bộ của các lớp mới và ghép hai khối khi cập nhật trọn phase. API thực tế tạo one-hot toàn cục Y có C cột, đệm W cũ bằng 0 thành W_pad rồi dùng:

$$
R_{new}=R_{old}-R_{old}H^T(I+HR_{old}H^T)^{-1}HR_{old},
$$

$$
W_{new}=W_{pad}+R_{new}H^T(Y_{global}-HW_{pad}).
$$

Công thức residual tương đương công thức ghép khối khi batch chỉ có lớp mới và xử lý được lớp xuất hiện lại. Hàm không sửa các mảng trạng thái đầu vào.

### 9.6. Điều kiện giữ nguyên và checkpoint

Sau base training, tắt gradient **và** chuyển mọi module sang `eval()`. BatchNorm không được cập nhật running statistics. Kiểm tra fingerprint của toàn bộ tham số/buffer trước và sau analytic; expansion cũng có checksum. Giữ nguyên feature map, gamma và preprocessing trong suốt chuỗi cập nhật.

Chỉ tái sử dụng checkpoint khi cấu hình base, class order, normalization và revision dữ liệu khớp. Nếu reuse, JSON phải ghi rõ base training không được thực thi lại trong lượt này.

### 9.7. Phạm vi kiểm chứng

`incremental_update` chỉ nhận R/W cũ và batch hiện tại. Bộ kiểm tra bên ngoài giữ feature train lịch sử để giải joint ridge làm tham chiếu; phần lịch sử này không nằm trong trạng thái analytic được lưu.

Đối chiếu R, W, score và argmax trên cùng feature map/gamma. Sau 50+10 lớp, đánh giá toàn bộ **6.000 ảnh test của 60 lớp**, gồm cả 50 lớp cũ; W phải có 60 cột. Smoke dùng float64 ở analytic và ngưỡng trong config. Kết quả này chưa phải tái lập benchmark paper.
