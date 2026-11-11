"""Analytic head ACIL với nhãn toàn cục, dùng được ở cấp phase và minibatch.

Tự cài từ phương trình (4), (9), (10) của ACIL, NeurIPS 2022.
Feature map và gamma phải giữ cố định; module chỉ hỗ trợ gamma > 0.
Không giữ hoặc đọc lại mẫu cũ trong incremental_update.
"""

from __future__ import annotations

import numpy as np


def _data(h, labels):
    """Kiểm tra ma trận đặc trưng H và nhãn toàn cục, chuyển về dtype chuẩn.

    Tham số:
        h: Mảng đặc trưng đầu vào dạng (n, d), hữu hạn và không rỗng.
        labels: Vector nhãn lớp nguyên không âm có độ dài n.

    Trả về:
        Tuple (h_float64, labels_int64) đã được kiểm tra tính hợp lệ.
    """
    # Kiểm tra đầu vào có sạch hay không.
    # H là ma trận hữu hạn, 2d, không rỗng
    # Nhãn là vector số nguyên, đúng n phần tử, không âm
    h = np.asarray(h, dtype=np.float64)
    labels = np.asarray(labels)
    if h.ndim != 2 or min(h.shape) < 1 or not np.isfinite(h).all():
        raise ValueError("H phải là ma trận (n,d) hữu hạn, không rỗng")
    if labels.ndim != 1 or len(labels) != len(h) or labels.dtype.kind not in "iu":
        raise ValueError("Nhãn phải là vector số nguyên có đúng n phần tử")
    if np.any(labels < 0) or np.any(labels > np.iinfo(np.int64).max):
        raise ValueError("ID lớp phải không âm và biểu diễn được bằng int64")
    return h, labels.astype(np.int64, copy=False)


def _classes(labels, n_classes=None, minimum=0):
    """Xác định số cột classifier hợp lệ, không thu hẹp lớp cũ và không sót nhãn mới.

    Tham số:
        labels: Vector nhãn toàn cục kiểu nguyên (n,).
        n_classes: Số cột lớp mong muốn (tùy chọn); nếu None sẽ tự suy từ nhãn lớn nhất.
        minimum: Số cột tối thiểu phải giữ lại từ trạng thái cũ.

    Trả về:
        Số nguyên dương biểu diễn tổng số cột của ma trận trọng số W.
    """
    #Tính số cột
    # Số cột phải lớn hơn bằng max(Nhãn) + 1, và không nhỏ hơn số cột cũ (minimum).
    required = max(minimum, int(labels.max()) + 1)
    if n_classes is None:
        return required
    if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)):
        raise ValueError("n_classes phải là số nguyên")
    if n_classes < required:
        raise ValueError("n_classes không đủ cho lớp cũ và nhãn mới")
    return int(n_classes)


def _gamma(gamma):
    """Kiểm tra và chuẩn hóa hệ số regularization dương hữu hạn.

    Tham số:
        gamma: Giá trị hệ số regularization cho bài toán ridge.

    Trả về:
        Giá trị float dương và hữu hạn của gamma.
    """
    # Kiểm tra gamma > 0 và hữu hạn.
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma phải hữu hạn và lớn hơn 0")
    return gamma


def _targets(labels, n_classes):
    """Tạo ma trận nhãn one-hot theo ID lớp toàn cục.

    Tham số:
        labels: Vector nhãn nguyên toàn cục có kích thước (n,).
        n_classes: Tổng số lớp C (số cột của ma trận đích).

    Trả về:
        Mảng float64 kích thước (n, C) chứa mã hóa one-hot.
    """
    y = np.zeros((len(labels), n_classes), dtype=np.float64)
    y[np.arange(len(labels)), labels] = 1.0
    return y


def joint_ridge(h, labels, n_classes=None, gamma=1.0):
    """Giải nghiệm joint ridge độc lập từ toàn bộ dữ liệu để làm chuẩn đối chiếu.

    Các lớp chưa có mẫu sẽ có cột target bằng 0 nếu ``n_classes`` được chỉ định trước.

    Tham số:
        h: Ma trận đặc trưng tổng hợp kích thước ``(n, d)``.
        labels: Vector nhãn toàn cục kiểu nguyên kích thước ``(n,)``.
        n_classes: Số cột lớp C tùy chọn; nếu None tự suy từ nhãn lớn nhất.
        gamma: Hệ số regularization dương (mặc định 1.0).

    Trả về:
        Ma trận trọng số ``W`` kích thước ``(d, C)`` giải từ ``(H^T H + gamma I) W = H^T Y``.
    """
    # Giải W để khi học tât cả các batch -> So sánh
    # H ma trận đặc trưng
    # labels là nhãn toàn cục
    # gamma*I thành phần điều tiết, ổn định ma trận -> tránh suy biến
    # Ma trận tối ưu cuối cùng
    h, labels = _data(h, labels)
    gamma = _gamma(gamma)
    count = _classes(labels, n_classes)
    gram = h.T @ h + gamma * np.eye(h.shape[1])
    return np.linalg.solve(gram, h.T @ _targets(labels, count))


def initialize(h, labels, gamma=1.0, n_classes=None):
    """Khởi tạo trạng thái analytic (R, W) từ tập base hoặc batch đầu tiên.

    Tham số:
        h: Ma trận đặc trưng sau expansion có kích thước (n, d).
        labels: Vector nhãn toàn cục kiểu nguyên có kích thước (n,).
        gamma: Hệ số regularization dương (mặc định 1.0); chỉ cộng tại bước khởi tạo.
        n_classes: Số cột lớp C tùy chọn; nếu None tự suy từ nhãn lớn nhất.

    Trả về:
        Tuple (r, w) gồm nghịch đảo Gram có điều chuẩn R (d, d) và trọng số W (d, C).
    """
    # thiết lập ma trận nghịch đảo ban đầu
    # đóng vai trò là nơi duy nhất tạo hệ só gamma vào hệ thống
    h, labels = _data(h, labels)
    gamma = _gamma(gamma)
    count = _classes(labels, n_classes)
    identity = np.eye(h.shape[1])
    r = np.linalg.solve(h.T @ h + gamma * identity, identity)
    w = r @ (h.T @ _targets(labels, count))
    return r, w


def initialize_batches(batches, n_features, n_classes, gamma=1.0, *, timings=None):
    """Tích lũy thống kê Gram/cross theo batch cho base phase và giải (R, W) bằng Cholesky.

    Tham số:
        batches: Iterator trả về các cặp (H, y) của tập base, chỉ duyệt một lượt.
        n_features: Số chiều đặc trưng d của H.
        n_classes: Tổng số lớp C của tập base.
        gamma: Hệ số regularization dương (mặc định 1.0), cộng đúng một lần vào đường chéo.
        timings: Dict tùy chọn để ghi thời gian tích lũy, giải hệ và tổng số mẫu base.

    Trả về:
        Tuple (r, w) gồm ma trận R (d, d) và trọng số W (d, C), không giữ lại lịch sử H.
    """
    from scipy.linalg import cho_factor, cho_solve
    import time
    if isinstance(n_features, bool) or not isinstance(n_features, (int, np.integer)) or n_features < 1:
        raise ValueError("n_features phải là số nguyên dương")
    if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)) or n_classes < 1:
        raise ValueError("n_classes phải là số nguyên dương")
    gamma = _gamma(gamma)
    gram = np.zeros((n_features, n_features), dtype=np.float64, order="F")
    cross = np.zeros((n_features, n_classes), dtype=np.float64, order="F")
    samples = 0
    start = time.perf_counter()
    for h, labels in batches:
        h, labels = _data(h, labels)
        if h.shape[1] != n_features:
            raise ValueError("Batch không khớp chiều feature")
        _classes(labels, n_classes)
        gram += h.T @ h
        cross += h.T @ _targets(labels, n_classes)
        samples += len(h)
    if samples == 0:
        raise ValueError("Base batches không được rỗng")
    if not np.isfinite(gram).all() or not np.isfinite(cross).all():
        raise FloatingPointError("Thống kê base không hữu hạn")
    accumulated = time.perf_counter() - start
    gram[np.diag_indices(n_features)] += gamma
    start = time.perf_counter()
    factor = cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
    w = cho_solve(factor, cross, check_finite=False)
    r = cho_solve(factor, np.eye(n_features, order="F"), overwrite_b=True, check_finite=False)
    if timings is not None:
        timings.update(base_accumulation_seconds=accumulated, base_solve_seconds=time.perf_counter()-start,
                       base_samples=samples)
    return r, w


def incremental_update(r_old, w_old, h, labels, n_classes=None):
    """Cập nhật trạng thái analytic theo batch hiện tại bằng Woodbury và phần dư.

    Hỗ trợ cả lớp mới lẫn lớp xuất hiện lại trong minibatch mà không làm thay đổi
    các mảng trạng thái đầu vào hay lưu trữ dữ liệu train của các phase cũ.

    Tham số:
        r_old: Ma trận nghịch đảo Gram từ bước trước có kích thước (d, d).
        w_old: Ma trận trọng số từ bước trước có kích thước (d, C_old).
        h: Ma trận đặc trưng của batch hiện tại kích thước (n, d).
        labels: Vector nhãn toàn cục kích thước (n,); không được đặt lại từ 0.
        n_classes: Tổng số lớp C sau cập nhật (tùy chọn) để mở trước cột cho cả phase.

    Trả về:
        Tuple (r_new, w_new) với R_new kích thước (d, d) và W_new kích thước (d, C).
    """
    h, labels = _data(h, labels)
    r_old = np.asarray(r_old, dtype=np.float64)
    w_old = np.asarray(w_old, dtype=np.float64)
    d = h.shape[1]
    if r_old.shape != (d, d) or w_old.ndim != 2 or w_old.shape[0] != d or w_old.shape[1] < 1:
        raise ValueError("Kích thước R/W cũ không khớp số chiều feature")
    if not np.isfinite(r_old).all() or not np.isfinite(w_old).all():
        raise ValueError("R/W cũ phải hữu hạn")
    count = _classes(labels, n_classes, minimum=w_old.shape[1])

    # Giữ nguyên cột lớp cũ, chỉ đệm 0 cho lớp mới; không thêm cột theo số batch( mở rộng W cho lớp mới)
    w_padded = np.zeros((d, count), dtype=np.float64)
    w_padded[:, :w_old.shape[1]] = w_old
    y_global = _targets(labels, count)

    # Woodbury: giải hệ ở không gian batch, không tạo nghịch đảo tường minh.
    #Chỉ cần giải phương trình kích thước nxn, kích thước batch nhỏ hơn nhiều so với d.
    left = r_old @ h.T
    middle = np.eye(len(h)) + h @ left
    r_new = r_old - left @ np.linalg.solve(middle, h @ r_old)

    # Dạng residual của (9). Dùng R MỚI; lớp lặp tiếp tục dùng đúng cột đã có.
    #cập nhật W mới
    w_new = w_padded + r_new @ h.T @ (y_global - h @ w_padded)
    return r_new, w_new
