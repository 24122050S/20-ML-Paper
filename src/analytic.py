"""Analytic head ACIL với nhãn toàn cục, dùng được ở cấp phase và minibatch.

Tự cài từ phương trình (4), (9), (10) của ACIL, NeurIPS 2022.
Feature map và gamma phải giữ cố định; module chỉ hỗ trợ gamma > 0.
Không giữ hoặc đọc lại mẫu cũ trong incremental_update.
"""

from __future__ import annotations

import numpy as np


def _data(h, labels):
    """Kiểm tra H (n,d), nhãn toàn cục (n,); trả mảng float64/int64 hợp lệ."""
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
    """Trả số cột classifier, không thu hẹp lớp cũ và không bỏ sót nhãn mới."""
    required = max(minimum, int(labels.max()) + 1)
    if n_classes is None:
        return required
    if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)):
        raise ValueError("n_classes phải là số nguyên")
    if n_classes < required:
        raise ValueError("n_classes không đủ cho lớp cũ và nhãn mới")
    return int(n_classes)


def _gamma(gamma):
    """Trả gamma hữu hạn, lớn hơn 0; gamma=0 cần phương pháp giải riêng."""
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma phải hữu hạn và lớn hơn 0")
    return gamma


def _targets(labels, n_classes):
    """Tạo one-hot (n,C) theo ID toàn cục; batch có thể chỉ chứa một số lớp."""
    y = np.zeros((len(labels), n_classes), dtype=np.float64)
    y[np.arange(len(labels)), labels] = 1.0
    return y


def joint_ridge(h, labels, n_classes=None, gamma=1.0):
    """Giải nghiệm joint độc lập từ toàn bộ dữ liệu được cung cấp.

    Tham số: H (n,d), labels nguyên toàn cục (n,), số lớp C tùy chọn và gamma>0.
    Trả về W (d,C) từ hệ (H.T H + gamma I) W = H.T Y.
    Các lớp chưa có mẫu có cột target bằng 0 nếu C được chỉ định trước.
    """
    h, labels = _data(h, labels)
    gamma = _gamma(gamma)
    count = _classes(labels, n_classes)
    gram = h.T @ h + gamma * np.eye(h.shape[1])
    return np.linalg.solve(gram, h.T @ _targets(labels, count))


def initialize(h, labels, gamma=1.0, n_classes=None):
    """Khởi tạo trạng thái từ base hoặc batch đầu tiên.

    Nhận H (n,d), nhãn toàn cục (n,), gamma>0 và số lớp tùy chọn.
    Trả (R,W), với R kích thước (d,d), W kích thước (d,C).
    Gamma chỉ được cộng tại khởi tạo, không cộng lại ở mỗi minibatch.
    """
    h, labels = _data(h, labels)
    gamma = _gamma(gamma)
    count = _classes(labels, n_classes)
    identity = np.eye(h.shape[1])
    r = np.linalg.solve(h.T @ h + gamma * identity, identity)
    w = r @ (h.T @ _targets(labels, count))
    return r, w


def incremental_update(r_old, w_old, h, labels, n_classes=None):
    """Cập nhật bằng H/nhãn hiện tại, hỗ trợ lớp mới hoặc lớp lặp trong minibatch.

    Tham số: R cũ (d,d), W cũ (d,C_old), H (n,d), ID toàn cục (n,).
    n_classes tùy chọn cho phép mở trước số cột của toàn phase; nếu bỏ qua,
    số cột tăng theo nhãn lớn nhất đã xuất hiện. Nhãn không được đặt lại từ 0.

    Trả (R_new,W_new). Không sửa các mảng đầu vào, không giữ dữ liệu train cũ.
    Backbone/feature expansion và gamma của trạng thái phải giữ nguyên.
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

    # Giữ nguyên cột lớp cũ, chỉ đệm 0 cho lớp mới; không thêm cột theo số batch.
    w_padded = np.zeros((d, count), dtype=np.float64)
    w_padded[:, :w_old.shape[1]] = w_old
    y_global = _targets(labels, count)

    # Woodbury: giải hệ ở không gian batch, không tạo nghịch đảo tường minh.
    left = r_old @ h.T
    middle = np.eye(len(h)) + h @ left
    r_new = r_old - left @ np.linalg.solve(middle, h @ r_old)

    # Dạng residual của (9). Dùng R MỚI; lớp lặp tiếp tục dùng đúng cột đã có.
    w_new = w_padded + r_new @ h.T @ (y_global - h @ w_padded)
    return r_new, w_new
