"""Điều phối thực nghiệm cuối kỳ E2 (E2a, E2b & nguyên mẫu FD2) kiểm chứng hạn chế P2.

Module này tái sử dụng trực tiếp 3 checkpoints ResNet-32 160 epochs của R1 (kiểm SHA-256,
chặn huấn luyện lại CNN) để đo:
1. Nhánh E2a: Đường cong đánh đổi độ chính xác - dung lượng trạng thái tính bằng Byte - thời gian
   trên lưới chiều mở rộng d (bao gồm trần công bằng 6.144 MB của 2,000 ảnh CIFAR-100 tại
   d=798 ở float64 và d=1160 ở float32) cùng nguyên mẫu FD2 (Block-Diagonal RFAM).
2. Nhánh E2b: Độ trôi số học của phép cập nhật đệ quy Woodbury giữa float32 và float64 theo
   hệ số điều chuẩn gamma và số bước lặp minibatch đối chiếu với nghiệm Joint Ridge float64.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
from unittest import mock

import numpy as np
import psutil
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import cifar_smoke
from src.analytic import joint_ridge
from src.cifar_data import audit_data, load_cifar100, make_protocol, sha256_file
from src.cifar_smoke import (
    extract_features,
    fingerprint,
    make_loader,
    seed_everything,
    validate_checkpoint,
)
from src.reporting import collect_provenance, save_json
from src.resnet import ResNet32, freeze_backbone
from src.run_attempt import RunAttempt
from src.streaming import StreamingHead


CIFAR100_EXEMPLAR_2000_BYTES: int = 2000 * 32 * 32 * 3  # 6,144,000 bytes
DEFAULT_E2A_DIMENSIONS: tuple[int, ...] = (
    64,
    128,
    256,
    512,
    768,
    798,
    1024,
    1160,
    2048,
    4096,
    8000,
)
DEFAULT_FD2_BLOCKS: tuple[int, ...] = (16, 80)
DEFAULT_E2B_GAMMAS: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)
DEFAULT_E2B_REGIMES: tuple[tuple[int, int], ...] = ((5, 128), (25, 16))


def max_iso_byte_dimension(
    budget_bytes: int = CIFAR100_EXEMPLAR_2000_BYTES,
    *,
    n_classes: int = 100,
    d_cnn: int = 64,
    bytes_per_elem: int = 8,
) -> int:
    """Tính chiều mở rộng tối đa d_max để tổng trạng thái (R, W_fe, W) không vượt ngân sách Byte.

    Giải bất phương trình bậc hai ``bytes_per_elem * (d^2 + (d_cnn + n_classes) * d) <= budget_bytes``.

    Tham số:
        budget_bytes: Ngân sách bộ nhớ tối đa tính bằng byte (mặc định 6,144,000 bytes).
        n_classes: Số cột lớp tối đa ở phase cuối (mặc định 100).
        d_cnn: Số chiều đặc trưng đầu ra của backbone CNN (mặc định 64).
        bytes_per_elem: Số byte trên mỗi phần tử thực (8 cho float64, 4 cho float32).

    Trả về:
        Số nguyên dương lớn nhất ``d_max`` thỏa mãn ràng buộc ngân sách byte.
    """
    if budget_bytes <= 0 or n_classes <= 0 or d_cnn <= 0 or bytes_per_elem not in (4, 8):
        raise ValueError("Tham số tính trần ngân sách byte không hợp lệ")
    max_elements = budget_bytes // bytes_per_elem
    linear_coeff = d_cnn + n_classes
    half_b = linear_coeff / 2.0
    approx = int(math.floor(math.sqrt(max_elements + half_b * half_b) - half_b))
    while (approx + 1) * (approx + 1 + linear_coeff) <= max_elements:
        approx += 1
    while approx > 0 and approx * (approx + linear_coeff) > max_elements:
        approx -= 1
    return approx


def compute_state_memory_budget(
    dimension: int,
    *,
    n_classes: int = 100,
    d_cnn: int = 64,
    feature_expansion: bool = True,
    num_blocks: int = 1,
    exemplar_budget_bytes: int = CIFAR100_EXEMPLAR_2000_BYTES,
) -> dict:
    """Tính chính xác số phần tử và số byte của trạng thái analytic lưu giữa các phase.

    Hỗ trợ cả ACIL dày đặc chuẩn (``num_blocks=1``), trường hợp không mở rộng đặc trưng
    (``feature_expansion=False``) và nguyên mẫu khối đường chéo FD2 (``num_blocks=B >= 1``).

    Tham số:
        dimension: Tổng số chiều đặc trưng analytic ``d``.
        n_classes: Số cột phân loại ở pha đánh giá (mặc định 100).
        d_cnn: Số chiều đặc trưng CNN trước khi mở rộng (mặc định 64).
        feature_expansion: Có lưu ma trận chiếu ngẫu nhiên ``W_fe`` hay không.
        num_blocks: Số khối đường chéo ``B`` chia đều ``d`` (mặc định 1 là ma trận dày đặc).
        exemplar_budget_bytes: Dung lượng kho mẫu đối chiếu tính bằng byte (mặc định 6,144,000).

    Trả về:
        Dict ghi chi tiết số phần tử, dung lượng Byte/MB/MiB ở float64 & float32 và tỷ số đối chiếu.
    """
    if dimension < 1 or n_classes < 1 or d_cnn < 1 or num_blocks < 1:
        raise ValueError("dimension, n_classes, d_cnn và num_blocks phải nguyên dương")
    if not feature_expansion and (dimension != d_cnn or num_blocks != 1):
        raise ValueError("Chế độ No-FE yêu cầu dimension == d_cnn và num_blocks == 1")
    if dimension % num_blocks != 0:
        raise ValueError("dimension phải chia hết cho num_blocks")

    block_dim = dimension // num_blocks
    r_elements = num_blocks * block_dim * block_dim
    fe_elements = (d_cnn * dimension) if feature_expansion else 0
    w_elements = dimension * n_classes
    total_elements = r_elements + fe_elements + w_elements
    bytes_fp64 = total_elements * 8
    bytes_fp32 = total_elements * 4
    return {
        "dimension": dimension,
        "num_blocks": num_blocks,
        "block_dim": block_dim,
        "feature_expansion": feature_expansion,
        "r_elements": r_elements,
        "fe_elements": fe_elements,
        "w_elements": w_elements,
        "total_elements": total_elements,
        "bytes_fp64": bytes_fp64,
        "bytes_fp32": bytes_fp32,
        "mb_fp64": bytes_fp64 / 1e6,
        "mib_fp64": bytes_fp64 / (1024.0 ** 2),
        "mb_fp32": bytes_fp32 / 1e6,
        "mib_fp32": bytes_fp32 / (1024.0 ** 2),
        "exemplar_2000_raw_bytes": exemplar_budget_bytes,
        "ratio_fp64_vs_2000_exemplars": bytes_fp64 / exemplar_budget_bytes,
        "ratio_fp32_vs_2000_exemplars": bytes_fp32 / exemplar_budget_bytes,
        "fits_2000_exemplar_budget_fp64": bytes_fp64 <= exemplar_budget_bytes,
        "fits_2000_exemplar_budget_fp32": bytes_fp32 <= exemplar_budget_bytes,
    }


class LearningWindow:
    """Ngữ cảnh đo đồng hồ thời gian thực và đỉnh RSS tiến trình (5 ms) riêng cho bước học analytic."""

    active: bool = False

    def __init__(self, interval: float = 0.005):
        """Khởi tạo cửa sổ đo thời gian và lấy mẫu bộ nhớ RSS định kỳ.

        Tham số:
            interval: Chu kỳ lấy mẫu RSS tính bằng giây (mặc định 0.005s = 5ms).

        Trả về:
            None.
        """
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.baseline = 0
        self.peak = 0
        self.seconds = 0.0
        self.thread: threading.Thread | None = None
        self._start = 0.0

    def _sample_loop(self) -> None:
        """Chạy luồng nền cập nhật đỉnh RSS của tiến trình cho đến khi nhận tín hiệu dừng.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            None.
        """
        while not self.stop_event.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "LearningWindow":
        """Bắt đầu cửa sổ đo, ghi nhận RSS nền và bật cờ ``LearningWindow.active``.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            Chính đối tượng ``LearningWindow`` hiện tại.
        """
        self.baseline = self.peak = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        LearningWindow.active = True
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_args) -> None:
        """Kết thúc cửa sổ đo, tắt cờ ``LearningWindow.active`` và chốt thời gian/đỉnh RSS.

        Tham số:
            *_args: Bộ đối số ngoại lệ tiêu chuẩn của context manager.

        Trả về:
            None.
        """
        self.seconds = time.perf_counter() - self._start
        LearningWindow.active = False
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


class BlockDiagonalStreamingHead:
    """Nguyên mẫu FD2: Xấp xỉ khối đường chéo cho ma trận tương quan nghịch đảo (Block-Diagonal RFAM).

    Chia không gian mở rộng ``d`` chiều thành ``B`` khối con độc lập cỡ ``m = d // B``, lưu
    ``R_blocks`` kích thước ``(B, m, m)`` (giảm dung lượng R đúng ``B`` lần) và cập nhật từng khối
    bằng công thức Woodbury chuẩn. Khi ``B = 1``, mô hình trùng khớp đại số với ``StreamingHead``.
    """

    def __init__(self, *, seed: int, expansion_dim: int, num_blocks: int, gamma: float = 0.1):
        """Khởi tạo đầu phân loại giải tích khối đường chéo với ma trận chiếu ngẫu nhiên cố định.

        Tham số:
            seed: Hạt giống cơ sở; ma trận ``W_fe`` được sinh từ ``default_rng(seed + 1000)``.
            expansion_dim: Tổng số chiều đặc trưng mở rộng ``d``.
            num_blocks: Số lượng khối đường chéo ``B`` (``expansion_dim`` phải chia hết cho ``B``).
            gamma: Hệ số điều chuẩn Ridge dương cộng một lần tại Phase 0.

        Trả về:
            None.
        """
        if expansion_dim < 1 or num_blocks < 1 or expansion_dim % num_blocks != 0:
            raise ValueError("expansion_dim phải dương và chia hết cho num_blocks")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma phải hữu hạn và lớn hơn 0")
        self.dimension = int(expansion_dim)
        self.num_blocks = int(num_blocks)
        self.block_dim = self.dimension // self.num_blocks
        self.gamma = float(gamma)
        self.expansion = np.random.default_rng(int(seed) + 1000).normal(size=(64, self.dimension))
        self.r_blocks: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.maximum_feature_batch_rows = 0

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Chiếu đặc trưng CNN 64 chiều sang không gian mở rộng ReLU(F @ W_fe) kiểu float64.

        Tham số:
            raw: Mảng đặc trưng 2D hữu hạn có kích thước ``(n, 64)``.

        Trả về:
            Mảng float64 kích thước ``(n, d)``.
        """
        raw_arr = np.asarray(raw, dtype=np.float64)
        if raw_arr.ndim != 2 or raw_arr.shape[1] != 64 or not np.isfinite(raw_arr).all():
            raise ValueError("Đặc trưng đầu vào phải hữu hạn và có kích thước (n, 64)")
        self.maximum_feature_batch_rows = max(self.maximum_feature_batch_rows, len(raw_arr))
        return np.maximum(raw_arr @ self.expansion, 0.0)

    def fit_base(self, batches, n_classes: int = 50) -> int:
        """Tích lũy Gram/cross theo từng khối đường chéo trên tập Base và giải Cholesky.

        Tham số:
            batches: Iterator sinh các cặp ``(F64_batch, labels_batch)`` của Phase 0.
            n_classes: Số lượng lớp của Phase 0 (mặc định 50).

        Trả về:
            Tổng số mẫu train Phase 0 đã xử lý (int).
        """
        b_cnt, m_dim = self.num_blocks, self.block_dim
        grams = np.zeros((b_cnt, m_dim, m_dim), dtype=np.float64)
        crosses = np.zeros((b_cnt, m_dim, n_classes), dtype=np.float64)
        samples = 0
        for raw, labels in batches:
            labels_arr = np.asarray(labels, dtype=np.int64)
            h_full = self.transform(raw)
            y_one_hot = np.zeros((len(labels_arr), n_classes), dtype=np.float64)
            y_one_hot[np.arange(len(labels_arr)), labels_arr] = 1.0
            h_blocks = h_full.reshape(len(labels_arr), b_cnt, m_dim)
            grams += np.einsum("nbi,nbj->bij", h_blocks, h_blocks, optimize=True)
            crosses += np.einsum("nbi,nc->bic", h_blocks, y_one_hot, optimize=True)
            samples += int(len(labels_arr))
        if samples == 0:
            raise ValueError("Dữ liệu Phase 0 không được rỗng")

        r_blocks = np.empty((b_cnt, m_dim, m_dim), dtype=np.float64)
        w_blocks = np.empty((b_cnt, m_dim, n_classes), dtype=np.float64)
        eye_m = np.eye(m_dim, dtype=np.float64)
        diag_idx = np.diag_indices(m_dim)
        for blk in range(b_cnt):
            gram_b = grams[blk]
            gram_b[diag_idx] += self.gamma
            factor = cho_factor(gram_b, lower=True, check_finite=True)
            w_blocks[blk] = cho_solve(factor, crosses[blk], check_finite=True)
            r_blocks[blk] = cho_solve(factor, eye_m, check_finite=True)
        self.r_blocks = r_blocks
        self.w = w_blocks.reshape(self.dimension, n_classes)
        return samples

    def update_phase(self, batches, n_classes: int) -> tuple[int, int]:
        """Cập nhật đệ quy Woodbury độc lập trên từng khối đường chéo cho phase mới.

        Tham số:
            batches: Iterator sinh các cặp ``(F64_batch, labels_batch)`` của phase hiện tại.
            n_classes: Tổng số lớp đã thấy tính đến hết phase hiện tại.

        Trả về:
            Tuple ``(sample_count, batch_count)`` của phase vừa cập nhật.
        """
        if self.r_blocks is None or self.w is None:
            raise ValueError("Phải gọi fit_base trước khi gọi update_phase")
        b_cnt, m_dim = self.num_blocks, self.block_dim
        old_cols = self.w.shape[1]
        if n_classes < old_cols:
            raise ValueError("n_classes không được nhỏ hơn số cột lớp đã có")
        w_blocks = np.zeros((b_cnt, m_dim, n_classes), dtype=np.float64)
        w_blocks[:, :, :old_cols] = self.w.reshape(b_cnt, m_dim, old_cols)

        samples = batch_count = 0
        for raw, labels in batches:
            labels_arr = np.asarray(labels, dtype=np.int64)
            h_full = self.transform(raw)
            batch_len = len(labels_arr)
            eye_b = np.eye(batch_len, dtype=np.float64)
            y_one_hot = np.zeros((batch_len, n_classes), dtype=np.float64)
            y_one_hot[np.arange(batch_len), labels_arr] = 1.0
            h_blocks = h_full.reshape(batch_len, b_cnt, m_dim)
            for blk in range(b_cnt):
                h_b = h_blocks[:, blk, :]
                r_old = self.r_blocks[blk]
                w_old = w_blocks[blk]
                left = r_old @ h_b.T
                middle = eye_b + h_b @ left
                r_new = r_old - left @ np.linalg.solve(middle, h_b @ r_old)
                w_new = w_old + r_new @ (h_b.T @ (y_one_hot - h_b @ w_old))
                self.r_blocks[blk] = r_new
                w_blocks[blk] = w_new
            samples += batch_len
            batch_count += 1
        if samples == 0:
            raise ValueError("Phase tăng dần không được rỗng")
        self.w = w_blocks.reshape(self.dimension, n_classes)
        return samples, batch_count

    def score(self, raw: np.ndarray) -> np.ndarray:
        """Tính điểm dự đoán trung bình của B bộ phân loại con trên đặc trưng đầu vào.

        Tham số:
            raw: Mảng đặc trưng CNN kích thước ``(n, 64)``.

        Trả về:
            Mảng điểm dự đoán float64 kích thước ``(n, C)``.
        """
        if self.w is None:
            raise ValueError("Chưa khởi tạo trọng số W")
        return (self.transform(raw) @ self.w) / float(self.num_blocks)

    def state_bytes(self) -> int:
        """Trả về tổng số byte thực tế của ``r_blocks``, ``w`` và ``expansion``.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            Số nguyên dương biểu diễn tổng số byte đang lưu trữ.
        """
        return sum(
            arr.nbytes for arr in (self.r_blocks, self.w, self.expansion) if arr is not None
        )


def load_r1_seed_reference(seed: int, project_root: Path = ROOT) -> dict:
    """Đọc và xác minh mã băm SHA-256 của gói tham chiếu R1 lịch sử cho một seed chuẩn.

    Tham số:
        seed: Hạt giống chuẩn của R1 (``42``, ``43`` hoặc ``44``).
        project_root: Đường dẫn thư mục gốc của repository.

    Trả về:
        Dict thông tin tham chiếu gồm ``checkpoint_sha256``, ``backbone_sha256``, ``source_sha256``...
    """
    if seed not in (42, 43, 44):
        raise ValueError("Tham chiếu R1 chỉ định nghĩa cho các seeds 42, 43, 44")
    report_path = project_root / f"reports/benchmark_seed{seed}_kaggle_results.json"
    manifest_path = project_root / f"reports/R1-SEED{seed}-KAGGLE-01-provenance.json"
    report = json.loads(report_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    expected_report_sha = manifest["artifacts"][report_path.name]["sha256"]
    if sha256_file(report_path) != expected_report_sha:
        raise ValueError(f"Tệp kết quả R1 seed {seed} không khớp mã băm trong provenance manifest")
    return {
        "seed": seed,
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "backbone_sha256": report["backbone_sha256_before"],
        "expansion_sha256": report["expansion_sha256"],
        "source_sha256": report["provenance"]["source_sha256"],
        "dataset_source": report["dataset_source"],
        "wandb_run_id": manifest["tracking_sync"]["wandb_run_id"],
        "baseline_config": report["config"],
        "baseline_metrics": report["metric_summary"],
    }


def find_verified_r1_checkpoint(
    seed: int,
    expected_sha256: str,
    *,
    checkpoint_path: str | Path | None = None,
    project_root: Path = ROOT,
) -> Path:
    """Tìm và xác minh tệp ``base_checkpoint.pt`` của R1 theo đúng mã băm SHA-256 đã khóa.

    Tuyệt đối từ chối chạy hoặc huấn luyện thay thế nếu không tìm thấy đúng checkpoint R1.

    Tham số:
        seed: Hạt giống cần tìm checkpoint (``42``, ``43`` hoặc ``44``).
        expected_sha256: Chuỗi SHA-256 chuẩn của checkpoint R1 lấy từ provenance.
        checkpoint_path: Đường dẫn tường minh do người dùng cung cấp (tùy chọn).
        project_root: Thư mục gốc repository để quét ``reports/local`` khi chạy cục bộ.

    Trả về:
        Đối tượng ``Path`` trỏ tới tệp checkpoint đã xác minh khớp SHA-256.
    """
    if checkpoint_path is not None:
        candidate = Path(checkpoint_path)
        if not candidate.is_file() or sha256_file(candidate) != expected_sha256:
            raise ValueError(
                f"Checkpoint truyền vào cho seed {seed} không tồn tại hoặc sai SHA-256 R1; cấm train fallback"
            )
        return candidate

    search_roots = [Path("/kaggle/input"), project_root / "reports/local"]
    for root in search_roots:
        if root.is_dir():
            for path in sorted(root.rglob("base_checkpoint.pt")):
                if sha256_file(path) == expected_sha256:
                    return path
    raise ValueError(
        f"Không tìm thấy base_checkpoint.pt khớp SHA-256 R1 cho seed {seed}; cấm train fallback"
    )


def prepare_or_load_seed_fixture(
    seed: int,
    fixture_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    data_dir: str | Path = ROOT / "data",
    project_root: Path = ROOT,
    device: str = "auto",
    threads: int = 4,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trích xuất một lần (hoặc nạp lại có kiểm SHA-256) đặc trưng F64 từ checkpoint R1 đã đóng băng.

    Mảng train ``(50000, 64)`` được lưu theo đúng thứ tự chỉ số gốc ``0..49999`` cùng nhãn toàn cục
    để có thể chia nhất quán sang cả ``K=5`` (E2a, E2b) lẫn ``K=25`` (E2b) mà không giữ lịch sử H.

    Tham số:
        seed: Hạt giống R1 (``42``, ``43`` hoặc ``44``).
        fixture_dir: Thư mục lưu hoặc nạp lại fixture đặc trưng ``F64`` của seed này.
        checkpoint_path: Đường dẫn tường minh tới ``base_checkpoint.pt`` (tùy chọn).
        data_dir: Thư mục cache dữ liệu CIFAR-100.
        project_root: Thư mục gốc repository.
        device: Thiết bị chạy forward CNN (``"auto"``, ``"cpu"`` hoặc ``"cuda"``).
        threads: Số luồng tính toán khi trích xuất đặc trưng.

    Trả về:
        Tuple ``(meta, train_features, train_labels, test_features, test_labels)`` đã kiểm SHA-256.
    """
    fixture_root = Path(fixture_dir)
    manifest_file = fixture_root / "fixture.json"
    reference = load_r1_seed_reference(seed, project_root)

    if manifest_file.is_file():
        meta = json.loads(manifest_file.read_bytes())
        if meta.get("seed") != seed or meta.get("checkpoint_sha256") != reference["checkpoint_sha256"]:
            raise ValueError("Fixture hiện có không khớp seed hoặc checkpoint_sha256 của R1")
        for fname, expected_hash in meta["files"].items():
            if sha256_file(fixture_root / fname) != expected_hash:
                raise ValueError(f"Tệp fixture bị thay đổi hoặc hỏng mã băm: {fname}")
        return (
            meta,
            np.load(fixture_root / "train_features.npy", allow_pickle=False),
            np.load(fixture_root / "train_labels.npy", allow_pickle=False),
            np.load(fixture_root / "test_features.npy", allow_pickle=False),
            np.load(fixture_root / "test_labels.npy", allow_pickle=False),
        )

    verified_ckpt = find_verified_r1_checkpoint(
        seed, reference["checkpoint_sha256"], checkpoint_path=checkpoint_path, project_root=project_root
    )
    saved = torch.load(verified_ckpt, map_location="cpu", weights_only=True)
    if saved.get("checkpoint_format_version") != 2 or not saved.get("training_complete"):
        raise ValueError("Checkpoint R1 không ở trạng thái hoàn tất hợp lệ")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    seed_everything(seed, threads)
    arrays = load_cifar100(data_dir)
    protocol = make_protocol(seed, 5)
    audit = audit_data(arrays, protocol)
    cfg = {**reference["baseline_config"], "seed": seed, "torch_threads": threads, "device": device}
    validate_checkpoint(saved, cfg, protocol, audit["normalization"], arrays.source["revision"])
    if arrays.source != reference["dataset_source"]:
        raise ValueError("Nguồn dữ liệu CIFAR-100 không khớp dataset_source của R1")

    provenance = collect_provenance(project_root)
    if set(provenance["source_sha256"].keys()) != set(reference["source_sha256"].keys()):
        raise ValueError("Danh sách tệp core trong src/ không khớp cấu trúc R1")

    fixture_root.mkdir(parents=True, exist_ok=True)
    with mock.patch.object(
        cifar_smoke, "train_base", side_effect=AssertionError("E2 cấm tuyệt đối gọi train_base")
    ):
        model = ResNet32(n_classes=50).to(torch_device)
        model.load_state_dict(saved["model_state_dict"])
        freeze_backbone(model)
        before_sha = fingerprint(model)
        if before_sha != reference["backbone_sha256"]:
            raise ValueError("Dấu vân tay trọng số ResNet-32 không khớp R1")

        started = time.perf_counter()
        train_loader = make_loader(
            arrays, protocol, audit["normalization"], range(100), "train", cfg, batch_size=256
        )
        train_features, train_labels = extract_features(
            model, train_loader, torch_device, f"E2 seed{seed} train all"
        )
        test_loader = make_loader(
            arrays, protocol, audit["normalization"], range(100), "test", cfg, batch_size=256
        )
        test_features, test_labels = extract_features(
            model, test_loader, torch_device, f"E2 seed{seed} test all"
        )
        extraction_seconds = time.perf_counter() - started
        after_sha = fingerprint(model)
        if before_sha != after_sha:
            raise AssertionError("Backbone bị thay đổi trong lúc trích xuất đặc trưng")

    mapping = np.asarray(protocol["original_to_global"], dtype=np.int64)
    if not np.array_equal(train_labels, mapping[arrays.train_labels]):
        raise AssertionError("Thứ tự nhãn train trích xuất lệch so với chỉ số gốc 0..49999")
    if not np.array_equal(test_labels, mapping[arrays.test_labels]):
        raise AssertionError("Thứ tự nhãn test trích xuất lệch so với chỉ số gốc 0..9999")

    files = {}
    for name, arr in (
        ("train_features.npy", train_features),
        ("train_labels.npy", train_labels),
        ("test_features.npy", test_features),
        ("test_labels.npy", test_labels),
    ):
        target = fixture_root / name
        np.save(target, arr, allow_pickle=False)
        files[name] = sha256_file(target)

    meta = {
        "scope": "e2_seed_feature_fixture",
        "seed": seed,
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "backbone_sha256": before_sha,
        "baseline_wandb_run_id": reference["wandb_run_id"],
        "baseline_metrics": reference["baseline_metrics"],
        "train_samples": int(len(train_labels)),
        "test_samples": int(len(test_labels)),
        "train_per_class": 500,
        "test_per_class": 100,
        "feature_extraction_seconds_excluded": extraction_seconds,
        "training_executed_this_run": False,
        "files": files,
    }
    save_json(meta, manifest_file)
    return meta, train_features, train_labels, test_features, test_labels


def slice_phase_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    base_classes: int = 50,
    n_incremental_phases: int = 5,
) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
    """Tách mảng đặc trưng/nhãn theo từng phase giữ nguyên thứ tự mẫu gốc tăng dần trong mỗi phase.

    Tham số:
        features: Mảng đặc trưng 64 chiều kích thước ``(N, 64)``.
        labels: Mảng nhãn toàn cục kiểu nguyên kích thước ``(N,)``.
        base_classes: Số lớp của Phase 0 (mặc định 50).
        n_incremental_phases: Số phase tăng dần ``K`` chia đều ``100 - base_classes`` lớp còn lại.

    Trả về:
        Danh sách ``1 + K`` bộ ``(phase_features, phase_labels, phase_classes)``.
    """
    rem = 100 - base_classes
    if n_incremental_phases < 1 or rem % n_incremental_phases != 0:
        raise ValueError("n_incremental_phases phải chia hết số lớp tăng dần")
    step = rem // n_incremental_phases
    groups = [list(range(base_classes))]
    for idx in range(n_incremental_phases):
        start = base_classes + idx * step
        groups.append(list(range(start, start + step)))
    output = []
    for group in groups:
        mask = np.isin(labels, np.asarray(group, dtype=np.int64))
        output.append((features[mask], labels[mask], group))
    return output


def iter_array_batches(features: np.ndarray, labels: np.ndarray, batch_size: int):
    """Sinh tuần tự từng minibatch lát cắt liên tiếp mà không sao chép hay xáo trộn thứ tự mẫu.

    Tham số:
        features: Mảng đặc trưng 2D của phase hiện tại.
        labels: Mảng nhãn 1D tương ứng.
        batch_size: Kích thước mỗi lô dữ liệu nguyên dương.

    Trả về:
        Generator trả về từng cặp ``(features_slice, labels_slice)``.
    """
    if batch_size < 1:
        raise ValueError("batch_size phải lớn hơn hoặc bằng 1")
    for offset in range(0, len(labels), batch_size):
        yield features[offset : offset + batch_size], labels[offset : offset + batch_size]


def evaluate_seen_classes(
    score_fn,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    n_classes: int,
    base_classes: int,
    new_classes: list[int],
    batch_size: int = 256,
    expected_per_class: int | None = 100,
) -> dict:
    """Đánh giá hàm tính điểm phân loại ngoài vùng đo thời gian/RSS trên toàn bộ lớp đã thấy.

    Tham số:
        score_fn: Hàm nhận mảng đặc trưng ``(b, 64)`` và trả mảng điểm số ``(b, n_classes)``.
        test_features: Mảng đặc trưng test đầy đủ ``(N_test, 64)``.
        test_labels: Mảng nhãn toàn cục test ``(N_test,)``.
        n_classes: Tổng số lớp đã thấy ``0..n_classes-1``.
        base_classes: Số lượng lớp Phase 0 (mặc định 50).
        new_classes: Danh sách lớp thuộc riêng phase mới vừa học.
        batch_size: Kích thước lô khi tính điểm dự đoán test.
        expected_per_class: Số mẫu test kỳ vọng của mỗi lớp (hoặc ``None`` khi test tổng hợp).

    Trả về:
        Dict các thước đo độ chính xác toàn cục, lớp Base, toàn bộ lớp Incremental và lớp mới.
    """
    if LearningWindow.active:
        raise AssertionError("Không được chạy bước đánh giá test bên trong cửa sổ đo LearningWindow")
    seen_mask = test_labels < n_classes
    seen_x = test_features[seen_mask]
    seen_y = test_labels[seen_mask]
    counts = np.bincount(seen_y, minlength=n_classes)
    if np.any(counts == 0) or (expected_per_class is not None and np.any(counts != expected_per_class)):
        raise AssertionError("Tập test đánh giá không phủ đúng số mẫu mỗi lớp đã thấy")

    preds = np.empty(len(seen_y), dtype=np.int64)
    for start in range(0, len(seen_y), batch_size):
        end = start + batch_size
        scores = score_fn(seen_x[start:end])
        if scores.shape != (len(seen_y[start:end]), n_classes) or not np.isfinite(scores).all():
            raise FloatingPointError("Điểm dự đoán test sai kích thước hoặc chứa NaN/Inf")
        preds[start:end] = scores.argmax(axis=1)

    matched = preds == seen_y
    base_mask = seen_y < base_classes
    inc_mask = seen_y >= base_classes
    new_mask = np.isin(seen_y, np.asarray(new_classes, dtype=np.int64))
    is_base = n_classes == base_classes
    acc_pct = float(100.0 * np.mean(matched))
    base_acc_pct = float(100.0 * np.mean(matched[base_mask]))
    inc_acc_pct = None if is_base else float(100.0 * np.mean(matched[inc_mask]))
    new_acc_pct = None if is_base else float(100.0 * np.mean(matched[new_mask]))
    return {
        "test_samples_all_seen": int(len(seen_y)),
        "accuracy_pct": acc_pct,
        "base_accuracy_pct": base_acc_pct,
        "inc_all_accuracy_pct": inc_acc_pct,
        "new_class_accuracy_pct": new_acc_pct,
    }


def _score_standard_head(head: StreamingHead, raw: np.ndarray) -> np.ndarray:
    """Tính điểm phân loại ``H @ W`` cho đối tượng ``StreamingHead`` chuẩn.

    Tham số:
        head: Đối tượng ``StreamingHead`` đã học xong phase hiện tại.
        raw: Mảng đặc trưng CNN ``(b, 64)``.

    Trả về:
        Mảng điểm số dự đoán ``(b, C)``.
    """
    return head.transform(raw) @ head.w


def compute_batched_dense_joint_oracle(
    *,
    seed: int,
    dimension: int,
    feature_expansion: bool,
    gamma: float,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    n_classes: int = 100,
    batch_size: int = 512,
) -> np.ndarray:
    """Giải nghiệm Joint Ridge float64 chính xác theo lô mà không cấp phát ma trận 50000 x d.

    Hàm này chỉ được gọi bên ngoài ``LearningWindow`` để đo độ lệch trọng số và bất đồng dự đoán
    của nghiệm đệ quy ACIL (hoặc xấp xỉ FD2) so với nghiệm học gộp đầy đủ ở pha cuối.

    Tham số:
        seed: Hạt giống sinh ``W_fe`` (``default_rng(seed + 1000)``).
        dimension: Số chiều đặc trưng ``d``.
        feature_expansion: ``True`` nếu bật ``ReLU(F @ W_fe)``, ``False`` nếu dùng trực tiếp ``F64``.
        gamma: Hệ số điều chuẩn Ridge dương.
        train_features: Mảng đặc trưng train ``(N_train, 64)``.
        train_labels: Mảng nhãn toàn cục train ``(N_train,)``.
        n_classes: Tổng số cột lớp ở pha cuối (mặc định 100).
        batch_size: Kích thước lô tích lũy ma trận Gram ``H^T H`` và ``H^T Y``.

    Trả về:
        Ma trận trọng số chuẩn ``w_oracle`` kiểu ``float64`` kích thước ``(dimension, n_classes)``.
    """
    if LearningWindow.active:
        raise AssertionError("Không được giải Joint Oracle bên trong cửa sổ đo LearningWindow")
    expansion = (
        np.random.default_rng(int(seed) + 1000).normal(size=(64, dimension))
        if feature_expansion
        else None
    )
    gram = np.zeros((dimension, dimension), dtype=np.float64, order="F")
    cross = np.zeros((dimension, n_classes), dtype=np.float64, order="F")
    for raw_b, y_b in iter_array_batches(train_features, train_labels, batch_size):
        raw_64 = np.asarray(raw_b, dtype=np.float64)
        h_b = np.maximum(raw_64 @ expansion, 0.0) if expansion is not None else raw_64
        y_oh = np.zeros((len(y_b), n_classes), dtype=np.float64)
        y_oh[np.arange(len(y_b)), y_b] = 1.0
        gram += h_b.T @ h_b
        cross += h_b.T @ y_oh
    gram[np.diag_indices(dimension)] += float(gamma)
    factor = cho_factor(gram, lower=True, overwrite_a=True, check_finite=True)
    return cho_solve(factor, cross, check_finite=True)


def run_e2a_case(
    *,
    seed: int,
    dimension: int,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    feature_expansion: bool = True,
    num_blocks: int = 1,
    gamma: float = 0.1,
    batch_size: int = 128,
    threads: int = 4,
    train_per_class: int = 500,
    test_per_class: int = 100,
    precomputed_dense_joint_w: np.ndarray | None = None,
) -> dict:
    """Chạy và đo tài nguyên một cấu hình E2a (ACIL chuẩn hoặc FD2 Block-Diagonal) trên fixture F64.

    Tham số:
        seed: Hạt giống ngẫu nhiên (sinh ``W_fe`` với ``seed + 1000``).
        dimension: Số chiều đặc trưng analytic ``d``.
        train_features: Mảng đặc trưng train ``(N_train, 64)``.
        train_labels: Mảng nhãn toàn cục train ``(N_train,)``.
        test_features: Mảng đặc trưng test ``(N_test, 64)``.
        test_labels: Mảng nhãn toàn cục test ``(N_test,)``.
        feature_expansion: ``False`` cho cấu hình No-FE ``d=64``, ``True`` cho các cấu hình có FE.
        num_blocks: Số khối đường chéo ``B`` (``1`` cho ACIL chuẩn, ``B > 1`` cho nguyên mẫu FD2).
        gamma: Hệ số điều chuẩn Ridge dương (mặc định 0.1).
        batch_size: Kích thước minibatch học giải tích (mặc định 128).
        threads: Số luồng CPU/BLAS giới hạn trong vùng đo.
        train_per_class: Số mẫu train mỗi lớp (500 trên CIFAR-100 đầy đủ).
        test_per_class: Số mẫu test mỗi lớp (100 trên CIFAR-100 đầy đủ).
        precomputed_dense_joint_w: Nghiệm Joint Ridge dày đặc truyền sẵn ở pha cuối (tùy chọn).

    Trả về:
        Dict kết quả chi tiết gồm dung lượng trạng thái Byte, RSS, thời gian học và độ chính xác.
    """
    variant = "FD2_block_diagonal" if num_blocks > 1 else ("ACIL_no_fe" if not feature_expansion else "ACIL_dense")
    phases = slice_phase_arrays(train_features, train_labels, base_classes=50, n_incremental_phases=5)
    gc.collect()

    with threadpool_limits(limits=threads):
        np.linalg.solve(np.eye(32, dtype=np.float64), np.ones((32, 2), dtype=np.float64))
        cho_factor(np.eye(32, dtype=np.float64))

        if num_blocks > 1:
            learner = BlockDiagonalStreamingHead(
                seed=seed, expansion_dim=dimension, num_blocks=num_blocks, gamma=gamma
            )
        else:
            learner = StreamingHead({
                "seed": seed,
                "expansion_dim": dimension,
                "feature_expansion": feature_expansion,
                "gamma": gamma,
            })

        rows = []
        windows = []
        for phase_idx, (phase_x, phase_y, classes) in enumerate(phases):
            n_classes = max(classes) + 1
            batch_iter = iter_array_batches(phase_x, phase_y, batch_size)
            with LearningWindow() as win:
                if phase_idx == 0:
                    seen_count = learner.fit_base(batch_iter, n_classes=n_classes)
                    batch_cnt = (seen_count + batch_size - 1) // batch_size
                else:
                    seen_count, batch_cnt = learner.update_phase(batch_iter, n_classes=n_classes)
            if seen_count != len(classes) * train_per_class:
                raise AssertionError(f"Phase {phase_idx} xử lý sai số lượng mẫu train")
            windows.append(win)

            score_fn = (
                learner.score
                if isinstance(learner, BlockDiagonalStreamingHead)
                else (lambda raw_b, h=learner: _score_standard_head(h, raw_b))
            )
            eval_metrics = evaluate_seen_classes(
                score_fn,
                test_features,
                test_labels,
                n_classes=n_classes,
                base_classes=50,
                new_classes=classes,
                expected_per_class=test_per_class,
            )
            rows.append({
                "phase": phase_idx,
                "n_classes": n_classes,
                "train_samples_current_phase": int(seen_count),
                "analytic_batches": int(batch_cnt),
                "phase_learning_seconds": win.seconds,
                "phase_peak_rss_bytes": win.peak,
                **eval_metrics,
            })

        w_joint = (
            precomputed_dense_joint_w
            if precomputed_dense_joint_w is not None
            else compute_batched_dense_joint_oracle(
                seed=seed,
                dimension=dimension,
                feature_expansion=feature_expansion,
                gamma=gamma,
                train_features=train_features,
                train_labels=train_labels,
                n_classes=100,
            )
        )
        effective_w = learner.w / float(num_blocks)
        rel_w_vs_joint = float(np.linalg.norm(effective_w - w_joint) / np.linalg.norm(w_joint))
        preds_learner = []
        preds_joint = []
        for start in range(0, len(test_labels), 256):
            raw_chunk = test_features[start : start + 256]
            h_chunk = learner.transform(raw_chunk)
            preds_learner.append((h_chunk @ effective_w).argmax(axis=1))
            preds_joint.append((h_chunk @ w_joint).argmax(axis=1))
        disagreements_vs_joint = int(
            np.count_nonzero(np.concatenate(preds_learner) != np.concatenate(preds_joint))
        )
        if num_blocks == 1 and disagreements_vs_joint != 0:
            raise AssertionError(
                f"ACIL chuẩn d={dimension} phát sinh {disagreements_vs_joint} dự đoán lệch Joint Oracle"
            )

    budget = compute_state_memory_budget(
        dimension,
        n_classes=100,
        d_cnn=64,
        feature_expansion=feature_expansion,
        num_blocks=num_blocks,
    )
    actual_bytes = learner.state_bytes()
    if actual_bytes != budget["bytes_fp64"]:
        raise AssertionError(
            f"Dung lượng mảng NumPy thực tế ({actual_bytes}) không khớp công thức ({budget['bytes_fp64']})"
        )

    base_seconds = windows[0].seconds
    inc_seconds = float(sum(w.seconds for w in windows[1:]))
    baseline_rss = windows[0].baseline
    peak_rss = max(w.peak for w in windows)
    avg_acc = float(np.mean([r["accuracy_pct"] for r in rows]))
    final_row = rows[-1]
    base_delta = float(final_row["base_accuracy_pct"] - rows[0]["base_accuracy_pct"])

    return {
        "branch": "FD2" if num_blocks > 1 else "E2a",
        "variant": variant,
        "seed": seed,
        "dimension": dimension,
        "feature_expansion": feature_expansion,
        "num_blocks": num_blocks,
        "block_dim": dimension // num_blocks,
        "gamma": gamma,
        "batch_size": batch_size,
        "state_memory": {**budget, "actual_nbytes_fp64": actual_bytes, "formula_exact_match": True},
        "timings": {
            "base_seconds": base_seconds,
            "incremental_seconds": inc_seconds,
            "learning_seconds": base_seconds + inc_seconds,
        },
        "rss": {
            "baseline_rss_bytes": baseline_rss,
            "peak_learning_rss_bytes": peak_rss,
            "rss_increase_bytes": peak_rss - baseline_rss,
        },
        "joint_oracle_comparison": {
            "final_relative_w_error_vs_dense_joint": rel_w_vs_joint,
            "final_prediction_disagreements_vs_dense_joint": disagreements_vs_joint,
        },
        "metric_summary": {
            "average_incremental_accuracy_pct": avg_acc,
            "final_accuracy_pct": final_row["accuracy_pct"],
            "initial_base_accuracy_pct": rows[0]["base_accuracy_pct"],
            "final_base_accuracy_pct": final_row["base_accuracy_pct"],
            "final_inc_accuracy_pct": final_row["inc_all_accuracy_pct"],
            "final_new_phase_accuracy_pct": final_row["new_class_accuracy_pct"],
            "base_forgetting_pp": -base_delta,
        },
        "results": rows,
        "verification_passed": True,
    }


def run_typed_woodbury_case(
    *,
    seed: int,
    dimension: int,
    gamma: float,
    dtype_name: str,
    n_incremental_phases: int,
    batch_size: int,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    threads: int = 4,
    compare_all_phases: bool = False,
) -> dict:
    """Thực thi cập nhật Woodbury ở đúng kiểu ``float32`` hoặc ``float64`` và đối chiếu Joint Oracle.

    Mọi ma trận và phép toán trung gian trong nhánh cập nhật được ép kiểu tường minh và kiểm tra
    ``dtype`` sau từng minibatch để ngăn NumPy âm thầm nâng kiểu lên ``float64``.

    Tham số:
        seed: Hạt giống sinh ma trận mở rộng đặc trưng ``W_fe``.
        dimension: Số chiều đặc trưng mở rộng ``d``.
        gamma: Hệ số điều chuẩn Ridge dương (ví dụ ``0.1, 1e-2, 1e-3, 1e-4``).
        dtype_name: Chuỗi kiểu số dấu phẩy động (``"float32"`` hoặc ``"float64"``).
        n_incremental_phases: Số phase tăng dần ``K`` (ví dụ ``5`` hoặc ``25``).
        batch_size: Kích thước minibatch cập nhật Woodbury (ví dụ ``128`` hoặc ``16``).
        train_features: Mảng đặc trưng train ``(N_train, 64)``.
        train_labels: Mảng nhãn toàn cục train ``(N_train,)``.
        test_features: Mảng đặc trưng test ``(N_test, 64)``.
        test_labels: Mảng nhãn toàn cục test ``(N_test,)``.
        threads: Số luồng CPU/BLAS giới hạn.
        compare_all_phases: Nếu ``True`` đối chiếu Joint Oracle ở mọi phase; nếu ``False`` đối chiếu
            tại Base, các mốc chia hết cho 5 và Phase cuối để tiết kiệm thời gian khi ``K=25``.

    Trả về:
        Dict kết quả chẩn đoán độ trôi số học, sai số tương đối ``W/R``, độ mất đối xứng và lệch nhãn.
    """
    if dtype_name not in ("float32", "float64"):
        raise ValueError("dtype_name phải là 'float32' hoặc 'float64'")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma phải hữu hạn và dương")
    target_dtype = np.float32 if dtype_name == "float32" else np.float64
    zero_scalar = target_dtype(0.0)
    one_scalar = target_dtype(1.0)
    gamma_scalar = target_dtype(gamma)

    phases = slice_phase_arrays(
        train_features, train_labels, base_classes=50, n_incremental_phases=n_incremental_phases
    )
    expansion_fp64 = np.random.default_rng(int(seed) + 1000).normal(size=(64, dimension))
    expansion_typed = expansion_fp64.astype(target_dtype, copy=True)

    oracle_h_history: list[np.ndarray] = []
    oracle_y_history: list[np.ndarray] = []
    phase_comparisons = []
    woodbury_steps = 0
    numerical_breakdown = False
    breakdown_phase = None
    breakdown_reason = None
    r_mat: np.ndarray | None = None
    w_mat: np.ndarray | None = None

    with threadpool_limits(limits=threads):
        started = time.perf_counter()
        for phase_idx, (phase_x, phase_y, classes) in enumerate(phases):
            n_classes = max(classes) + 1
            oracle_h_history.append(np.maximum(phase_x.astype(np.float64) @ expansion_fp64, 0.0))
            oracle_y_history.append(phase_y.astype(np.int64))

            try:
                if phase_idx == 0:
                    gram = np.zeros((dimension, dimension), dtype=target_dtype)
                    cross = np.zeros((dimension, n_classes), dtype=target_dtype)
                    for raw_b, y_b in iter_array_batches(phase_x, phase_y, batch_size):
                        x_b = raw_b.astype(target_dtype, copy=False)
                        h_b = np.maximum(x_b @ expansion_typed, zero_scalar)
                        if h_b.dtype != target_dtype:
                            raise AssertionError("Phát hiện nâng kiểu ngầm ở đặc trưng H")
                        y_oh = np.zeros((len(y_b), n_classes), dtype=target_dtype)
                        y_oh[np.arange(len(y_b)), y_b] = one_scalar
                        gram += h_b.T @ h_b
                        cross += h_b.T @ y_oh
                    gram[np.diag_indices(dimension)] += gamma_scalar
                    eye_d = np.eye(dimension, dtype=target_dtype)
                    r_mat = np.linalg.solve(gram, eye_d).astype(target_dtype, copy=False)
                    w_mat = (r_mat @ cross).astype(target_dtype, copy=False)
                    if r_mat.dtype != target_dtype or w_mat.dtype != target_dtype:
                        raise AssertionError("Phát hiện nâng kiểu ngầm ở trạng thái khởi tạo Phase 0")
                    if not (np.isfinite(r_mat).all() and np.isfinite(w_mat).all()):
                        raise FloatingPointError("NaN/Inf xuất hiện ngay tại nghiệm Phase 0")
                else:
                    assert r_mat is not None and w_mat is not None
                    w_pad = np.zeros((dimension, n_classes), dtype=target_dtype)
                    w_pad[:, : w_mat.shape[1]] = w_mat
                    w_mat = w_pad
                    for raw_b, y_b in iter_array_batches(phase_x, phase_y, batch_size):
                        x_b = raw_b.astype(target_dtype, copy=False)
                        h_b = np.maximum(x_b @ expansion_typed, zero_scalar)
                        b_len = len(y_b)
                        eye_b = np.eye(b_len, dtype=target_dtype)
                        left = r_mat @ h_b.T
                        middle = eye_b + h_b @ left
                        rhs = h_b @ r_mat
                        delta_r = left @ np.linalg.solve(middle, rhs)
                        r_mat = (r_mat - delta_r).astype(target_dtype, copy=False)
                        y_oh = np.zeros((b_len, n_classes), dtype=target_dtype)
                        y_oh[np.arange(b_len), y_b] = one_scalar
                        residual = y_oh - h_b @ w_mat
                        w_mat = (w_mat + r_mat @ (h_b.T @ residual)).astype(target_dtype, copy=False)
                        woodbury_steps += 1
                        if r_mat.dtype != target_dtype or w_mat.dtype != target_dtype:
                            raise AssertionError("Phát hiện nâng kiểu ngầm trong vòng lặp Woodbury")
                        if not (np.isfinite(r_mat).all() and np.isfinite(w_mat).all()):
                            raise FloatingPointError("NaN/Inf phát sinh trong cập nhật Woodbury")
            except (np.linalg.LinAlgError, FloatingPointError) as err:
                numerical_breakdown = True
                breakdown_phase = phase_idx
                breakdown_reason = type(err).__name__
                break

            should_check = (
                compare_all_phases
                or phase_idx == 0
                or phase_idx == n_incremental_phases
                or phase_idx % 5 == 0
            )
            if should_check:
                pooled_h = np.vstack(oracle_h_history)
                pooled_y = np.concatenate(oracle_y_history)
                gram_fp64 = pooled_h.T @ pooled_h + float(gamma) * np.eye(dimension, dtype=np.float64)
                eye_fp64 = np.eye(dimension, dtype=np.float64)
                r_oracle = np.linalg.solve(gram_fp64, eye_fp64)
                w_oracle = joint_ridge(pooled_h, pooled_y, n_classes=n_classes, gamma=float(gamma))

                r_f64 = r_mat.astype(np.float64)
                w_f64 = w_mat.astype(np.float64)
                rel_w = float(np.linalg.norm(w_f64 - w_oracle) / np.linalg.norm(w_oracle))
                rel_r = float(np.linalg.norm(r_f64 - r_oracle) / np.linalg.norm(r_oracle))
                sym_err = float(np.linalg.norm(r_f64 - r_f64.T))
                rel_sym_err = float(sym_err / max(np.linalg.norm(r_f64), 1e-30))

                seen_mask = test_labels < n_classes
                seen_x = test_features[seen_mask]
                seen_y = test_labels[seen_mask]
                test_h_typed = np.maximum(seen_x.astype(target_dtype) @ expansion_typed, zero_scalar)
                scores_typed = test_h_typed @ w_mat
                if scores_typed.dtype != target_dtype:
                    raise AssertionError("Điểm dự đoán test bị nâng kiểu ngầm")
                test_h_fp64 = np.maximum(seen_x.astype(np.float64) @ expansion_fp64, 0.0)
                scores_oracle = test_h_fp64 @ w_oracle

                preds_typed = scores_typed.argmax(axis=1)
                preds_oracle = scores_oracle.argmax(axis=1)
                disagree = int(np.count_nonzero(preds_typed != preds_oracle))
                acc_typed = float(100.0 * np.mean(preds_typed == seen_y))
                acc_oracle = float(100.0 * np.mean(preds_oracle == seen_y))
                max_score_err = float(np.max(np.abs(scores_typed.astype(np.float64) - scores_oracle)))

                phase_comparisons.append({
                    "phase": phase_idx,
                    "n_classes": n_classes,
                    "cumulative_woodbury_steps": woodbury_steps,
                    "dtype_verified": str(w_mat.dtype),
                    "relative_w_error": rel_w,
                    "relative_r_error": rel_r,
                    "symmetry_frobenius_error": sym_err,
                    "relative_symmetry_error": rel_sym_err,
                    "max_abs_score_error": max_score_err,
                    "prediction_disagreements": disagree,
                    "typed_accuracy_pct": acc_typed,
                    "oracle_fp64_accuracy_pct": acc_oracle,
                    "accuracy_drop_vs_oracle_pp": acc_oracle - acc_typed,
                })
        elapsed = time.perf_counter() - started

    if numerical_breakdown or not phase_comparisons:
        summary = {
            "numerical_breakdown": True,
            "breakdown_phase": breakdown_phase,
            "breakdown_reason": breakdown_reason,
            "total_woodbury_steps_completed": woodbury_steps,
            "max_relative_w_error": None,
            "final_relative_w_error": None,
            "max_relative_r_error": None,
            "max_symmetry_frobenius_error": None,
            "final_prediction_disagreements": None,
            "total_prediction_disagreements": None,
            "typed_final_accuracy_pct": None,
            "oracle_fp64_final_accuracy_pct": None,
            "final_accuracy_drop_pp": None,
        }
    else:
        final_cmp = phase_comparisons[-1]
        summary = {
            "numerical_breakdown": False,
            "breakdown_phase": None,
            "breakdown_reason": None,
            "total_woodbury_steps_completed": woodbury_steps,
            "max_relative_w_error": max(r["relative_w_error"] for r in phase_comparisons),
            "final_relative_w_error": final_cmp["relative_w_error"],
            "max_relative_r_error": max(r["relative_r_error"] for r in phase_comparisons),
            "final_relative_r_error": final_cmp["relative_r_error"],
            "max_symmetry_frobenius_error": max(r["symmetry_frobenius_error"] for r in phase_comparisons),
            "max_relative_symmetry_error": max(r["relative_symmetry_error"] for r in phase_comparisons),
            "max_abs_score_error": max(r["max_abs_score_error"] for r in phase_comparisons),
            "final_prediction_disagreements": final_cmp["prediction_disagreements"],
            "total_prediction_disagreements": sum(r["prediction_disagreements"] for r in phase_comparisons),
            "typed_final_accuracy_pct": final_cmp["typed_accuracy_pct"],
            "oracle_fp64_final_accuracy_pct": final_cmp["oracle_fp64_accuracy_pct"],
            "final_accuracy_drop_pp": final_cmp["accuracy_drop_vs_oracle_pp"],
        }

    return {
        "branch": "E2b",
        "seed": seed,
        "dimension": dimension,
        "gamma": float(gamma),
        "dtype": dtype_name,
        "n_incremental_phases": n_incremental_phases,
        "batch_size": batch_size,
        "wall_seconds": elapsed,
        "summary": summary,
        "comparisons": phase_comparisons,
        "strict_dtype_preserved": True,
        "verification_passed": True,
    }


def log_e2_case_wandb(
    case_payload: dict,
    *,
    wandb_mode: str = "disabled",
    entity: str = "hsangg-2006",
    project: str = "acil-reproduction",
    wandb_dir: Path,
    provenance: dict,
    wandb_client=None,
) -> dict:
    """Đồng bộ một trường hợp thực nghiệm E2a/FD2/E2b lên W&B với bộ nhãn riêng của cuối kỳ.

    Tham số:
        case_payload: Dict kết quả của một trường hợp ``run_e2a_case`` hoặc ``run_typed_woodbury_case``.
        wandb_mode: Chế độ ghi W&B (``"disabled"``, ``"offline"`` hoặc ``"online"``).
        entity: Tên entity W&B.
        project: Tên project W&B.
        wandb_dir: Thư mục lưu tệp đồng bộ W&B.
        provenance: Dict thông tin truy nguyên mã nguồn Git/SHA-256.
        wandb_client: Client W&B giả lập phục vụ kiểm thử đơn vị (tùy chọn).

    Trả về:
        Dict trạng thái đồng bộ W&B gồm ``wandb_status``, ``wandb_run_id``, ``wandb_run_url`` và ``tags``.
    """
    if wandb_mode == "disabled":
        return {"wandb_status": "not_logged", "wandb_run_id": None, "wandb_run_url": None, "tags": []}
    if wandb_mode not in ("offline", "online"):
        raise ValueError("wandb_mode phải thuộc {'disabled', 'offline', 'online'}")

    branch = case_payload["branch"]
    seed = case_payload["seed"]
    if branch in ("E2a", "FD2"):
        dim = case_payload["dimension"]
        blocks = case_payload["num_blocks"]
        sub_tag = "FD2-block-diagonal" if branch == "FD2" else "E2a-memory-budget"
        run_name = (
            f"FD2-d{dim}-B{blocks}-seed{seed}"
            if branch == "FD2"
            else f"E2a-d{dim}-seed{seed}"
        )
        tags = [
            "final-empirics",
            "P2-memory-precision",
            "P2-memory-scaling",
            sub_tag,
            f"d{dim}",
            f"seed{seed}",
        ]
        step_rows = case_payload["results"]
        summary_dict = {
            **case_payload["metric_summary"],
            **case_payload["timings"],
            **case_payload["rss"],
            "bytes_fp64": case_payload["state_memory"]["bytes_fp64"],
            "bytes_fp32": case_payload["state_memory"]["bytes_fp32"],
            "mb_fp64": case_payload["state_memory"]["mb_fp64"],
            "ratio_fp64_vs_2000_exemplars": case_payload["state_memory"]["ratio_fp64_vs_2000_exemplars"],
        }
    else:
        dtype_str = case_payload["dtype"]
        gamma_val = case_payload["gamma"]
        k_val = case_payload["n_incremental_phases"]
        b_val = case_payload["batch_size"]
        run_name = f"E2b-{dtype_str}-g{gamma_val:g}-K{k_val}b{b_val}-seed{seed}"
        tags = [
            "final-empirics",
            "P2-memory-precision",
            "P2-memory-scaling",
            "E2b-woodbury-precision",
            dtype_str,
            f"K{k_val}b{b_val}",
            f"seed{seed}",
        ]
        step_rows = case_payload["comparisons"]
        summary_dict = {k: v for k, v in case_payload["summary"].items() if v is not None}

    if wandb_client is None:
        import wandb as wandb_client
    with wandb_client.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="final-empirics-e2",
        mode=wandb_mode,
        dir=str(wandb_dir),
        tags=tags,
        config={
            "branch": branch,
            "seed": seed,
            "dimension": case_payload["dimension"],
            "gamma": case_payload["gamma"],
            "batch_size": case_payload["batch_size"],
            "source_commit": provenance.get("git_commit"),
            "runner_script": "scripts/e2_memory_precision_runner.py",
        },
    ) as run:
        for row in step_rows:
            metrics = {k: v for k, v in row.items() if isinstance(v, (int, float, bool))}
            run.log(metrics, step=row["phase"])
        run.summary.update(summary_dict)
        run.summary["verification_passed"] = True
        run_id = getattr(run, "id", None)
        run_url = getattr(run, "url", None)
    if wandb_mode == "online" and (not run_id or not run_url):
        raise RuntimeError("W&B online không trả về ID/URL hợp lệ")
    return {
        "wandb_status": "logged" if wandb_mode == "online" else "offline_pending_sync",
        "wandb_run_id": run_id,
        "wandb_run_url": run_url,
        "tags": tags,
    }


def log_e2_suite_wandb(
    *,
    seed: int,
    e2a_cases: list[dict],
    fd2_cases: list[dict],
    e2b_cases: list[dict],
    wandb_mode: str,
    entity: str,
    project: str,
    wandb_dir: Path,
    provenance: dict,
    wandb_client=None,
) -> dict:
    """Ghi gộp 2 runs W&B gọn cho mỗi seed: một run đường cong E2a+FD2 và một run khảo sát E2b.

    Tham số:
        seed: Hạt giống chuẩn của R1 (42, 43 hoặc 44).
        e2a_cases: Danh sách kết quả quét chiều d của E2a.
        fd2_cases: Danh sách kết quả nguyên mẫu khối đường chéo FD2.
        e2b_cases: Danh sách kết quả khảo sát số học float32 vs float64 của E2b.
        wandb_mode: Chế độ ghi W&B (``"disabled"``, ``"offline"`` hoặc ``"online"``).
        entity: Tên entity W&B.
        project: Tên project W&B.
        wandb_dir: Thư mục lưu đệm tệp W&B.
        provenance: Dict thông tin truy nguyên Git/SHA-256.
        wandb_client: Client W&B giả lập cho kiểm thử (tùy chọn).

    Trả về:
        Dict ghi nhận trạng thái và URL của ``e2a_run`` và ``e2b_run`` trên W&B.
    """
    if wandb_mode == "disabled":
        return {"e2a_run": {"wandb_status": "not_logged"}, "e2b_run": {"wandb_status": "not_logged"}}
    if wandb_client is None:
        import wandb as wandb_client

    out = {"e2a_run": {"wandb_status": "not_logged"}, "e2b_run": {"wandb_status": "not_logged"}}
    if e2a_cases or fd2_cases:
        tags_a = [
            "final-empirics",
            "P2-memory-scaling",
            "P2-memory-precision",
            "E2a-memory-budget",
            f"seed{seed}",
        ]
        if fd2_cases:
            tags_a.append("FD2-block-diagonal")
        with wandb_client.init(
            entity=entity,
            project=project,
            name=f"E2a-memory-scaling-seed{seed}",
            job_type="final-empirics-e2a",
            mode=wandb_mode,
            dir=str(wandb_dir),
            tags=tags_a,
            config={
                "branch": "E2a_and_FD2",
                "seed": seed,
                "dimensions": [c["dimension"] for c in e2a_cases],
                "fd2_blocks": [c["num_blocks"] for c in fd2_cases],
                "gamma": 0.1,
                "batch_size": 128,
                "source_commit": provenance.get("git_commit"),
                "runner_script": "scripts/e2_memory_precision_runner.py",
            },
        ) as run_a:
            for item in e2a_cases:
                dim = item["dimension"]
                run_a.log(
                    {
                        "dimension": dim,
                        "state_mb_fp64": item["state_memory"]["mb_fp64"],
                        "state_mb_fp32": item["state_memory"]["mb_fp32"],
                        "ratio_fp64_vs_2000_exemplars": item["state_memory"]["ratio_fp64_vs_2000_exemplars"],
                        "peak_learning_rss_mib": item["rss"]["peak_learning_rss_bytes"] / (1024.0 ** 2),
                        "learning_seconds": item["timings"]["learning_seconds"],
                        "average_incremental_accuracy_pct": item["metric_summary"]["average_incremental_accuracy_pct"],
                        "final_accuracy_pct": item["metric_summary"]["final_accuracy_pct"],
                        "final_inc_accuracy_pct": item["metric_summary"]["final_inc_accuracy_pct"],
                        "final_relative_w_error_vs_dense_joint": item["joint_oracle_comparison"]["final_relative_w_error_vs_dense_joint"],
                        "final_prediction_disagreements_vs_dense_joint": item["joint_oracle_comparison"]["final_prediction_disagreements_vs_dense_joint"],
                    },
                    step=dim,
                )
                if dim in (64, 798, 1024, 1160, 8000):
                    run_a.summary[f"d{dim}_abar_pct"] = item["metric_summary"]["average_incremental_accuracy_pct"]
                    run_a.summary[f"d{dim}_afinal_pct"] = item["metric_summary"]["final_accuracy_pct"]
                    run_a.summary[f"d{dim}_mb_fp64"] = item["state_memory"]["mb_fp64"]
            for fd2_item in fd2_cases:
                blk = fd2_item["num_blocks"]
                run_a.summary[f"FD2_d8000_B{blk}_abar_pct"] = fd2_item["metric_summary"]["average_incremental_accuracy_pct"]
                run_a.summary[f"FD2_d8000_B{blk}_afinal_pct"] = fd2_item["metric_summary"]["final_accuracy_pct"]
                run_a.summary[f"FD2_d8000_B{blk}_mb_fp64"] = fd2_item["state_memory"]["mb_fp64"]
            run_a.summary["verification_passed"] = True
            id_a, url_a = getattr(run_a, "id", None), getattr(run_a, "url", None)
        if wandb_mode == "online" and (not id_a or not url_a):
            raise RuntimeError("W&B online E2a không trả về ID/URL")
        out["e2a_run"] = {
            "wandb_status": "logged" if wandb_mode == "online" else "offline_pending_sync",
            "wandb_run_id": id_a,
            "wandb_run_url": url_a,
            "tags": tags_a,
        }

    if e2b_cases:
        tags_b = [
            "final-empirics",
            "P2-memory-scaling",
            "P2-memory-precision",
            "E2b-woodbury-precision",
            f"seed{seed}",
        ]
        with wandb_client.init(
            entity=entity,
            project=project,
            name=f"E2b-woodbury-precision-seed{seed}",
            job_type="final-empirics-e2b",
            mode=wandb_mode,
            dir=str(wandb_dir),
            tags=tags_b,
            config={
                "branch": "E2b",
                "seed": seed,
                "dimension": e2b_cases[0]["dimension"],
                "n_cases": len(e2b_cases),
                "source_commit": provenance.get("git_commit"),
                "runner_script": "scripts/e2_memory_precision_runner.py",
            },
        ) as run_b:
            for idx, item in enumerate(e2b_cases):
                sm = item["summary"]
                payload = {
                    "is_float32": item["dtype"] == "float32",
                    "gamma": item["gamma"],
                    "n_incremental_phases": item["n_incremental_phases"],
                    "batch_size": item["batch_size"],
                    "numerical_breakdown": sm["numerical_breakdown"],
                }
                for key in (
                    "max_relative_w_error",
                    "max_relative_r_error",
                    "max_symmetry_frobenius_error",
                    "final_prediction_disagreements",
                    "total_prediction_disagreements",
                    "final_accuracy_drop_pp",
                ):
                    if sm.get(key) is not None:
                        payload[key] = sm[key]
                run_b.log(payload, step=idx)
                label = f"{item['dtype']}_K{item['n_incremental_phases']}b{item['batch_size']}_g{item['gamma']:g}"
                run_b.summary[f"{label}_max_rel_w"] = sm["max_relative_w_error"]
                run_b.summary[f"{label}_disagree"] = sm["final_prediction_disagreements"]
            run_b.summary["verification_passed"] = True
            id_b, url_b = getattr(run_b, "id", None), getattr(run_b, "url", None)
        if wandb_mode == "online" and (not id_b or not url_b):
            raise RuntimeError("W&B online E2b không trả về ID/URL")
        out["e2b_run"] = {
            "wandb_status": "logged" if wandb_mode == "online" else "offline_pending_sync",
            "wandb_run_id": id_b,
            "wandb_run_url": url_b,
            "tags": tags_b,
        }
    return out


def run_e2_seed_suite(
    *,
    seed: int = 42,
    output_dir: str | Path,
    fixture_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    data_dir: str | Path = ROOT / "data",
    project_root: Path = ROOT,
    run_e2a: bool = True,
    include_fd2: bool = True,
    run_e2b: bool = True,
    e2a_dimensions: tuple[int, ...] = DEFAULT_E2A_DIMENSIONS,
    fd2_blocks: tuple[int, ...] = DEFAULT_FD2_BLOCKS,
    e2b_dimension: int = 1024,
    e2b_gammas: tuple[float, ...] = DEFAULT_E2B_GAMMAS,
    e2b_regimes: tuple[tuple[int, int], ...] = DEFAULT_E2B_REGIMES,
    device: str = "auto",
    threads: int = 4,
    wandb_mode: str = "disabled",
    wandb_granularity: str = "suite",
    entity: str = "hsangg-2006",
    project: str = "acil-reproduction",
    wandb_client=None,
) -> dict:
    """Thực thi trọn bộ thực nghiệm E2 (E2a, FD2 và/hoặc E2b) cho một seed R1 trong ``RunAttempt``.

    Tham số:
        seed: Hạt giống chuẩn của R1 (``42``, ``43`` hoặc ``44``).
        output_dir: Thư mục gốc ghi báo cáo ``attempts/<id>/results.json``.
        fixture_dir: Thư mục lưu/tái dùng đặc trưng ``F64`` của seed này (tùy chọn).
        checkpoint_path: Đường dẫn tường minh tới ``base_checkpoint.pt`` của R1 (tùy chọn).
        data_dir: Thư mục dữ liệu CIFAR-100.
        project_root: Thư mục gốc repository.
        run_e2a: Bật chạy lưới chiều mở rộng ``e2a_dimensions`` nếu ``True``.
        include_fd2: Bật chạy nguyên mẫu ``BlockDiagonalStreamingHead`` tại ``d=8000`` nếu ``True``.
        run_e2b: Bật chạy khảo sát số học ``float32`` vs ``float64`` nếu ``True``.
        e2a_dimensions: Tuple các chiều ``d`` cần quét trong nhánh E2a.
        fd2_blocks: Tuple các số khối ``B`` cho nguyên mẫu FD2 ở ``d=8000``.
        e2b_dimension: Chiều mở rộng ``d`` dùng cho khảo sát E2b (mặc định 1024).
        e2b_gammas: Tuple các hệ số ``gamma`` cần quét trong nhánh E2b.
        e2b_regimes: Tuple các cặp ``(K, batch_size)`` cần đo trong nhánh E2b.
        device: Thiết bị chạy bước trích xuất đặc trưng CNN ban đầu.
        threads: Số luồng CPU/BLAS giới hạn cho mọi phép tính giải tích.
        wandb_mode: Chế độ ghi W&B (``"disabled"``, ``"offline"`` hoặc ``"online"``).
        wandb_granularity: ``"suite"`` (mặc định: 2 runs gọn/seed) hoặc ``"case"`` (1 run/cấu hình).
        entity: Tên entity W&B.
        project: Tên project W&B.
        wandb_client: Client W&B tùy chọn cho kiểm thử.

    Trả về:
        Dict báo cáo toàn bộ suite E2 của seed đã hoàn tất và vượt qua mọi kiểm tra.
    """
    if wandb_granularity not in ("suite", "case"):
        raise ValueError("wandb_granularity phải là 'suite' hoặc 'case'")
    out_root = Path(output_dir)
    resolved_fixture = Path(fixture_dir) if fixture_dir is not None else (out_root / f"fixture_seed{seed}")

    with RunAttempt(out_root, seed=seed) as attempt:
        attempt.progress("prepare_fixture", seed=seed)
        provenance = collect_provenance(project_root)
        runner_path = Path(__file__).resolve()
        provenance["runner_script"] = runner_path.relative_to(project_root).as_posix()
        provenance["runner_sha256"] = sha256_file(runner_path)

        meta, train_x, train_y, test_x, test_y = prepare_or_load_seed_fixture(
            seed,
            resolved_fixture,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            project_root=project_root,
            device=device,
            threads=threads,
        )

        cached_d8000_joint: np.ndarray | None = None
        if (run_e2a and 8000 in e2a_dimensions) or include_fd2:
            with threadpool_limits(limits=threads):
                cached_d8000_joint = compute_batched_dense_joint_oracle(
                    seed=seed,
                    dimension=8000,
                    feature_expansion=True,
                    gamma=0.1,
                    train_features=train_x,
                    train_labels=train_y,
                    n_classes=100,
                )

        case_wandb_mode = wandb_mode if wandb_granularity == "case" else "disabled"
        e2a_cases = []
        if run_e2a:
            for dim in e2a_dimensions:
                attempt.progress("e2a_dimension", dimension=dim)
                fe_enabled = dim != 64
                case_out = run_e2a_case(
                    seed=seed,
                    dimension=dim,
                    train_features=train_x,
                    train_labels=train_y,
                    test_features=test_x,
                    test_labels=test_y,
                    feature_expansion=fe_enabled,
                    num_blocks=1,
                    gamma=0.1,
                    batch_size=128,
                    threads=threads,
                    precomputed_dense_joint_w=cached_d8000_joint if dim == 8000 else None,
                )
                case_out["tracking"] = log_e2_case_wandb(
                    case_out,
                    wandb_mode=case_wandb_mode,
                    entity=entity,
                    project=project,
                    wandb_dir=attempt.directory,
                    provenance=provenance,
                    wandb_client=wandb_client,
                )
                e2a_cases.append(case_out)

        fd2_cases = []
        if include_fd2:
            for blk in fd2_blocks:
                attempt.progress("fd2_block_diagonal", num_blocks=blk)
                fd2_out = run_e2a_case(
                    seed=seed,
                    dimension=8000,
                    train_features=train_x,
                    train_labels=train_y,
                    test_features=test_x,
                    test_labels=test_y,
                    feature_expansion=True,
                    num_blocks=blk,
                    gamma=0.1,
                    batch_size=128,
                    threads=threads,
                    precomputed_dense_joint_w=cached_d8000_joint,
                )
                fd2_out["tracking"] = log_e2_case_wandb(
                    fd2_out,
                    wandb_mode=case_wandb_mode,
                    entity=entity,
                    project=project,
                    wandb_dir=attempt.directory,
                    provenance=provenance,
                    wandb_client=wandb_client,
                )
                fd2_cases.append(fd2_out)

        e2b_cases = []
        if run_e2b:
            for k_phases, b_size in e2b_regimes:
                for gamma_val in e2b_gammas:
                    for dtype_str in ("float64", "float32"):
                        attempt.progress(
                            "e2b_woodbury",
                            dtype=dtype_str,
                            gamma=gamma_val,
                            phases=k_phases,
                            batch_size=b_size,
                        )
                        e2b_out = run_typed_woodbury_case(
                            seed=seed,
                            dimension=e2b_dimension,
                            gamma=gamma_val,
                            dtype_name=dtype_str,
                            n_incremental_phases=k_phases,
                            batch_size=b_size,
                            train_features=train_x,
                            train_labels=train_y,
                            test_features=test_x,
                            test_labels=test_y,
                            threads=threads,
                        )
                        e2b_out["tracking"] = log_e2_case_wandb(
                            e2b_out,
                            wandb_mode=case_wandb_mode,
                            entity=entity,
                            project=project,
                            wandb_dir=attempt.directory,
                            provenance=provenance,
                            wandb_client=wandb_client,
                        )
                        e2b_cases.append(e2b_out)

        suite_tracking = (
            log_e2_suite_wandb(
                seed=seed,
                e2a_cases=e2a_cases,
                fd2_cases=fd2_cases,
                e2b_cases=e2b_cases,
                wandb_mode=wandb_mode,
                entity=entity,
                project=project,
                wandb_dir=attempt.directory,
                provenance=provenance,
                wandb_client=wandb_client,
            )
            if wandb_granularity == "suite"
            else {"mode": "per_case"}
        )

        r1_cross_check_passed = True
        for item in e2a_cases:
            if item["dimension"] == 8000 and item["num_blocks"] == 1:
                expected_abar = meta["baseline_metrics"]["average_incremental_accuracy_pct"]
                if abs(item["metric_summary"]["average_incremental_accuracy_pct"] - expected_abar) > 1e-6:
                    r1_cross_check_passed = False

        report = {
            "experiment": "ACIL-E2-Memory-And-Precision",
            "seed": seed,
            "iso_byte_anchors": {
                "cifar100_2000_exemplars_bytes": CIFAR100_EXEMPLAR_2000_BYTES,
                "d_max_fp64": max_iso_byte_dimension(bytes_per_elem=8),
                "d_max_fp32": max_iso_byte_dimension(bytes_per_elem=4),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "threads": threads,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("scipy", "psutil", "threadpoolctl")
                },
            },
            "provenance": provenance,
            "fixture_metadata": meta,
            "suite_tracking": suite_tracking,
            "e2a_cases": e2a_cases,
            "fd2_cases": fd2_cases,
            "e2b_cases": e2b_cases,
            "checks": {
                "training_never_executed": meta["training_executed_this_run"] is False,
                "all_e2a_verified": all(c["verification_passed"] for c in e2a_cases),
                "all_fd2_verified": all(c["verification_passed"] for c in fd2_cases),
                "all_e2b_verified": all(c["verification_passed"] for c in e2b_cases),
                "d8000_matches_r1_baseline": r1_cross_check_passed,
            },
        }
        report["verification_passed"] = all(report["checks"].values())
        save_json(report, attempt.directory / "results.json")
        return attempt.complete(report)


def aggregate_e2_reports(report_paths: list[str | Path], output_path: str | Path | None = None) -> dict:
    """Tổng hợp kết quả nhiều seeds (mean ± sample SD, ddof=1) từ các tệp ``results.json`` của E2.

    Tham số:
        report_paths: Danh sách đường dẫn tới các tệp ``results.json`` hoàn tất của ``run_e2_seed_suite``.
        output_path: Đường dẫn tệp JSON đích lưu bảng tổng hợp (tùy chọn).

    Trả về:
        Dict tổng hợp đường cong E2a, nguyên mẫu FD2 và bảng chẩn đoán E2b.
    """
    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in report_paths]
    for path, item in zip(report_paths, payloads):
        if item.get("status") != "complete" or not item.get("verification_passed"):
            raise ValueError(f"Báo cáo E2 chưa hoàn tất hợp lệ: {path}")

    curve_groups: dict[str, list[dict]] = {}
    for item in payloads:
        for case in [*item.get("e2a_cases", []), *item.get("fd2_cases", [])]:
            key = f"{case['branch']}__{case['variant']}__d{case['dimension']}__B{case['num_blocks']}"
            curve_groups.setdefault(key, []).append(case)

    summarized_curves = {}
    for key, cases in sorted(curve_groups.items(), key=lambda kv: (kv[1][0]["branch"], kv[1][0]["dimension"])):
        cases_sorted = sorted(cases, key=lambda c: c["seed"])
        seeds = [c["seed"] for c in cases_sorted]
        metrics_out = {}
        for m_name in (
            "average_incremental_accuracy_pct",
            "final_accuracy_pct",
            "final_base_accuracy_pct",
            "final_inc_accuracy_pct",
            "base_forgetting_pp",
        ):
            vals = np.asarray([c["metric_summary"][m_name] for c in cases_sorted], dtype=np.float64)
            metrics_out[m_name] = {
                "mean": float(np.mean(vals)),
                "sample_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
                "values_by_seed": {str(c["seed"]): c["metric_summary"][m_name] for c in cases_sorted},
            }
        learn_secs = np.asarray([c["timings"]["learning_seconds"] for c in cases_sorted], dtype=np.float64)
        peak_rss_mib = np.asarray(
            [c["rss"]["peak_learning_rss_bytes"] / (1024.0 ** 2) for c in cases_sorted],
            dtype=np.float64,
        )
        rel_w_joint = np.asarray(
            [c["joint_oracle_comparison"]["final_relative_w_error_vs_dense_joint"] for c in cases_sorted],
            dtype=np.float64,
        )
        disagree_joint = [
            c["joint_oracle_comparison"]["final_prediction_disagreements_vs_dense_joint"]
            for c in cases_sorted
        ]
        summarized_curves[key] = {
            "branch": cases_sorted[0]["branch"],
            "variant": cases_sorted[0]["variant"],
            "dimension": cases_sorted[0]["dimension"],
            "num_blocks": cases_sorted[0]["num_blocks"],
            "seeds": seeds,
            "state_memory": cases_sorted[0]["state_memory"],
            "learning_seconds": {
                "mean": float(np.mean(learn_secs)),
                "sample_sd": float(np.std(learn_secs, ddof=1)) if len(learn_secs) > 1 else 0.0,
            },
            "peak_learning_rss_mib": {
                "mean": float(np.mean(peak_rss_mib)),
                "sample_sd": float(np.std(peak_rss_mib, ddof=1)) if len(peak_rss_mib) > 1 else 0.0,
            },
            "joint_oracle_comparison": {
                "max_relative_w_error": float(np.max(rel_w_joint)),
                "mean_relative_w_error": float(np.mean(rel_w_joint)),
                "total_prediction_disagreements": int(sum(disagree_joint)),
            },
            "metrics": metrics_out,
        }

    e2b_groups: dict[str, list[dict]] = {}
    for item in payloads:
        for case in item.get("e2b_cases", []):
            g_key = f"{case['dtype']}__K{case['n_incremental_phases']}b{case['batch_size']}__g{case['gamma']:g}"
            e2b_groups.setdefault(g_key, []).append(case)

    e2b_summarized = {}
    for g_key, cases in e2b_groups.items():
        cases_sorted = sorted(cases, key=lambda c: c["seed"])
        breakdowns = [c["summary"]["numerical_breakdown"] for c in cases_sorted]
        valid = [c for c in cases_sorted if not c["summary"]["numerical_breakdown"]]
        rel_w_vals = np.asarray([c["summary"]["max_relative_w_error"] for c in valid], dtype=np.float64)
        rel_r_vals = np.asarray([c["summary"]["max_relative_r_error"] for c in valid], dtype=np.float64)
        sym_vals = np.asarray([c["summary"]["max_symmetry_frobenius_error"] for c in valid], dtype=np.float64)
        drop_vals = np.asarray([c["summary"]["final_accuracy_drop_pp"] for c in valid], dtype=np.float64)
        final_disagrees = [c["summary"]["final_prediction_disagreements"] for c in valid]
        e2b_summarized[g_key] = {
            "dtype": cases_sorted[0]["dtype"],
            "n_incremental_phases": cases_sorted[0]["n_incremental_phases"],
            "batch_size": cases_sorted[0]["batch_size"],
            "gamma": cases_sorted[0]["gamma"],
            "dimension": cases_sorted[0]["dimension"],
            "seeds": [c["seed"] for c in cases_sorted],
            "n_breakdowns": int(sum(breakdowns)),
            "max_relative_w_error_mean": float(np.mean(rel_w_vals)) if len(valid) else None,
            "max_relative_w_error_sd": float(np.std(rel_w_vals, ddof=1)) if len(valid) > 1 else 0.0,
            "max_relative_w_error_worst": float(np.max(rel_w_vals)) if len(valid) else None,
            "max_relative_r_error_mean": float(np.mean(rel_r_vals)) if len(valid) else None,
            "max_symmetry_error_mean": float(np.mean(sym_vals)) if len(valid) else None,
            "final_prediction_disagreements_mean": float(np.mean(final_disagrees)) if len(valid) else None,
            "final_prediction_disagreements_total": int(sum(final_disagrees)) if len(valid) else None,
            "final_accuracy_drop_pp_mean": float(np.mean(drop_vals)) if len(valid) else None,
            "values_by_seed": {
                str(c["seed"]): c["summary"] for c in cases_sorted
            },
        }

    e2b_rows = [case for item in payloads for case in item.get("e2b_cases", [])]
    summary = {
        "status": "complete",
        "experiment_family": "ACIL-Final-Empirics-E2",
        "iso_byte_anchors": payloads[0]["iso_byte_anchors"],
        "e2a_and_fd2_summary": summarized_curves,
        "e2b_summary": e2b_summarized,
        "e2b_cases": e2b_rows,
    }
    if output_path is not None:
        save_json(summary, Path(output_path))
    return summary


def main() -> None:
    """Điểm vào CLI thực thi suite E2 theo từng seed R1 hoặc gộp kết quả nhiều seeds.

    Tham số:
        Không nhận tham số trực tiếp; đọc đối số dòng lệnh qua ``argparse``.

    Trả về:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Chạy thực nghiệm E2 trên checkpoint R1 đã kiểm SHA-256")
    run_p.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    run_p.add_argument("--checkpoint", type=Path)
    run_p.add_argument("--fixture-dir", type=Path)
    run_p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    run_p.add_argument("--output-dir", type=Path, required=True)
    run_p.add_argument("--skip-e2a", action="store_true")
    run_p.add_argument("--skip-fd2", action="store_true")
    run_p.add_argument("--skip-e2b", action="store_true")
    run_p.add_argument("--e2b-dimension", type=int, default=1024)
    run_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run_p.add_argument("--threads", type=int, default=4)
    run_p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    run_p.add_argument("--entity", default="hsangg-2006")
    run_p.add_argument("--project", default="acil-reproduction")

    agg_p = subparsers.add_parser("aggregate", help="Gộp kết quả E2 từ danh sách tệp results.json")
    agg_p.add_argument("--inputs", type=Path, nargs="+", required=True)
    agg_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        run_e2_seed_suite(
            seed=args.seed,
            output_dir=args.output_dir,
            fixture_dir=args.fixture_dir,
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            run_e2a=not args.skip_e2a,
            include_fd2=not args.skip_fd2,
            run_e2b=not args.skip_e2b,
            e2b_dimension=args.e2b_dimension,
            device=args.device,
            threads=args.threads,
            wandb_mode=args.wandb_mode,
            entity=args.entity,
            project=args.project,
        )
    else:
        aggregate_e2_reports(args.inputs, args.output)


if __name__ == "__main__":
    main()
