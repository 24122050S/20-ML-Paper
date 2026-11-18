"""Điều phối thực nghiệm cuối kỳ E1 (E1a & E1b) kiểm chứng giả định tập Base đóng băng (P1).

Module này mở rộng giao thức chia lớp sang quy mô lớp Base tùy biến c_0 in {10, 20, 30, 50}
(nhánh E1a) và phân hoạch theo 20 nhóm ngữ nghĩa thô coarse superclasses của CIFAR-100
(nhánh E1b), đồng thời giữ nguyên toàn bộ mã nguồn src/*.py và dữ liệu lịch sử R1-R3.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytic import joint_ridge
from src.base_training import train_base
from src.cifar_data import CifarArrays, DATASET_REVISION, load_cifar100, sha256_file
from src.cifar_smoke import fingerprint, iter_feature_batches, make_loader, seed_everything
from src.profile_analytic import MemoryMonitor
from src.reporting import collect_provenance, save_json
from src.resnet import ResNet32, freeze_backbone
from src.run_attempt import RunAttempt
from src.streaming import StreamingHead


CANONICAL_COARSE_NAMES: tuple[str, ...] = (
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
)

COARSE_TO_FINE_GROUPS: dict[int, tuple[int, ...]] = {
    0: (4, 30, 55, 72, 95),
    1: (1, 32, 67, 73, 91),
    2: (54, 62, 70, 82, 92),
    3: (9, 10, 16, 28, 61),
    4: (0, 51, 53, 57, 83),
    5: (22, 39, 40, 86, 87),
    6: (5, 20, 25, 84, 94),
    7: (6, 7, 14, 18, 24),
    8: (3, 42, 43, 88, 97),
    9: (12, 17, 37, 68, 76),
    10: (23, 33, 49, 60, 71),
    11: (15, 19, 21, 31, 38),
    12: (34, 63, 64, 66, 75),
    13: (26, 45, 77, 79, 99),
    14: (2, 11, 35, 46, 98),
    15: (27, 29, 44, 78, 93),
    16: (36, 50, 65, 74, 80),
    17: (47, 52, 56, 59, 96),
    18: (8, 13, 48, 58, 90),
    19: (41, 69, 81, 85, 89),
}

NATURAL_SUPERCLASSES: tuple[int, ...] = (0, 1, 2, 4, 7, 8, 10, 11, 12, 17)
SHIFTED_SUPERCLASSES: tuple[int, ...] = (3, 5, 6, 9, 13, 14, 15, 16, 18, 19)
PURE_MANMADE_SUPERCLASSES: tuple[int, ...] = (3, 5, 6, 9, 18, 19)


def build_canonical_fine_to_coarse() -> tuple[int, ...]:
    """Xây dựng bảng ánh xạ chuẩn từ 100 nhãn fine (0..99) sang 20 nhãn coarse (0..19).

    Tham số:
        Không nhận tham số; đọc trực tiếp từ hằng số ``COARSE_TO_FINE_GROUPS``.

    Trả về:
        Tuple gồm 100 số nguyên trong phạm vi ``0..19`` ứng với từng ``fine_label``.
    """
    mapping = [-1] * 100
    for coarse_id, fine_ids in COARSE_TO_FINE_GROUPS.items():
        if len(fine_ids) != 5:
            raise ValueError(f"Nhóm coarse {coarse_id} không có đúng 5 lớp fine")
        for fine_id in fine_ids:
            if not 0 <= fine_id < 100 or mapping[fine_id] != -1:
                raise ValueError(f"Nhãn fine {fine_id} không hợp lệ hoặc bị trùng")
            mapping[fine_id] = coarse_id
    if any(value < 0 for value in mapping):
        raise ValueError("Bảng ánh xạ fine -> coarse chưa phủ đủ 100 lớp")
    return tuple(mapping)


CANONICAL_FINE_TO_COARSE: tuple[int, ...] = build_canonical_fine_to_coarse()


def verify_or_load_coarse_map(data_dir: str | Path | None = None) -> tuple[int, ...]:
    """Kiểm tra bảng ánh xạ fine -> coarse với tệp Parquet gốc nếu có trong cache địa phương.

    Tham số:
        data_dir: Đường dẫn thư mục dữ liệu chứa ``hf_cache`` (hoặc ``None`` khi chạy test giả lập).

    Trả về:
        Tuple gồm 100 nhãn ``coarse_label`` đã xác minh khớp hoàn toàn với ``CANONICAL_FINE_TO_COARSE``.
    """
    if data_dir is None:
        return CANONICAL_FINE_TO_COARSE
    cache_root = Path(data_dir) / "hf_cache"
    if not cache_root.is_dir():
        return CANONICAL_FINE_TO_COARSE
    snapshots = sorted(cache_root.rglob(f"*{DATASET_REVISION}*/cifar100/test-00000-of-00001.parquet"))
    if not snapshots:
        snapshots = sorted(cache_root.rglob("cifar100/test-00000-of-00001.parquet"))
    if not snapshots:
        return CANONICAL_FINE_TO_COARSE
    parquet = pq.ParquetFile(snapshots[0])
    metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
    coarse_names = tuple(metadata["info"]["features"]["coarse_label"]["names"])
    if coarse_names != CANONICAL_COARSE_NAMES:
        raise ValueError("Danh sách tên coarse_label trong Parquet không khớp bảng hằng số")
    columns = parquet.read(columns=["fine_label", "coarse_label"]).to_pydict()
    observed = [-1] * 100
    for fine_id, coarse_id in zip(columns["fine_label"], columns["coarse_label"]):
        if observed[fine_id] not in (-1, coarse_id):
            raise ValueError(f"Nhãn fine {fine_id} ánh xạ sang nhiều coarse_label khác nhau")
        observed[fine_id] = int(coarse_id)
    observed_tuple = tuple(observed)
    if observed_tuple != CANONICAL_FINE_TO_COARSE:
        raise ValueError("Ánh xạ fine -> coarse từ Parquet không khớp CANONICAL_FINE_TO_COARSE")
    return observed_tuple


def default_e1_config(
    *,
    branch: str = "E1a",
    seed: int = 42,
    base_classes: int = 50,
    phase_size: int = 10,
    split_mode: str = "uniform",
) -> dict:
    """Tạo dict cấu hình mặc định chuẩn benchmark cho một trường hợp thực nghiệm E1.

    Tham số:
        branch: Mã nhánh thực nghiệm (``"E1a"`` hoặc ``"E1b"``).
        seed: Hạt giống ngẫu nhiên cho hoán vị lớp, huấn luyện BP và chiếu ngẫu nhiên FE.
        base_classes: Số lượng lớp ở Phase 0 (ví dụ ``10, 20, 30, 50``).
        phase_size: Số lượng lớp mới thêm vào ở mỗi phase tăng dần (mặc định ``10``).
        split_mode: Quy tắc chia lớp (``"uniform"``, ``"stratified"``, ``"superclass_disjoint"``,
            hoặc ``"superclass_disjoint_reverse"``).

    Trả về:
        Dict cấu hình đầy đủ sẵn sàng truyền vào ``run_e1_case``.
    """
    if branch not in ("E1a", "E1b"):
        raise ValueError("branch phải là 'E1a' hoặc 'E1b'")
    if base_classes < 1 or base_classes >= 100 or (100 - base_classes) % phase_size != 0:
        raise ValueError("base_classes phải thuộc 1..99 và (100 - base_classes) chia hết cho phase_size")
    inc_phases = (100 - base_classes) // phase_size
    return {
        "experiment": f"ACIL-{branch}-C100",
        "branch": branch,
        "split_mode": split_mode,
        "seed": seed,
        "base_classes": base_classes,
        "phase_size": phase_size,
        "incremental_phases": inc_phases,
        "run_incremental_phases": inc_phases,
        "epochs": 160,
        "batch_size": 128,
        "eval_batch_size": 256,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "lr_milestones": [80, 120],
        "expansion_dim": 8000,
        "feature_expansion": True,
        "gamma": 0.1,
        "analytic_batch_size": 128,
        "base_statistics_batch_size": 128,
        "num_workers": 0,
        "torch_threads": 4,
        "device": "auto",
        "rtol": 1e-6,
        "atol": 1e-8,
        "train_per_class": 500,
        "test_per_class": 100,
    }


def make_e1_protocol(
    seed: int = 42,
    *,
    base_classes: int = 50,
    phase_size: int = 10,
    split_mode: str = "uniform",
    coarse_map: tuple[int, ...] = CANONICAL_FINE_TO_COARSE,
) -> dict:
    """Sinh giao thức phân chia 100 lớp CIFAR-100 cho thực nghiệm E1a và E1b.

    Ở chế độ ``uniform`` với ``base_classes=50, phase_size=10``, hàm sinh ra chính xác cùng
    ``class_order`` và danh sách lớp từng phase như ``src.cifar_data.make_protocol``.

    Tham số:
        seed: Hạt giống nguyên điều khiển bộ sinh số ngẫu nhiên ``default_rng(seed)``.
        base_classes: Số lượng lớp ở Phase 0 (``10, 20, 30, 50`` cho E1a; ``50`` cho E1b).
        phase_size: Số lượng lớp học mới ở mỗi phase tăng dần (mặc định ``10``).
        split_mode: Chế độ chia lớp gồm ``"uniform"``, ``"stratified"``,
            ``"superclass_disjoint"`` hoặc ``"superclass_disjoint_reverse"``.
        coarse_map: Tuple 100 phần tử ánh xạ từ ``fine_label`` sang ``coarse_label``.

    Trả về:
        Dict giao thức chứa ``class_order``, ``original_to_global``, ``phase_global_classes``,
        ``phase_original_classes`` và thống kê độ phủ superclass giữa Base và Incremental.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed phải là số nguyên")
    if isinstance(base_classes, bool) or not isinstance(base_classes, int) or not 1 <= base_classes < 100:
        raise ValueError("base_classes phải là số nguyên trong 1..99")
    if isinstance(phase_size, bool) or not isinstance(phase_size, int) or phase_size < 1:
        raise ValueError("phase_size phải là số nguyên dương")
    if (100 - base_classes) % phase_size != 0:
        raise ValueError("(100 - base_classes) phải chia hết cho phase_size")
    if len(coarse_map) != 100:
        raise ValueError("coarse_map phải có đúng 100 phần tử")

    rng = np.random.default_rng(int(seed))
    if split_mode == "uniform":
        order = rng.permutation(100).astype(np.int64)
    elif split_mode == "stratified":
        if base_classes != 50:
            raise ValueError("Chế độ stratified khóa cố định cho base_classes=50")
        coarse_perm = rng.permutation(20)
        base_pool: list[int] = []
        inc_pool: list[int] = []
        for rank, coarse_id in enumerate(coarse_perm.tolist()):
            members = np.array([fine for fine, cid in enumerate(coarse_map) if cid == coarse_id], dtype=np.int64)
            shuffled = rng.permutation(members).tolist()
            take_base = 3 if rank < 10 else 2
            base_pool.extend(shuffled[:take_base])
            inc_pool.extend(shuffled[take_base:])
        order = np.concatenate([
            rng.permutation(np.array(base_pool, dtype=np.int64)),
            rng.permutation(np.array(inc_pool, dtype=np.int64)),
        ])
    elif split_mode in ("superclass_disjoint", "superclass_disjoint_reverse"):
        if base_classes != 50:
            raise ValueError(f"Chế độ {split_mode} khóa cố định cho base_classes=50")
        base_coarse = (
            set(NATURAL_SUPERCLASSES)
            if split_mode == "superclass_disjoint"
            else set(SHIFTED_SUPERCLASSES)
        )
        base_pool = [fine for fine, cid in enumerate(coarse_map) if cid in base_coarse]
        inc_pool = [fine for fine, cid in enumerate(coarse_map) if cid not in base_coarse]
        if len(base_pool) != 50 or len(inc_pool) != 50:
            raise ValueError("Phân hoạch superclass_disjoint phải cho đúng 50 lớp mỗi phía")
        order = np.concatenate([
            rng.permutation(np.array(base_pool, dtype=np.int64)),
            rng.permutation(np.array(inc_pool, dtype=np.int64)),
        ])
    else:
        raise ValueError(f"split_mode không hợp lệ: {split_mode}")

    mapping = np.empty(100, dtype=np.int64)
    mapping[order] = np.arange(100, dtype=np.int64)
    inc_phases = (100 - base_classes) // phase_size
    phase_global = [list(range(base_classes))]
    for phase_idx in range(inc_phases):
        start = base_classes + phase_idx * phase_size
        phase_global.append(list(range(start, start + phase_size)))
    phase_original = [[int(order[g]) for g in group] for group in phase_global]

    base_superclasses = sorted({int(coarse_map[fine]) for fine in phase_original[0]})
    inc_superclasses = sorted(
        {int(coarse_map[fine]) for group in phase_original[1:] for fine in group}
    )
    shared_superclasses = sorted(set(base_superclasses).intersection(inc_superclasses))
    return {
        "protocol_version": "ACIL-E1-v1.0",
        "seed": int(seed),
        "split_mode": split_mode,
        "base_classes": base_classes,
        "phase_size": phase_size,
        "incremental_phases": inc_phases,
        "class_order": order.tolist(),
        "original_to_global": mapping.tolist(),
        "phase_global_classes": phase_global,
        "phase_original_classes": phase_original,
        "base_superclasses": base_superclasses,
        "base_superclass_names": [CANONICAL_COARSE_NAMES[c] for c in base_superclasses],
        "incremental_superclasses": inc_superclasses,
        "incremental_superclass_names": [CANONICAL_COARSE_NAMES[c] for c in inc_superclasses],
        "shared_superclass_count": len(shared_superclasses),
    }


def audit_e1_data(
    arrays: CifarArrays,
    protocol: dict,
    *,
    train_per_class: int = 500,
    test_per_class: int = 100,
) -> dict:
    """Kiểm tra toàn vẹn dữ liệu E1 và tính hằng số chuẩn hóa mean/std chỉ từ tập train Phase 0.

    Bảo đảm tuyệt đối không đọc ảnh từ các lớp tăng dần ``>= base_classes`` hay tập ``test``
    khi ước lượng ``mean`` và ``std`` tiền xử lý.

    Tham số:
        arrays: Đối tượng ``CifarArrays`` chứa dữ liệu ảnh uint8 NHWC và nhãn gốc.
        protocol: Dict giao thức chia lớp được tạo bởi ``make_e1_protocol``.
        train_per_class: Số mẫu train kỳ vọng của mỗi lớp (500 với CIFAR-100 chuẩn).
        test_per_class: Số mẫu test kỳ vọng của mỗi lớp (100 với CIFAR-100 chuẩn).

    Trả về:
        Dict kết quả kiểm toán gồm số mẫu từng phase, thông tin nguồn và thống kê ``normalization``.
    """
    base_classes = int(protocol["base_classes"])
    mapping = np.asarray(protocol["original_to_global"], dtype=np.int64)
    order = np.asarray(protocol["class_order"], dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(100)):
        raise ValueError("class_order phải là một hoán vị đầy đủ của 0..99")
    if not np.array_equal(mapping[order], np.arange(100)):
        raise ValueError("original_to_global không phải ánh xạ ngược của class_order")

    expected_train = 100 * train_per_class
    expected_test = 100 * test_per_class
    for images, labels, total_expected, per_class in (
        (arrays.train_images, arrays.train_labels, expected_train, train_per_class),
        (arrays.test_images, arrays.test_labels, expected_test, test_per_class),
    ):
        if images.shape != (total_expected, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError("Mảng ảnh không đúng kích thước (N, 32, 32, 3) uint8")
        if labels.shape != (total_expected,) or not np.array_equal(np.unique(labels), np.arange(100)):
            raise ValueError("Mảng nhãn thiếu lớp trong 0..99")
        if not np.all(np.bincount(labels, minlength=100) == per_class):
            raise ValueError("Số lượng mẫu mỗi lớp không đồng đều theo cấu hình")

    flat = [label for group in protocol["phase_global_classes"] for label in group]
    if sorted(flat) != list(range(100)) or len(flat) != len(set(flat)):
        raise ValueError("Các phase không rời nhau hoặc không phủ đủ 100 lớp")
    if protocol["phase_global_classes"][0] != list(range(base_classes)):
        raise ValueError("Phase 0 phải tương ứng với các nhãn toàn cục 0..base_classes-1")

    mapped_train = mapping[arrays.train_labels]
    mapped_test = mapping[arrays.test_labels]
    phases = []
    for index, group in enumerate(protocol["phase_global_classes"]):
        phases.append({
            "phase": index,
            "classes": group,
            "original_classes": protocol["phase_original_classes"][index],
            "train_samples": int(np.isin(mapped_train, group).sum()),
            "test_samples": int(np.isin(mapped_test, group).sum()),
        })

    base_indices = np.flatnonzero(mapped_train < base_classes)
    if len(base_indices) != base_classes * train_per_class:
        raise AssertionError("Số lượng mẫu train Phase 0 không khớp base_classes * train_per_class")
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    n_chunks = min(len(base_indices), max(1, base_classes))
    for chunk in np.array_split(base_indices, n_chunks):
        pixels = arrays.train_images[chunk].astype(np.float64) / 255.0
        sums += pixels.sum(axis=(0, 1, 2))
        squares += np.square(pixels).sum(axis=(0, 1, 2))
    pixel_count = len(base_indices) * 32 * 32
    mean = sums / pixel_count
    std = np.sqrt(np.maximum(squares / pixel_count - np.square(mean), 1e-12))
    return {
        "passed": True,
        "base_classes": base_classes,
        "split_mode": protocol["split_mode"],
        "train_samples": expected_train,
        "test_samples": expected_test,
        "image_shape": [32, 32, 3],
        "classes": 100,
        "train_samples_per_class": train_per_class,
        "test_samples_per_class": test_per_class,
        "phase_classes_disjoint": True,
        "label_mapping_bijective": True,
        "phases": phases,
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "source": "base_train_only",
            "base_classes": base_classes,
            "samples": int(len(base_indices)),
        },
        "source": arrays.source,
    }


def validate_e1_config(config: dict) -> None:
    """Kiểm tra tính hợp lệ của dict cấu hình E1 trước khi khởi tạo mạng và huấn luyện.

    Tham số:
        config: Dict cấu hình thực nghiệm E1.

    Trả về:
        None. Phát sinh ``ValueError`` nếu có trường cấu hình sai kiểu hoặc không nhất quán.
    """
    if config.get("branch") not in ("E1a", "E1b"):
        raise ValueError("branch phải là 'E1a' hoặc 'E1b'")
    for key in (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "analytic_batch_size",
        "base_statistics_batch_size",
        "torch_threads",
        "expansion_dim",
        "base_classes",
        "phase_size",
        "incremental_phases",
        "run_incremental_phases",
        "train_per_class",
        "test_per_class",
    ):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} phải là số nguyên dương")
    base_classes = config["base_classes"]
    phase_size = config["phase_size"]
    if not 1 <= base_classes < 100 or (100 - base_classes) % phase_size != 0:
        raise ValueError("base_classes và phase_size không chia chẵn 100 lớp")
    expected_k = (100 - base_classes) // phase_size
    if config["incremental_phases"] != expected_k:
        raise ValueError("incremental_phases không khớp (100 - base_classes) // phase_size")
    if not 1 <= config["run_incremental_phases"] <= expected_k:
        raise ValueError("run_incremental_phases nằm ngoài 1..incremental_phases")
    if config.get("num_workers", 0) != 0:
        raise ValueError("Pipeline yêu cầu num_workers=0 để đảm bảo tính tất định")
    for key in ("gamma", "rtol", "atol"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} phải hữu hạn và dương")
    if not isinstance(config.get("feature_expansion", True), bool):
        raise ValueError("feature_expansion phải là kiểu bool")


def validate_e1_checkpoint(
    checkpoint: dict,
    config: dict,
    protocol: dict,
    normalization: dict,
    dataset_revision: str,
) -> None:
    """Kiểm tra tính khớp ngữ cảnh của checkpoint ResNet-32 Phase 0 khi tái sử dụng trong E1.

    Tham số:
        checkpoint: Dict checkpoint đã nạp từ tệp ``base_checkpoint.pt``.
        config: Dict cấu hình lượt chạy hiện tại.
        protocol: Dict giao thức chia lớp E1 hiện tại.
        normalization: Dict thống kê chuẩn hóa ``mean``/``std`` của Phase 0 hiện tại.
        dataset_revision: Chuỗi mã revision của tập dữ liệu đang sử dụng.

    Trả về:
        None. Phát sinh ``ValueError`` nếu checkpoint chưa hoàn tất hoặc lệch cấu hình/protocol.
    """
    if checkpoint.get("checkpoint_format_version") == 2 and checkpoint.get("training_complete") is not True:
        raise ValueError("Checkpoint Phase 0 chưa hoàn tất; cần dùng resume thay vì reuse")
    expected_samples = config["base_classes"] * config["train_per_class"]
    if (
        checkpoint.get("checkpoint_format_version") == 2
        and checkpoint.get("loader_signature", {}).get("dataset_size") != expected_samples
    ):
        raise ValueError(f"Checkpoint không khớp số mẫu Phase 0 kỳ vọng ({expected_samples})")
    for key in ("seed", "epochs", "learning_rate", "momentum", "weight_decay", "batch_size", "lr_milestones"):
        if checkpoint["config"].get(key) != config.get(key):
            raise ValueError(f"Checkpoint không khớp tham số huấn luyện {key}")
    if checkpoint.get("class_order") != protocol["class_order"]:
        raise ValueError("Checkpoint không khớp class_order của giao thức E1")
    if checkpoint.get("normalization") != normalization:
        raise ValueError("Checkpoint không khớp thống kê chuẩn hóa base_train_only")
    if checkpoint.get("dataset_revision") != dataset_revision:
        raise ValueError("Checkpoint không khớp dataset_revision")
    history = checkpoint.get("training_history", [])
    if len(history) != config["epochs"] or not all(np.isfinite(row["train_loss"]) for row in history):
        raise ValueError("Checkpoint thiếu lịch sử loss hợp lệ đủ số epochs")
    if not np.isfinite(checkpoint.get("parameter_change_l2", 0.0)) or checkpoint["parameter_change_l2"] <= 0:
        raise ValueError("Checkpoint không có bằng chứng cập nhật trọng số dương")


def evaluate_e1_phase(
    head: StreamingHead,
    batches,
    *,
    n_classes: int,
    base_classes: int,
    current_phase_classes: list[int] | range,
    class_order: list[int],
    coarse_map: tuple[int, ...] = CANONICAL_FINE_TO_COARSE,
    expected_per_class: int | None = 100,
) -> dict:
    """Đánh giá mô hình ACIL trên mọi lớp đã thấy không dùng task ID, tách riêng Base và Incremental.

    Ngoài độ chính xác toàn cục và lớp Base (``labels < base_classes``), hàm tính thêm độ chính xác
    trên toàn bộ các lớp tăng dần đã thấy (``labels >= base_classes``) và phân rã chẩn đoán giữa
    nhóm đồ vật nhân tạo thuần túy (``PURE_MANMADE_SUPERCLASSES``) với nhóm sinh học/con người.

    Tham số:
        head: Đối tượng ``StreamingHead`` chứa trạng thái ``W`` hiện tại.
        batches: Iterator sinh các cặp ``(F_batch, labels_batch)`` của tập test đã thấy.
        n_classes: Tổng số lớp đã thấy tính đến hết phase hiện tại.
        base_classes: Số lượng lớp gốc của Phase 0 (``c_0``).
        current_phase_classes: Tập nhãn toàn cục của riêng phase vừa cập nhật.
        class_order: Danh sách ánh xạ nhãn toàn cục ``g`` về ``fine_label`` gốc ``class_order[g]``.
        coarse_map: Tuple 100 phần tử ánh xạ từ ``fine_label`` gốc sang ``coarse_label``.
        expected_per_class: Số mẫu test kỳ vọng trên mỗi lớp đã thấy (hoặc ``None``).

    Trả về:
        Dict các chỉ số đánh giá của phase hiện tại kèm kiểm tra phủ đủ các lớp đã thấy.
    """
    if head.w is None or head.w.shape != (head.dimension, n_classes):
        raise ValueError("Kích thước ma trận W không khớp số lớp cần đánh giá")
    order_arr = np.asarray(class_order, dtype=np.int64)
    coarse_arr = np.asarray(coarse_map, dtype=np.int64)
    manmade_set = np.array(PURE_MANMADE_SUPERCLASSES, dtype=np.int64)
    current_arr = np.asarray(list(current_phase_classes), dtype=np.int64)

    counts = np.zeros(n_classes, dtype=np.int64)
    total = correct = 0
    base_total = base_correct = 0
    inc_total = inc_correct = 0
    new_total = new_correct = 0
    inc_manmade_total = inc_manmade_correct = 0
    inc_bio_total = inc_bio_correct = 0

    for raw, labels in batches:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= n_classes):
            raise ValueError("Nhãn test nằm ngoài phạm vi các lớp đã thấy")
        scores = head.transform(raw) @ head.w
        if len(scores) != len(labels) or not np.isfinite(scores).all():
            raise FloatingPointError("Điểm dự đoán test sai kích thước hoặc không hữu hạn")
        matched = scores.argmax(axis=1) == labels
        base_mask = labels < base_classes
        inc_mask = labels >= base_classes
        new_mask = np.isin(labels, current_arr)

        counts += np.bincount(labels, minlength=n_classes)
        total += int(len(labels))
        correct += int(matched.sum())
        base_total += int(base_mask.sum())
        base_correct += int(matched[base_mask].sum())
        inc_total += int(inc_mask.sum())
        inc_correct += int(matched[inc_mask].sum())
        new_total += int(new_mask.sum())
        new_correct += int(matched[new_mask].sum())

        if np.any(inc_mask):
            inc_labels = labels[inc_mask]
            inc_matched = matched[inc_mask]
            inc_coarse = coarse_arr[order_arr[inc_labels]]
            is_manmade = np.isin(inc_coarse, manmade_set)
            inc_manmade_total += int(is_manmade.sum())
            inc_manmade_correct += int(inc_matched[is_manmade].sum())
            inc_bio_total += int((~is_manmade).sum())
            inc_bio_correct += int(inc_matched[~is_manmade].sum())

    if np.any(counts == 0) or (expected_per_class is not None and np.any(counts != expected_per_class)):
        raise AssertionError("Tập test đánh giá không phủ đúng số mẫu kỳ vọng trên mọi lớp đã thấy")

    acc_pct = 100.0 * correct / total
    base_acc_pct = 100.0 * base_correct / base_total
    is_base_phase = n_classes == base_classes
    inc_acc_pct = None if is_base_phase else (100.0 * inc_correct / inc_total)
    new_acc_pct = None if is_base_phase else (100.0 * new_correct / new_total)
    gap_pp = None if inc_acc_pct is None else (base_acc_pct - inc_acc_pct)
    manmade_acc_pct = (
        None if is_base_phase or inc_manmade_total == 0 else (100.0 * inc_manmade_correct / inc_manmade_total)
    )
    bio_acc_pct = (
        None if is_base_phase or inc_bio_total == 0 else (100.0 * inc_bio_correct / inc_bio_total)
    )
    return {
        "test_samples_all_seen": total,
        "accuracy_pct": float(acc_pct),
        "base_accuracy_pct": float(base_acc_pct),
        "inc_all_accuracy_pct": None if inc_acc_pct is None else float(inc_acc_pct),
        "new_class_accuracy_pct": None if new_acc_pct is None else float(new_acc_pct),
        "base_minus_inc_gap_pp": None if gap_pp is None else float(gap_pp),
        "inc_manmade_accuracy_pct": None if manmade_acc_pct is None else float(manmade_acc_pct),
        "inc_biological_accuracy_pct": None if bio_acc_pct is None else float(bio_acc_pct),
        "inc_manmade_test_samples": 0 if is_base_phase else inc_manmade_total,
        "inc_biological_test_samples": 0 if is_base_phase else inc_bio_total,
    }


def summarize_e1_metrics(rows: list[dict], config: dict, protocol: dict) -> dict:
    """Tổng hợp bộ chỉ số kết quả cuối kỳ cho một lượt chạy E1 từ danh sách kết quả từng phase.

    Tham số:
        rows: Danh sách dict kết quả theo thứ tự liên tiếp từ Phase 0 đến Phase cuối.
        config: Dict cấu hình của lượt chạy E1.
        protocol: Dict giao thức chia lớp E1 tương ứng.

    Trả về:
        Dict tóm tắt các thước đo chính (``final_accuracy_pct``, ``final_inc_accuracy_pct``,
        ``final_base_minus_inc_gap_pp``, ``average_incremental_accuracy_pct``, ``base_forgetting_pp``...).
    """
    if not rows or [row["phase"] for row in rows] != list(range(len(rows))):
        raise ValueError("Danh sách kết quả phải bắt đầu từ phase 0 và liên tiếp")
    total_k = config["incremental_phases"]
    requested_k = config["run_incremental_phases"]
    base_classes = config["base_classes"]
    phase_size = config["phase_size"]
    per_class = config["test_per_class"]
    if len(rows) != requested_k + 1:
        raise ValueError("Số dòng kết quả không khớp run_incremental_phases + 1")
    for phase, row in enumerate(rows):
        expected_classes = base_classes + phase * phase_size
        if row["n_classes"] != expected_classes or row["test_samples_all_seen"] != expected_classes * per_class:
            raise ValueError(f"Phase {phase} sai số lượng lớp hoặc số mẫu test")
        if row.get("checks_passed") is not True:
            raise ValueError(f"Phase {phase} chưa vượt qua kiểm tra toàn vẹn")
    complete = requested_k == total_k
    average = float(np.mean([row["accuracy_pct"] for row in rows]))
    final_row = rows[-1]
    base_delta = float(final_row["base_accuracy_pct"] - rows[0]["base_accuracy_pct"])
    return {
        "protocol_version": protocol["protocol_version"],
        "metric_version": "ACIL-E1-metrics-v1.0",
        "branch": config["branch"],
        "split_mode": config["split_mode"],
        "seed": config["seed"],
        "base_classes": base_classes,
        "phase_size": phase_size,
        "n_incremental_phases": total_k,
        "n_evaluations": len(rows),
        "complete_protocol": complete,
        "shared_superclass_count": protocol["shared_superclass_count"],
        "average_incremental_accuracy_pct": average if complete else None,
        "partial_average_accuracy_pct": None if complete else average,
        "final_accuracy_pct": final_row["accuracy_pct"],
        "final_n_classes": final_row["n_classes"],
        "initial_base_accuracy_pct": rows[0]["base_accuracy_pct"],
        "final_base_accuracy_pct": final_row["base_accuracy_pct"],
        "final_inc_accuracy_pct": final_row["inc_all_accuracy_pct"],
        "final_new_phase_accuracy_pct": final_row["new_class_accuracy_pct"],
        "final_base_minus_inc_gap_pp": final_row["base_minus_inc_gap_pp"],
        "final_inc_manmade_accuracy_pct": final_row["inc_manmade_accuracy_pct"],
        "final_inc_biological_accuracy_pct": final_row["inc_biological_accuracy_pct"],
        "base_forgetting_pp": -base_delta,
        "base_accuracy_delta_pp": base_delta,
    }


def log_e1_wandb(
    report: dict,
    output_path: Path,
    *,
    wandb_mode: str = "disabled",
    entity: str = "hsangg-2006",
    project: str = "acil-reproduction",
    wandb_dir: Path | None = None,
    wandb_client=None,
) -> dict:
    """Ghi nhận kết quả E1 lên W&B với bộ nhãn tách biệt hoàn toàn khỏi giữa kỳ rồi lưu JSON.

    Tham số:
        report: Dict báo cáo kết quả hoàn chỉnh của một lượt chạy E1.
        output_path: Đường dẫn tệp JSON đích cần lưu.
        wandb_mode: Chế độ ghi W&B gồm ``"disabled"``, ``"offline"`` hoặc ``"online"``.
        entity: Tên tài khoản/nhóm trên W&B.
        project: Tên project W&B công khai (``"acil-reproduction"``).
        wandb_dir: Thư mục lưu đệm các tệp W&B local/offline (tùy chọn).
        wandb_client: Đối tượng giả lập SDK ``wandb`` dùng trong kiểm thử đơn vị (tùy chọn).

    Trả về:
        Bản sao dict báo cáo đã đính kèm trạng thái đồng bộ W&B và lưu ra ``output_path``.
    """
    if wandb_mode not in ("disabled", "offline", "online"):
        raise ValueError("wandb_mode phải thuộc {'disabled', 'offline', 'online'}")
    result = copy.deepcopy(report)
    result.update(wandb_status="not_logged", wandb_run_id=None, wandb_run_url=None)
    if wandb_mode != "disabled":
        cfg = result["config"]
        branch = cfg["branch"]
        branch_tag = "E1a-base-ratio" if branch == "E1a" else "E1b-superclass-shift"
        tags = [
            "final-empirics",
            "P1-base-assumption",
            "P1-base-shift",
            branch_tag,
            cfg["split_mode"],
            f"c0_{cfg['base_classes']}",
            f"seed{cfg['seed']}",
        ]
        run_name = f"{branch}-{cfg['split_mode']}-c0_{cfg['base_classes']}-seed{cfg['seed']}"
        try:
            if wandb_client is None:
                import wandb as wandb_client
            with wandb_client.init(
                entity=entity,
                project=project,
                name=run_name,
                job_type="final-empirics-e1",
                mode=wandb_mode,
                dir=str(wandb_dir or output_path.parent),
                tags=tags,
                config={
                    **cfg,
                    "source_commit": result.get("provenance", {}).get("git_commit"),
                    "source_sha256": result.get("provenance", {}).get("source_sha256", {}),
                    "runner_script": "scripts/e1_base_assumption_runner.py",
                },
            ) as run:
                for row in result["results"]:
                    metrics = {k: v for k, v in row.items() if isinstance(v, (int, float, bool))}
                    run.log(metrics, step=row["phase"])
                run.summary.update(result["metric_summary"])
                run.summary["verification_passed"] = result["verification_passed"]
                run_id = getattr(run, "id", None)
                run_url = getattr(run, "url", None)
            if wandb_mode == "online" and (not run_id or not run_url):
                raise RuntimeError("W&B chế độ online không trả về ID/URL hợp lệ")
            result.update(
                wandb_status="logged" if wandb_mode == "online" else "offline_pending_sync",
                wandb_run_id=run_id,
                wandb_run_url=run_url,
                wandb_tags=tags,
            )
        except Exception as error:
            result.update(
                wandb_status="failed",
                wandb_run_id=None,
                wandb_run_url=None,
                tracking_error_type=type(error).__name__,
            )
            save_json(result, output_path)
            raise
    save_json(result, output_path)
    return result


class _PhaseBatchFactory:
    """Bộ tạo luồng batch đặc trưng CNN cho từng phase train hoặc tập test đã thấy trong E1."""

    def __init__(self, model, arrays, protocol, normalization, config, device, timings):
        """Khởi tạo bộ cung cấp iterator đặc trưng cho ``StreamingHead``.

        Tham số:
            model: Mạng ResNet-32 đã đóng băng trọng số và khóa BatchNorm ở ``eval()``.
            arrays: Đối tượng ``CifarArrays`` chứa dữ liệu ảnh/nhãn.
            protocol: Dict giao thức chia lớp E1.
            normalization: Dict thống kê chuẩn hóa ``mean`` và ``std`` của Phase 0.
            config: Dict cấu hình lượt chạy E1.
            device: Thiết bị PyTorch chạy trích xuất đặc trưng.
            timings: Dict ghi nhận tổng thời gian trích xuất đặc trưng CNN.

        Trả về:
            None.
        """
        self.model = model
        self.arrays = arrays
        self.protocol = protocol
        self.normalization = normalization
        self.config = config
        self.device = device
        self.timings = timings
        self._cached_test: tuple[np.ndarray, np.ndarray] | None = None

    def train_batches(self, phase: int):
        """Tạo iterator sinh các batch đặc trưng 64 chiều của tập train tại ``phase``.

        Tham số:
            phase: Chỉ số nguyên của phase cần đọc (``0`` cho Base, ``1..K`` cho Incremental).

        Trả về:
            Generator trả về từng cặp ``(F64_batch, labels_batch)``.
        """
        batch_size = (
            self.config["base_statistics_batch_size"]
            if phase == 0
            else self.config["analytic_batch_size"]
        )
        classes = self.protocol["phase_global_classes"][phase]
        loader = make_loader(
            self.arrays, self.protocol, self.normalization, classes, "train", self.config, batch_size=batch_size
        )
        return iter_feature_batches(
            self.model, loader, self.device, f"E1 train phase {phase}", timings=self.timings
        )

    def test_batches(self, n_classes: int):
        """Tạo iterator sinh các batch đặc trưng 64 chiều của toàn bộ các lớp test đã thấy.

        Đặc trưng của tập test trên backbone đã đóng băng được trích xuất một lần duy nhất
        (không giữ bất kỳ mẫu huấn luyện nào) để tránh lặp lại forward CNN qua các mốc đánh giá.

        Tham số:
            n_classes: Tổng số lớp đã thấy ``0..n_classes-1``.

        Trả về:
            Generator trả về từng cặp ``(F64_batch, labels_batch)``.
        """
        if self._cached_test is None:
            full_loader = make_loader(
                self.arrays, self.protocol, self.normalization, range(100), "test", self.config
            )
            chunks = list(
                iter_feature_batches(
                    self.model, full_loader, self.device, "E1 test all100", timings=self.timings
                )
            )
            self._cached_test = (
                np.concatenate([x for x, _ in chunks], axis=0),
                np.concatenate([y for _, y in chunks], axis=0),
            )
        all_test_x, all_test_y = self._cached_test
        seen_mask = all_test_y < n_classes
        seen_x = all_test_x[seen_mask]
        seen_y = all_test_y[seen_mask]
        step = self.config["eval_batch_size"]
        for offset in range(0, len(seen_y), step):
            yield seen_x[offset : offset + step], seen_y[offset : offset + step]


def run_e1_case(
    config: dict | None = None,
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    config_overrides: dict | None = None,
    arrays_override: CifarArrays | None = None,
    reuse_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    stop_after_epoch: int | None = None,
    verify_joint_oracle: bool = False,
    save_large_r_npz: bool = True,
    wandb_mode: str = "disabled",
    entity: str = "hsangg-2006",
    project: str = "acil-reproduction",
    wandb_client=None,
) -> dict:
    """Thực thi một lượt thực nghiệm E1 trọn vẹn trong thư mục cách ly của ``RunAttempt``.

    Hỗ trợ cả huấn luyện ResNet-32 đầu ra ``c_0`` lớp ở Phase 0, cơ chế resume an toàn tại biên
    epoch, đóng băng mạng, học giải tích streaming và đối chiếu nghiệm Joint Ridge tùy chọn.

    Tham số:
        config: Dict cấu hình cơ sở (nếu ``None`` sẽ khởi tạo từ ``default_e1_config``).
        data_dir: Thư mục cache dữ liệu CIFAR-100.
        output_dir: Thư mục gốc ghi nhận ``attempts/<id>/`` và ``results.json``.
        config_overrides: Dict ghi đè các trường cấu hình cụ thể (tùy chọn).
        arrays_override: Đối tượng ``CifarArrays`` truyền sẵn phục vụ kiểm thử đơn vị nhanh (tùy chọn).
        reuse_checkpoint: Đường dẫn checkpoint Phase 0 đã hoàn tất để tái sử dụng (tùy chọn).
        resume_checkpoint: Đường dẫn checkpoint v2 đang dở để tiếp tục huấn luyện (tùy chọn).
        stop_after_epoch: Mốc epoch dừng sớm có chủ đích phục vụ kiểm tra resume (tùy chọn).
        verify_joint_oracle: Nếu ``True``, giải thêm nghiệm Joint Ridge tại từng phase để xác nhận
            Định lý 3.1 vẫn giữ nguyên ngay cả khi biểu diễn đóng băng suy giảm.
        save_large_r_npz: Nếu ``True`` lưu toàn bộ ``inverse_gram`` vào ``analytic_state.npz``;
            nếu ``False`` kiểm tra đối xứng và băm SHA-256 của ``R`` trong RAM, chỉ ghi ``weights``
            để tiết kiệm băng thông tải lên của phiên nhiều cấu hình trên Kaggle.
        wandb_mode: Chế độ đồng bộ W&B (``"disabled"``, ``"offline"`` hoặc ``"online"``).
        entity: Tên entity W&B.
        project: Tên project W&B.
        wandb_client: Client W&B tùy chọn cho kiểm thử.

    Trả về:
        Dict báo cáo kết quả hoàn chỉnh với ``status="complete"`` và ``verification_passed=True``.
    """
    merged = {**(config or default_e1_config()), **(config_overrides or {})}
    with RunAttempt(output_dir, seed=merged.get("seed")) as attempt:
        attempt.progress("validating")
        if reuse_checkpoint is not None and resume_checkpoint is not None:
            raise ValueError("Chỉ được chọn một trong hai: reuse_checkpoint hoặc resume_checkpoint")
        validate_e1_config(merged)
        save_json(merged, attempt.directory / "config.json")
        seed_everything(merged["seed"], merged["torch_threads"])

        device_name = merged["device"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Đã yêu cầu thiết bị CUDA nhưng hệ thống không có GPU khả dụng")
        device = torch.device(device_name)

        provenance = collect_provenance(ROOT)
        runner_path = Path(__file__).resolve()
        provenance["runner_script"] = runner_path.relative_to(ROOT).as_posix()
        provenance["runner_sha256"] = sha256_file(runner_path)
        provenance["effective_config_sha256"] = hashlib.sha256(
            json.dumps(merged, sort_keys=True).encode("utf-8")
        ).hexdigest()

        started = time.perf_counter()
        with threadpool_limits(limits=merged["torch_threads"]), MemoryMonitor(device) as memory_probe:
            attempt.progress("data_and_protocol")
            coarse_map = verify_or_load_coarse_map(None if arrays_override is not None else data_dir)
            arrays = arrays_override if arrays_override is not None else load_cifar100(data_dir)
            protocol = make_e1_protocol(
                merged["seed"],
                base_classes=merged["base_classes"],
                phase_size=merged["phase_size"],
                split_mode=merged["split_mode"],
                coarse_map=coarse_map,
            )
            audit = audit_e1_data(
                arrays,
                protocol,
                train_per_class=merged["train_per_class"],
                test_per_class=merged["test_per_class"],
            )
            normalization = audit["normalization"]
            save_json(protocol, attempt.directory / "protocol.json")
            save_json(audit, attempt.directory / "data_audit.json")

            base_classes = merged["base_classes"]
            model = ResNet32(n_classes=base_classes).to(device)
            checkpoint_path = attempt.directory / "base_checkpoint.pt"
            origin = {"reused": reuse_checkpoint is not None, "resumed": resume_checkpoint is not None}

            if reuse_checkpoint is not None:
                attempt.progress("checkpoint_load")
                reused_path = Path(reuse_checkpoint).resolve()
                saved = torch.load(reused_path, map_location="cpu", weights_only=True)
                validate_e1_checkpoint(saved, merged, protocol, normalization, arrays.source["revision"])
                model.load_state_dict(saved["model_state_dict"])
                history = saved["training_history"]
                change = float(saved["parameter_change_l2"])
                origin.update(input_sha256=sha256_file(reused_path), training_executed_this_run=False)
                if reused_path != checkpoint_path:
                    shutil.copy2(reused_path, checkpoint_path)
            else:
                attempt.progress("base_training", base_classes=base_classes)
                base_loader = make_loader(
                    arrays, protocol, normalization, range(base_classes), "train", merged, augment=True
                )
                if resume_checkpoint is not None:
                    origin["resume_input_sha256"] = sha256_file(Path(resume_checkpoint))
                history, change = train_base(
                    model,
                    base_loader,
                    merged,
                    device,
                    attempt.directory,
                    progress=attempt.progress,
                    checkpoint_context={
                        "class_order": protocol["class_order"],
                        "normalization": normalization,
                        "dataset_revision": arrays.source["revision"],
                    },
                    resume_checkpoint=resume_checkpoint,
                    stop_after_epoch=stop_after_epoch,
                )
                origin["training_executed_this_run"] = True

            attempt.progress("checkpoint_reload_verify")
            model.eval()
            probe_loader = make_loader(arrays, protocol, normalization, range(base_classes), "test", merged)
            probe_images, _ = next(iter(probe_loader))
            with torch.inference_mode():
                logits_before = model(probe_images.to(device))
            restored = ResNet32(n_classes=base_classes).to(device)
            restored.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state_dict"]
            )
            restored.eval()
            with torch.inference_mode():
                logits_after = restored(probe_images.to(device))
            torch.testing.assert_close(logits_before, logits_after, rtol=0, atol=0)
            reload_error = float((logits_before - logits_after).abs().max().item())

            model = freeze_backbone(restored)
            backbone_before = fingerprint(model)
            stage_timings = {"feature_extraction_seconds": 0.0, "incremental_analytic_seconds": 0.0}
            factory = _PhaseBatchFactory(
                model, arrays, protocol, normalization, merged, device, stage_timings
            )
            head = StreamingHead(merged)
            expansion_sha_before = (
                hashlib.sha256(head.expansion.tobytes()).hexdigest()
                if head.expansion is not None
                else None
            )

            oracle_h_chunks: list[np.ndarray] = []
            oracle_y_chunks: list[np.ndarray] = []
            rows: list[dict] = []
            analytic_wall_start = time.perf_counter()

            for phase in range(merged["run_incremental_phases"] + 1):
                attempt.progress("analytic_phase", phase=phase)
                classes = protocol["phase_global_classes"][phase]
                n_classes = max(classes) + 1
                if verify_joint_oracle:
                    phase_pairs = list(factory.train_batches(phase))
                    for raw_chunk, y_chunk in phase_pairs:
                        oracle_h_chunks.append(head.transform(raw_chunk))
                        oracle_y_chunks.append(np.asarray(y_chunk, dtype=np.int64))
                    phase_iter = iter(phase_pairs)
                else:
                    phase_iter = factory.train_batches(phase)

                if phase == 0:
                    seen_train = head.fit_base(phase_iter, n_classes=n_classes)
                    n_batches = None
                else:
                    seen_train, n_batches = head.update_phase(phase_iter, n_classes=n_classes)
                    stage_timings["incremental_analytic_seconds"] += head.timings["last_incremental_seconds"]

                if seen_train != len(classes) * merged["train_per_class"]:
                    raise AssertionError(f"Phase {phase} xử lý sai số lượng mẫu train")

                evaluation = evaluate_e1_phase(
                    head,
                    factory.test_batches(n_classes),
                    n_classes=n_classes,
                    base_classes=base_classes,
                    current_phase_classes=classes,
                    class_order=protocol["class_order"],
                    coarse_map=coarse_map,
                    expected_per_class=merged["test_per_class"],
                )

                rel_w_err = max_abs_r_err = max_abs_score_err = disagreements = None
                if verify_joint_oracle:
                    pooled_h = np.vstack(oracle_h_chunks)
                    pooled_y = np.concatenate(oracle_y_chunks)
                    w_ref = joint_ridge(pooled_h, pooled_y, n_classes=n_classes, gamma=merged["gamma"])
                    eye = np.eye(head.dimension, dtype=np.float64)
                    r_ref = np.linalg.solve(pooled_h.T @ pooled_h + merged["gamma"] * eye, eye)
                    np.testing.assert_allclose(head.w, w_ref, rtol=merged["rtol"], atol=merged["atol"])
                    np.testing.assert_allclose(head.r, r_ref, rtol=merged["rtol"], atol=merged["atol"])
                    test_pairs = list(factory.test_batches(n_classes))
                    test_h = np.vstack([head.transform(raw_b) for raw_b, _ in test_pairs])
                    scores_acil = test_h @ head.w
                    scores_ref = test_h @ w_ref
                    disagreements = int(np.count_nonzero(scores_acil.argmax(1) != scores_ref.argmax(1)))
                    if disagreements != 0:
                        raise AssertionError("Nghiệm đệ quy và nghiệm Joint Ridge bất đồng nhãn dự đoán")
                    rel_w_err = float(np.linalg.norm(head.w - w_ref) / np.linalg.norm(w_ref))
                    max_abs_r_err = float(np.max(np.abs(head.r - r_ref)))
                    max_abs_score_err = float(np.max(np.abs(scores_acil - scores_ref)))

                row = {
                    "phase": phase,
                    "n_classes": n_classes,
                    "weights_shape": list(head.w.shape),
                    "inverse_gram_shape": list(head.r.shape),
                    "train_samples_current_phase": int(seen_train),
                    "analytic_batches": n_batches,
                    "joint_reference_status": "checked" if verify_joint_oracle else "not_run",
                    "relative_w_error": rel_w_err,
                    "max_abs_r_error": max_abs_r_err,
                    "max_abs_score_error": max_abs_score_err,
                    "prediction_disagreements": disagreements,
                    "checks_passed": True,
                    **evaluation,
                }
                rows.append(row)

            stage_timings.update(head.timings)
            stage_timings["streaming_phases_wall_seconds"] = time.perf_counter() - analytic_wall_start
            backbone_after = fingerprint(model)
            expansion_sha_after = (
                hashlib.sha256(head.expansion.tobytes()).hexdigest()
                if head.expansion is not None
                else None
            )
            if backbone_before != backbone_after or expansion_sha_before != expansion_sha_after:
                raise AssertionError("Trọng số ResNet-32/BatchNorm hoặc ma trận FE bị biến đổi sau Phase 0")
            if not np.isfinite(head.r).all() or not np.isfinite(head.w).all():
                raise FloatingPointError("Trạng thái analytic (R, W) chứa giá trị không hữu hạn")

            for start in range(0, head.dimension, 256):
                end = min(start + 256, head.dimension)
                if not np.allclose(
                    head.r[start:end], head.r[:, start:end].T, rtol=merged["rtol"], atol=merged["atol"]
                ):
                    raise AssertionError("Ma trận R sau cập nhật mất tính đối xứng")
            provenance["inverse_gram_sha256"] = hashlib.sha256(head.r.tobytes()).hexdigest()
            provenance["weights_sha256"] = hashlib.sha256(head.w.tobytes()).hexdigest()

            state_payload = {
                "weights": head.w,
                "gamma": merged["gamma"],
                "class_order": np.asarray(protocol["class_order"], dtype=np.int64),
                "base_classes": base_classes,
                "n_classes": head.w.shape[1],
            }
            if save_large_r_npz:
                state_payload["inverse_gram"] = head.r
                if head.expansion is not None:
                    state_payload["expansion_matrix"] = head.expansion
            np.savez_compressed(attempt.directory / "analytic_state.npz", **state_payload)

            expected_final_cols = base_classes + merged["run_incremental_phases"] * merged["phase_size"]
            checks = {
                "data_audit_passed": bool(audit["passed"]),
                "base_only_normalization_samples_exact": (
                    normalization["samples"] == base_classes * merged["train_per_class"]
                ),
                "checkpoint_reload_exact": reload_error == 0.0,
                "finite_training_loss": all(np.isfinite(item["train_loss"]) for item in history),
                "base_parameters_updated": change > 0.0,
                "backbone_batchnorm_and_expansion_frozen": True,
                "finite_analytic_state": True,
                "all_seen_test_classes_evaluated": True,
                "final_columns_correct": head.w.shape[1] == expected_final_cols,
            }
            if verify_joint_oracle:
                checks["theorem_3_1_joint_oracle_matched"] = all(
                    row["prediction_disagreements"] == 0 for row in rows
                )

            provenance["base_checkpoint_sha256"] = sha256_file(checkpoint_path)

        report = {
            "experiment": merged["experiment"],
            "description": (
                f"Final-term E1 ({merged['branch']} / {merged['split_mode']}, c0={base_classes}) "
                "testing frozen base representation assumption P1"
            ),
            "config": merged,
            "protocol_summary": {
                "split_mode": protocol["split_mode"],
                "base_classes": base_classes,
                "phase_size": protocol["phase_size"],
                "incremental_phases": protocol["incremental_phases"],
                "base_superclasses": protocol["base_superclasses"],
                "incremental_superclasses": protocol["incremental_superclasses"],
                "shared_superclass_count": protocol["shared_superclass_count"],
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "device": str(device),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("scipy", "pyarrow", "psutil", "threadpoolctl")
                },
            },
            "provenance": provenance,
            "checkpoint_origin": origin,
            "dataset_source": arrays.source,
            "training_history": history,
            "stem_weight_change_l2": change,
            "checkpoint_reload_max_logit_error": reload_error,
            "backbone_sha256_before": backbone_before,
            "backbone_sha256_after": backbone_after,
            "expansion_sha256": expansion_sha_before,
            "streaming": {
                "stored_training_samples": 0,
                "maximum_feature_batch_rows": head.maximum_feature_batch_rows,
                "persistent_analytic_state_bytes": head.state_bytes(),
            },
            "stage_timings": stage_timings,
            "memory": memory_probe.result,
            "results": rows,
            "metric_summary": summarize_e1_metrics(rows, merged, protocol),
            "checks": checks,
            "verification_passed": all(checks.values()),
            "attempt_id": attempt.id,
            "seconds_this_run": time.perf_counter() - started,
        }

        attempt.progress("tracking")
        logged = log_e1_wandb(
            report,
            attempt.directory / "results.json",
            wandb_mode=wandb_mode,
            entity=entity,
            project=project,
            wandb_dir=Path(output_dir),
            wandb_client=wandb_client,
        )
        completed = attempt.complete(logged)
        return completed


def aggregate_e1_reports(report_paths: list[str | Path], output_path: str | Path | None = None) -> dict:
    """Tổng hợp thống kê mean ± sample SD (ddof=1) từ danh sách các tệp kết quả E1 theo nhóm cấu hình.

    Tham số:
        report_paths: Danh sách đường dẫn tới các tệp ``results.json`` hoàn tất của E1.
        output_path: Đường dẫn tệp JSON lưu bản tổng hợp (tùy chọn).

    Trả về:
        Dict tổng hợp kết quả theo từng nhóm ``(branch, split_mode, base_classes)``.
    """
    grouped: dict[str, list[dict]] = {}
    for raw_path in report_paths:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or not payload.get("verification_passed"):
            raise ValueError(f"Báo cáo E1 chưa hoàn tất hợp lệ: {raw_path}")
        summary = payload["metric_summary"]
        key = f"{summary['branch']}__{summary['split_mode']}__c0_{summary['base_classes']}"
        grouped.setdefault(key, []).append(payload)

    metric_fields = (
        "average_incremental_accuracy_pct",
        "final_accuracy_pct",
        "initial_base_accuracy_pct",
        "final_base_accuracy_pct",
        "final_inc_accuracy_pct",
        "final_new_phase_accuracy_pct",
        "final_base_minus_inc_gap_pp",
        "final_inc_manmade_accuracy_pct",
        "final_inc_biological_accuracy_pct",
        "base_forgetting_pp",
    )
    groups_out = {}
    for group_key, items in sorted(grouped.items()):
        items_sorted = sorted(items, key=lambda item: item["config"]["seed"])
        seeds = [item["config"]["seed"] for item in items_sorted]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Nhóm {group_key} bị trùng lặp seed: {seeds}")
        stats = {}
        for field in metric_fields:
            by_seed = {
                str(item["config"]["seed"]): item["metric_summary"].get(field)
                for item in items_sorted
            }
            valid_vals = [val for val in by_seed.values() if val is not None]
            if not valid_vals:
                stats[field] = None
            else:
                arr = np.asarray(valid_vals, dtype=np.float64)
                sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                stats[field] = {
                    "mean": float(np.mean(arr)),
                    "sample_sd": sd,
                    "n": int(len(arr)),
                    "ddof": 1 if len(arr) > 1 else 0,
                    "values_by_seed": by_seed,
                }
        first_meta = items_sorted[0]["metric_summary"]
        groups_out[group_key] = {
            "branch": first_meta["branch"],
            "split_mode": first_meta["split_mode"],
            "base_classes": first_meta["base_classes"],
            "n_incremental_phases": first_meta["n_incremental_phases"],
            "shared_superclass_count": first_meta["shared_superclass_count"],
            "seeds": seeds,
            "metrics": stats,
        }

    aggregate = {
        "status": "complete",
        "experiment_family": "ACIL-Final-Empirics-E1",
        "groups": groups_out,
    }
    if output_path is not None:
        save_json(aggregate, Path(output_path))
    return aggregate


def main() -> None:
    """Điểm vào dòng lệnh (CLI) để thực thi một trường hợp E1 hoặc gộp kết quả nhiều seeds.

    Tham số:
        Không nhận tham số trực tiếp; đọc đối số từ ``sys.argv`` qua ``argparse``.

    Trả về:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Chạy một cấu hình thực nghiệm E1a hoặc E1b")
    run_parser.add_argument("--branch", choices=["E1a", "E1b"], required=True)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--base-classes", type=int, default=50)
    run_parser.add_argument("--phase-size", type=int, default=10)
    run_parser.add_argument(
        "--split-mode",
        choices=["uniform", "stratified", "superclass_disjoint", "superclass_disjoint_reverse"],
        default="uniform",
    )
    run_parser.add_argument("--epochs", type=int, default=160)
    run_parser.add_argument("--expansion-dim", type=int, default=8000)
    run_parser.add_argument("--gamma", type=float, default=0.1)
    run_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run_parser.add_argument("--threads", type=int, default=4)
    run_parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--reuse-checkpoint", type=Path)
    run_parser.add_argument("--resume-checkpoint", type=Path)
    run_parser.add_argument("--stop-after-epoch", type=int)
    run_parser.add_argument("--verify-joint-oracle", action="store_true")
    run_parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    run_parser.add_argument("--entity", default="hsangg-2006")
    run_parser.add_argument("--project", default="acil-reproduction")

    agg_parser = subparsers.add_parser("aggregate", help="Tổng hợp mean ± SD từ các tệp results.json của E1")
    agg_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    agg_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        cfg = default_e1_config(
            branch=args.branch,
            seed=args.seed,
            base_classes=args.base_classes,
            phase_size=args.phase_size,
            split_mode=args.split_mode,
        )
        cfg.update(
            epochs=args.epochs,
            expansion_dim=args.expansion_dim,
            gamma=args.gamma,
            device=args.device,
            torch_threads=args.threads,
        )
        run_e1_case(
            cfg,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            reuse_checkpoint=args.reuse_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
            stop_after_epoch=args.stop_after_epoch,
            verify_joint_oracle=args.verify_joint_oracle,
            wandb_mode=args.wandb_mode,
            entity=args.entity,
            project=args.project,
        )
    else:
        aggregate_e1_reports(args.inputs, args.output)


if __name__ == "__main__":
    main()
