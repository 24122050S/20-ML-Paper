"""Tải CIFAR-100 từ Hugging Face, kiểm tra và chia lớp có thể tái lập."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DATASET_ID = "uoft-cs/cifar100"
DATASET_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"


@dataclass(frozen=True)
class CifarArrays:
    """Ảnh uint8 NHWC, nhãn fine nguyên và thông tin nguồn của hai split gốc."""

    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    class_names: tuple[str, ...]
    source: dict


def sha256_file(path: Path) -> str:
    """Tính mã băm SHA-256 của một file bằng cách đọc theo từng khối 1 MiB.

    Tham số:
        path: Đường dẫn ``Path`` tới file cần băm.

    Trả về:
        Chuỗi hex SHA-256 gồm 64 ký tự.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cifar100(data_dir: str | Path = "data") -> CifarArrays:
    """Tải CIFAR-100 từ Hugging Face đúng revision đã chốt, giải mã và cache dạng NPY.

    Lần đầu tải Parquet và giải mã toàn bộ ảnh PNG; các lần sau đọc NPY qua memory map.

    Tham số:
        data_dir: Đường dẫn thư mục gốc chứa cache tải về và dữ liệu đã chuẩn bị.

    Trả về:
        Đối tượng ``CifarArrays`` chứa mảng train/test, tên lớp và metadata nguồn kèm SHA-256.
    """
    data_dir = Path(data_dir)
    prepared = data_dir / "prepared" / DATASET_REVISION
    manifest_path = prepared / "source.json"
    paths = {split: prepared / f"{split}_images.npy" for split in ("train", "test")}
    label_paths = {split: prepared / f"{split}_labels.npy" for split in paths}
    expected = [*paths.values(), *label_paths.values(), manifest_path]
    if not all(path.is_file() for path in expected):
        prepared.mkdir(parents=True, exist_ok=True)
        source = {"dataset_id": DATASET_ID, "revision": DATASET_REVISION, "files": {}}
        class_names = None
        for split, count in (("train", 50000), ("test", 10000)):
            filename = f"cifar100/{split}-00000-of-00001.parquet"
            print(f"Dataset: download/cache {filename}", flush=True)
            downloaded = Path(hf_hub_download(
                repo_id=DATASET_ID, repo_type="dataset", filename=filename,
                revision=DATASET_REVISION, cache_dir=str(data_dir / "hf_cache"),
            ))
            parquet = pq.ParquetFile(downloaded)
            if parquet.metadata.num_rows != count:
                raise ValueError(f"Sai số lượng mẫu trong split {split}")
            metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
            names = metadata["info"]["features"]["fine_label"]["names"]
            if class_names is not None and class_names != names:
                raise ValueError("Tên lớp giữa train/test không khớp")
            class_names = names
            images = np.lib.format.open_memmap(
                paths[split], mode="w+", dtype=np.uint8, shape=(count, 32, 32, 3)
            )
            labels = np.empty(count, dtype=np.int64)
            offset = 0
            for batch in parquet.iter_batches(batch_size=1024, columns=["img", "fine_label"]):
                for row in batch.to_pylist():
                    with Image.open(io.BytesIO(row["img"]["bytes"])) as image:
                        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    if array.shape != (32, 32, 3):
                        raise ValueError(f"Ảnh {split}/{offset} sai kích thước")
                    images[offset] = array
                    labels[offset] = row["fine_label"]
                    offset += 1
            if offset != count:
                raise ValueError("Giải mã thiếu ảnh")
            images.flush()
            del images
            np.save(label_paths[split], labels, allow_pickle=False)
            source["files"][split] = {
                "hf_filename": filename, "sha256": sha256_file(downloaded), "rows": count,
            }
        source["class_names"] = class_names
        manifest_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source["revision"] != DATASET_REVISION:
        raise ValueError("Cache không khớp revision đã chốt")
    return CifarArrays(
        np.load(paths["train"], mmap_mode="r", allow_pickle=False),
        np.load(label_paths["train"], allow_pickle=False),
        np.load(paths["test"], mmap_mode="r", allow_pickle=False),
        np.load(label_paths["test"], allow_pickle=False),
        tuple(source["class_names"]), source,
    )


def make_protocol(seed: int = 42, n_incremental_phases: int = 5) -> dict:
    """Sinh giao thức chia 100 lớp CIFAR-100 thành 50 lớp base và K phase tăng dần rời nhau.

    Base gồm 50 lớp đầu trong permutation; 50 lớp còn lại chia đều cho K phase.
    ID toàn cục là vị trí trong permutation nên không đặt lại nhãn từ 0 ở từng phase.

    Tham số:
        seed: Hạt giống ngẫu nhiên để sinh hoán vị ``class_order`` của 100 lớp.
        n_incremental_phases: Số phase tăng dần K (phải dương và là ước của 50).

    Trả về:
        Dict chứa ``class_order``, ánh xạ ``original_to_global`` và danh sách lớp từng phase.
    """
    if n_incremental_phases < 1 or 50 % n_incremental_phases:
        raise ValueError("K phải dương và chia hết 50")
    order = np.random.default_rng(seed).permutation(100)
    mapping = np.empty(100, dtype=np.int64)
    mapping[order] = np.arange(100)
    phase_classes = [list(range(50))]
    phase_classes.extend(chunk.tolist() for chunk in np.array_split(np.arange(50, 100), n_incremental_phases))
    return {
        "seed": seed, "class_order": order.tolist(), "original_to_global": mapping.tolist(),
        "base_classes": 50, "incremental_phases": n_incremental_phases,
        "phase_global_classes": phase_classes,
        "phase_original_classes": [[int(order[i]) for i in group] for group in phase_classes],
        "class_order_source": "NumPy default_rng(seed).permutation(100); chưa đối chiếu thứ tự tác giả",
    }


def audit_data(arrays: CifarArrays, protocol: dict) -> dict:
    """Kiểm tra toàn vẹn dữ liệu CIFAR-100, tính rời nhau các phase và tính mean/std chỉ từ base train.

    Tham số:
        arrays: Đối tượng ``CifarArrays`` chứa dữ liệu ảnh và nhãn gốc.
        protocol: Dict giao thức chia lớp được tạo bởi ``make_protocol``.

    Trả về:
        Dict bằng chứng kiểm tra gồm số mẫu từng phase, thống kê chuẩn hóa base-only và thông tin nguồn.
    """
    mapping = np.asarray(protocol["original_to_global"])
    order = np.asarray(protocol["class_order"])
    if not np.array_equal(np.sort(order), np.arange(100)):
        raise ValueError("class_order phải là hoán vị 0..99")
    if not np.array_equal(mapping[order], np.arange(100)):
        raise ValueError("Ánh xạ nhãn không nghịch đảo đúng class_order")
    for images, labels, count, per_class in (
        (arrays.train_images, arrays.train_labels, 50000, 500),
        (arrays.test_images, arrays.test_labels, 10000, 100),
    ):
        if images.shape != (count, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError("Sai định dạng ảnh CIFAR-100")
        if labels.shape != (count,) or not np.array_equal(np.unique(labels), np.arange(100)):
            raise ValueError("Cần đủ 100 fine labels trong mỗi split")
        if not np.all(np.bincount(labels, minlength=100) == per_class):
            raise ValueError("Số mẫu mỗi lớp không đúng dữ liệu gốc")
    flat = [label for group in protocol["phase_global_classes"] for label in group]
    if sorted(flat) != list(range(100)) or len(flat) != len(set(flat)):
        raise ValueError("Các phase phải rời nhau và phủ đủ 100 lớp")
    train_labels = mapping[arrays.train_labels]
    test_labels = mapping[arrays.test_labels]
    phases = []
    for index, group in enumerate(protocol["phase_global_classes"]):
        phases.append({
            "phase": index, "classes": group,
            "original_classes": protocol["phase_original_classes"][index],
            "train_samples": int(np.isin(train_labels, group).sum()),
            "test_samples": int(np.isin(test_labels, group).sum()),
        })
    # Chỉ train base được dùng để chọn thống kê tiền xử lý; không đọc test/future.
    indices = np.flatnonzero(train_labels < 50)
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    for chunk in np.array_split(indices, 50):
        pixels = arrays.train_images[chunk].astype(np.float64) / 255.0
        sums += pixels.sum(axis=(0, 1, 2))
        squares += np.square(pixels).sum(axis=(0, 1, 2))
    pixel_count = len(indices) * 32 * 32
    mean = sums / pixel_count
    std = np.sqrt(squares / pixel_count - np.square(mean))
    return {
        "passed": True, "train_samples": 50000, "test_samples": 10000,
        "image_shape": [32, 32, 3], "label_field": "fine_label", "classes": 100,
        "train_samples_per_class": 500, "test_samples_per_class": 100,
        "phase_classes_disjoint": True, "label_mapping_bijective": True,
        "phases": phases,
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "source": "base_train_only", "samples": len(indices)},
        "source": arrays.source,
    }


class CifarSubset(Dataset):
    """Subset theo ID lớp toàn cục, trả tensor CHW đã chuẩn hóa và nhãn nguyên.

    images/labels là split gốc; indices chọn mẫu; mapping chuyển fine ID sang
    ID toàn cục. Chỉ augment=True mới crop 32 sau padding 4 và lật ngang.
    """

    def __init__(self, images, labels, indices, mapping, mean, std, augment=False):
        """Khởi tạo tập con CIFAR-100 gắn ánh xạ nhãn toàn cục và thống kê chuẩn hóa.

        Tham số:
            images: Mảng ảnh NHWC uint8.
            labels: Mảng nhãn fine gốc.
            indices: Mảng chỉ số mẫu thuộc tập con cần lấy.
            mapping: Bảng ánh xạ từ nhãn gốc sang nhãn toàn cục.
            mean: Bộ 3 giá trị trung bình kênh RGB để chuẩn hóa.
            std: Bộ 3 giá trị độ lệch chuẩn kênh RGB để chuẩn hóa.
            augment: Bật padding/random crop/flip ngang nếu True (chỉ dùng lúc train base).

        Trả về:
            None.
        """
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mapping = np.asarray(mapping, dtype=np.int64)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.augment = augment

    def __len__(self):
        """Trả về tổng số mẫu trong tập con.

        Tham số:
            Không nhận tham số ngoài ``self``.

        Trả về:
            Số nguyên là độ dài của ``self.indices``.
        """
        return len(self.indices)

    def __getitem__(self, index):
        """Đọc, tùy chọn augment, chuẩn hóa một ảnh và ánh xạ nhãn sang ID toàn cục.

        Tham số:
            index: Vị trí mẫu trong phạm vi ``0 .. len(self) - 1``.

        Trả về:
            Tuple ``(image_tensor, global_label)`` với ảnh float32 dạng CHW và nhãn nguyên.
        """
        original_index = int(self.indices[index])
        # tensor() tạo bản sao, tránh ghi lên memory map chỉ đọc của ảnh gốc.
        image = torch.tensor(self.images[original_index], dtype=torch.float32).permute(2, 0, 1) / 255.0
        if self.augment:
            image = F.pad(image, (4, 4, 4, 4))
            top, left = torch.randint(0, 9, (2,)).tolist()
            image = image[:, top:top + 32, left:left + 32]
            if torch.rand(()).item() < 0.5:
                image = image.flip(2)
        return (image - self.mean) / self.std, int(self.mapping[self.labels[original_index]])
