"""Huấn luyện base với checkpoint atomic và resume đúng tại biên epoch."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import platform
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn


class EpochLimitReached(RuntimeError):
    """Dừng có chủ đích sau checkpoint epoch; lượt pipeline vẫn chưa complete."""


def atomic_torch_save(payload, path):
    """Ghi checkpoint PyTorch an toàn qua file tạm cùng thư mục rồi replace atomic.

    Chỉ thay thế file cũ sau khi đã flush và fsync thành công lên đĩa.

    Tham số:
        payload: Dict hoặc đối tượng PyTorch hợp lệ cần tuần tự hóa.
        path: Đường dẫn file checkpoint đích.

    Trả về:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".checkpoint-",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _runtime(device):
    """Thu thập dấu vân tay môi trường và hash các file nguồn huấn luyện để kiểm tra resume.

    Tham số:
        device: Thiết bị PyTorch mục tiêu (``cpu``, ``cuda`` hoặc ``torch.device``).

    Trả về:
        Dict thông tin phiên bản thư viện, thiết bị, cờ deterministic và ``training_source_sha256``.
    """
    device = torch.device(device)
    root = Path(__file__).resolve().parent
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "device_type": device.type,
            "cuda": torch.version.cuda if device.type == "cuda" else None,
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "training_source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                      for name in ("base_training.py", "cifar_smoke.py", "cifar_data.py", "resnet.py")}}


def capture_rng(device):
    """Đóng gói toàn bộ trạng thái bộ sinh số ngẫu nhiên Python, NumPy và PyTorch CPU/CUDA.

    Dữ liệu đóng gói chỉ dùng tensor và kiểu nguyên thủy để đọc được qua ``weights_only=True``.

    Tham số:
        device: Thiết bị PyTorch hiện đang chạy.

    Trả về:
        Dict lưu trạng thái RNG của Python, NumPy, Torch CPU và Torch CUDA (nếu có).
    """
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": state[0], "keys": torch.tensor(state[1].astype(np.int64)),
                      "position": int(state[2]), "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else []}


def restore_rng(state, device):
    """Khôi phục trạng thái RNG đã lưu sau khi khởi tạo model, optimizer và DataLoader.

    Tham số:
        state: Dict trạng thái RNG được tạo bởi ``capture_rng``.
        device: Thiết bị PyTorch hiện tại cần khôi phục RNG.

    Trả về:
        None.
    """
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], numpy_state["keys"].numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _validate_resume(checkpoint, model, loader, config, context, device):
    """Kiểm tra chặt chẽ tính tương thích của checkpoint v2 trước khi khôi phục huấn luyện.

    Chặn checkpoint thiếu state, khác cấu hình, khác protocol/normalization/runtime
    hoặc đã hoàn tất toàn bộ target epochs.

    Tham số:
        checkpoint: Dict checkpoint đã đọc từ đĩa.
        model: Mô hình ``nn.Module`` hiện tại.
        loader: ``DataLoader`` huấn luyện base.
        config: Dict cấu hình lượt chạy hiện tại.
        context: Dict chứa ``class_order``, ``normalization`` và ``dataset_revision``.
        device: Thiết bị PyTorch hiện tại.

    Trả về:
        None. Phát sinh ``ValueError`` nếu checkpoint không hợp lệ để resume.
    """
    if checkpoint.get("checkpoint_format_version") != 2:
        raise ValueError("Resume cần checkpoint v2 có optimizer/scheduler/RNG; checkpoint cũ chỉ dùng reuse")
    required = ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state",
                "dataloader_generator_state", "initial_stem_weight", "runtime", "loader_signature")
    if any(key not in checkpoint for key in required):
        raise ValueError("Checkpoint resume thiếu state bắt buộc")
    for key in ("seed", "epochs", "learning_rate", "momentum", "weight_decay", "batch_size",
                "lr_milestones", "torch_threads", "num_workers"):
        if checkpoint["config"].get(key) != config.get(key):
            raise ValueError(f"Checkpoint resume không khớp {key}")
    for key in ("class_order", "normalization", "dataset_revision"):
        if checkpoint.get(key) != context[key]:
            raise ValueError(f"Checkpoint resume không khớp {key}")
    completed = checkpoint.get("completed_epoch")
    if not isinstance(completed, int) or isinstance(completed, bool) or not 1 <= completed < config["epochs"]:
        raise ValueError("Checkpoint phải có epoch đã hoàn thành và chưa đạt target; base hoàn tất dùng reuse")
    if checkpoint.get("target_epochs") != config["epochs"] or checkpoint.get("training_complete") is not False:
        raise ValueError("Checkpoint resume không khớp target/training_complete")
    history = checkpoint.get("training_history", [])
    if len(history) != completed or [row["epoch"] for row in history] != list(range(1, completed + 1)):
        raise ValueError("Checkpoint resume thiếu lịch sử epoch liên tiếp")
    if not all(np.isfinite(row["train_loss"]) for row in history):
        raise ValueError("Checkpoint resume có loss không hữu hạn")
    expected_loader = {"dataset_size": len(loader.dataset), "batch_size": loader.batch_size,
                       "num_workers": loader.num_workers, "drop_last": loader.drop_last,
                       "sampler": type(loader.sampler).__qualname__}
    if checkpoint["loader_signature"] != expected_loader:
        raise ValueError("Checkpoint resume không khớp DataLoader")
    if checkpoint["runtime"] != _runtime(device):
        raise ValueError("Checkpoint resume không khớp runtime/source; cần cùng môi trường huấn luyện")
    current = model.state_dict()
    saved = checkpoint["model_state_dict"]
    if current.keys() != saved.keys() or any(current[n].shape != saved[n].shape or current[n].dtype != saved[n].dtype for n in current):
        raise ValueError("Checkpoint resume không khớp kiến trúc model")


def train_base(model, loader, config, device, output, *, progress=None,
               checkpoint_context=None, resume_checkpoint=None, stop_after_epoch=None):
    """Huấn luyện backbone giai đoạn base theo số epoch mục tiêu, ghi checkpoint v2 sau từng epoch.

    Nếu gặp lỗi giữa epoch, checkpoint của epoch hoàn tất trước đó vẫn được giữ nguyên.

    Tham số:
        model: Mô hình backbone ResNet-32 cần huấn luyện trên các lớp base.
        loader: ``DataLoader`` tập train base có gắn ``generator`` và ``num_workers=0``.
        config: Dict cấu hình hyperparameters (``epochs``, ``learning_rate``, ``momentum``, ...).
        device: Thiết bị PyTorch thực thi (CPU hoặc CUDA).
        output: Thư mục ghi file ``base_checkpoint.pt``.
        progress: Callback tùy chọn báo tiến độ theo epoch/batch.
        checkpoint_context: Dict metadata gắn với checkpoint (``class_order``, ``normalization``, ``dataset_revision``).
        resume_checkpoint: Đường dẫn tới checkpoint v2 đang dở để tiếp tục (tùy chọn).
        stop_after_epoch: Mốc epoch dừng có chủ đích (tùy chọn) để kiểm thử cơ chế resume.

    Trả về:
        Tuple ``(history, change)`` gồm danh sách dict lịch sử từng epoch và chuẩn L2
        độ thay đổi trọng số lớp conv đầu.
    """
    if checkpoint_context is None or loader.generator is None or loader.num_workers != 0:
        raise ValueError("Cần checkpoint context, DataLoader generator và num_workers=0 để resume")
    if stop_after_epoch is not None and (isinstance(stop_after_epoch, bool) or
            not isinstance(stop_after_epoch, int) or not 1 <= stop_after_epoch < config["epochs"]):
        raise ValueError("stop_after_epoch phải nằm trong 1..target-1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "base_checkpoint.pt"
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config["lr_milestones"], gamma=0.1)
    before = model.conv1.weight.detach().cpu().clone()
    history, start_epoch = [], 1
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
        _validate_resume(checkpoint, model, loader, config, checkpoint_context, device)
        if stop_after_epoch is not None and stop_after_epoch <= checkpoint["completed_epoch"]:
            raise ValueError("stop_after_epoch phải lớn hơn epoch đã hoàn thành")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        before = checkpoint["initial_stem_weight"]
        history = copy.deepcopy(checkpoint["training_history"])
        start_epoch = checkpoint["completed_epoch"] + 1
        loader.generator.set_state(checkpoint["dataloader_generator_state"])
        restore_rng(checkpoint["rng_state"], device)
        # Lượt mới có bản checkpoint an toàn riêng ngay cả khi lỗi trước epoch kế tiếp.
        atomic_torch_save(checkpoint, target_path)
    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        total_loss, correct, samples = 0.0, 0, 0
        start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        for step, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Loss không hữu hạn")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise FloatingPointError("Gradient không hữu hạn")
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            samples += len(labels)
            if step == 1 or step % 50 == 0 or step == len(loader):
                print(f"Base epoch {epoch}/{config['epochs']}, batch {step}/{len(loader)}, loss={total_loss/samples:.4f}", flush=True)
                if progress is not None:
                    progress("base_training", epoch=epoch, batch=step, checkpoint=str(target_path))
        history.append({"epoch": epoch, "train_loss": total_loss / samples,
                        "train_accuracy_pct": 100 * correct / samples, "samples": samples,
                        "learning_rate": learning_rate, "seconds": time.perf_counter() - start})
        scheduler.step()
        change = float(torch.linalg.vector_norm(model.conv1.weight.detach().cpu() - before).item())
        if not np.isfinite(change) or not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
            raise FloatingPointError("Model không hữu hạn; giữ checkpoint epoch trước")
        payload = {"checkpoint_format_version": 2, "kind": "base_training",
                   "completed_epoch": epoch, "target_epochs": config["epochs"],
                   "training_complete": epoch == config["epochs"],
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "config": copy.deepcopy(config),
                   **copy.deepcopy(checkpoint_context), "training_history": copy.deepcopy(history),
                   "initial_stem_weight": before, "parameter_change_l2": change,
                   "rng_state": capture_rng(device), "dataloader_generator_state": loader.generator.get_state(),
                   "loader_signature": {"dataset_size": len(loader.dataset), "batch_size": loader.batch_size,
                                        "num_workers": loader.num_workers, "drop_last": loader.drop_last,
                                        "sampler": type(loader.sampler).__qualname__},
                   "runtime": _runtime(device)}
        atomic_torch_save(payload, target_path)
        if stop_after_epoch is not None and epoch == stop_after_epoch:
            raise EpochLimitReached(f"Đã lưu epoch {epoch}; resume từ {target_path}")
    if change <= 0 or not np.isfinite(change):
        raise AssertionError("Trọng số không đổi hoặc không hữu hạn")
    return history, change
