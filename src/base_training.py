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
    """Ghi checkpoint qua file tạm cùng thư mục; chỉ thay bản cũ sau flush/fsync."""
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
    """Nhận device; trả dấu môi trường cần trùng để resume deterministic."""
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
    """Đóng gói RNG thành tensor/primitive, đọc được bằng torch.load weights_only=True."""
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": state[0], "keys": torch.tensor(state[1].astype(np.int64)),
                      "position": int(state[2]), "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else []}


def restore_rng(state, device):
    """Khôi phục RNG sau khi dựng model/optimizer/loader để bỏ qua RNG đã tiêu lúc khởi tạo."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], numpy_state["keys"].numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _validate_resume(checkpoint, model, loader, config, context, device):
    """Chặn checkpoint thiếu state, đổi target/protocol/preprocessing/runtime trước restore."""
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
    """Train tới target epochs; trả history/change và ghi v2 sau từng epoch hoàn tất.

    context chứa class_order, normalization, dataset_revision. Resume nhận path
    checkpoint đang dở; stop_after_epoch phục vụ kiểm tra chạy ngắt có chủ đích.
    Exception giữa epoch giữ checkpoint epoch trước, không lưu batch chưa hoàn tất.
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
