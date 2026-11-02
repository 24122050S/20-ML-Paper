"""Ghép CIFAR-100/ResNet-32 với analytic.py; mặc định chạy 50 base + 10 lớp mới.

Đây là smoke test với 2 epochs và expansion 256, không phải benchmark paper.
Joint giữ lịch sử chỉ để chẩn đoán; incremental_update chỉ nhận batch hiện tại.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.utils.data import DataLoader

from .analytic import initialize, incremental_update, joint_ridge
from .cifar_data import CifarSubset, audit_data, load_cifar100, make_protocol, sha256_file
from .reporting import collect_provenance, log_and_save_report, save_json
from .resnet import ResNet32, freeze_backbone


def fingerprint(model):
    """Băm cả trọng số và buffer BatchNorm để kiểm tra backbone thật sự cố định."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed, threads):
    """Cố định RNG và thuật toán; không bảo đảm bitwise giống giữa CPU và GPU."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(arrays, protocol, normalization, classes, split, config, augment=False):
    """Chọn ID toàn cục, cùng mapping train/test; chỉ augment khi huấn luyện base."""
    if split not in ("train", "test"):
        raise ValueError("split phải là train hoặc test")
    images = arrays.train_images if split == "train" else arrays.test_images
    labels = arrays.train_labels if split == "train" else arrays.test_labels
    mapping = np.asarray(protocol["original_to_global"], dtype=np.int64)
    indices = np.flatnonzero(np.isin(mapping[labels], classes))
    subset = CifarSubset(images, labels, indices, mapping,
                         normalization["mean"], normalization["std"], augment)
    return DataLoader(subset, batch_size=config["batch_size"] if augment else config["eval_batch_size"],
                      shuffle=augment, num_workers=0,
                      generator=torch.Generator().manual_seed(config["seed"]))


def train_base(model, loader, config, device, output):
    """Huấn luyện đúng base train; chọn epoch cuối theo config, không chọn bằng test."""
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config["lr_milestones"], gamma=0.1)
    before = model.conv1.weight.detach().clone()
    history = []
    for epoch in range(1, config["epochs"] + 1):
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
                save_json({"stage": "base_training", "epoch": epoch, "batch": step}, output / "progress.json")
        history.append({"epoch": epoch, "train_loss": total_loss / samples,
                        "train_accuracy_pct": 100 * correct / samples, "samples": samples,
                        "learning_rate": learning_rate, "seconds": time.perf_counter() - start})
        scheduler.step()
    change = torch.linalg.vector_norm(model.conv1.weight.detach() - before).item()
    if change <= 0:
        raise AssertionError("Trọng số không đổi")
    return history, change


def validate_checkpoint(checkpoint, config, protocol, normalization, dataset_revision):
    """Không dùng checkpoint của class order, preprocessing hoặc cấu hình base khác."""
    for key in ("seed", "epochs", "learning_rate", "momentum", "weight_decay", "batch_size", "lr_milestones"):
        if checkpoint["config"][key] != config[key]:
            raise ValueError(f"Checkpoint không khớp {key}")
    if checkpoint["class_order"] != protocol["class_order"]:
        raise ValueError("Checkpoint không khớp class_order")
    if checkpoint["normalization"] != normalization:
        raise ValueError("Checkpoint không khớp normalization")
    if checkpoint["dataset_revision"] != dataset_revision:
        raise ValueError("Checkpoint không khớp revision dữ liệu")
    history = checkpoint["training_history"]
    if len(history) != config["epochs"] or not all(np.isfinite(row["train_loss"]) for row in history):
        raise ValueError("Checkpoint thiếu lịch sử train hợp lệ")
    if not np.isfinite(checkpoint["parameter_change_l2"]) or checkpoint["parameter_change_l2"] <= 0:
        raise ValueError("Checkpoint thiếu bằng chứng cập nhật trọng số")


@torch.inference_mode()
def extract_features(model, loader, device, name):
    """Trả F (n,64) float64 và y toàn cục; từ chối backbone còn train/unfrozen."""
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Phải freeze backbone và eval tất cả module")
    features, labels = [], []
    for step, (images, y) in enumerate(loader, 1):
        f = model.forward_features(images.to(device)).cpu().numpy().astype(np.float64)
        if f.ndim != 2 or f.shape[1] != 64 or not np.isfinite(f).all():
            raise FloatingPointError("Đặc trưng sai shape hoặc không hữu hạn")
        features.append(f)
        labels.append(y.numpy())
        if step == 1 or step % 50 == 0 or step == len(loader):
            print(f"Features {name}: {step}/{len(loader)}", flush=True)
    return np.vstack(features), np.concatenate(labels)


def validate_config(config):
    """Chặn cấu hình rỗng hoặc ngoài phạm vi smoke test trước khi tốn thời gian train."""
    for key in ("epochs", "batch_size", "eval_batch_size", "analytic_batch_size", "torch_threads", "expansion_dim"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} phải là số nguyên dương")
    if config["num_workers"] != 0 or config["incremental_phases"] != 5:
        raise ValueError("Smoke protocol dùng num_workers=0 và 50 base + 5 phase x 10 lớp")
    if not isinstance(config["run_incremental_phases"], int) or not 1 <= config["run_incremental_phases"] <= 5:
        raise ValueError("Cần chạy 1..5 phase incremental")
    for key in ("gamma", "rtol", "atol"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} phải hữu hạn và dương")


def run(config, *, data_dir, output_dir, reuse_checkpoint=None, log_wandb=False,
        entity="hsangg-2006", project="acil-reproduction", config_path=None):
    """Chạy thử đầu-cuối, trả report sau khi kiểm tra và lưu JSON tracking hoàn tất."""
    validate_config(config)
    seed_everything(config["seed"], config["torch_threads"])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng môi trường không có GPU CUDA")
    device = torch.device(device_name)
    root = Path(__file__).resolve().parents[1]
    provenance = collect_provenance(root)
    if config_path is not None:
        provenance["input_config_sha256"] = sha256_file(Path(config_path))
    provenance["effective_config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    start = time.perf_counter()
    with threadpool_limits(limits=config["torch_threads"]):
        arrays = load_cifar100(data_dir)
        protocol = make_protocol(config["seed"], config["incremental_phases"])
        audit = audit_data(arrays, protocol)
        normalization = audit["normalization"]
        save_json(config, output / "config.json")
        save_json(protocol, output / "protocol.json")
        save_json(audit, output / "data_audit.json")
        print("Data PASS: 50000 train, 10000 test; shared mapping; disjoint 50+10 phases.", flush=True)
        model = ResNet32(n_classes=50).to(device)
        checkpoint_path = output / "base_checkpoint.pt"
        origin = {"reused": reuse_checkpoint is not None}
        if reuse_checkpoint is not None:
            reused = Path(reuse_checkpoint).resolve()
            checkpoint = torch.load(reused, map_location="cpu", weights_only=True)
            validate_checkpoint(checkpoint, config, protocol, normalization, arrays.source["revision"])
            model.load_state_dict(checkpoint["model_state_dict"])
            history = checkpoint["training_history"]
            change = checkpoint["parameter_change_l2"]
            origin.update(input_sha256=sha256_file(reused), training_executed_this_run=False)
            if reused != checkpoint_path:
                shutil.copy2(reused, checkpoint_path)
            print("Reused checkpoint: protocol, normalization, revision and base config verified.", flush=True)
        else:
            loader = make_loader(arrays, protocol, normalization, range(50), "train", config, augment=True)
            history, change = train_base(model, loader, config, device, output)
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "class_order": protocol["class_order"], "normalization": normalization,
                        "dataset_revision": arrays.source["revision"], "training_history": history,
                        "parameter_change_l2": change}, checkpoint_path)
            origin["training_executed_this_run"] = True
        model.eval()
        base_test = make_loader(arrays, protocol, normalization, range(50), "test", config)
        probe, _ = next(iter(base_test))
        with torch.inference_mode():
            before_reload = model(probe.to(device))
        restored = ResNet32(n_classes=50).to(device)
        restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state_dict"])
        restored.eval()
        with torch.inference_mode():
            after_reload = restored(probe.to(device))
        torch.testing.assert_close(before_reload, after_reload, rtol=0, atol=0)
        reload_error = float((before_reload - after_reload).abs().max().item())
        model = freeze_backbone(restored)
        frozen_before = fingerprint(model)
        expansion = np.random.default_rng(config["seed"] + 1000).normal(size=(64, config["expansion_dim"]))
        expansion_before = hashlib.sha256(expansion.tobytes()).hexdigest()
        joint_features, joint_labels, rows = [], [], []
        r = w = None
        for phase in range(config["run_incremental_phases"] + 1):
            save_json({"stage": "analytic", "phase": phase}, output / "progress.json")
            classes = protocol["phase_global_classes"][phase]
            n_classes = max(classes) + 1
            train_loader = make_loader(arrays, protocol, normalization, classes, "train", config)
            raw, labels = extract_features(model, train_loader, device, f"train phase {phase}")
            h = np.maximum(raw @ expansion, 0.0)
            if phase == 0:
                r, w = initialize(h, labels, gamma=config["gamma"], n_classes=n_classes)
                batch_count = 1
            else:
                batch_count = 0
                for offset in range(0, len(h), config["analytic_batch_size"]):
                    end = offset + config["analytic_batch_size"]
                    r, w = incremental_update(r, w, h[offset:end], labels[offset:end], n_classes=n_classes)
                    assert w.shape == (config["expansion_dim"], n_classes)
                    batch_count += 1
            # Dữ liệu lịch sử dưới đây chỉ phục vụ oracle joint, không đi vào update.
            joint_features.append(h)
            joint_labels.append(labels)
            pooled = np.vstack(joint_features)
            reference = joint_ridge(pooled, np.concatenate(joint_labels), n_classes, config["gamma"])
            identity = np.eye(config["expansion_dim"])
            r_reference = np.linalg.solve(pooled.T @ pooled + config["gamma"] * identity, identity)
            np.testing.assert_allclose(w, reference, rtol=config["rtol"], atol=config["atol"])
            np.testing.assert_allclose(r, r_reference, rtol=config["rtol"], atol=config["atol"])
            test_loader = make_loader(arrays, protocol, normalization, range(n_classes), "test", config)
            test_raw, test_labels = extract_features(model, test_loader, device, f"test seen {n_classes}")
            test_h = np.maximum(test_raw @ expansion, 0.0)
            scores, joint_scores = test_h @ w, test_h @ reference
            np.testing.assert_allclose(scores, joint_scores, rtol=config["rtol"], atol=config["atol"])
            predictions = scores.argmax(1)
            disagreements = int(np.count_nonzero(predictions != joint_scores.argmax(1)))
            if disagreements:
                raise AssertionError("Minibatch và joint có dự đoán khác nhau")
            if not np.array_equal(np.unique(test_labels), np.arange(n_classes)) or len(test_labels) != 100 * n_classes:
                raise AssertionError("Evaluation phải phủ tất cả lớp đã thấy")
            base_mask = test_labels < 50
            new_mask = np.isin(test_labels, classes)
            row = {"phase": phase, "n_classes": n_classes, "weights_shape": list(w.shape),
                   "inverse_gram_shape": list(r.shape), "train_samples_current_phase": len(h),
                   "test_samples_all_seen": len(test_labels), "analytic_batches": batch_count,
                   "relative_w_error": float(np.linalg.norm(w-reference) / np.linalg.norm(reference)),
                   "max_abs_r_error": float(np.max(np.abs(r-r_reference))),
                   "max_abs_score_error": float(np.max(np.abs(scores-joint_scores))),
                   "prediction_disagreements": disagreements,
                   "accuracy_pct": float(100*np.mean(predictions == test_labels)),
                   "base_accuracy_pct": float(100*np.mean(predictions[base_mask] == test_labels[base_mask])),
                   "new_class_accuracy_pct": None if phase == 0 else float(100*np.mean(predictions[new_mask] == test_labels[new_mask])),
                   "checks_passed": True}
            rows.append(row)
            print(f"Phase {phase}: W={w.shape}, test={len(test_labels)}, accuracy={row['accuracy_pct']:.2f}%, relative_W={row['relative_w_error']:.3e}", flush=True)
        frozen_after = fingerprint(model)
        if frozen_before != frozen_after or expansion_before != hashlib.sha256(expansion.tobytes()).hexdigest():
            raise AssertionError("Backbone/BatchNorm hoặc expansion đã thay đổi")
        np.savez_compressed(output / "analytic_state.npz", inverse_gram=r, weights=w,
                            expansion_matrix=expansion, gamma=config["gamma"],
                            class_order=np.asarray(protocol["class_order"]), n_classes=w.shape[1])
        checks = {"data_audit_passed": audit["passed"], "checkpoint_reload_exact": reload_error == 0,
                  "finite_training_loss": all(np.isfinite(row["train_loss"]) for row in history),
                  "base_parameters_updated": change > 0, "backbone_batchnorm_and_expansion_frozen": True,
                  "all_seen_test_classes_evaluated": True, "analytic_minibatch_matches_joint": True,
                  "final_columns_correct": w.shape[1] == 50 + 10*config["run_incremental_phases"]}
        provenance["base_checkpoint_sha256"] = sha256_file(checkpoint_path)
        report = {"experiment": config["experiment"], "description": "Real CIFAR-100 smoke test; not a paper benchmark",
                  "config": config, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "device": str(device),
                    "packages": {name: importlib.metadata.version(name) for name in ("huggingface_hub", "pyarrow", "Pillow", "threadpoolctl")}},
                  "provenance": provenance, "checkpoint_origin": origin,
                  "dataset_source": arrays.source, "training_history": history,
                  "stem_weight_change_l2": change, "checkpoint_reload_max_logit_error": reload_error,
                  "backbone_sha256_before": frozen_before, "backbone_sha256_after": frozen_after,
                  "expansion_sha256": expansion_before, "results": rows, "checks": checks,
                  "verification_passed": all(checks.values()), "seconds_this_run": time.perf_counter()-start}
    report = log_and_save_report(report, output / "results.json", log_wandb=log_wandb,
                                 entity=entity, project=project,
                                 tags=["smoke", "cifar100", "float64", "minibatch"])
    save_json({"stage": "complete", "checks_passed": report["verification_passed"]}, output / "progress.json")
    print(f"CIFAR SMOKE PASS; tracking={report['wandb_status']}; report={output / 'results.json'}", flush=True)
    return report


def main():
    """CLI từ gốc repo; checkpoint reuse là tùy chọn, mặc định huấn luyện base mới."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/cifar100_smoke.json")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/local/cifar100_smoke")
    parser.add_argument("--reuse-checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--entity", default="hsangg-2006")
    parser.add_argument("--project", default="acil-reproduction")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    run(config, data_dir=args.data_dir, output_dir=args.output_dir, reuse_checkpoint=args.reuse_checkpoint,
        log_wandb=args.wandb, entity=args.entity, project=args.project, config_path=args.config)


if __name__ == "__main__":
    main()
