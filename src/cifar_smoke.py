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
from .run_attempt import RunAttempt
from .base_training import train_base
from .streaming import run_streaming_phases
from .metrics import summarize


def fingerprint(model):
    """Băm cả trọng số và buffer BatchNorm để kiểm tra backbone thật sự cố định."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed, threads):
    """Cố định RNG và thuật toán; không bảo đảm bitwise giống giữa CPU và GPU."""
    #Cố định seed của random, numpy, torch; cố định số luồng; bật deterministic algorithms.
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


def make_loader(arrays, protocol, normalization, classes, split, config, augment=False, *, batch_size=None):
    """Chọn ID toàn cục, cùng mapping train/test; chỉ augment khi huấn luyện base."""
    #Lọc mẫu ID toàn cục, nên train và test thử dụng cùng mapping
    #augment = True chỉ bật khi huấn luyện base; batch_size khác nhau cho train/test.
    if split not in ("train", "test"):
        raise ValueError("split phải là train hoặc test")
    images = arrays.train_images if split == "train" else arrays.test_images
    labels = arrays.train_labels if split == "train" else arrays.test_labels
    mapping = np.asarray(protocol["original_to_global"], dtype=np.int64)
    indices = np.flatnonzero(np.isin(mapping[labels], classes))
    subset = CifarSubset(images, labels, indices, mapping,
                         normalization["mean"], normalization["std"], augment)
    selected_batch = batch_size or (config["batch_size"] if augment else config["eval_batch_size"])
    return DataLoader(subset, batch_size=selected_batch,
                      shuffle=augment, num_workers=0,
                      generator=torch.Generator().manual_seed(config["seed"]))


def validate_checkpoint(checkpoint, config, protocol, normalization, dataset_revision):
    """Không dùng checkpoint của class order, preprocessing hoặc cấu hình base khác."""
    if checkpoint.get("checkpoint_format_version") == 2 and checkpoint.get("training_complete") is not True:
        raise ValueError("Checkpoint base chưa hoàn tất; dùng resume thay vì reuse")
    if checkpoint.get("checkpoint_format_version") == 2 and checkpoint.get("loader_signature", {}).get("dataset_size") != 25000:
        raise ValueError("Checkpoint CIFAR base phải train đủ25000 ảnh, không dùng checkpoint subset")
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


def iter_feature_batches(model, loader, device, name, *, timings=None):
    """Yield F64/y từng batch, không vstack; từ chối backbone chưa freeze/eval."""
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Phải freeze backbone và eval tất cả module")
    for step, (images, labels) in enumerate(loader, 1):
        started = time.perf_counter()
        with torch.inference_mode():
            raw = model.forward_features(images.to(device)).cpu().numpy().astype(np.float64)
        if raw.ndim != 2 or raw.shape[1] != 64 or not np.isfinite(raw).all():
            raise FloatingPointError("Feature batch sai shape hoặc không hữu hạn")
        if timings is not None:
            timings["feature_extraction_seconds"] += time.perf_counter() - started
        if step == 1 or step % 50 == 0 or step == len(loader):
            print(f"Features {name}: {step}/{len(loader)}", flush=True)
        yield raw, labels.numpy()


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
    if config.get("mode", "smoke") not in ("smoke", "benchmark"):
        raise ValueError("mode phải là smoke hoặc benchmark")
    if not isinstance(config.get("feature_expansion", True), bool):
        raise ValueError("feature_expansion phải là bool")
    if config.get("mode", "smoke") == "smoke" and not config.get("feature_expansion", True):
        raise ValueError("No-FE dùng mode benchmark")
    size = config.get("base_statistics_batch_size", 1024)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("base_statistics_batch_size phải nguyên dương")


def run(config, *, data_dir, output_dir, reuse_checkpoint=None, log_wandb=False,
        entity="hsangg-2006", project="acil-reproduction", config_path=None,
        config_overrides=None, prepare_tracking=None, resume_checkpoint=None, stop_after_epoch=None):
    """Chạy một attempt riêng; trả report complete hoặc ghi failed rồi raise.

    output_dir là root chứa latest/progress/results summary và attempts/<id>/.
    reuse_checkpoint là đường dẫn tường minh của checkpoint đã hoàn thành.
    config=None đọc config_path bên trong attempt; config_overrides ghi các
    tùy chọn CLI/notebook. prepare_tracking là callback đăng nhập, chỉ gọi khi
    bật W&B và được bảo vệ bởi cùng trạng thái attempt.
    """
    seed = config.get("seed") if isinstance(config, dict) else None
    with RunAttempt(output_dir, seed=seed) as attempt:
        attempt.progress("config_load")
        if reuse_checkpoint is not None and resume_checkpoint is not None:
            raise ValueError("Chọn reuse hoặc resume, không dùng đồng thời")
        if reuse_checkpoint is not None and stop_after_epoch is not None:
            raise ValueError("stop_after_epoch chỉ áp dụng khi đang huấn luyện BP")
        if config is None:
            if config_path is None:
                raise ValueError("Cần config hoặc config_path")
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        config = {**config, **(config_overrides or {})}
        seed = config.get("seed")
        attempt.seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
        if log_wandb and prepare_tracking is not None:
            attempt.progress("tracking_setup")
            prepare_tracking()
        report = _compute_report(config, data_dir=data_dir, output=attempt.directory,
                                 reuse_checkpoint=reuse_checkpoint, config_path=config_path,
                                 progress=attempt.progress, resume_checkpoint=resume_checkpoint,
                                 stop_after_epoch=stop_after_epoch)
        report.update(attempt_id=attempt.id, artifacts_dir=attempt.relative_dir, status="running")
        attempt.progress("tracking")
        report = log_and_save_report(report, attempt.directory / "results.json", log_wandb=log_wandb,
                                     entity=entity, project=project,
                                     tags=[config.get("mode", "smoke"), "cifar100", "float64", "minibatch"])
        report = attempt.complete(report)
        print(f"CIFAR SMOKE PASS; tracking={report['wandb_status']}; report={attempt.directory / 'results.json'}", flush=True)
        return report


def _compute_report(config, *, data_dir, output, reuse_checkpoint, config_path, progress,
                    resume_checkpoint=None, stop_after_epoch=None):
    """Tính numeric report vào output của attempt; progress nhận stage và metadata."""
    progress("validating")
    validate_config(config)
    save_json(config, output / "config.json")
    seed_everything(config["seed"], config["torch_threads"])
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
        progress("download_and_audit")
        arrays = load_cifar100(data_dir)
        protocol = make_protocol(config["seed"], config["incremental_phases"])
        audit = audit_data(arrays, protocol)
        normalization = audit["normalization"]
        save_json(protocol, output / "protocol.json")
        save_json(audit, output / "data_audit.json")
        print("Data PASS: 50000 train, 10000 test; shared mapping; disjoint 50+10 phases.", flush=True)
        model = ResNet32(n_classes=50).to(device)
        checkpoint_path = output / "base_checkpoint.pt"
        origin = {"reused": reuse_checkpoint is not None, "resumed": resume_checkpoint is not None}
        if reuse_checkpoint is not None:
            progress("checkpoint_load")
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
            progress("base_training")
            loader = make_loader(arrays, protocol, normalization, range(50), "train", config, augment=True)
            if resume_checkpoint is not None:
                origin["resume_input_sha256"] = sha256_file(Path(resume_checkpoint))
            history, change = train_base(model, loader, config, device, output, progress=progress,
                checkpoint_context={"class_order": protocol["class_order"], "normalization": normalization,
                                    "dataset_revision": arrays.source["revision"]},
                resume_checkpoint=resume_checkpoint, stop_after_epoch=stop_after_epoch)
            origin["training_executed_this_run"] = True
        progress("checkpoint_reload")
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
        enabled = config.get("feature_expansion", True)
        expansion = (np.random.default_rng(config["seed"] + 1000).normal(size=(64, config["expansion_dim"]))
                     if enabled else None)
        expansion_before = hashlib.sha256(expansion.tobytes()).hexdigest() if enabled else None
        mode = config.get("mode", "smoke")
        stage_timings = {"feature_extraction_seconds": 0.0}
        if mode == "benchmark":
            def train_batches(phase):
                size = config.get("base_statistics_batch_size", 1024) if phase == 0 else config["analytic_batch_size"]
                loader = make_loader(arrays, protocol, normalization, protocol["phase_global_classes"][phase],
                                     "train", config, batch_size=size)
                return iter_feature_batches(model, loader, device, f"train phase {phase}", timings=stage_timings)
            def test_batches(n_classes):
                loader = make_loader(arrays, protocol, normalization, range(n_classes), "test", config)
                return iter_feature_batches(model, loader, device, f"test seen {n_classes}", timings=stage_timings)
            analytic_start = time.perf_counter()
            head, rows = run_streaming_phases(config, protocol, train_batches, test_batches, progress=progress)
            stage_timings.update(head.timings)
            stage_timings["streaming_phases_wall_seconds"] = time.perf_counter() - analytic_start
            r, w, expansion = head.r, head.w, head.expansion
            streaming_metadata = {"stored_training_samples": 0,
                                  "maximum_feature_batch_rows": head.maximum_feature_batch_rows,
                                  "persistent_analytic_state_bytes": head.state_bytes()}
        else:
            joint_features, joint_labels, rows = [], [], []
            r = w = None
            for phase in range(config["run_incremental_phases"] + 1):
                progress("analytic", phase=phase)
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
        if frozen_before != frozen_after or expansion_before != (hashlib.sha256(expansion.tobytes()).hexdigest() if expansion is not None else None):
            raise AssertionError("Backbone/BatchNorm hoặc expansion đã thay đổi")
        if not np.isfinite(r).all() or not np.isfinite(w).all():
            raise FloatingPointError("Trạng thái analytic không hữu hạn")
        state = {"inverse_gram": r, "weights": w, "gamma": config["gamma"],
                 "class_order": np.asarray(protocol["class_order"]), "n_classes": w.shape[1]}
        if expansion is not None:
            state["expansion_matrix"] = expansion
        np.savez_compressed(output / "analytic_state.npz", **state)
        checks = {"data_audit_passed": audit["passed"], "checkpoint_reload_exact": reload_error == 0,
                  "finite_training_loss": all(np.isfinite(row["train_loss"]) for row in history),
                  "base_parameters_updated": change > 0, "backbone_batchnorm_and_expansion_frozen": True,
                  "finite_analytic_state": True,
                  "all_seen_test_classes_evaluated": True,
                  "final_columns_correct": w.shape[1] == 50 + 10*config["run_incremental_phases"]}
        if mode == "smoke":
            checks["analytic_minibatch_matches_joint"] = True
        else:
            checks["streaming_phases_complete"] = len(rows) == config["run_incremental_phases"] + 1
        provenance["base_checkpoint_sha256"] = sha256_file(checkpoint_path)
        report = {"experiment": config["experiment"], "description": "CIFAR-100 pipeline; experiment scope is defined by config",
                  "mode": mode, "joint_reference_status": "checked" if mode == "smoke" else "not_run",
                  "analytic_feature_dim": w.shape[0], "feature_expansion_enabled": enabled,
                  "config": config, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "device": str(device),
                    "packages": {name: importlib.metadata.version(name) for name in ("huggingface_hub", "pyarrow", "Pillow", "threadpoolctl")}},
                  "provenance": provenance, "checkpoint_origin": origin,
                  "dataset_source": arrays.source, "training_history": history,
                  "stem_weight_change_l2": change, "checkpoint_reload_max_logit_error": reload_error,
                  "backbone_sha256_before": frozen_before, "backbone_sha256_after": frozen_after,
                  "expansion_sha256": expansion_before, "results": rows, "checks": checks,
                  "verification_passed": all(checks.values()), "seconds_this_run": time.perf_counter()-start}
        if mode == "benchmark":
            report["environment"]["packages"]["scipy"] = importlib.metadata.version("scipy")
        report["metric_summary"] = summarize(rows, config)
        if mode == "benchmark":
            report["streaming"] = streaming_metadata
            report["stage_timings"] = stage_timings
    return report


def main():
    """CLI từ gốc repo; checkpoint reuse là tùy chọn, mặc định huấn luyện base mới."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/cifar100_smoke.json")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/local/cifar100_smoke")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--reuse-checkpoint", type=Path)
    checkpoint_group.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--mode", choices=["smoke", "benchmark"])
    parser.add_argument("--expansion-dim", type=int)
    parser.add_argument("--run-phases", type=int)
    parser.add_argument("--no-feature-expansion", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--entity", default="hsangg-2006")
    parser.add_argument("--project", default="acil-reproduction")
    args = parser.parse_args()
    overrides = {key: value for key, value in (("device", args.device), ("mode", args.mode),
                 ("expansion_dim", args.expansion_dim), ("run_incremental_phases", args.run_phases)) if value is not None}
    if args.no_feature_expansion:
        overrides["feature_expansion"] = False
    run(None, data_dir=args.data_dir, output_dir=args.output_dir, reuse_checkpoint=args.reuse_checkpoint,
        log_wandb=args.wandb, entity=args.entity, project=args.project,
        config_path=args.config, config_overrides=overrides, resume_checkpoint=args.resume_checkpoint,
        stop_after_epoch=args.stop_after_epoch)


if __name__ == "__main__":
    main()
