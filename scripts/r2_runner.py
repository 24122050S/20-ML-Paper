"""Điều phối R2 từ core ACIL đã nghiệm thu; không huấn luyện lại backbone."""

from pathlib import Path
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
import uuid
from unittest import mock


CASES = (
    {"key": "no_fe_g0p1", "feature_expansion": False, "expansion_dim": 64, "gamma": 0.1},
    {"key": "fe8000_g0p01", "feature_expansion": True, "expansion_dim": 8000, "gamma": 0.01},
    {"key": "fe8000_g0p001", "feature_expansion": True, "expansion_dim": 8000, "gamma": 0.001},
)


def sha256_file(path):
    """Băm theo chunk để không giữ thêm một bản sao ma trận lớn trong RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_suite(*, seed, checkpoint, reference, project_root, working, test_proof, device="cuda"):
    """Chạy3 case; chỉ complete khi đúng checkpoint/FE, đủ6 phase và đã log offline."""
    project_root, working = Path(project_root), Path(working)
    sys.path.insert(0, str(project_root))
    import numpy as np
    import torch
    import wandb
    from src import cifar_smoke
    from src.cifar_data import make_protocol
    from src.profile_analytic import MemoryMonitor
    from src.reporting import save_json

    ready_path = working / f"r2_seed{seed}_ready.json"
    state = {"status": "running", "verification_passed": False, "seed": seed,
             "suite_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10],
             "scope": "R2:3 new cases; FE8000/gamma0.1 baseline reused from R1",
             "expected_cases": [case["key"] for case in CASES], "cases": [], "tests": test_proof,
             "baseline": reference, "stage": "input_validation"}
    save_json(state, ready_path)
    try:
        if seed not in (42, 43, 44) or reference["seed"] != seed:
            raise ValueError("Seed và R1 reference không khớp")
        if device == "cuda" and (not torch.cuda.is_available() or not test_proof.get("cuda_resume_passed")):
            raise RuntimeError("R2 cần CUDA và GPU test proof đạt")
        if checkpoint is None:
            matches = [path for path in sorted(Path("/kaggle/input").rglob("base_checkpoint.pt"))
                       if sha256_file(path) == reference["checkpoint_sha256"]]
            if not matches:
                raise ValueError("Không có checkpoint Input đúng hash R1; không được train fallback")
            checkpoint = matches[0]
            state["matching_input_checkpoint_files"] = len(matches)
        checkpoint = Path(checkpoint)
        if sha256_file(checkpoint) != reference["checkpoint_sha256"]:
            raise ValueError("Checkpoint Input không đúng hash R1; không được train fallback")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["checkpoint_format_version"] == 2 and saved["training_complete"]
        assert saved["completed_epoch"] == saved["target_epochs"] == 160 and saved["config"]["seed"] == seed
        assert saved["class_order"] == make_protocol(seed, 5)["class_order"]
        shared_context = {key: saved[key] for key in ("class_order", "normalization", "dataset_revision")}
        del saved
        for case in CASES:
            key, dimension = case["key"], case["expansion_dim"]
            state.update(stage="numeric", current_case=key)
            save_json(state, ready_path)
            print(f"R2 START seed={seed} case={key} checkpoint={reference['checkpoint_sha256']}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with mock.patch.object(cifar_smoke, "train_base", side_effect=AssertionError("R2 must never train a backbone")) as blocked:
                with MemoryMonitor(device) as resources:
                    report = cifar_smoke.run(None, config_path=project_root / "configs/cifar100_benchmark_k5.json",
                        config_overrides={"experiment": "ACIL-R2-C100-K5", "purpose": "ablation", "seed": seed,
                            "device": device, "torch_threads": 4, "run_incremental_phases": 5,
                            **{field: case[field] for field in ("gamma", "feature_expansion", "expansion_dim")}},
                        data_dir=working / "cifar-data", output_dir=working / f"r2_seed{seed}" / key,
                        reuse_checkpoint=checkpoint, log_wandb=False)
                blocked.assert_not_called()
            assert report["status"] == "complete" and report["verification_passed"] and all(report["checks"].values())
            assert report["checkpoint_origin"]["reused"] and not report["checkpoint_origin"]["training_executed_this_run"]
            assert report["provenance"]["base_checkpoint_sha256"] == reference["checkpoint_sha256"]
            assert report["backbone_sha256_before"] == report["backbone_sha256_after"] == reference["backbone_sha256"]
            assert report["provenance"]["source_sha256"] == reference["source_sha256"]
            assert report["dataset_source"] == reference["dataset_source"]
            assert len(report["training_history"]) == 160 and len(report["results"]) == 6
            assert report["metric_summary"]["complete_protocol"] and report["metric_summary"]["seed"] == seed
            assert report["streaming"]["stored_training_samples"] == 0 and report["joint_reference_status"] == "not_run"
            assert all(row["weights_shape"] == [dimension, 50 + 10 * phase] and row["test_samples_all_seen"] == (50 + 10 * phase) * 100
                       for phase, row in enumerate(report["results"]))
            assert report["expansion_sha256"] == (reference["expansion_sha256"] if case["feature_expansion"] else None)
            assert report["analytic_feature_dim"] == dimension
            artifact_dir = Path(report["output_dir"])
            protocol = json.loads((artifact_dir / "protocol.json").read_text(encoding="utf-8"))
            audit = json.loads((artifact_dir / "data_audit.json").read_text(encoding="utf-8"))
            assert protocol["class_order"] == shared_context["class_order"]
            assert audit["normalization"] == shared_context["normalization"] and audit["source"]["revision"] == shared_context["dataset_revision"]
            # Kiểm NPZ độc lập sau numeric timer; không giải oracle joint d8000.
            state.update(stage="artifact_verification")
            save_json(state, ready_path)
            with np.load(artifact_dir / "analytic_state.npz") as archive:
                inverse, weights = archive["inverse_gram"], archive["weights"]
                assert inverse.shape == (dimension, dimension) and weights.shape == (dimension, 100)
                assert inverse.dtype == weights.dtype == np.float64 and np.isfinite(inverse).all() and np.isfinite(weights).all()
                assert np.array_equal(archive["class_order"], shared_context["class_order"]) and float(archive["gamma"]) == case["gamma"]
                for start in range(0, dimension, 256):
                    end = min(start + 256, dimension)
                    assert np.allclose(inverse[start:end], inverse[:, start:end].T, rtol=1e-6, atol=1e-8)
                if case["feature_expansion"]:
                    expansion = archive["expansion_matrix"]
                    assert expansion.shape == (64, 8000) and np.isfinite(expansion).all()
                    assert hashlib.sha256(expansion.tobytes()).hexdigest() == reference["expansion_sha256"]
                    del expansion
                else:
                    assert "expansion_matrix" not in archive.files
                del inverse, weights
            artifact_proof = {"status": "verified", "verification_location": "Kaggle CPU after numeric timer",
                "checkpoint_sha256": sha256_file(artifact_dir / "base_checkpoint.pt"),
                "analytic_state_sha256": sha256_file(artifact_dir / "analytic_state.npz"),
                "inverse_gram_shape": [dimension, dimension], "weights_shape": [dimension, 100],
                "finite_float64_and_symmetric_r": True, "context_and_fe_match_r1": True,
                "training_calls": 0, "oracle_joint_run": False}
            save_json(artifact_proof, artifact_dir / "r2_artifact_proof.json")
            state.update(stage="offline_tracking")
            save_json(state, ready_path)
            with wandb.init(entity="hsangg-2006", project="acil-reproduction", mode="offline", dir=str(working),
                    name=f"R2-seed{seed}-{key}", job_type="ablation",
                    tags=["R2", f"seed{seed}", key, "K5", "offline-to-sync"],
                    config={**report["config"], "source_sha256": report["provenance"]["source_sha256"],
                        "kaggle_attempt_id": report["attempt_id"], "r2_case_key": key,
                        "baseline_r1_run_id": reference["wandb_run_id"],
                        "base_checkpoint_sha256": reference["checkpoint_sha256"],
                        "logging_origin": "Kaggle offline; sync exact payload after run"}) as tracked:
                for row in report["results"]:
                    tracked.log({field: value for field, value in row.items() if isinstance(value, (int, float, bool))}, step=row["phase"])
                tracked.summary.update(report["metric_summary"])
                tracked.summary.update({"verification_passed": True, "artifact_checks_passed": True,
                    "training_executed_this_run": False, "numeric_pipeline_seconds": report["seconds_this_run"],
                    "reused_base_training_seconds": sum(row["seconds"] for row in report["training_history"]),
                    "profile/peak_cpu_rss_bytes": resources.result["peak_cpu_rss_bytes"],
                    "profile/peak_cuda_allocated_bytes": resources.result["peak_cuda_allocated_bytes"]})
                offline_id, offline_dir = tracked.id, str(Path(tracked.dir).parent)
            wrapper = {"status": "complete", "verification_passed": True, "seed": seed, "case_key": key,
                "suite_id": state["suite_id"], "attempt_id": report["attempt_id"], "report": report,
                "artifact_proof": artifact_proof,
                "memory": {"scope": "case numeric pipeline; artifact check and logging excluded", **resources.result},
                "tracking": {"status": "offline_pending_sync", "offline_run_id": offline_id, "offline_directory": offline_dir}}
            save_json(wrapper, artifact_dir / "r2_case_ready.json")
            state["cases"].append(wrapper)
            save_json(state, ready_path)
            print(f"R2 COMPLETE seed={seed} case={key} Abar={report['metric_summary']['average_incremental_accuracy_pct']:.6f} final={report['metric_summary']['final_accuracy_pct']:.4f} offline={offline_id}", flush=True)
            del report, wrapper
            gc.collect()
        assert [item["case_key"] for item in state["cases"]] == state["expected_cases"]
        state.update(status="complete", verification_passed=True, stage="complete", current_case=None)
        save_json(state, ready_path)
        print(f"R2 SUITE COMPLETE seed={seed} cases=3 path={ready_path}", flush=True)
        return state
    except BaseException as error:
        state.update(status="failed", verification_passed=False, error_type=type(error).__name__)
        save_json(state, ready_path)
        raise
