"""Tổng hợp đúng ba seed R1 đã nghiệm thu; không chọn seed tốt hoặc dùng SD population."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
METRICS = ("average_incremental_accuracy_pct", "final_accuracy_pct",
           "base_forgetting_pp", "base_accuracy_delta_pp")


def aggregate(reports):
    """Đối chiếu input/provenance trước khi tính metric trong từng seed rồi mean±SD."""
    runs = []
    for seed in SEEDS:
        path = reports / f"benchmark_seed{seed}_kaggle_results.json"
        raw = path.read_bytes()
        report = json.loads(raw)
        manifest_name = f"R1-SEED{seed}-KAGGLE-01-provenance.json"
        manifest = json.loads((reports / manifest_name).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == manifest["artifacts"][path.name]["sha256"]
        assert report["status"] == "complete" and report["verification_passed"]
        assert all(report["checks"].values())
        assert report["config"]["seed"] == seed and manifest["metrics"]["seed"] == seed
        assert report["attempt_id"] == manifest["attempt_id"]
        assert manifest["tracking_sync"]["status"] == "synced"
        assert manifest["tracking_sync"]["verified_state"] == "finished"
        assert manifest["tracking_sync"]["public_api_verified"]
        assert manifest["checkpoint"]["fresh_training"] and manifest["checkpoint"]["completed_epoch"] == 160
        assert report["checkpoint_origin"] == {"reused": False, "resumed": False, "training_executed_this_run": True}
        assert report["config"]["epochs"] == 160 and report["config"]["expansion_dim"] == 8000
        assert report["config"]["incremental_phases"] == report["config"]["run_incremental_phases"] == 5
        assert report["config"]["gamma"] == 0.1 and report["mode"] == "benchmark"
        assert report["streaming"]["stored_training_samples"] == 0 and report["joint_reference_status"] == "not_run"
        summary = report["metric_summary"]
        assert summary == manifest["metrics"] and summary["complete_protocol"]
        assert summary["n_completed_seeds"] == 1 and summary["between_seed_sd"] is None
        rows = report["results"]
        assert len(rows) == 6 and [row["phase"] for row in rows] == list(range(6))
        for phase, row in enumerate(rows):
            classes = 50 + phase * 10
            assert row["weights_shape"] == [8000, classes]
            assert row["n_classes"] == classes and row["test_samples_all_seen"] == classes * 100
            assert row["checks_passed"]
        assert np.isclose(summary["average_incremental_accuracy_pct"], np.mean([row["accuracy_pct"] for row in rows]), rtol=0, atol=1e-12)
        assert summary["final_accuracy_pct"] == rows[-1]["accuracy_pct"]
        assert np.isclose(summary["base_forgetting_pp"], rows[0]["base_accuracy_pct"] - rows[-1]["base_accuracy_pct"], rtol=0, atol=1e-12)
        assert all(np.isfinite(summary[key]) for key in METRICS)
        runs.append({"seed": seed, "report": report, "manifest": manifest, "manifest_file": manifest_name})

    first = runs[0]["report"]
    recipe = lambda run: {key: value for key, value in run["config"].items() if key not in ("seed", "torch_threads")}
    assert all(recipe(run["report"]) == recipe(first) for run in runs)
    assert all(run["report"]["provenance"]["source_sha256"] == first["provenance"]["source_sha256"] for run in runs)
    assert all(run["report"]["dataset_source"] == first["dataset_source"] for run in runs)
    for extractor in (lambda run: run["report"]["provenance"]["base_checkpoint_sha256"],
                      lambda run: run["report"]["backbone_sha256_before"],
                      lambda run: run["report"]["expansion_sha256"]):
        assert len({extractor(run) for run in runs}) == 3, "Backbone/checkpoint/FE must be independent for the three seeds"

    def stats(values):
        values = [float(value) for value in values]
        mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
        # Kiểm chéo bằng thư viện chuẩn, không dùng lại cùng công thức NumPy.
        assert np.isclose(mean, statistics.mean(values), rtol=0, atol=1e-12)
        assert np.isclose(sd, statistics.stdev(values), rtol=0, atol=1e-12)
        return {"mean": mean, "sample_sd": sd, "n": 3, "ddof": 1, "values_by_seed": dict(zip(map(str, SEEDS), values))}

    metrics = {key: stats([run["report"]["metric_summary"][key] for run in runs]) for key in METRICS}
    phases = []
    for phase in range(6):
        entry = {"phase": phase, "n_classes": 50 + 10 * phase, "test_samples_all_seen": (50 + 10 * phase) * 100}
        for key in ("accuracy_pct", "base_accuracy_pct", "new_class_accuracy_pct"):
            values = [run["report"]["results"][phase][key] for run in runs]
            entry[key] = None if all(value is None for value in values) else stats(values)
        phases.append(entry)
    # Mean của các A-bar theo seed phải bằng mean của đường cong trung bình sáu phase.
    assert np.isclose(metrics[METRICS[0]]["mean"], np.mean([row["accuracy_pct"]["mean"] for row in phases]), rtol=0, atol=1e-12)
    average = metrics["average_incremental_accuracy_pct"]["mean"]
    forgetting = metrics["base_forgetting_pp"]["mean"]
    inputs = [{"seed": run["seed"], "attempt_id": run["report"]["attempt_id"],
               "results_file": f"benchmark_seed{run['seed']}_kaggle_results.json", "manifest_file": run["manifest_file"],
               "results_sha256": run["manifest"]["artifacts"][f"benchmark_seed{run['seed']}_kaggle_results.json"]["sha256"],
               "checkpoint_sha256": run["manifest"]["checkpoint"]["sha256"],
               "kaggle_url": run["manifest"]["kaggle"]["url"],
               "wandb_run_id": run["manifest"]["tracking_sync"]["wandb_run_id"],
               "wandb_run_url": run["manifest"]["tracking_sync"]["wandb_run_url"]} for run in runs]
    return {"status": "complete", "scope": "R1 CIFAR100/K5, seeds42/43/44 only; R2/R3 tracked separately",
            "protocol_version": "ACIL-C100-v1.0", "metric_version": "ACIL-metrics-v1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS), "n_completed_seeds": 3, "ddof": 1,
            "statistical_unit": "one complete seed with its own class order/backbone/FE; not one phase",
            "metrics": metrics, "phase_metrics": phases, "inputs": inputs,
            "source_sha256": first["provenance"]["source_sha256"],
            "recipe": recipe(first), "torch_threads_by_seed": {str(run["seed"]): run["report"]["config"]["torch_threads"] for run in runs},
            "paper_comparison": {"paper_average_incremental_accuracy_pct": 66.30, "paper_base_forgetting_pp": 9.00,
                "paper_final_accuracy_pct": None, "average_deviation_pp": average - 66.30,
                "average_relative_deviation_pct": 100 * (average - 66.30) / 66.30,
                "base_forgetting_deviation_pp": forgetting - 9.00,
                "reference": "Table1 CIFAR100/K5 as pinned in docs/protocol_chot.md and docs/metrics.md"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/r1_cifar100_k5_summary.json")
    args = parser.parse_args()
    def save(value):
        raw = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(raw, encoding="utf-8", newline="\n")
        os.replace(temporary, args.output)
    save({"status": "running", "expected_seeds": list(SEEDS), "n_completed_seeds": None})
    try:
        result = aggregate(args.reports_dir)
    except BaseException as error:
        save({"status": "failed", "expected_seeds": list(SEEDS), "n_completed_seeds": None,
              "error_type": type(error).__name__})
        raise
    save(result)
    print("R1 complete: seeds42/43/44, mean and sample_SD(ddof=1),", args.output)


if __name__ == "__main__":
    main()
