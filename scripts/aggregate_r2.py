"""R2: đủ4 hàng/3 seeds, giữ paired controls và dùng sample SD(ddof=1)."""

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
ROWS = {
    "no_fe_g0p1": {"feature_expansion": False, "dimension": 64, "gamma": 0.1, "paper_average": 52.99},
    "fe8000_g0p1": {"feature_expansion": True, "dimension": 8000, "gamma": 0.1, "paper_average": 66.30},
    "fe8000_g0p01": {"feature_expansion": True, "dimension": 8000, "gamma": 0.01, "paper_average": 66.25},
    "fe8000_g0p001": {"feature_expansion": True, "dimension": 8000, "gamma": 0.001, "paper_average": 66.23},
}
METRICS = ("average_incremental_accuracy_pct", "final_accuracy_pct", "base_forgetting_pp")


def statistics_for(values):
    values = [float(value) for value in values]
    assert len(values) == 3 and np.isfinite(values).all()
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    assert np.isclose(mean, statistics.mean(values), rtol=0, atol=1e-12)
    assert np.isclose(sd, statistics.stdev(values), rtol=0, atol=1e-12)
    return {"mean": mean, "sample_sd": sd, "n": 3, "ddof": 1, "values_by_seed": dict(zip(map(str, SEEDS), values))}


def aggregate(reports):
    """Không chọn gamma theo test; so từng hàng với đúng baseline cùng seed."""
    load = lambda path: json.loads(path.read_bytes())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    collected = {key: [] for key in ROWS}
    input_manifests = []
    for seed in SEEDS:
        baseline = load(reports / f"benchmark_seed{seed}_kaggle_results.json")
        baseline_manifest = load(reports / f"R1-SEED{seed}-KAGGLE-01-provenance.json")
        assert baseline["status"] == "complete" and baseline["config"]["seed"] == seed
        assert baseline["metric_summary"]["complete_protocol"] and len(baseline["results"]) == 6
        assert baseline["config"]["epochs"] == 160 and baseline["config"]["gamma"] == 0.1 and baseline["feature_expansion_enabled"]
        assert baseline_manifest["tracking_sync"]["verified_state"] == "finished" and baseline_manifest["tracking_sync"]["public_api_verified"]
        assert sha(reports / f"benchmark_seed{seed}_kaggle_results.json") == baseline_manifest["artifacts"][f"benchmark_seed{seed}_kaggle_results.json"]["sha256"]
        path = reports / f"r2_seed{seed}_kaggle_results.json"
        suite = load(path); manifest_name = f"R2-SEED{seed}-KAGGLE-01-provenance.json"
        manifest = load(reports / manifest_name)
        assert sha(path) == manifest["artifacts"][path.name]["sha256"]
        assert suite["status"] == "complete" and suite["verification_passed"] and suite["seed"] == seed
        assert manifest["suite_id"] == suite["suite_id"] and manifest["tracking_status"] == "synced"
        cases = {case["case_key"]: case for case in suite["cases"]}
        assert set(cases) == set(ROWS) - {"fe8000_g0p1"}
        collected["fe8000_g0p1"].append({"seed": seed, "report": baseline,
            "wandb_run_id": baseline_manifest["tracking_sync"]["wandb_run_id"],
            "wandb_run_url": baseline_manifest["tracking_sync"]["wandb_run_url"],
            "origin": "reused_R1", "kaggle_url": baseline_manifest["kaggle"]["url"]})
        for key, case in cases.items():
            setting = ROWS[key]; r = case["report"]; tracked = manifest["cases"][key]["tracking"]
            assert tracked["verified_state"] == "finished" and tracked["public_api_verified"]
            assert case["status"] == "complete" and case["verification_passed"] and r["status"] == "complete" and r["verification_passed"]
            assert r["config"]["seed"] == seed and r["config"]["gamma"] == setting["gamma"]
            assert r["config"]["feature_expansion"] == setting["feature_expansion"] and r["analytic_feature_dim"] == setting["dimension"]
            assert r["checkpoint_origin"]["reused"] and not r["checkpoint_origin"]["training_executed_this_run"]
            assert r["provenance"]["base_checkpoint_sha256"] == baseline["provenance"]["base_checkpoint_sha256"]
            assert r["backbone_sha256_before"] == r["backbone_sha256_after"] == baseline["backbone_sha256_before"]
            assert r["provenance"]["source_sha256"] == baseline["provenance"]["source_sha256"]
            assert r["dataset_source"] == baseline["dataset_source"]
            assert r["expansion_sha256"] == (baseline["expansion_sha256"] if setting["feature_expansion"] else None)
            assert r["metric_summary"]["complete_protocol"] and r["metric_summary"]["n_completed_seeds"] == 1
            assert len(r["results"]) == 6 and r["streaming"]["stored_training_samples"] == 0 and r["joint_reference_status"] == "not_run"
            assert np.isclose(r["metric_summary"]["average_incremental_accuracy_pct"], np.mean([row["accuracy_pct"] for row in r["results"]]), rtol=0, atol=1e-12)
            collected[key].append({"seed": seed, "report": r, "wandb_run_id": tracked["wandb_run_id"],
                "wandb_run_url": tracked["wandb_run_url"], "origin": "new_R2", "kaggle_url": manifest["kaggle"]["url"]})
        input_manifests.append({"seed": seed, "file": manifest_name, "sha256": sha(reports / manifest_name),
            "r1_manifest": f"R1-SEED{seed}-KAGGLE-01-provenance.json", "r1_manifest_sha256": sha(reports / f"R1-SEED{seed}-KAGGLE-01-provenance.json")})
    output_rows = []
    for key, setting in ROWS.items():
        runs = sorted(collected[key], key=lambda item: item["seed"])
        assert [run["seed"] for run in runs] == list(SEEDS)
        metrics = {name: statistics_for([run["report"]["metric_summary"][name] for run in runs]) for name in METRICS}
        deltas = [run["report"]["metric_summary"]["average_incremental_accuracy_pct"] - collected["fe8000_g0p1"][i]["report"]["metric_summary"]["average_incremental_accuracy_pct"] for i, run in enumerate(runs)]
        mean = metrics[METRICS[0]]["mean"]
        output_rows.append({"case_key": key, **setting, "metrics": metrics,
            "paired_delta_average_vs_r1_pp": statistics_for(deltas),
            "paper_comparison": {"average_deviation_pp": mean - setting["paper_average"],
                "average_relative_deviation_pct": 100 * (mean - setting["paper_average"]) / setting["paper_average"],
                "paper_final_accuracy_pct": None, "paper_base_forgetting_pp": None},
            "runs": [{key: value for key, value in run.items() if key != "report"} for run in runs]})
    return {"status": "complete", "scope": "R2:4/5 Table2 rows, gamma0 excluded; no additional backbone training",
        "seeds": list(SEEDS), "n_completed_seeds": 3, "n_new_cases": 9, "n_reused_r1_cases": 3, "ddof": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "rows": output_rows, "inputs": input_manifests,
        "statistical_unit": "seed; paired changes use same seed checkpoint, mapping, normalization and FE",
        "excluded": {"gamma0": "outside fixed core scope; no epsilon/pseudoinverse substitution"}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/r2_cifar100_k5_summary.json"); args = parser.parse_args()
    def save(value):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, args.output)
    save({"status": "running", "expected_seeds": list(SEEDS)})
    try:result = aggregate(args.reports_dir)
    except BaseException as error:
        save({"status": "failed", "expected_seeds": list(SEEDS), "error_type": type(error).__name__}); raise
    save(result); print("R2 complete:9 new cases +3 reused R1 baselines;4 rows, n=3, sample SD and paired deltas.")


if __name__ == "__main__":
    main()
