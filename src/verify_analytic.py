"""CLI kiểm chứng: python -m src.verify_analytic; mặc định chỉ chạy cục bộ."""

import argparse
import json
from pathlib import Path

from .reporting import collect_provenance, log_and_save_report
from .verification import run_verification


def main():
    """Điểm vào CLI chạy kiểm chứng số học ``VERIFY-02`` và xuất báo cáo JSON.

    Tham số:
        Không nhận tham số hàm; đọc ``--config``, ``--output``, ``--wandb``,
        ``--entity`` và ``--project`` từ dòng lệnh.

    Trả về:
        Mã thoát nguyên 0 khi kiểm chứng và lưu báo cáo thành công.
    """
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/verify_analytic_update.json")
    parser.add_argument("--output", type=Path, default=root / "reports/local/verify_02_results.json")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--entity", default="hsangg-2006")
    parser.add_argument("--project", default="acil-reproduction")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_verification(config)
    report["provenance"] = collect_provenance(root)
    report = log_and_save_report(report, args.output, log_wandb=args.wandb, entity=args.entity, project=args.project)
    for row in report["results"]:
        print(f"phase={row['phase']} W={row['weights_shape']} "
              f"relative_W={row['relative_w_error']:.3e} minibatch_W={row['minibatch_relative_w_error']:.3e}")
    print("PASS", report["experiment"], "tracking=" + report["wandb_status"])
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
