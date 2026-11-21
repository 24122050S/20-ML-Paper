"""Đối chiếu SHA-256 của tệp công bố trong các manifest ở reports/."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


REPORTS = Path(__file__).resolve().parents[1] / "reports"


def artifact_hashes(manifest):
    """Đọc bảng artifacts hoặc cặp tên tệp và checksum từ manifest cũ."""
    artifacts = manifest.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a JSON object")
    if artifacts:
        return {
            name: entry.get("sha256") if isinstance(entry, dict) else entry
            for name, entry in artifacts.items()
        }
    for field in ("result", "results"):
        if f"{field}_file" in manifest:
            return {manifest[f"{field}_file"]: manifest.get(f"{field}_sha256")}
    raise ValueError("manifest has no published file checksums")


def main():
    """Kiểm tra tất cả manifest hoặc danh sách được truyền từ dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="*", type=Path, help="paths to provenance JSON files")
    args = parser.parse_args()
    manifests = args.manifests or sorted(REPORTS.glob("*-provenance.json"))
    if not manifests:
        parser.error(f"no provenance manifests found in {REPORTS}")

    checked = 0
    failures = 0
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_bytes())
            if not isinstance(data, dict):
                raise ValueError("manifest must be a JSON object")
            artifacts = artifact_hashes(data)
        except (OSError, ValueError, TypeError) as exc:
            print(f"{manifest}: {exc}", file=sys.stderr)
            failures += 1
            continue

        reports = manifest.resolve().parent
        for name, expected in artifacts.items():
            try:
                if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                    raise ValueError("missing or invalid SHA-256")
                path = (reports / name).resolve()
                if not path.is_relative_to(reports):
                    raise ValueError("artifact path must stay inside the manifest directory")
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected.lower():
                    raise ValueError(f"SHA-256 mismatch (expected {expected}, got {actual})")
            except (OSError, ValueError, TypeError) as exc:
                print(f"{manifest.name} / {name}: {exc}", file=sys.stderr)
                failures += 1
            else:
                checked += 1

    print(f"{checked} checksum(s) matched across {len(manifests)} manifest(s), {failures} error(s).")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
