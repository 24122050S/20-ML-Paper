# Chuẩn hóa report JSON và sửa checksum CPU

Cập nhật 03/11/2026. Đây là sửa serialization/provenance; không train lại và không thay số liệu của lượt CPU.

## Nguyên nhân và cách sửa

`Path.write_text` trước đây dùng newline mặc định của hệ điều hành. Trên Windows, chuỗi LF thành bytes CRLF; Git chuẩn hóa về LF theo `.gitattributes`. Hai file parse thành JSON giống hệt nhưng SHA-256 khác nhau.

`src/reporting.py::save_json` hiện ghi UTF-8 không BOM, ép `newline="\n"` và giữ một LF cuối file. Bộ ghi trạng thái trong `src/run_attempt.py` đã ép LF từ PR #7. Notebook 03 được đóng gói lại với source mới; notebook 02 và bundle của lượt Kaggle Version 2 được giữ nguyên.

## Checksum trong manifest CPU

| Trường | File được băm | SHA-256 |
|---|---|---|
| `results_sha256` | JSON được công bố trong Git, UTF-8/LF | `31b43db9b024f3e55895ff6d76e0475afc63e115ec9efdf2525866d4717a9b10` |
| `results_original_raw_sha256` | JSON raw local ban đầu, UTF-8/CRLF | `c64bfac8d21635938e0a08d9732556d28b9ce84ea45594c5f2e6b3127850223e` |

[Manifest CPU](../reports/CIFAR-SMOKE-01-provenance.json) bổ sung serialization, ngày sửa và lý do. Các trường execution/source/tracking/unit_tests của lượt cũ giữ nguyên. Blob [JSON kết quả CPU](../reports/cifar100_smoke_results.json) không thay đổi so với main trước bản sửa; không gán source mới hay lượt training mới cho kết quả cũ.

Bản CRLF gốc được giữ trên máy tại `reports/local/checksum_original/cifar100_smoke_results.raw_crlf.json`, bị Git bỏ qua. File CPU trong working tree đã được chuẩn LF để khớp file công bố.

## Kiểm tra

- Test byte-level mới tái hiện lỗi CRLF trên Windows trước bản sửa, sau đó đạt với UTF-8/LF; chuỗi tiếng Việt và newline bên trong giá trị JSON được giữ đúng.
- Suite đầy đủ có 38 tests đạt. Kiểm tra source bundle bảo đảm notebook 03 dùng đúng bộ ghi mới.
- SHA-256 được đối chiếu với **Git blob**, không chỉ file working tree. Nội dung JSON CPU giống bản trước; các JSON/manifest VERIFY-02 và Kaggle giữ nguyên.

Để kiểm manifest sau khi checkout commit đã sửa, chạy đoạn Python sau tại gốc repo:

```python
import hashlib
import json
from pathlib import Path
import subprocess

manifest = json.loads(Path("reports/CIFAR-SMOKE-01-provenance.json").read_text(encoding="utf-8"))
blob = subprocess.check_output(["git", "show", "HEAD:reports/" + manifest["results_file"]])
assert hashlib.sha256(blob).hexdigest() == manifest["results_sha256"]
assert b"\r\n" not in blob
print("Checksum JSON CPU công bố: khớp")
```
