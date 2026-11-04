"""Hồi quy bundle triển khai: snapshot cũ bất biến, bản mới dùng đúng config."""

import ast
import base64
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import zipfile


HAS_NBFORMAT = importlib.util.find_spec("nbformat") is not None
if HAS_NBFORMAT:
    import nbformat
    from src.notebook_bundle import build


@unittest.skipUnless(HAS_NBFORMAT, "Cài requirements-dev.txt để kiểm tra notebook distribution")
class NotebookDistributionTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "notebooks/02_cifar100_smoke.ipynb").is_file(),
                         "Snapshot 02 chỉ có trong checkout repo, không nằm trong source bundle")
    def test_cannot_overwrite_published_notebook02(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("02_cifar100_smoke.ipynb", "03_cifar100_safe_attempts.ipynb"):
            old = root / "notebooks" / name
            before = old.read_bytes()
            with self.assertRaises(ValueError):
                build(old)
            self.assertEqual(old.read_bytes(), before)

    def test_bundle_has_current_attempt_code_and_tests(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = build(Path(directory) / "safe.ipynb")
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    compile(cell.source, "notebook-cell", "exec")
            tree = ast.parse(notebook.cells[1].source)
            encoded = next(n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr == "b64decode")
            raw = base64.b64decode(encoded)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), notebook.metadata.acil_source_bundle_sha256)
            with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
                self.assertIn("src/run_attempt.py", bundle.namelist())
                self.assertIn("tests/test_run_attempts.py", bundle.namelist())
                for name in bundle.namelist():
                    self.assertEqual(bundle.read(name), (root / name).read_bytes().replace(b"\r\n", b"\n"))

    def test_final_cell_accepts_one_or_five_phases_and_configured_feature_dimension(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = build(Path(directory) / "safe.ipynb")
            final_cell = nbformat.read(path, as_version=4).cells[-1].source
            for phases, dimension in ((1, 256), (5, 128)):
                rows = [{"phase": i, "n_classes": 50+10*i, "weights_shape": [dimension, 50+10*i],
                         "test_samples_all_seen": (50+10*i)*100, "accuracy_pct": 0.0,
                         "relative_w_error": 0.0, "prediction_disagreements": 0}
                        for i in range(phases+1)]
                report = {"status": "complete", "verification_passed": True,
                          "config": {"incremental_phases": 5, "run_incremental_phases": phases,
                                     "expansion_dim": dimension}, "results": rows,
                          "attempt_id": "test", "output_dir": "test-artifacts",
                          "wandb_status": "not_logged", "wandb_run_url": None}
                exec(compile(final_cell, "notebook-final-cell", "exec"), {"report": report})
            report["config"].update(feature_expansion=False, expansion_dim=8000)
            report["results"][-1]["weights_shape"] = [64, 100]
            exec(compile(final_cell, "notebook-final-cell", "exec"), {"report": report})


if __name__ == "__main__":
    unittest.main()
