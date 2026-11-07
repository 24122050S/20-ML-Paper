"""Regression tests cho điều kiện số học và failure state của comparison runner."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scripts.compare_author_acil import comparison


class AuthorComparisonTests(unittest.TestCase):
    def test_comparison_checks_predictions_and_numeric_tolerance(self):
        r = np.eye(2); w = np.array([[1., 0.], [0., 1.]])
        scores = np.array([[2., 1.], [1., 2.]])
        self.assertTrue(comparison(r,w,r,w,scores,scores)['passed'])
        perturbed = w.copy(); perturbed[0,0] += .01
        result = comparison(r,perturbed,r,w,scores,scores)
        self.assertFalse(result['passed'])
        self.assertTrue(result['checks']['same_predictions'])
        swapped = scores[:,::-1]
        result = comparison(r,w,r,w,swapped,scores)
        self.assertEqual(result['prediction_disagreements'],2)
        self.assertFalse(result['passed'])

    def test_input_failure_replaces_old_complete_and_keeps_attempts(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'results.json').write_text(json.dumps({'status':'complete','verification_passed':True,'attempt_id':'old'}))
            command = [sys.executable,'-B',str(root/'scripts/compare_author_acil.py'),
                '--author-root',str(directory/'missing-author'), '--r3-root',str(directory/'missing-r3'),
                '--checkpoint',str(directory/'missing.pt'),'--output-dir',str(directory)]
            result = subprocess.run(command,capture_output=True,text=True,timeout=45)
            self.assertNotEqual(result.returncode,0)
            report = json.loads((directory/'results.json').read_bytes())
            progress = json.loads((directory/'progress.json').read_bytes())
            self.assertEqual(report['status'],'failed')
            self.assertFalse(report['verification_passed'])
            self.assertNotEqual(report['attempt_id'],'old')
            self.assertEqual(report['attempt_id'],progress['attempt_id'])
            self.assertTrue((Path(report['output_dir'])/'traceback.txt').is_file())


if __name__=='__main__': unittest.main()
