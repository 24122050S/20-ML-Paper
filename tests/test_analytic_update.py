"""Kiểm tra joint/phase/minibatch bằng nghiệm tham chiếu và dữ liệu nhiều tình huống."""

import json
from pathlib import Path
import unittest

import numpy as np

from src.analytic import initialize, incremental_update, joint_ridge
from src.verification import make_synthetic_phases, run_verification


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/verify_analytic_update.json").read_text(encoding="utf-8"))


class AnalyticTests(unittest.TestCase):
    def test_three_phases_and_minibatches_match_joint_across_seeds(self):
        for seed in (7, 42, 2026):
            with self.subTest(seed=seed):
                report = run_verification({**CONFIG, "seed": seed})
                self.assertTrue(report["verification_passed"])
                self.assertEqual([r["weights_shape"] for r in report["results"]], [[24,2],[24,4],[24,6]])
                self.assertTrue(all(r["minibatch_prediction_disagreements"] == 0 for r in report["results"]))

    def test_two_chunks_of_one_phase_do_not_add_duplicate_class_columns(self):
        base, new, _ = make_synthetic_phases(CONFIG)
        r, w = initialize(base[0], base[1], gamma=1.0)
        r, w = incremental_update(r, w, new[0][:20], new[1][:20])
        self.assertEqual(w.shape, (24,3))
        r, w = incremental_update(r, w, new[0][20:], new[1][20:])
        self.assertEqual(w.shape, (24,4))
        expected = joint_ridge(np.vstack([base[0],new[0]]), np.concatenate([base[1],new[1]]), gamma=1.0)
        np.testing.assert_allclose(w,expected,rtol=1e-8,atol=1e-10)

    def test_repeated_old_classes_keep_same_columns(self):
        rng = np.random.default_rng(9)
        h = rng.normal(size=(30,5))
        labels = np.tile([0,1,2],10)
        r,w = initialize(h[:12],labels[:12],gamma=0.2)
        r,w = incremental_update(r,w,h[12:],labels[12:])
        self.assertEqual(w.shape,(5,3))
        np.testing.assert_allclose(w,joint_ridge(h,labels,gamma=0.2),rtol=1e-9,atol=1e-11)

    def test_update_does_not_mutate_old_state(self):
        base,new,_ = make_synthetic_phases(CONFIG)
        r,w = initialize(base[0],base[1])
        old_r,old_w = r.copy(),w.copy()
        r_new,w_new = incremental_update(r,w,new[0],new[1])
        np.testing.assert_array_equal(r,old_r)
        np.testing.assert_array_equal(w,old_w)
        self.assertFalse(np.shares_memory(r,r_new))
        self.assertFalse(np.shares_memory(w,w_new))

    def test_positive_gamma_handles_rank_deficient_features(self):
        rng=np.random.default_rng(3)
        raw=rng.normal(size=(18,2))
        h=np.column_stack([raw,raw[:,0],np.zeros(18)])
        labels=np.repeat([0,1,2],6)
        r,w=initialize(h[:6],labels[:6],gamma=0.1)
        r,w=incremental_update(r,w,h[6:],labels[6:])
        np.testing.assert_allclose(w,joint_ridge(h,labels,gamma=0.1),rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(r@(h.T@h+0.1*np.eye(4)),np.eye(4),rtol=1e-9,atol=1e-11)

    def test_invalid_inputs_are_rejected(self):
        h=np.ones((3,2)); labels=np.array([0,0,1])
        for gamma in (0,-1,float("nan"),float("inf")):
            with self.subTest(gamma=gamma),self.assertRaises(ValueError):initialize(h,labels,gamma)
        for bad_h,bad_y in ((np.empty((0,2)),np.array([],dtype=int)),(h,labels[:2]),(h,np.array([0,-1,1])),(h,labels.astype(float)),(h*np.nan,labels)):
            with self.subTest(shape=bad_h.shape),self.assertRaises(ValueError):joint_ridge(bad_h,bad_y)
        r,w=initialize(h,labels)
        with self.assertRaises(ValueError):incremental_update(r,w,np.ones((2,3)),np.array([0,1]))
        with self.assertRaises(ValueError):incremental_update(r,w,h,labels,n_classes=1)
        with self.assertRaises(ValueError):joint_ridge(h,labels,n_classes=True)


if __name__ == "__main__":
    unittest.main()
