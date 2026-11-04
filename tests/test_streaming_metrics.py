"""Kiểm objective streaming, không giữ lịch sử và các metric đã chốt."""

import gc
import importlib.util
import unittest
from unittest import mock
import weakref

import numpy as np

from src.analytic import initialize, initialize_batches, joint_ridge
from src.metrics import summarize
from src.streaming import StreamingHead, run_streaming_phases


@unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'Cần SciPy để kiểm streaming base')
class StreamingTests(unittest.TestCase):
    def config(self,expand=True):
        return {'seed':42,'expansion_dim':12,'gamma':0.1,'feature_expansion':expand,
                'mode':'benchmark','incremental_phases':5,'run_incremental_phases':1}
    def test_base_batches_match_direct_and_do_not_retain_inputs(self):
        rng=np.random.default_rng(7)
        h=rng.normal(size=(90,12));labels=np.arange(90)%4
        actual=initialize_batches(((h[i:i+13],labels[i:i+13]) for i in range(0,90,13)),12,4,0.1)
        expected=initialize(h,labels,0.1,4)
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
        refs=[]
        def raw_batches():
            for i in range(4):
                raw=rng.normal(size=(10,64));refs.append(weakref.ref(raw))
                yield raw,np.arange(10)%4
        head=StreamingHead(self.config())
        self.assertEqual(head.fit_base(raw_batches(),4),40)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
    def test_no_fe_is_raw_feature_including_negative_values(self):
        raw=np.arange(128,dtype=np.float64).reshape(2,64)-100
        head=StreamingHead(self.config(False))
        self.assertIsNone(head.expansion)
        self.assertEqual(head.dimension,64)
        np.testing.assert_array_equal(head.transform(raw),raw)
    def test_benchmark_never_calls_joint_and_rows_have_null_reference_metrics(self):
        rng=np.random.default_rng(8)
        raw=[rng.normal(size=(25000,64)),rng.normal(size=(5000,64))]
        labels=[np.arange(25000)%50,50+np.arange(5000)%10]
        test=rng.normal(size=(6000,64));test_labels=np.arange(6000)%60
        def train_batches(phase):
            for i in range(0,len(raw[phase]),128):yield raw[phase][i:i+128],labels[phase][i:i+128]
        def test_batches(classes):
            keep=test_labels<classes
            x,y=test[keep],test_labels[keep]
            for i in range(0,len(x),256):yield x[i:i+256],y[i:i+256]
        protocol={'phase_global_classes':[list(range(50)),list(range(50,60))]}
        with mock.patch('src.analytic.joint_ridge',side_effect=AssertionError('oracle forbidden')):
            head,rows=run_streaming_phases(self.config(),protocol,train_batches,test_batches,progress=lambda *a,**k:None)
        h=head.transform(np.vstack(raw))
        expected=joint_ridge(h,np.concatenate(labels),60,0.1)
        np.testing.assert_allclose(head.w,expected,rtol=1e-6,atol=1e-8)
        self.assertEqual(head.w.shape,(12,60))
        for row in rows:
            self.assertIsNone(row['prediction_disagreements'])
            self.assertIsNone(row['relative_w_error'])
            self.assertEqual(row['joint_reference_status'],'not_run')
        self.assertEqual(rows[-1]['analytic_batches'],40)
    def test_invalid_base_batches_are_rejected(self):
        for batches in ([],[(np.ones((3,2)),np.array([0,1,4]))],[(np.ones((3,3)),np.array([0,1,2]))]):
            with self.assertRaises(ValueError):initialize_batches(iter(batches),2,3,0.1)


class MetricTests(unittest.TestCase):
    def rows(self):
        accuracy=[33.46,29.916666666666668,27.5,25.225,23.31111111111111,22.05]
        base=[33.46,30.6,28.74,27.32,26.58,25.56]
        return [dict(phase=i,n_classes=50+10*i,test_samples_all_seen=(50+10*i)*100,
                     accuracy_pct=a,base_accuracy_pct=b,checks_passed=True) for i,(a,b) in enumerate(zip(accuracy,base))]
    def config(self,phases=5):
        return dict(seed=42,incremental_phases=5,run_incremental_phases=phases,mode='benchmark')
    def test_complete_summary_includes_base_and_signed_forgetting(self):
        result=summarize(self.rows(),self.config())
        self.assertAlmostEqual(result['average_incremental_accuracy_pct'],26.910462962962963)
        self.assertAlmostEqual(result['base_forgetting_pp'],7.9)
        self.assertAlmostEqual(result['base_accuracy_delta_pp'],-7.9)
        self.assertEqual(result['final_accuracy_pct'],22.05)
        self.assertIsNone(result['between_seed_sd'])
    def test_partial_protocol_does_not_claim_full_average(self):
        result=summarize(self.rows()[:2],self.config(1))
        self.assertFalse(result['complete_protocol'])
        self.assertIsNone(result['average_incremental_accuracy_pct'])
        self.assertIsNotNone(result['partial_average_accuracy_pct'])
    def test_missing_phase_bad_sample_count_and_nan_are_rejected(self):
        for change in (lambda rows:rows.pop(2),lambda rows:rows[-1].__setitem__('test_samples_all_seen',1000),
                       lambda rows:rows[-1].__setitem__('base_accuracy_pct',float('nan'))):
            rows=self.rows();change(rows)
            with self.assertRaises(ValueError):summarize(rows,self.config())


if __name__=='__main__':unittest.main()
