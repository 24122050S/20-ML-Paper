"""Kiểm driver R3 bằng fixture tổng hợp nhỏ; không gọi đây là benchmark CIFAR."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from src.reporting import save_json
from scripts import r3_worker as r3


class R3Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.fixture=self.root/'fixture';self.fixture.mkdir()
        rng=np.random.default_rng(5);files={}
        def store(name,value):
            np.save(self.fixture/name,value,allow_pickle=False);files[name]=r3.sha(self.fixture/name)
        for phase in range(6):
            start,end=(0,50) if phase==0 else (40+10*phase,50+10*phase)
            labels=np.repeat(np.arange(start,end,dtype=np.int64),2)
            store(f'phase{phase}_features.npy',rng.normal(size=(len(labels),64)))
            store(f'phase{phase}_labels.npy',labels)
        store('test_features.npy',rng.normal(size=(200,64)))
        store('test_labels.npy',np.repeat(np.arange(100,dtype=np.int64),2))
        store('expansion.npy',np.random.default_rng(1042).normal(size=(64,12)))
        meta={'scope':'synthetic_test_only','config':{'seed':42,'expansion_dim':12,'feature_expansion':True,
              'gamma':0.1,'torch_threads':1,'rtol':1e-6,'atol':1e-8},
              'train_counts':[100,20,20,20,20,20],'train_per_class':2,'test_per_class':2,'files':files}
        save_json(meta,self.fixture/'fixture.json');self.digest=r3.sha(self.fixture/'fixture.json')

    def test_remainders_and_all6_phases_match_independent_joint(self):
        r3.run_oracle(self.fixture,self.digest,self.root/'oracle')
        for batch in (7,16,64):
            folder=self.root/f'b{batch}'
            report=r3.run_worker(self.fixture,self.digest,folder,batch,1)
            proof=r3.compare_case(self.fixture,self.digest,folder,self.root/'oracle')
            self.assertEqual(proof['status'],'verified')
            self.assertEqual([row['analytic_batches'] for row in report['results']],[(n+batch-1)//batch for n in [100,20,20,20,20,20]])
            self.assertTrue(all(row['prediction_disagreements']==0 for row in proof['results']))
            self.assertEqual(report['learner_stored_training_samples'],0)
            self.assertGreater(report['learning_seconds'],0)
            self.assertAlmostEqual(report['learning_seconds'],report['base_seconds']+report['incremental_seconds'])

    def test_joint_evaluation_and_io_are_outside_measured_windows(self):
        original_save=np.savez_compressed;original_eval=r3.StreamingHead.evaluate
        def save(*args,**kwargs):
            self.assertFalse(r3.LearningWindow.active);return original_save(*args,**kwargs)
        def evaluate(head,*args,**kwargs):
            self.assertFalse(r3.LearningWindow.active);return original_eval(head,*args,**kwargs)
        with (mock.patch.object(r3,'joint_ridge',side_effect=AssertionError('oracle forbidden in timed worker')) as blocked,
              mock.patch.object(r3.np,'savez_compressed',side_effect=save),
              mock.patch.object(r3.StreamingHead,'evaluate',new=evaluate)):
            r3.run_worker(self.fixture,self.digest,self.root/'case',7,1)
        blocked.assert_not_called()

    def test_new_failure_overwrites_old_complete_marker(self):
        import json
        folder=self.root/'case';save_json({'status':'complete','verification_passed':True},folder/'results.json')
        with self.assertRaises(ValueError):r3.run_worker(self.fixture,'0'*64,folder,32,1)
        result=json.loads((folder/'results.json').read_bytes())
        self.assertEqual(result['status'],'failed');self.assertFalse(result['verification_passed'])

    def test_changed_fixture_is_rejected(self):
        file=self.fixture/'phase0_features.npy';data=np.load(file);data[0,0]+=1;np.save(file,data)
        with self.assertRaises(ValueError):r3.load_fixture(self.fixture,self.digest)

    def test_corrupt_state_cannot_pass_comparison(self):
        r3.run_oracle(self.fixture,self.digest,self.root/'oracle')
        folder=self.root/'case';r3.run_worker(self.fixture,self.digest,folder,16,1)
        with (folder/'phase2_state.npz').open('ab') as stream:stream.write(b'changed')
        with self.assertRaises(AssertionError):r3.compare_case(self.fixture,self.digest,folder,self.root/'oracle')


if __name__=='__main__':unittest.main()
