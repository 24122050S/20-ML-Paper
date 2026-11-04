"""Profile chỉ là subset và lỗi đầu vào không lấy lại kết quả pass cũ."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

HAS_PROFILE=all(importlib.util.find_spec(n) is not None for n in
    ('torch','scipy','psutil','huggingface_hub','pyarrow','PIL','threadpoolctl','filelock'))
if HAS_PROFILE:
    import torch
    from src.profile_analytic import profile
    from src.cifar_data import CifarArrays,make_protocol
    from src.resnet import ResNet32


@unittest.skipUnless(HAS_PROFILE,'Cần requirements-cifar để kiểm profile')
class ProfileTests(unittest.TestCase):
    def test_missing_checkpoint_replaces_old_success_with_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'results.json').write_text(json.dumps({'status':'complete','verification_passed':True}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'CHECKPOINT'):
                profile(None,data_dir=root/'data',output_dir=root,dimension=16)
            result=json.loads((root/'results.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'failed')
            self.assertFalse(result['verification_passed'])
            self.assertEqual(result['failed_stage'],'profile_checkpoint_load')

    def test_profile_is_subset_resource_probe_not_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=json.loads((Path(__file__).resolve().parents[1]/'configs/cifar100_smoke.json').read_text(encoding='utf-8'))
            protocol=make_protocol(42)
            norm={'mean':[0.5]*3,'std':[0.25]*3}
            path=root/'base.pt'
            torch.save({'model_state_dict':ResNet32().state_dict(),'config':config,
                        'class_order':protocol['class_order'],'normalization':norm,'dataset_revision':'test',
                        'training_history':[{'train_loss':1.0}]*2,'parameter_change_l2':1.0},path)
            images=np.zeros((100,32,32,3),dtype=np.uint8);labels=np.arange(100)
            arrays=CifarArrays(images,labels,images,labels,tuple(map(str,labels)),{'revision':'test'})
            with mock.patch('src.profile_analytic.load_cifar100',return_value=arrays),mock.patch(
                    'src.profile_analytic.audit_data',return_value={'normalization':norm,'passed':True}):
                result=profile(path,data_dir=root/'data',output_dir=root/'profile',dimension=16,threads=2,
                               base_per_class=1,incremental_per_class=1,test_per_class=1)
            self.assertEqual(result['status'],'complete')
            self.assertFalse(result['profile_scope']['full_protocol_evaluated'])
            self.assertFalse(result['profile_scope']['gpu_benchmark_validated'])
            self.assertEqual(result['state_shapes']['W'],[16,100])
            self.assertEqual(result['results'],[])
            self.assertNotIn('metric_summary',result)
            self.assertEqual(result['wandb_status'],'not_logged')
            self.assertGreater(result['memory']['peak_cpu_rss_bytes'],0)


if __name__=='__main__':unittest.main()
