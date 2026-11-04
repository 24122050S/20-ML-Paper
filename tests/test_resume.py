"""Resume thật: so optimizer, scheduler, RNG, shuffle và trọng số sau chạy ngắt."""

import copy
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in
                ("torch","huggingface_hub","pyarrow","PIL","threadpoolctl","filelock"))
if HAS_TORCH:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from src.base_training import train_base, EpochLimitReached, atomic_torch_save
    from src.cifar_smoke import seed_everything
    from src.cifar_smoke import validate_checkpoint

    class RandomDataset(Dataset):
        def __init__(self):
            self.images = np.random.default_rng(123).normal(size=(16, 3, 8, 8)).astype(np.float32)
            self.seen = []
        def __len__(self):
            return len(self.images)
        def __getitem__(self, index):
            self.seen.append(index)
            noise = random.random() + np.random.random() + torch.rand(()).item()
            return torch.tensor(self.images[index]) + noise*0.01, index % 3

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.drop = nn.Dropout(0.25)
            self.fc = nn.Linear(4, 3)
        def forward(self, x):
            return self.fc(self.drop(torch.relu(self.bn(self.conv1(x))).mean((2,3))))


@unittest.skipUnless(HAS_TORCH, "Cần PyTorch để kiểm resume")
class ResumeTests(unittest.TestCase):
    def config(self):
        return {"seed":42,"epochs":2,"batch_size":4,"learning_rate":0.1,"momentum":0.9,
                "weight_decay":0.0005,"lr_milestones":[1],"torch_threads":2,"num_workers":0}
    def context(self):
        return {"class_order":list(range(100)),"normalization":{"mean":[0.5]*3,"std":[0.25]*3},
                "dataset_revision":"test-revision"}
    def setup_training(self, device="cpu"):
        seed_everything(42,2)
        model=TinyModel().to(device)
        dataset=RandomDataset()
        loader=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0,
                          generator=torch.Generator().manual_seed(42))
        return model,loader
    def load(self, path):
        return torch.load(path,map_location="cpu",weights_only=True)
    def assert_nested_equal(self, left, right):
        if isinstance(left,torch.Tensor):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        elif isinstance(left,dict):
            self.assertEqual(left.keys(),right.keys())
            for k in left:self.assert_nested_equal(left[k],right[k])
        elif isinstance(left,(tuple,list)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.assert_nested_equal(a,b)
        else:self.assertEqual(left,right)

    def compare_restart(self, device):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            model,loader=self.setup_training(device)
            history,_=train_base(model,loader,self.config(),device,root/'continuous',checkpoint_context=self.context())
            expected_order=loader.dataset.seen.copy()
            expected=self.load(root/'continuous/base_checkpoint.pt')
            model,loader=self.setup_training(device)
            with self.assertRaises(EpochLimitReached):
                train_base(model,loader,self.config(),device,root/'stopped',checkpoint_context=self.context(),stop_after_epoch=1)
            first_order=loader.dataset.seen.copy()
            old_bytes=(root/'stopped/base_checkpoint.pt').read_bytes()
            # Mô phỏng process/kernel mới và tiêu RNG trước restore.
            random.seed(999);np.random.seed(999);torch.manual_seed(999)
            model=TinyModel().to(device)
            dataset=RandomDataset()
            loader=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0,
                              generator=torch.Generator().manual_seed(999))
            resumed,_=train_base(model,loader,self.config(),device,root/'resumed',checkpoint_context=self.context(),
                                 resume_checkpoint=root/'stopped/base_checkpoint.pt')
            actual=self.load(root/'resumed/base_checkpoint.pt')
            self.assertEqual(first_order+dataset.seen,expected_order)
            for key in ("model_state_dict","optimizer_state_dict","scheduler_state_dict","rng_state",
                        "dataloader_generator_state","initial_stem_weight"):
                self.assert_nested_equal(expected[key],actual[key])
            semantic=lambda rows:[{k:v for k,v in r.items() if k!='seconds'} for r in rows]
            self.assertEqual(semantic(history),semantic(resumed))
            self.assertEqual([r['learning_rate'] for r in resumed],[0.1,0.010000000000000002])
            self.assertTrue(actual['training_complete'])
            self.assertEqual(actual['completed_epoch'],2)
            self.assertEqual(old_bytes,(root/'stopped/base_checkpoint.pt').read_bytes())

    def test_cpu_restart_matches_continuous_exactly(self):
        self.compare_restart("cpu")

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(),"Không có CUDA trên máy kiểm tra")
    def test_cuda_restart_matches_continuous_exactly(self):
        self.compare_restart("cuda")

    def test_mid_epoch_interrupt_preserves_previous_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            model,loader=self.setup_training()
            def interrupt(stage,**detail):
                if detail['epoch']==2:raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                train_base(model,loader,self.config(),'cpu',root,checkpoint_context=self.context(),progress=interrupt)
            saved=self.load(root/'base_checkpoint.pt')
            self.assertEqual(saved['completed_epoch'],1)
            self.assertFalse(saved['training_complete'])
            self.assertEqual(len(saved['training_history']),1)

    def test_resume_rejects_changed_config_context_and_missing_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            model,loader=self.setup_training()
            with self.assertRaises(EpochLimitReached):
                train_base(model,loader,self.config(),'cpu',root/'source',checkpoint_context=self.context(),stop_after_epoch=1)
            original=self.load(root/'source/base_checkpoint.pt')
            changes=[lambda c:c['config'].__setitem__('seed',43),
                     lambda c:c['config'].__setitem__('lr_milestones',[2]),
                     lambda c:c.__setitem__('target_epochs',3),
                     lambda c:c['class_order'].reverse(),
                     lambda c:c['normalization']['mean'].__setitem__(0,0.1),
                     lambda c:c.__setitem__('dataset_revision','other'),
                     lambda c:c.pop('optimizer_state_dict'),
                     lambda c:c['loader_signature'].__setitem__('sampler','SequentialSampler'),
                     lambda c:c['runtime'].__setitem__('torch','other')]
            for i,change in enumerate(changes):
                with self.subTest(change=i):
                    altered=copy.deepcopy(original);change(altered)
                    path=root/f'bad{i}.pt';torch.save(altered,path)
                    model,loader=self.setup_training()
                    with self.assertRaises(ValueError):
                        train_base(model,loader,self.config(),'cpu',root/f'run{i}',checkpoint_context=self.context(),resume_checkpoint=path)

    def test_resume_rejects_legacy_and_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            torch.save({'model_state_dict':{}},root/'legacy.pt')
            model,loader=self.setup_training()
            with self.assertRaisesRegex(ValueError,'v2'):
                train_base(model,loader,self.config(),'cpu',root/'bad',checkpoint_context=self.context(),resume_checkpoint=root/'legacy.pt')
            model,loader=self.setup_training()
            train_base(model,loader,self.config(),'cpu',root/'done',checkpoint_context=self.context())
            model,loader=self.setup_training()
            with self.assertRaises(ValueError):
                train_base(model,loader,self.config(),'cpu',root/'bad2',checkpoint_context=self.context(),resume_checkpoint=root/'done/base_checkpoint.pt')

    def test_atomic_checkpoint_failure_keeps_previous_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'checkpoint.pt'
            atomic_torch_save({'epoch':1},path)
            before=path.read_bytes()
            def fail_save(payload,stream):
                stream.write(b'partial');raise OSError('simulated disk failure')
            with mock.patch('src.base_training.torch.save',side_effect=fail_save),self.assertRaises(OSError):
                atomic_torch_save({'epoch':2},path)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(path.parent.glob('.checkpoint-*.tmp')),[])

    def test_reuse_rejects_incomplete_or_subset_v2_checkpoint(self):
        context=self.context()
        for checkpoint in ({'checkpoint_format_version':2,'training_complete':False},
                           {'checkpoint_format_version':2,'training_complete':True,'loader_signature':{'dataset_size':200}}):
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint,self.config(),{'class_order':context['class_order']},
                                    context['normalization'],context['dataset_revision'])


if __name__=='__main__':unittest.main()
