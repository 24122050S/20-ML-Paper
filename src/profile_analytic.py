"""Profile analytic d lớn bằng feature CIFAR thật trên subset, không train BP dài."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import threading
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits
import torch
from torch.utils.data import DataLoader

from .cifar_data import CifarSubset, load_cifar100, audit_data, make_protocol, sha256_file
from .cifar_smoke import validate_checkpoint, seed_everything, fingerprint, iter_feature_batches
from .resnet import ResNet32, freeze_backbone
from .streaming import StreamingHead
from .reporting import collect_provenance, save_json, log_and_save_report
from .run_attempt import RunAttempt


class MemoryMonitor:
    """Lấy mẫu RSS process theo khoảng20ms; ghi peak trong đúng cửa sổ profile."""
    def __init__(self, device, interval=0.02):
        self.device=torch.device(device)
        self.interval=interval
        self.process=psutil.Process(os.getpid())
        self.stopped=threading.Event()
    def _sample(self):
        while not self.stopped.wait(self.interval):
            self.peak=max(self.peak,self.process.memory_info().rss)
    def __enter__(self):
        self.baseline=self.peak=self.process.memory_info().rss
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.thread=threading.Thread(target=self._sample,daemon=True)
        self.thread.start()
        return self
    def __exit__(self,*args):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        self.peak=max(self.peak,self.process.memory_info().rss)
        self.stopped.set();self.thread.join()
        self.result={'baseline_cpu_rss_bytes':self.baseline,'peak_cpu_rss_bytes':self.peak,
                     'sampling_interval_seconds':self.interval,
                     'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None}


def balanced_loader(arrays, protocol, norm, classes, split, per_class, config):
    """Chọn số ảnh cố định mỗi lớp theo index gốc; trả loader không augment/shuffle."""
    labels=arrays.train_labels if split=='train' else arrays.test_labels
    images=arrays.train_images if split=='train' else arrays.test_images
    mapped=np.asarray(protocol['original_to_global'])[labels]
    indices=np.sort(np.concatenate([np.flatnonzero(mapped==c)[:per_class] for c in classes]))
    if len(indices)!=len(classes)*per_class:raise ValueError('Không đủ ảnh cho profile')
    dataset=CifarSubset(images,labels,indices,np.asarray(protocol['original_to_global']),norm['mean'],norm['std'])
    return DataLoader(dataset,batch_size=config['analytic_batch_size'],shuffle=False,num_workers=0,
                      generator=torch.Generator().manual_seed(config['seed']))


def profile(checkpoint_path, *, data_dir, output_dir, dimension=8000, threads=4,
            device='cpu', base_per_class=20, incremental_per_class=4, test_per_class=6,
            log_wandb=False, prepare_tracking=None):
    """Profile d8000 bằng subset thật; trả report tài nguyên, không phải benchmark accuracy.

    Base1000 ảnh, mới200 ảnh phủ50 lớp mới, test600 ảnh là mặc định. Ghi rõ
    scope, số mẫu, thời gian từng stage và RSS; không lưu ma trận R lớn ra NPZ.
    """
    with RunAttempt(output_dir) as attempt:
        attempt.progress('profile_checkpoint_load')
        if checkpoint_path is None:
            raise ValueError('Cần CHECKPOINT: attach Output notebook03 làm Input, hoặc điền path base_checkpoint.pt đã hoàn tất')
        if dimension<1 or threads<1 or not 1<=base_per_class<=500 or not 1<=incremental_per_class<=500 or not 1<=test_per_class<=100:
            raise ValueError('Tham số profile ngoài phạm vi')
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        config={**checkpoint['config'],'experiment':'ACIL-RESOURCE-PROFILE','mode':'profile',
                'expansion_dim':dimension,'torch_threads':threads,'device':device,
                'feature_expansion':True,'analytic_batch_size':128}
        attempt.seed=config['seed']
        save_json(config,attempt.directory/'config.json')
        if log_wandb and prepare_tracking is not None:
            attempt.progress('tracking_setup');prepare_tracking()
        seed_everything(config['seed'],threads)
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA không khả dụng')
        device=torch.device(device)
        arrays=load_cifar100(data_dir)
        protocol=make_protocol(config['seed'],5)
        audit=audit_data(arrays,protocol)
        norm=audit['normalization']
        validate_checkpoint(checkpoint,config,protocol,norm,arrays.source['revision'])
        model=ResNet32().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        freeze_backbone(model)
        before=fingerprint(model)
        save_json(protocol,attempt.directory/'protocol.json')
        save_json(audit,attempt.directory/'data_audit.json')
        timings={}
        started=time.perf_counter()
        with threadpool_limits(limits=threads),MemoryMonitor(device) as memory:
            # Cache chỉ F64 của subset để tách thời gian CNN khỏi phép analytic.
            attempt.progress('profile_features')
            features=[]
            feature_started=time.perf_counter()
            for classes,split,count in ((range(50),'train',base_per_class),
                                        (range(50,100),'train',incremental_per_class),
                                        (range(100),'test',test_per_class)):
                loader=balanced_loader(arrays,protocol,norm,classes,split,count,config)
                chunks=list(iter_feature_batches(model,loader,device,'profile '+split))
                features.append((np.concatenate([x for x,y in chunks]),np.concatenate([y for x,y in chunks])))
            timings['feature_extraction_seconds']=time.perf_counter()-feature_started
            def batches(pair):
                raw,labels=pair
                for i in range(0,len(raw),128):yield raw[i:i+128],labels[i:i+128]
            head=StreamingHead(config)
            attempt.progress('profile_base_analytic',dimension=dimension)
            print(f'Profile d={dimension}: base={50*base_per_class}, new={50*incremental_per_class}, test={100*test_per_class}',flush=True)
            head.fit_base(batches(features[0]),50)
            timings.update(head.timings)
            attempt.progress('profile_incremental_analytic',dimension=dimension)
            head.update_phase(batches(features[1]),100)
            timings['incremental_analytic_seconds']=head.timings['last_incremental_seconds']
            attempt.progress('profile_evaluation')
            start=time.perf_counter()
            evaluation=head.evaluate(batches(features[2]),100,range(50,100),expected_per_class=test_per_class)
            timings['evaluation_seconds']=time.perf_counter()-start
            checks={'finite_analytic_state':bool(np.isfinite(head.r).all() and np.isfinite(head.w).all()),
                    'weights_shape_correct':head.w.shape==(dimension,100),
                    'all_profile_classes_evaluated':evaluation['test_samples_all_seen']==100*test_per_class,
                    'backbone_frozen':fingerprint(model)==before}
            persistent=head.state_bytes()
            state_shapes={'R':list(head.r.shape),'W':list(head.w.shape),'FE':list(head.expansion.shape)}
        report={'experiment':config['experiment'],'description':'Resource probe on a balanced CIFAR subset; not benchmark accuracy',
                'config':config,'environment':{'python':platform.python_version(),'numpy':np.__version__,
                    'torch':str(torch.__version__),'device':str(device),'cuda_device':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                    'packages':{n:importlib.metadata.version(n) for n in ('scipy','psutil','threadpoolctl')}},
                'provenance':{**collect_provenance(Path(__file__).resolve().parents[1]),
                    'input_checkpoint_sha256':sha256_file(Path(checkpoint_path))},
                'profile_scope':{'full_protocol_evaluated':False,'base_samples':50*base_per_class,
                    'incremental_samples':50*incremental_per_class,'test_samples':100*test_per_class,
                    'class_transition':[50,100],'stored_raw_feature_scope':'subset F64 only; no expanded-feature history',
                    'analytic_state_saved':False,'gpu_benchmark_validated':False},
                'timings':timings,'memory':{**memory.result,'persistent_analytic_state_bytes':persistent},
                'state_shapes':state_shapes,'profile_subset_evaluation':evaluation,
                'rough_extrapolation':{'base_full25000_seconds':timings['base_accumulation_seconds']*(500/base_per_class)+timings['base_solve_seconds'],
                    'incremental_full25000_seconds':timings['incremental_analytic_seconds']*(500/incremental_per_class),
                    'note':'Linear subset extrapolation only; excludes full evaluation, artifact I/O and BP. Not a Kaggle runtime guarantee.'},
                'results':[],'checks':checks,'verification_passed':all(checks.values()),
                'attempt_id':attempt.id,'seconds_this_run':time.perf_counter()-started}
        attempt.progress('tracking')
        report=log_and_save_report(report,attempt.directory/'results.json',log_wandb=log_wandb,
                                   entity='hsangg-2006',project='acil-reproduction',tags=['profile','subset','d'+str(dimension)])
        result=attempt.complete(report)
        print('PROFILE PASS:',attempt.directory/'results.json',flush=True)
        return result


def main():
    """CLI profile offline; nhận checkpoint base đã hoàn tất và thư mục dữ liệu/output."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--output-dir',type=Path,default=Path('reports/local/profile_d8000'))
    parser.add_argument('--dimension',type=int,default=8000)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    args=parser.parse_args()
    profile(args.checkpoint,data_dir=args.data_dir,output_dir=args.output_dir,dimension=args.dimension,
            threads=args.threads,device=args.device)


if __name__=='__main__':main()
