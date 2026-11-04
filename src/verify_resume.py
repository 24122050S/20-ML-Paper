"""Kiểm ResNet32/CIFAR thật: hai epochs liên tục so với1+resume1 trên subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base_training import train_base, EpochLimitReached
from .cifar_smoke import seed_everything, fingerprint
from .cifar_data import load_cifar100, make_protocol, audit_data, CifarSubset, sha256_file
from .resnet import ResNet32
from .reporting import collect_provenance, save_json
from .run_attempt import RunAttempt


def assert_state_equal(left, right):
    """So đệ quy tensor/primitive của checkpoint; không dùng tolerance cho CPU."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def verify(*, data_dir, output_dir):
    """Chạy CPU200 ảnh/epoch, lưu proof/hashes; không phải training benchmark.

    Dữ liệu dùng50 lớp base,4 ảnh/lớp theo index gốc, crop/flip bật.
    Unit test bổ sung sample order trực tiếp và biến thể CUDA nếu khả dụng.
    """
    root = Path(__file__).resolve().parents[1]
    with RunAttempt(output_dir) as attempt:
        attempt.seed = 42
        config = json.loads((root / 'configs/cifar100_smoke.json').read_text(encoding='utf-8'))
        config.update(epochs=2, batch_size=64, lr_milestones=[1], torch_threads=4, device='cpu')
        arrays = load_cifar100(data_dir)
        protocol = make_protocol(42, 5)
        norm = audit_data(arrays, protocol)['normalization']
        mapping = np.asarray(protocol['original_to_global'])
        mapped = mapping[arrays.train_labels]
        indices = np.sort(np.concatenate([np.flatnonzero(mapped == c)[:4] for c in range(50)]))
        context = {'class_order': protocol['class_order'], 'normalization': norm,
                   'dataset_revision': arrays.source['revision']}

        def setup():
            seed_everything(42, 4)
            model = ResNet32()
            dataset = CifarSubset(arrays.train_images, arrays.train_labels, indices, mapping,
                                  norm['mean'], norm['std'], augment=True)
            loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(42))
            return model, loader

        attempt.progress('resume_verification_continuous')
        continuous, loader = setup()
        history, _ = train_base(continuous, loader, config, 'cpu', attempt.directory / 'continuous',
                               checkpoint_context=context)
        stopped, loader = setup()
        try:
            train_base(stopped, loader, config, 'cpu', attempt.directory / 'stopped',
                       checkpoint_context=context, stop_after_epoch=1)
        except EpochLimitReached:
            pass
        partial = attempt.directory / 'stopped/base_checkpoint.pt'
        before = sha256_file(partial)
        attempt.progress('resume_verification_restart')
        resumed, loader = setup()
        actual_history, _ = train_base(resumed, loader, config, 'cpu', attempt.directory / 'resumed',
                                      checkpoint_context=context, resume_checkpoint=partial)
        expected = torch.load(attempt.directory / 'continuous/base_checkpoint.pt', weights_only=True)
        actual = torch.load(attempt.directory / 'resumed/base_checkpoint.pt', weights_only=True)
        keys = ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'rng_state',
                'dataloader_generator_state', 'initial_stem_weight')
        for key in keys:
            assert_state_equal(expected[key], actual[key])
        semantic = lambda rows: [{k: v for k, v in r.items() if k != 'seconds'} for r in rows]
        assert semantic(history) == semantic(actual_history)
        assert before == sha256_file(partial)
        checks = {key + '_exact': True for key in keys}
        checks.update(history_excluding_time_exact=True, input_checkpoint_unchanged=True)
        report = {'experiment': 'ACIL-RESUME-CPU-VERIFY', 'config': config,
                  'scope': '200 real base images/epoch; CPU2 versus1+resume1; not benchmark accuracy',
                  'train_samples_per_epoch': len(indices), 'dataset_source': arrays.source,
                  'class_order': protocol['class_order'], 'normalization': norm,
                  'continuous_model_sha256': fingerprint(continuous), 'resumed_model_sha256': fingerprint(resumed),
                  'training_history': actual_history, 'checkpoint_runtime': actual['runtime'],
                  'partial_checkpoint_sha256': before, 'compared_checkpoint_fields': list(keys),
                  'provenance': collect_provenance(root), 'checks': checks, 'verification_passed': True,
                  'results': [], 'wandb_status': 'not_logged', 'wandb_run_id': None, 'wandb_run_url': None}
        save_json(report, attempt.directory / 'results.json')
        result = attempt.complete(report)
        print('RESUME CPU PASS:', attempt.directory / 'results.json', flush=True)
        return result


def main():
    """CLI kiểm resume CIFAR thật, mặc định offline/CPU/subset để chạy nhanh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/local/resume_cpu_verification'))
    args = parser.parse_args()
    verify(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
