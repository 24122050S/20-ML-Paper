"""Thực thi source tác giả ở checkout riêng; chỉ adapter/observer, không chép core.

Xem docs/DOI_CHIEU_CODE_TAC_GIA_KE_HOACH.md trước khi chạy. Kết quả mới có
attempt riêng. Source tác giả, checkpoint và fixture đều được pin bằng hash.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

AUTHOR_COMMIT = '0afd6bf87622e808ce7f730f8556e78229b9e758'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    """Trả SHA-256 của bytes file theo chunk, không đọc cả checkpoint vào RAM."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    """Ghi JSON UTF-8/LF atomically; NaN không được xuất như số hợp lệ."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temp.replace(path)


def git(folder, *args):
    """Chạy Git chỉ đọc ở checkout chỉ định và trả stdout."""
    return subprocess.check_output(['git', '-C', str(folder), *args], text=True, encoding='utf-8').strip()


def array_sha(array):
    """Băm nội dung C-order để đối chiếu buffer giữa NumPy/PyTorch."""
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def comparison(author_r, author_w, reference_r, reference_w, actual_scores, reference_scores):
    """So R/W/score theo ngưỡng chốt; trả cả lỗi số và disagreements."""
    import numpy as np
    relative = lambda actual, ref: float(np.linalg.norm(actual-ref) / max(np.linalg.norm(ref), np.finfo(np.float64).eps))
    checks = {name: bool(np.allclose(actual, ref, rtol=1e-6, atol=1e-8)) for name, actual, ref in (
        ('r_allclose', author_r, reference_r), ('w_allclose', author_w, reference_w),
        ('score_allclose', actual_scores, reference_scores))}
    disagreements = int(np.count_nonzero(actual_scores.argmax(1) != reference_scores.argmax(1)))
    checks['same_predictions'] = disagreements == 0
    return {'passed': all(checks.values()), 'checks': checks,
            'relative_w_error': relative(author_w, reference_w), 'relative_r_error': relative(author_r, reference_r),
            'max_abs_w_error': float(np.max(np.abs(author_w-reference_w))),
            'max_abs_score_error': float(np.max(np.abs(actual_scores-reference_scores))),
            'prediction_disagreements': disagreements}


def build_backbone(saved, author_models):
    """Đổi tên key của checkpoint, strict-load tensor vào backbone tác giả."""
    import torch
    model = author_models.resnet32(num_classes=50).eval()
    converted = {}
    mapping = {}
    for name, tensor in saved['model_state_dict'].items():
        key = name
        for old, new in [('stage1.', 'layer1.'), ('stage2.', 'layer2.'), ('stage3.', 'layer3.'), ('classifier.', 'fc.')]:
            if key.startswith(old): key = new + key[len(old):]; break
        assert key not in converted
        converted[key] = tensor
        mapping[name] = key
    model.load_state_dict(converted, strict=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, mapping


def diagnostics(author, fixture, r3_root, saved, directory):
    """Chạy mapping/transform/backbone thật; diagnostic lỗi có điều kiện được giữ."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.resnet import ResNet32, freeze_backbone
    from PIL import Image
    from datasets.Features import Features
    from datasets.CIFAR import CIFAR100_
    author_models = importlib.import_module('models.CifarResNet')
    from analytic import ACILLearner

    directory.mkdir(parents=True, exist_ok=True)
    mapping_cache = directory/'mapping_fixture'; mapping_cache.mkdir()
    labels = torch.arange(100, dtype=torch.long)
    for split in ('train', 'test'):
        torch.save(torch.zeros(100, 64), mapping_cache/f'X_{split}.pt')
        torch.save(labels, mapping_cache/f'y_{split}.pt')
    shuffled = Features(str(mapping_cache), True, .5, 5, shuffle_seed=42)
    selected = shuffled.subset_at_phase(0)
    selected_labels = [int(selected[i][1]) for i in range(len(selected))]
    mapping_info = {'scope': '100 samples / one per class, actual author Features/DatasetWrapper',
        'dataset_seed': 42, 'selected_base_classes': shuffled.real_labels[:50],
        'returned_labels': selected_labels, 'returned_labels_are_contiguous_0_to49': sorted(selected_labels) == list(range(50)),
        'actual_max_label': max(selected_labels), 'group_maps_original_ids_to_class_order_positions': True}
    guard_args = {'learning_rate': .1, 'buffer_size': 64, 'gamma': .1, 'base_epochs': 1, 'warmup_epochs': 0,
                  'separate_decay': False, 'weight_decay': .0005, 'momentum': .9, 'label_smoothing': 0.,
                  'saving_root': str(directory)}
    learner = ACILLearner(guard_args, torch.nn.Identity(), 64, torch.device('cpu'))
    try:
        learner.base_training(DataLoader(selected, batch_size=16), DataLoader(selected, batch_size=16), 50)
    except (IndexError, RuntimeError, AssertionError) as error:
        mapping_info['base_training_reproduction'] = {'failure_reproduced': True, 'exception_type': type(error).__name__,
            'message': str(error), 'where': 'author ACILLearner.base_training epoch0 validation before optimizer steps',
            'scope_limit': 'shuffle_seed=42 with labels not remapped; not a claim all default configurations fail'}
    else:
        mapping_info['base_training_reproduction'] = {'failure_reproduced': False}
    assert mapping_info['base_training_reproduction']['failure_reproduced']
    assert not mapping_info['returned_labels_are_contiguous_0_to49']

    prepared = r3_root/'cifar-data/prepared'/saved['dataset_revision']
    images = np.load(prepared/'test_images.npy', mmap_mode='r')
    original_labels = np.load(prepared/'test_labels.npy')
    indices = np.concatenate([np.flatnonzero(original_labels == label)[:10] for label in range(100)])
    assert len(indices) == len(set(indices.tolist())) == 1000
    group = ResNet32(); group.load_state_dict(saved['model_state_dict']); freeze_backbone(group)
    original, key_mapping = build_backbone(saved, author_models)
    means = torch.tensor(saved['normalization']['mean'], dtype=torch.float32)[None, :, None, None]
    stds = torch.tensor(saved['normalization']['std'], dtype=torch.float32)[None, :, None, None]
    raw = torch.from_numpy(np.array(images[indices])).permute(0, 3, 1, 2).float().div_(255)
    inputs = (raw-means)/stds
    native_inputs = torch.stack([CIFAR100_.basic_transform(Image.fromarray(np.array(images[index]))) for index in indices])
    feature_max = logits_max = 0.; prediction_differences = 0; native_feature_max = 0.
    features_passed = logits_passed = True
    import copy
    feature_model = copy.deepcopy(original); feature_model.fc = torch.nn.Identity()
    with torch.no_grad():
        for start in range(0, 1000, 100):
            x = inputs[start:start+100]
            f1, f2 = group.forward_features(x), feature_model(x)
            y1, y2 = group(x), original(x)
            features_passed &= bool(torch.allclose(f1, f2, rtol=1e-5, atol=1e-6))
            logits_passed &= bool(torch.allclose(y1, y2, rtol=1e-5, atol=1e-6))
            feature_max = max(feature_max, float((f1-f2).abs().max()))
            logits_max = max(logits_max, float((y1-y2).abs().max()))
            prediction_differences += int((y1.argmax(1) != y2.argmax(1)).sum())
            native_feature_max = max(native_feature_max, float((feature_model(native_inputs[start:start+100])-f2).abs().max()))
    assert features_passed and logits_passed and prediction_differences == 0
    backbone_info = {'scope': '1000 real test images, 10 per original class; same normalized tensors and mapped R1 checkpoint',
        'indices_sha256': array_sha(indices), 'n_samples': 1000, 'strict_state_load': True, 'key_mapping': key_mapping,
        'feature_allclose': features_passed, 'logits_allclose': logits_passed, 'prediction_disagreements': prediction_differences,
        'max_abs_feature_error': feature_max, 'max_abs_logits_error': logits_max, 'rtol': 1e-5, 'atol': 1e-6}
    preprocessing = {'group_mean': saved['normalization']['mean'], 'group_std': saved['normalization']['std'],
        'group_statistics_source': 'base train seed42 only', 'author_fixed_mean': list(CIFAR100_.mean),
        'author_fixed_std': list(CIFAR100_.std), 'same_image_input_max_abs_difference': float((inputs-native_inputs).abs().max()),
        'feature_max_abs_difference_when_only_input_normalization_changes': native_feature_max,
        'scope_limit': 'diagnostic of input transform, not a native backbone retraining or full accuracy ablation'}
    result = {'status': 'complete', 'mapping': mapping_info, 'backbone': backbone_info, 'preprocessing': preprocessing}
    save(directory/'results.json', result)
    return result, feature_model


def execute(args):
    """Chạy C1/C2, giữ source/fixture/checkpoint và mọi phase comparison."""
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from src.analytic import initialize_batches, incremental_update, joint_ridge

    torch.set_num_threads(4); torch.set_num_interop_threads(4)
    author = Path(args.author_root).resolve(); r3_root = Path(args.r3_root).resolve()
    fixture = r3_root/'fixture'; checkpoint = Path(args.checkpoint).resolve()
    suite = json.loads((r3_root.parents[1]/'r3_ready.json').read_bytes())
    meta = json.loads((fixture/'fixture.json').read_bytes())
    assert sha(fixture/'fixture.json') == suite['fixture_sha256']
    assert git(author, 'rev-parse', 'HEAD') == AUTHOR_COMMIT and not git(author, 'status', '--porcelain')
    for name, expected in meta['files'].items(): assert sha(fixture/name) == expected, name
    for name, expected in meta['provenance']['source_sha256'].items(): assert sha(ROOT/name) == expected, name
    assert sha(checkpoint) == meta['reference']['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['completed_epoch'] == saved['target_epochs'] == 160 and saved['training_complete']
    assert saved['class_order'] == meta['class_order'] and saved['normalization'] == meta['normalization']
    source_names = git(author, 'ls-files').splitlines()
    original_hashes = {name: sha(author/name) for name in source_names}
    sys.path.insert(0, str(author))
    from analytic import ACIL, ACILLearner
    from datasets.Features import Features
    author_main = importlib.import_module('main')
    assert Path(author_main.__file__).resolve() == author/'main.py'

    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]
    root = Path(args.output_dir).resolve(); output = root/'attempts'/attempt; output.mkdir(parents=True)
    state = {'status': 'running', 'verification_passed': False, 'attempt_id': attempt, 'output_dir': str(output),
        'author_repository': 'https://github.com/ZHUANGHP/Analytic-continual-learning', 'author_commit': AUTHOR_COMMIT,
        'author_source_sha256': original_hashes, 'fixture_sha256': suite['fixture_sha256'],
        'base_checkpoint_sha256': sha(checkpoint), 'n_train': 50000, 'n_test': 10000, 'seed': 42, 'dimension': 1024,
        'gamma': .1, 'batch_size': 128, 'threads': 4, 'training_executed': False, 'cases': [],
        'execution_source': {'runner_sha256': sha(__file__), 'plan_sha256': sha(ROOT/'docs/DOI_CHIEU_CODE_TAC_GIA_KE_HOACH.md'),
            'group_git_commit': git(ROOT, 'rev-parse', 'HEAD'), 'group_source_dirty': bool(git(ROOT, 'status', '--porcelain'))},
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
            'packages': {name: importlib.metadata.version(name) for name in ('torchvision','scipy','scikit-learn','Pillow','tqdm','PyYAML','threadpoolctl')},
            'platform': platform.platform(), 'device': 'cpu', 'separate_venv': sys.prefix,
            'author_yaml_python': '3.11', 'python_deviation': '3.12 instead of3.11; torch2.5.1 meets declared torch>=2.2'}}
    def progress(stage):
        state['stage'] = stage; save(root/'progress.json', state); save(output/'progress.json', state)
        print('AUTHOR_COMPARE', stage, flush=True)
    progress('input_validated')
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=4):
            state['diagnostics'], backbone = diagnostics(author, fixture, r3_root, saved, output/'diagnostics')
            progress('diagnostics_complete')
            train = [(np.load(fixture/f'phase{k}_features.npy'), np.load(fixture/f'phase{k}_labels.npy')) for k in range(6)]
            test_raw, test_labels = np.load(fixture/'test_features.npy'), np.load(fixture/'test_labels.npy')
            r3_expansion = np.load(fixture/'expansion.npy')
            r3_case = next(case for case in suite['cases'] if case['key'] == 'b128_r1')

            def record(model, phase, expansion, directory, reference=None, own_state=None):
                classes = 50 + phase*10; keep = test_labels < classes
                with torch.no_grad(): scores = model(torch.from_numpy(test_raw[keep])).numpy()
                inverse = model.analytic_linear.R.numpy(); weights = model.analytic_linear.weight.numpy()
                assert inverse.shape == (1024,1024) and weights.shape == (1024,classes)
                assert np.isfinite(inverse).all() and np.isfinite(weights).all()
                test_h = np.maximum(test_raw[keep] @ expansion, 0)
                if reference is not None:
                    with np.load(r3_root/'cases/b128_r1'/reference['state_file']) as old:
                        ref_r, ref_w = old['inverse_gram'], old['weights']
                        group_check = comparison(inverse, weights, ref_r, ref_w, scores, test_h @ ref_w)
                    oracle_row = suite['oracle']['results'][phase]
                    with np.load(r3_root/'oracle'/oracle_row['state_file']) as old:
                        joint_check = comparison(inverse, weights, old['inverse_gram'], old['weights'], scores, test_h @ old['weights'])
                else:
                    ref_r, ref_w = own_state
                    group_check = comparison(inverse, weights, ref_r, ref_w, scores, test_h @ ref_w)
                    all_h = np.maximum(np.concatenate([raw for raw, labels in train[:phase+1]]) @ expansion, 0)
                    all_y = np.concatenate([labels for raw, labels in train[:phase+1]])
                    joint_w = joint_ridge(all_h, all_y, classes, .1)
                    joint_r = np.linalg.solve(all_h.T @ all_h + .1*np.eye(1024), np.eye(1024))
                    joint_check = comparison(inverse, weights, joint_r, joint_w, scores, test_h @ joint_w)
                filename = f'phase{phase}.npz'
                np.savez_compressed(directory/filename, inverse_gram=inverse, weights=weights, expansion=expansion)
                pred = scores.argmax(1); base = test_labels[keep] < 50
                row = {'phase': phase, 'n_classes': classes, 'test_samples': int(keep.sum()),
                    'accuracy_pct': 100*int((pred==test_labels[keep]).sum())/len(pred),
                    'base_accuracy_pct': 100*int((pred[base]==test_labels[keep][base]).sum())/int(base.sum()),
                    'weights_shape': list(weights.shape), 'state_file': filename, 'state_sha256': sha(directory/filename),
                    'group_comparison': group_check, 'joint_comparison': joint_check}
                row['passed'] = group_check['passed'] and joint_check['passed']
                return row

            controlled_dir = output/'controlled'; controlled_dir.mkdir()
            torch.manual_seed(42)
            controlled = ACIL(64, torch.nn.Identity(), 1024, .1, device='cpu', dtype=torch.float64)
            controlled.buffer.weight.copy_(torch.from_numpy(r3_expansion.T.copy()))
            controlled_rows = []; fe_error = 0.; fe_pass = True
            for phase, (raw, labels) in enumerate(train):
                progress(f'C1_phase{phase}')
                for offset in range(0, len(raw), 128):
                    x = torch.from_numpy(raw[offset:offset+128]); y = torch.from_numpy(labels[offset:offset+128])
                    actual_h = controlled.feature_expansion(x).numpy(); expected_h = np.maximum(x.numpy()@r3_expansion,0)
                    fe_error = max(fe_error,float(np.max(np.abs(actual_h-expected_h))))
                    fe_pass &= bool(np.allclose(actual_h, expected_h, rtol=1e-12, atol=1e-10))
                    controlled.fit(x,y)
                controlled.update()
                controlled_rows.append(record(controlled,phase,r3_expansion,controlled_dir,reference=r3_case['report']['results'][phase]))
            assert fe_pass
            c1 = {'case_key':'controlled','entry_point':'analytic.ACIL.ACIL.fit/forward/update',
                'status':'complete','results':controlled_rows,'expansion_sha256':array_sha(r3_expansion),
                'feature_expansion_allclose':fe_pass,'feature_expansion_max_abs_error':fe_error,
                'expected_accuracy_matches_r3':all(row['accuracy_pct']==ref['accuracy_pct'] for row,ref in zip(controlled_rows,suite['oracle']['results'])),
                'verification_passed':all(row['passed'] for row in controlled_rows)}
            save(controlled_dir/'results.json',c1);state['cases'].append(c1)

            native_dir = output/'native_main'; native_dir.mkdir(); cache = native_dir/'cache'; cache.mkdir()
            torch.save(torch.from_numpy(np.concatenate([raw for raw,labels in train])),cache/'X_train.pt')
            torch.save(torch.from_numpy(np.concatenate([labels for raw,labels in train])),cache/'y_train.pt')
            torch.save(torch.from_numpy(test_raw),cache/'X_test.pt');torch.save(torch.from_numpy(test_labels),cache/'y_test.pt')
            torch.save((backbone,32,64),cache/'backbone.pth')
            native_rows=[];observed={};group_r=group_w=None
            def factory(*factory_args,**factory_kwargs):
                nonlocal group_r,group_w
                learner = ACILLearner(*factory_args,**factory_kwargs)
                observed['learner']=learner
                expansion=learner.model.buffer.weight.numpy().T.copy();observed['expansion']=expansion
                assert learner.model.buffer.bias is None
                original_before=learner.before_validation
                def before_validation():
                    nonlocal group_r,group_w
                    original_before();phase=len(native_rows);progress(f'C2_phase{phase}')
                    raw,labels=train[phase]
                    if phase==0:
                        batches=((np.maximum(raw[i:i+128]@expansion,0),labels[i:i+128]) for i in range(0,len(raw),128))
                        group_r,group_w=initialize_batches(batches,1024,50,.1)
                    else:
                        for i in range(0,len(raw),128):
                            group_r,group_w=incremental_update(group_r,group_w,np.maximum(raw[i:i+128]@expansion,0),labels[i:i+128],n_classes=50+phase*10)
                    native_rows.append(record(learner.model,phase,expansion,native_dir,own_state=(group_r,group_w)))
                learner.before_validation=before_validation
                def no_training(*a,**kw): raise AssertionError('Comparison must never train new backbone')
                learner.base_training=no_training
                return learner
            original_dataset=author_main.load_dataset;original_factory=author_main.ALL_METHODS['ACIL']
            def cached_dataset(*a,**kw):
                assert not a and kw['name']=='CIFAR-100'
                return Features(str(cache),kw['train'],kw['base_ratio'],kw['num_phases'],augment=False)
            main_args={'backbone':'resnet32','cpu_only':True,'gpus':None,'seed':42,'dataset_seed':None,
                'backbone_path':str(cache/'backbone.pth'),'cache_path':str(cache),'cache_features':True,
                'dataset':'CIFAR-100','data_root':str(cache),'base_ratio':.5,'phases':5,'method':'ACIL',
                'learning_rate':.1,'buffer_size':1024,'gamma':.1,'base_epochs':160,'warmup_epochs':0,
                'num_workers':0,'batch_size':128,'IL_batch_size':128,'momentum':.9,'weight_decay':.0005,
                'separate_decay':False,'label_smoothing':0.,'saving_root':str(native_dir)}
            save(native_dir/'args.json',main_args)
            author_main.load_dataset=cached_dataset;author_main.ALL_METHODS['ACIL']=factory
            try: author_main.main(main_args)
            finally: author_main.load_dataset=original_dataset;author_main.ALL_METHODS['ACIL']=original_factory
            assert len(native_rows)==6
            with (native_dir/'IL.csv').open(newline='') as stream: csv_rows=list(csv.DictReader(stream))
            assert len(csv_rows)==6
            for phase,(csv_row,row) in enumerate(zip(csv_rows,native_rows)):
                assert np.isclose(100*float(csv_row['acc@1']),row['accuracy_pct'],rtol=0,atol=1e-12)
                assert np.isclose(100*float(csv_row['acc@avg']),np.mean([x['accuracy_pct'] for x in native_rows[:phase+1]]),rtol=0,atol=1e-12)
            expansion=observed['expansion'];native_stats={'mean':float(expansion.mean()),'std':float(expansion.std()),
                'min':float(expansion.min()),'max':float(expansion.max()),'sha256':array_sha(expansion),'bias':False}
            c2={'case_key':'native_main','entry_point':'main.main(args), original ACILLearner/Features/metrics',
                'status':'complete','results':native_rows,'expansion':native_stats,'csv_metrics_match':True,
                'verification_passed':all(row['passed'] for row in native_rows),
                'boundary_adapters':['source Features cache derived from R1/R3, mapped global labels',
                    'strict-converted author backbone from R1 checkpoint, no fresh BP',
                    'main.load_dataset returns cached Features for initial dataset construction',
                    'factory observer saves per-phase states without changing fit/forward/update'],
                'source_algorithm_edits':False,'native_minibatch_shuffle':True,'args_file':'native_main/args.json'}
            save(native_dir/'results.json',c2);state['cases'].append(c2)

            for case in state['cases']:
                rows=case['results'];case['metrics']={'average_incremental_accuracy_pct':float(np.mean([r['accuracy_pct'] for r in rows])),
                    'final_accuracy_pct':rows[-1]['accuracy_pct'],'base_forgetting_pp':rows[0]['base_accuracy_pct']-rows[-1]['base_accuracy_pct'],
                    'n_independent_seeds':1,'between_seed_sd':None}
                save(output/case['case_key']/'results.json',case)
            state['author_source_unchanged_after_execution']=original_hashes=={name:sha(author/name) for name in source_names}
            assert state['author_source_unchanged_after_execution'] and not git(author,'status','--porcelain')
            state['verification_passed']=all(case['verification_passed'] for case in state['cases'])
            state['seconds_this_suite']=time.perf_counter()-started
            state['status']='complete' if state['verification_passed'] else 'failed'
            save(output/'results.json',state);save(root/'results.json',state);progress(state['status'])
            if not state['verification_passed']:raise RuntimeError('Author comparison differs beyond preregistered thresholds; evidence preserved')
            return state
    except BaseException as error:
        state.update(status='failed',verification_passed=False,error_type=type(error).__name__,error=str(error))
        save(output/'results.json',state);save(root/'results.json',state);progress('failed')
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8',newline='\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author-root',required=True,type=Path)
    parser.add_argument('--r3-root',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    options=parser.parse_args()
    sys.path.insert(0,str(ROOT))
    os.environ.setdefault('TQDM_DISABLE','1')
    execute(options)
