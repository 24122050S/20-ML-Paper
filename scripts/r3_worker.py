"""R3: mỗi tiến trình đo một batch/repeat; oracle chạy ở tiến trình riêng."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

import numpy as np
import psutil
from scipy.linalg import cho_factor
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analytic import joint_ridge
from src.reporting import save_json
from src.streaming import StreamingHead


def sha(path):
    """Băm file theo khối; không giữ thêm bản sao dữ liệu lớn."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def batches(pair, size):
    """Chỉ yield lát cắt của phase hiện tại, không thay thứ tự mẫu."""
    raw, labels = pair
    for start in range(0, len(raw), size):
        yield raw[start:start+size], labels[start:start+size]


def load_fixture(directory, expected_sha):
    """Kiểm checksum trước đo; tất cả worker nhận cùng F64/y/FE."""
    directory = Path(directory)
    if sha(directory/'fixture.json') != expected_sha:
        raise ValueError('Fixture manifest không đúng hash đã pin')
    meta = json.loads((directory/'fixture.json').read_bytes())
    for name, value in meta['files'].items():
        if sha(directory/name) != value:
            raise ValueError('Fixture file không đúng hash: '+name)
    train = [(np.load(directory/f'phase{i}_features.npy', allow_pickle=False),
              np.load(directory/f'phase{i}_labels.npy', allow_pickle=False)) for i in range(6)]
    test = (np.load(directory/'test_features.npy', allow_pickle=False),
            np.load(directory/'test_labels.npy', allow_pickle=False))
    expansion = np.load(directory/'expansion.npy', allow_pickle=False)
    assert expansion.shape == (64, meta['config']['expansion_dim'])
    for phase, (raw, labels) in enumerate(train):
        classes = range(50) if phase == 0 else range(40+10*phase, 50+10*phase)
        assert raw.dtype == np.float64 and raw.shape == (meta['train_counts'][phase], 64)
        assert np.isfinite(raw).all() and labels.dtype == np.int64
        assert np.array_equal(np.unique(labels), np.array(list(classes)))
        assert np.all(np.bincount(labels, minlength=100)[list(classes)] == meta['train_per_class'])
    assert test[0].shape == (100*meta['test_per_class'], 64) and test[0].dtype == np.float64
    assert np.isfinite(test[0]).all() and np.all(np.bincount(test[1], minlength=100) == meta['test_per_class'])
    expected = np.random.default_rng(meta['config']['seed']+1000).normal(size=expansion.shape)
    assert np.array_equal(expansion, expected)
    return meta, train, test, expansion


class LearningWindow:
    """Đo perf_counter và RSS lấy mẫu 5ms, chỉ trong lệnh học analytic."""
    active = False

    def __init__(self, interval=0.005):
        self.interval = interval
        self.process = psutil.Process()
        self.stop = threading.Event()

    def sample(self):
        while not self.stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.baseline = self.peak = self.process.memory_info().rss
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        LearningWindow.active = True
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.seconds = time.perf_counter()-self.start
        LearningWindow.active = False
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def run_worker(fixture, expected_sha, output, batch_size, repeat):
    """Đo base +5 phase; evaluation/NPZ/kiểm nghiệm nằm ngoài timer và RSS window."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result = {'status':'running', 'verification_passed':False, 'batch_size':batch_size,
              'repeat':repeat, 'pid':os.getpid(), 'fixture_sha256':expected_sha, 'results':[]}
    save_json(result, output/'results.json')
    try:
        if batch_size < 1 or repeat < 1:
            raise ValueError('Batch/repeat phải dương')
        meta, train, test, expansion = load_fixture(fixture, expected_sha)
        config = {**meta['config'], 'analytic_batch_size':batch_size, 'base_statistics_batch_size':batch_size}
        result.update(config=config, fixture_scope=meta['scope'], environment={'python':platform.python_version(),
                      'numpy':np.__version__, 'cpu_count':os.cpu_count(), 'device':'cpu'})
        with threadpool_limits(limits=config['torch_threads']):
            # Warm-up BLAS/LAPACK nhỏ, chung mọi case; không phải lượt đo toàn bộ.
            np.linalg.solve(np.eye(32), np.ones((32,2))); cho_factor(np.eye(32))
            head = StreamingHead(config)
            assert np.array_equal(head.expansion, expansion)
            result['threadpools'] = [{key:item.get(key) for key in ('internal_api','user_api','num_threads','version','architecture')}
                                     for item in threadpool_info()]
            assert result['threadpools'] and all(item['num_threads']==config['torch_threads'] for item in result['threadpools'])
            gc.collect()
            initial_rss = psutil.Process().memory_info().rss
            peaks = []; learned = []
            for phase, pair in enumerate(train):
                classes = 50+10*phase
                with LearningWindow() as window:
                    if phase == 0:
                        count = head.fit_base(batches(pair,batch_size), classes)
                        number_batches = (count+batch_size-1)//batch_size
                    else:
                        count, number_batches = head.update_phase(batches(pair,batch_size), classes)
                peaks.append(window.peak); learned.append(window.seconds)
                assert count == meta['train_counts'][phase]
                assert np.isfinite(head.r).all() and np.isfinite(head.w).all()
                keep = test[1] < classes
                selected = (test[0][keep], test[1][keep])
                started = time.perf_counter()
                evaluation = head.evaluate(batches(selected,256), classes,
                    range(50) if phase==0 else range(classes-10,classes), expected_per_class=meta['test_per_class'])
                evaluation_seconds = time.perf_counter()-started
                path = output/f'phase{phase}_state.npz'
                started = time.perf_counter()
                np.savez_compressed(path, inverse_gram=head.r, weights=head.w)
                io_seconds = time.perf_counter()-started
                row = {'phase':phase, 'n_classes':classes, 'train_samples_current_phase':count,
                    'analytic_batches':number_batches, 'learning_seconds':window.seconds,
                    'learning_peak_rss_bytes':window.peak, 'learning_baseline_rss_bytes':window.baseline,
                    'evaluation_seconds_excluded':evaluation_seconds, 'state_io_seconds_excluded':io_seconds,
                    'weights_shape':list(head.w.shape), 'inverse_gram_shape':list(head.r.shape),
                    'state_file':path.name, 'state_sha256':sha(path), 'checks_passed':True, **evaluation}
                result['results'].append(row); save_json(result,output/'results.json')
                print(f'R3 batch={batch_size} repeat={repeat} phase={phase} seconds={window.seconds:.3f} accuracy={evaluation["accuracy_pct"]:.4f}',flush=True)
            result.update(status='complete', verification_passed=True,
                base_seconds=learned[0], incremental_seconds=sum(learned[1:]), learning_seconds=sum(learned),
                baseline_rss_bytes=initial_rss, peak_learning_rss_bytes=max(peaks),
                rss_increase_bytes=max(0,max(peaks)-initial_rss), rss_sampling_interval_seconds=0.005,
                persistent_analytic_state_bytes=head.state_bytes(), learner_stored_training_samples=0,
                timing_scope='FE transform + base Gram/solve + incremental RLS; sum of6 isolated learning windows',
                excluded_from_timing=['checkpoint/CNN','data loading','fixture checksum','warm-up','evaluation','oracle','state I/O','tracking'],
                input_fixture_bytes=sum(raw.nbytes+y.nbytes for raw,y in train)+test[0].nbytes+test[1].nbytes,
                worker_source_sha256=sha(Path(__file__)))
            save_json(result,output/'results.json')
            return result
    except BaseException as error:
        result.update(status='failed', verification_passed=False, error_type=type(error).__name__)
        save_json(result,output/'results.json'); raise


def run_oracle(fixture, expected_sha, output):
    """Joint độc lập cho6 mốc; chạy sau toàn bộ9 worker, không thuộc vùng đo."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result={'status':'running','verification_passed':False,'fixture_sha256':expected_sha,'results':[]}
    save_json(result,output/'results.json')
    try:
        meta,train,test,expansion=load_fixture(fixture,expected_sha)
        with threadpool_limits(limits=meta['config']['torch_threads']):
            features=[];labels=[]
            for phase,(raw,y) in enumerate(train):
                features.append(np.maximum(raw@expansion,0.0));labels.append(y)
                h=np.concatenate(features);seen_labels=np.concatenate(labels);classes=50+10*phase
                started=time.perf_counter()
                w=joint_ridge(h,seen_labels,classes,meta['config']['gamma'])
                gram=h.T@h+meta['config']['gamma']*np.eye(h.shape[1])
                r=np.linalg.solve(gram,np.eye(h.shape[1]))
                seconds=time.perf_counter()-started
                test_h=np.maximum(test[0][test[1]<classes]@expansion,0.0)
                scores=test_h@w;targets=test[1][test[1]<classes];pred=scores.argmax(1)
                path=output/f'phase{phase}_joint.npz'
                np.savez_compressed(path,inverse_gram=r,weights=w,scores=scores,predictions=pred)
                row={'phase':phase,'n_classes':classes,'train_samples_all_seen':len(h),
                     'test_samples_all_seen':len(targets),'accuracy_pct':100*int(np.count_nonzero(pred==targets))/len(targets),
                     'base_accuracy_pct':100*int(np.count_nonzero(pred[targets<50]==targets[targets<50]))/int(np.count_nonzero(targets<50)),
                     'solve_seconds_excluded':seconds,'file':path.name,'sha256':sha(path)}
                result['results'].append(row);save_json(result,output/'results.json')
                del h,seen_labels,gram,w,r,test_h,scores,pred
                print('R3 ORACLE phase',phase,'complete (outside all timed workers)',flush=True)
        result.update(status='complete',verification_passed=True,method='joint_ridge solve for W; independent full Gram solve for R',
                      keeps_history=True,used_only_for_verification=True)
        save_json(result,output/'results.json');return result
    except BaseException as error:
        result.update(status='failed',verification_passed=False,error_type=type(error).__name__)
        save_json(result,output/'results.json');raise


def compare_case(fixture, expected_sha, case_root, oracle_root):
    """Đối chiếu R/W/score/argmax ở đủ6 phase, không nằm trong timer."""
    case_root,oracle_root=Path(case_root),Path(oracle_root)
    meta,_,test,expansion=load_fixture(fixture,expected_sha)
    case=json.loads((case_root/'results.json').read_bytes());oracle=json.loads((oracle_root/'results.json').read_bytes())
    assert case['status']==oracle['status']=='complete' and len(case['results'])==len(oracle['results'])==6
    config=meta['config'];rows=[]
    with threadpool_limits(limits=config['torch_threads']):
        for actual,reference in zip(case['results'],oracle['results']):
            phase=actual['phase'];classes=50+10*phase
            assert reference['phase']==phase and actual['n_classes']==classes
            assert sha(case_root/actual['state_file'])==actual['state_sha256']
            assert sha(oracle_root/reference['file'])==reference['sha256']
            with np.load(case_root/actual['state_file']) as a,np.load(oracle_root/reference['file']) as b:
                r,w=a['inverse_gram'],a['weights'];rj,wj=b['inverse_gram'],b['weights']
                assert r.shape==rj.shape==(config['expansion_dim'],config['expansion_dim']) and w.shape==wj.shape==(config['expansion_dim'],classes)
                assert all(value.dtype==np.float64 and np.isfinite(value).all() for value in (r,w,rj,wj))
                h=np.maximum(test[0][test[1]<classes]@expansion,0.0);scores=h@w;pred=scores.argmax(1)
                disagreements=int(np.count_nonzero(pred!=b['predictions']))
                checks={'r_allclose':bool(np.allclose(r,rj,rtol=config['rtol'],atol=config['atol'])),
                        'w_allclose':bool(np.allclose(w,wj,rtol=config['rtol'],atol=config['atol'])),
                        'scores_allclose':bool(np.allclose(scores,b['scores'],rtol=config['rtol'],atol=config['atol'])),
                        'predictions_identical':disagreements==0,
                        'evaluation_matches_oracle':actual['accuracy_pct']==reference['accuracy_pct'] and actual['base_accuracy_pct']==reference['base_accuracy_pct']}
                rows.append({'phase':phase,'n_classes':classes,'relative_w_error':float(np.linalg.norm(w-wj)/max(np.linalg.norm(wj),np.finfo(float).tiny)),
                    'relative_r_error':float(np.linalg.norm(r-rj)/max(np.linalg.norm(rj),np.finfo(float).tiny)),
                    'max_abs_r_error':float(np.max(np.abs(r-rj))),'max_abs_score_error':float(np.max(np.abs(scores-b['scores']))),
                    'prediction_disagreements':disagreements,'checks':checks,'passed':all(checks.values())})
    result={'status':'verified' if all(row['passed'] for row in rows) else 'failed','fixture_sha256':expected_sha,
            'batch_size':case['batch_size'],'repeat':case['repeat'],'rtol':config['rtol'],'atol':config['atol'],
            'verification_location':'separate comparator after all timed workers and joint oracle','results':rows}
    save_json(result,case_root/'comparison.json')
    if result['status']!='verified':raise AssertionError('R3 sai lệch ngoài ngưỡng; giữ kết quả và không sửa tolerance theo score')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['worker','oracle'],required=True)
    parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--fixture-sha',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--batch',type=int);parser.add_argument('--repeat',type=int)
    args=parser.parse_args()
    if args.mode=='worker':run_worker(args.fixture,args.fixture_sha,args.output,args.batch,args.repeat)
    else:run_oracle(args.fixture,args.fixture_sha,args.output)


if __name__=='__main__':main()
