"""Chẩn đoán hậu nghiệm: nạp state base nhóm rồi chỉ chạy incremental tác giả.

Không thay thế C1 cold-start: giữ nguyên kết quả fail pointwise của C1.
Không có BP mới. Mọi phép RLS gọi source tác giả ở checkout ngoài repo.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_author_acil import AUTHOR_COMMIT,comparison,sha,save,git


def run(suite_path,r3_root,author_root,output):
    """Đếm các score fail C1; kiểm5 cập nhật sau một base state đã căn chỉnh."""
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    suite_path,r3_root,author_root,output=map(Path,(suite_path,r3_root,author_root,output))
    suite=json.loads(suite_path.read_bytes());assert len(suite['cases'])==2
    assert suite['author_commit']==git(author_root,'rev-parse','HEAD')==AUTHOR_COMMIT
    assert suite['author_source_unchanged_after_execution']
    for name,expected in suite['author_source_sha256'].items():assert sha(author_root/name)==expected
    sys.path.insert(0,str(author_root));ACIL=importlib.import_module('analytic').ACIL
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    meta=json.loads((r3_root/'fixture/fixture.json').read_bytes());assert sha(r3_root/'fixture/fixture.json')==suite['fixture_sha256']
    for name,expected in meta['files'].items():assert sha(r3_root/'fixture'/name)==expected
    source=json.loads((r3_root.parents[1]/'r3_ready.json').read_bytes())
    refs=next(case for case in source['cases'] if case['key']=='b128_r1')['report']['results']
    expansion=np.load(r3_root/'fixture/expansion.npy');raw=np.load(r3_root/'fixture/test_features.npy');labels=np.load(r3_root/'fixture/test_labels.npy')
    output.mkdir(parents=True,exist_ok=False)
    result={'status':'running','verification_passed':False,'purpose':'posthoc localization; not an independent cold-start benchmark',
        'source_suite_sha256':sha(suite_path),'fixture_sha256':suite['fixture_sha256'],'author_commit':AUTHOR_COMMIT,
        'script_sha256':sha(__file__),'source_commit':git(ROOT,'rev-parse','HEAD'),'n_independent_seeds':1,
        'dimension':1024,'gamma':.1,'batch_size':128,'base_state_origin':'R3 b128_r1 phase0, injected, not learned by author in this diagnostic',
        'original_c1_pointwise_score_counts':[],'results':[]}
    save(output/'results.json',result)
    try:
        with threadpool_limits(limits=4):
            model=ACIL(64,torch.nn.Identity(),1024,.1,device='cpu',dtype=torch.float64)
            model.buffer.weight.copy_(torch.from_numpy(expansion.T.copy()))
            c1=next(case for case in suite['cases'] if case['case_key']=='controlled')
            for row,ref in zip(c1['results'],refs):
                phase=row['phase'];keep=labels<50+phase*10
                path=Path(suite['output_dir'])/'controlled'/row['state_file'];assert sha(path)==row['state_sha256']
                with np.load(path) as state:
                    model.analytic_linear.R=torch.from_numpy(state['inverse_gram'].copy())
                    model.analytic_linear.weight=torch.from_numpy(state['weights'].copy())
                with torch.no_grad():scores=model(torch.from_numpy(raw[keep])).numpy()
                with np.load(r3_root/'cases/b128_r1'/ref['state_file']) as state:
                    reference=np.maximum(raw[keep]@expansion,0)@state['weights']
                count=int(np.count_nonzero(~np.isclose(scores,reference,rtol=1e-6,atol=1e-8)))
                result['original_c1_pointwise_score_counts'].append({'phase':phase,'n_score_elements':int(scores.size),
                    'n_elements_outside_original_tolerance':count,
                    'max_error_over_allowed_error':float(np.max(np.abs(scores-reference)/(1e-8+1e-6*np.abs(reference))))})
            base_path=r3_root/'cases/b128_r1'/refs[0]['state_file'];assert sha(base_path)==refs[0]['state_sha256']
            with np.load(base_path) as state:
                model.analytic_linear.R=torch.from_numpy(state['inverse_gram'].copy())
                model.analytic_linear.weight=torch.from_numpy(state['weights'].copy())
            result['base_state_sha256']=sha(base_path)
            for phase in range(1,6):
                x=np.load(r3_root/f'fixture/phase{phase}_features.npy');y=np.load(r3_root/f'fixture/phase{phase}_labels.npy')
                for start in range(0,len(x),128):model.fit(torch.from_numpy(x[start:start+128]),torch.from_numpy(y[start:start+128]))
                model.update();keep=labels<50+phase*10
                with torch.no_grad():scores=model(torch.from_numpy(raw[keep])).numpy()
                inverse,weights=model.analytic_linear.R.numpy(),model.analytic_linear.weight.numpy()
                with np.load(r3_root/'cases/b128_r1'/refs[phase]['state_file']) as state:
                    ref_r,ref_w=state['inverse_gram'],state['weights']
                    proof=comparison(inverse,weights,ref_r,ref_w,scores,np.maximum(raw[keep]@expansion,0)@ref_w)
                path=output/f'phase{phase}.npz';np.savez_compressed(path,inverse_gram=inverse,weights=weights)
                result['results'].append({'phase':phase,'n_classes':50+phase*10,'comparison':proof,
                    'accuracy_pct':100*int((scores.argmax(1)==labels[keep]).sum())/int(keep.sum()),'state_file':path.name,'state_sha256':sha(path)})
                save(output/'results.json',result)
                print('BASE_STATE_DIAGNOSTIC phase',phase,'passed',proof['passed'],flush=True)
            result.update(status='complete',verification_passed=all(row['comparison']['passed'] for row in result['results']),
                          completed_at_utc=datetime.now(timezone.utc).isoformat())
            assert all(sha(author_root/name)==expected for name,expected in suite['author_source_sha256'].items())
            save(output/'results.json',result);return result
    except BaseException as error:
        result.update(status='failed',verification_passed=False,error_type=type(error).__name__,error=str(error));save(output/'results.json',result);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('suite','r3-root','author-root','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();run(args.suite,args.r3_root,args.author_root,args.output)
