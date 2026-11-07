"""Chốt summary R3 sau sync: tách tổng cả chuỗi khỏi metric history từng phase."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def expected_summary(case):
    """Chỉ lấy số từ report hoàn tất đã so joint; không sửa dữ liệu thực nghiệm."""
    r,p=case['report'],case['comparison']
    assert r['status']=='complete' and r['verification_passed'] and p['status']=='verified'
    assert len(r['results'])==len(p['results'])==6
    assert all(row['passed'] and row['prediction_disagreements']==0 for row in p['results'])
    assert math.isclose(r['learning_seconds'],sum(row['learning_seconds'] for row in r['results']),rel_tol=0,abs_tol=1e-10)
    value={field:r[field] for field in ('base_seconds','incremental_seconds','learning_seconds','baseline_rss_bytes',
        'peak_learning_rss_bytes','rss_increase_bytes','persistent_analytic_state_bytes')}
    value.update(verification_passed=True,joint_comparison_passed=True,training_executed_this_run=False,
        average_incremental_accuracy_pct=sum(row['accuracy_pct'] for row in r['results'])/6,
        final_accuracy_pct=r['results'][-1]['accuracy_pct'],max_relative_w_error=max(row['relative_w_error'] for row in p['results']),
        max_relative_r_error=max(row['relative_r_error'] for row in p['results']),
        max_abs_score_error=max(row['max_abs_score_error'] for row in p['results']),prediction_disagreements=0)
    value['r3/learning_seconds_total']=value['learning_seconds']
    value['r3/max_abs_score_error_all_phases']=value['max_abs_score_error']
    return value


def reconcile(run,case,raw_suite_sha256):
    """Sau khi run finished, sửa summary từ JSON và tạo alias không trùng history."""
    assert run.state=='finished' and run.id==case['tracking']['offline_run_id']
    assert len(raw_suite_sha256)==64
    expected=expected_summary(case)
    expected.update({'r3/summary_reconciled_from_raw':True,'r3/raw_suite_sha256':raw_suite_sha256})
    before={name:run.summary.get(name) for name in ('learning_seconds','max_abs_score_error')}
    run.summary.update(expected)
    return {'before':before,'expected':expected,'method':'after Finished/sync; summary derived from raw six-phase report, unique r3 aliases added'}


def summary_matches(actual,expected):
    """JSON service có thể đổi1 ULP: float rtol1e-14/atol0; int/bool/hash đúng tuyệt đối."""
    for key,value in expected.items():
        observed=actual.get(key)
        if isinstance(value,float):
            if isinstance(observed,bool) or not isinstance(observed,(int,float)):
                return False
            if not math.isclose(observed,value,rel_tol=1e-14,abs_tol=0):return False
        elif isinstance(value,bool):
            if observed is not value:return False
        elif observed!=value:return False
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',type=Path,required=True);args=parser.parse_args()
    data=args.suite.read_bytes();suite=json.loads(data);digest=hashlib.sha256(data).hexdigest()
    assert suite['status']=='complete' and suite['verification_passed'] and len(suite['cases'])==9
    import wandb
    api=wandb.Api(timeout=30)
    for case in suite['cases']:
        run=api.run('hsangg-2006/acil-reproduction/'+case['tracking']['offline_run_id'])
        reconcile(run,case,digest)
        print('Reconciled',case['key'],run.url)


if __name__=='__main__':main()
