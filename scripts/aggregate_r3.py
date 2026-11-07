"""Tổng hợp R3: thống kê3 timing repeats trên1 seed, giữ đủ9 case/54 so sánh."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def stats(values):
    values=[float(value) for value in values]
    assert len(values)==3 and np.isfinite(values).all()
    mean,sd=statistics.mean(values),statistics.stdev(values)
    assert np.isclose(mean,np.mean(values),rtol=1e-12,atol=1e-10)
    assert np.isclose(sd,np.std(values,ddof=1),rtol=1e-12,atol=1e-10)
    return {'mean':mean,'sample_sd':sd,'n':3,'ddof':1,'values_by_repeat':dict(zip(['1','2','3'],values))}


def aggregate(reports):
    load=lambda name:json.loads((reports/name).read_bytes())
    manifest=load('R3-KAGGLE-01-provenance.json');suite=load('r3_cifar100_k5_results.json')
    assert manifest['status']=='complete' and manifest['tracking_status']=='synced'
    assert suite['status']=='complete' and suite['verification_passed'] and len(suite['cases'])==9
    for name,value in manifest['artifacts'].items():
        assert hashlib.sha256((reports/name).read_bytes()).hexdigest()==value['sha256']
    assert suite['fixture_metadata']['scope']=='full_cifar100_r3'
    config=suite['fixture_metadata']['config'];assert config['seed']==42 and config['expansion_dim']==1024 and config['gamma']==0.1
    assert suite['fixture_metadata']['train_counts']==[25000,5000,5000,5000,5000,5000]
    assert suite['fixture_metadata']['train_per_class']==500 and suite['fixture_metadata']['test_per_class']==100
    assert not suite['fixture_metadata']['training_executed_this_run']
    assert suite['fixture_metadata']['backbone_sha256_before']==suite['fixture_metadata']['backbone_sha256_after']==suite['reference']['backbone_sha256']
    assert set(case['key'] for case in suite['cases'])=={f'b{b}_r{r}' for b in (32,128,512) for r in (1,2,3)}
    assert len({case['report']['pid'] for case in suite['cases']})==9
    oracle=suite['oracle'];assert oracle['status']=='complete' and len(oracle['results'])==6
    for case in suite['cases']:
        r,p=case['report'],case['comparison'];tracking=manifest['cases'][case['key']]['tracking']
        assert r['status']=='complete' and r['verification_passed'] and p['status']=='verified'
        assert tracking['verified_state']=='finished' and tracking['public_api_verified'] and tracking['six_history_rows_match']
        assert r['fixture_sha256']==p['fixture_sha256']==suite['fixture_sha256']
        assert len(r['results'])==len(p['results'])==6 and r['config']['torch_threads']==4
        assert r['learner_stored_training_samples']==0 and r['rss_sampling_interval_seconds']==0.005
        assert r['persistent_analytic_state_bytes']==9732096
        assert all(row['passed'] and all(row['checks'].values()) and row['prediction_disagreements']==0 for row in p['results'])
        assert all(row['accuracy_pct']==reference['accuracy_pct'] for row,reference in zip(r['results'],oracle['results']))
        assert np.isclose(r['learning_seconds'],r['base_seconds']+r['incremental_seconds'],rtol=0,atol=1e-10)
    rows=[]
    fields=('base_seconds','incremental_seconds','learning_seconds','baseline_rss_bytes','peak_learning_rss_bytes','rss_increase_bytes')
    for batch in (32,128,512):
        members=sorted([case for case in suite['cases'] if case['report']['batch_size']==batch],key=lambda case:case['report']['repeat'])
        assert [case['report']['repeat'] for case in members]==[1,2,3]
        for case in members:
            assert [row['analytic_batches'] for row in case['report']['results']]==[(25000+batch-1)//batch]+[(5000+batch-1)//batch]*5
        proofs=[row for case in members for row in case['comparison']['results']]
        rows.append({'batch_size':batch,'timing_repeats':3,'metrics':{name:stats([case['report'][name] for case in members]) for name in fields},
            'accuracy':{'average_incremental_accuracy_pct':statistics.mean(row['accuracy_pct'] for row in oracle['results']),
                'final_accuracy_pct':oracle['results'][-1]['accuracy_pct'],
                'base_forgetting_pp':oracle['results'][0]['base_accuracy_pct']-oracle['results'][-1]['base_accuracy_pct'],
                'n_independent_seeds':1,'between_seed_sd':None,'same_all_repeats_and_joint':True},
            'comparison':{'phase_comparisons':18,'max_relative_w_error':max(row['relative_w_error'] for row in proofs),
                'max_relative_r_error':max(row['relative_r_error'] for row in proofs),
                'max_abs_r_error':max(row['max_abs_r_error'] for row in proofs),
                'max_abs_score_error':max(row['max_abs_score_error'] for row in proofs),'prediction_disagreements':0,'passed':True},
            'base_batches':(25000+batch-1)//batch,'incremental_batches_per_phase':(5000+batch-1)//batch,
            'persistent_analytic_state_bytes':9732096,
            'runs':[{'case_key':case['key'],'repeat':case['report']['repeat'],**manifest['cases'][case['key']]['tracking']} for case in members]})
    baseline=rows[1]['metrics']['learning_seconds']['mean']
    for row in rows:row['learning_time_ratio_vs_batch128']=row['metrics']['learning_seconds']['mean']/baseline
    return {'status':'complete','seed':42,'dimension':1024,'gamma':0.1,'n_seeds':1,'n_cases':9,'n_phase_comparisons':54,
        'timing_repeats_per_batch':3,'ddof':1,'statistical_unit':'timing repeat on one seed; accuracy is not a3-seed benchmark',
        'rows':rows,'oracle_phase_accuracy':[{'phase':row['phase'],'n_classes':row['n_classes'],'accuracy_pct':row['accuracy_pct'],
            'base_accuracy_pct':row['base_accuracy_pct']} for row in oracle['results']],
        'fixture_sha256':suite['fixture_sha256'],'input_manifest':'R3-KAGGLE-01-provenance.json',
        'input_manifest_sha256':hashlib.sha256((reports/'R3-KAGGLE-01-provenance.json').read_bytes()).hexdigest(),
        'same_batch_repeats_bitwise_equal':suite['same_batch_repeats_bitwise_equal'],
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'timing_scope':suite['cases'][0]['report']['timing_scope'],
        'limitations':['one data/model seed','sampled RSS can miss short peaks; includes runtime/input/allocator',
                      'cached F64 is research fixture, not learner replay memory','d1024 analysis is separate from d8000 benchmark']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reports-dir',type=Path,default=ROOT/'reports')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/r3_cifar100_k5_summary.json');args=parser.parse_args()
    def save(value):
        args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');os.replace(tmp,args.output)
    save({'status':'running'})
    try:value=aggregate(args.reports_dir)
    except BaseException as error:
        save({'status':'failed','verification_passed':False,'error_type':type(error).__name__});raise
    save(value);print('R3 complete:3 batches x3 timing repeats,54 joint comparisons; n_seeds=1, sample SD ddof1.')


if __name__=='__main__':main()
