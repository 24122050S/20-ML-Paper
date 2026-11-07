"""Tổng hợp đối chiếu tác giả; giữ C1 failed gate, không giả tạo nhiều seed."""
from pathlib import Path
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]


def aggregate(reports):
    """Đọc raw/tracking/diagnostic, kiểm lại metric và xuất summary có giới hạn."""
    reports=Path(reports);load=lambda name:json.loads((reports/name).read_bytes())
    digest=lambda name:hashlib.sha256((reports/name).read_bytes()).hexdigest()
    raw=load('author_comparison_results.json');diagnostic=load('author_comparison_base_diagnostic.json');ready=load('author_comparison_ready.json')
    assert raw['status']=='failed' and not raw['verification_passed']
    assert raw['error_type']=='RuntimeError' and raw['error']=='Author comparison differs beyond preregistered thresholds; evidence preserved'
    assert raw['author_source_unchanged_after_execution']
    assert ready['status']=='complete' and ready['tracking_status']=='synced' and ready['n_new_runs']==3
    assert ready['numeric_sha256']==digest('author_comparison_results.json')==diagnostic['source_suite_sha256']
    assert ready['diagnostic_sha256']==digest('author_comparison_base_diagnostic.json')
    assert [r['analytic_equivalence_passed'] for r in ready['runs']]==[False,True,True]
    assert all(r['public_api_verified'] and r['config_source_summary_match'] and r['verified_state']=='finished' for r in ready['runs'])
    rows=[]
    for case in raw['cases']:
        result=case['results'];assert case['status']=='complete' and len(result)==6
        metrics={'average_incremental_accuracy_pct':statistics.mean(r['accuracy_pct'] for r in result),
            'final_accuracy_pct':result[-1]['accuracy_pct'],'base_forgetting_pp':result[0]['base_accuracy_pct']-result[-1]['base_accuracy_pct']}
        assert all(abs(value-case['metrics'][key])<1e-12 for key,value in metrics.items())
        assert bool(case['verification_passed'])==all(r['passed'] for r in result)
        proof={key:sum(r['group_comparison']['checks'][key] for r in result) for key in ('r_allclose','w_allclose','score_allclose','same_predictions')}
        rows.append({'case_key':case['case_key'],'n_phase_comparisons':6,'metrics':metrics,'between_seed_sd':None,
            'equivalence_passed':case['verification_passed'],'group_checks_pass_count':proof,
            'max_relative_w_error_vs_group':max(r['group_comparison']['relative_w_error'] for r in result),
            'max_relative_w_error_vs_joint':max(r['joint_comparison']['relative_w_error'] for r in result),
            'max_abs_score_error_vs_group':max(r['group_comparison']['max_abs_score_error'] for r in result),
            'prediction_disagreements':sum(r['group_comparison']['prediction_disagreements'] for r in result),
            'tracking':next(r for r in ready['runs'] if r['case_key']==case['case_key'])})
    assert [r['group_checks_pass_count']['score_allclose'] for r in rows]==[4,6]
    assert sum(row['n_elements_outside_original_tolerance'] for row in diagnostic['original_c1_pointwise_score_counts'])==2
    assert diagnostic['status']=='complete' and diagnostic['verification_passed'] and len(diagnostic['results'])==5
    assert all(row['comparison']['passed'] for row in diagnostic['results'])
    with (reports/'author_comparison_native_IL.csv').open(newline='') as stream:csv_rows=list(csv.DictReader(stream))
    assert len(csv_rows)==6
    for item,row in zip(csv_rows,raw['cases'][1]['results']):assert abs(100*float(item['acc@1'])-row['accuracy_pct'])<1e-12
    return {'status':'complete_with_differences','scope':'two primary cases plus one posthoc diagnostic; one seed/d1024, no fresh BP',
        'raw_gate_status':raw['status'],'raw_gate_passed':raw['verification_passed'],'author_commit':raw['author_commit'],
        'n_independent_seeds':1,'between_seed_sd':None,'n_primary_cases':2,'n_posthoc_diagnostics':1,
        'n_primary_phase_comparisons':12,'n_posthoc_incremental_comparisons':5,'rows':rows,
        'c1_score_elements_outside_tolerance':2,'c1_total_score_elements':sum(r['n_score_elements'] for r in diagnostic['original_c1_pointwise_score_counts']),
        'posthoc_diagnostic':{'base_state_injected':True,'equivalence_passed':True,'n_incremental_updates':5,
            'max_relative_w_error':max(row['comparison']['relative_w_error'] for row in diagnostic['results']),
            'max_abs_score_error':max(row['comparison']['max_abs_score_error'] for row in diagnostic['results']),
            'tracking':next(r for r in ready['runs'] if r['case_key']=='base_state_diagnostic')},
        'c2_minus_c1_single_seed':{key:rows[1]['metrics'][key]-rows[0]['metrics'][key] for key in metrics},
        'backbone_fp32_checks_all_passed':raw['diagnostics']['all_fp32_backbone_checks_passed'],
        'backbone_prediction_disagreements':raw['diagnostics']['backbone']['prediction_disagreements'],
        'mapping_failure_reproduced':raw['diagnostics']['mapping']['base_training_reproduction'],
        'inputs_sha256':{name:digest(name) for name in ('author_comparison_results.json','author_comparison_base_diagnostic.json','author_comparison_ready.json','author_comparison_native_IL.csv')},
        'limitations':['not native fresh training300epochs/d8192','shared group representation/cache and mapping',
            'C1R is posthoc with injected base, not replacement for C1','Finished tracking does not imply equivalence passed'],
        'created_at_utc':datetime.now(timezone.utc).isoformat()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-dir',type=Path,default=ROOT/'reports');parser.add_argument('--output',type=Path,default=ROOT/'reports/author_comparison_summary.json')
    args=parser.parse_args();value=aggregate(args.reports_dir)
    args.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(value['status'],'C1 gate=false;2 score elements outside original tolerance;no threshold changed')
