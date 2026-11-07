"""Log/sync hai kết quả đã kiểm; không gọi training hoặc sửa raw numeric JSON."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys


def matches(actual, expected):
    """Số thực chấp nhận round-trip JSON1e-14; bool/int/hash phải chính xác."""
    for key,value in expected.items():
        if key not in actual:return False
        if isinstance(value,bool) or isinstance(value,(int,str)):
            if type(actual[key]) is not type(value) or actual[key]!=value:return False
        elif isinstance(value,float):
            if not math.isclose(actual[key],value,rel_tol=1e-14,abs_tol=0):return False
        elif actual[key]!=value:return False
    return True


def track(suite_path, output):
    """Tạo payload một lần; sync lại payload nếu cần; chỉ ready khi public/finished."""
    import requests
    import wandb
    suite_path,output=Path(suite_path),Path(output)
    raw=suite_path.read_bytes();suite=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    assert suite['status']=='complete' and suite['verification_passed']
    assert suite['verification_scope'].startswith('analytic C1/C2')
    assert {case['case_key'] for case in suite['cases']}=={'controlled','native_main'}
    work=output.parent;work.mkdir(parents=True,exist_ok=True)
    receipt=work/'tracking_capture.json'
    if receipt.exists():
        capture=json.loads(receipt.read_bytes());assert capture['numeric_sha256']==digest
    else:
        capture={'numeric_sha256':digest,'runs':[]}
    def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    api=wandb.Api(timeout=45);verified=[]
    for case in suite['cases']:
        rows=case['results'];assert len(rows)==6 and all(row['passed'] for row in rows)
        key=case['case_key']
        config={'experiment':'ACIL-author-comparison','case_key':key,'seed':42,'dimension':1024,'gamma':.1,
            'analytic_batch_size':128,'n_train':50000,'n_test':10000,'n_phases':5,'dtype':'float64','device':'cpu',
            'author_commit':suite['author_commit'],'author_source_sha256':suite['author_source_sha256'],
            'fixture_sha256':suite['fixture_sha256'],'base_checkpoint_sha256':suite['base_checkpoint_sha256'],
            'runner_sha256':suite['execution_source']['runner_sha256'],'execution_source_commit':suite['execution_source']['group_git_commit'],
            'numeric_results_sha256':digest,'numeric_environment':suite['environment'],'training_executed':False,
            'scope':'one seed/d1024; same completed group checkpoint; not native fresh training benchmark',
            'feature_expansion_origin':'R3 N01 seed1042' if key=='controlled' else 'author native RandomBuffer seed42',
            'backbone_fp32_all_checks_passed':suite['diagnostics']['all_fp32_backbone_checks_passed']}
        expected={**{k:v for k,v in case['metrics'].items() if v is not None},
            'author_compare/analytic_equivalence_passed':True,'author_compare/n_phase_comparisons':6,
            'author_compare/total_prediction_disagreements_vs_group':sum(r['group_comparison']['prediction_disagreements'] for r in rows),
            'author_compare/max_relative_w_error_vs_group':max(r['group_comparison']['relative_w_error'] for r in rows),
            'author_compare/max_relative_w_error_vs_joint':max(r['joint_comparison']['relative_w_error'] for r in rows),
            'author_compare/max_abs_score_error_vs_group':max(r['group_comparison']['max_abs_score_error'] for r in rows),
            'author_compare/raw_results_sha256':digest,'author_compare/backbone_fp32_all_checks_passed':suite['diagnostics']['all_fp32_backbone_checks_passed']}
        prior=next((r for r in capture['runs'] if r['case_key']==key),None)
        if prior is None:
            with wandb.init(entity='hsangg-2006',project='acil-reproduction',mode='offline',dir=str(work),
                name='AUTHOR-COMPARE-'+key,job_type='author-code-comparison',tags=['author-comparison',key,'seed42','d1024'],config=config) as run:
                for r in rows:
                    run.log({'phase':r['phase'],'n_classes':r['n_classes'],'test_samples':r['test_samples'],
                        'accuracy_pct':r['accuracy_pct'],'base_accuracy_pct':r['base_accuracy_pct'],
                        'group_relative_w_error':r['group_comparison']['relative_w_error'],
                        'joint_relative_w_error':r['joint_comparison']['relative_w_error'],
                        'group_max_abs_score_error':r['group_comparison']['max_abs_score_error'],
                        'prediction_disagreements':r['group_comparison']['prediction_disagreements'],'phase_comparison_passed':r['passed']},step=r['phase'])
                run.summary.update(expected)
                prior={'case_key':key,'offline_run_id':run.id,'offline_dir':str(Path(run.dir).parent)}
            payload=Path(prior['offline_dir'])/('run-'+prior['offline_run_id']+'.wandb')
            prior['offline_payload_sha256']=hashlib.sha256(payload.read_bytes()).hexdigest()
            capture['runs'].append(prior);save(receipt,capture)
        payload=Path(prior['offline_dir'])/('run-'+prior['offline_run_id']+'.wandb')
        assert hashlib.sha256(payload.read_bytes()).hexdigest()==prior['offline_payload_sha256']
        if not payload.with_suffix('.wandb.synced').exists():
            result=subprocess.run([str(Path(sys.executable).with_name('wandb.exe')),'sync','--entity','hsangg-2006','--project','acil-reproduction',str(payload)],
                env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'),capture_output=True,text=True,encoding='utf-8',timeout=180)
            (work/f'sync_{key}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8',newline='\n')
            assert result.returncode==0,'Sync failed; keep raw evidence and exact payload, retry tracking only'
        actual=api.run('hsangg-2006/acil-reproduction/'+prior['offline_run_id'])
        assert actual.state=='finished' and all(actual.config[k]==v for k,v in config.items())
        assert matches(dict(actual.summary),expected),'Tracking summary mismatch; numerical results unchanged'
        history=sorted(actual.scan_history(keys=['phase','accuracy_pct','base_accuracy_pct','group_relative_w_error','prediction_disagreements']),key=lambda r:r['phase'])
        assert len(history)==6
        for item,row in zip(history,rows):
            assert matches(item,{'phase':row['phase'],'accuracy_pct':row['accuracy_pct'],'base_accuracy_pct':row['base_accuracy_pct'],
                'group_relative_w_error':row['group_comparison']['relative_w_error'],'prediction_disagreements':row['group_comparison']['prediction_disagreements']})
        session=requests.Session();session.trust_env=False
        query='query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){name state config summaryMetrics}}}'
        response=session.post('https://api.wandb.ai/graphql',json={'query':query,'variables':{'e':'hsangg-2006','p':'acil-reproduction','r':actual.id}},timeout=45)
        assert response.status_code==200 and not any(k in response.request.headers for k in ('Authorization','Cookie'))
        body=response.json();assert not body.get('errors');public=body['data']['project']['run']
        assert public['name']==actual.id and public['state']=='finished'
        public_config={k:v['value'] for k,v in json.loads(public['config']).items() if isinstance(v,dict) and 'value'in v}
        assert all(public_config[k]==v for k,v in config.items()) and matches(json.loads(public['summaryMetrics']),expected)
        verified.append({'case_key':key,'wandb_run_id':actual.id,'wandb_run_url':actual.url,'status':'synced','verified_state':'finished',
            'public_api_verified':True,'six_history_rows_match':True,'config_source_summary_match':True,
            'offline_payload_sha256':prior['offline_payload_sha256'],'checked_at_utc':datetime.now(timezone.utc).isoformat()})
        print(key,actual.url,'Finished/public,6 rows matched',flush=True)
    ready={'status':'complete','tracking_status':'synced','numeric_source_file':suite_path.name,'numeric_sha256':digest,
        'n_new_runs':2,'verification_scope':suite['verification_scope'],'runs':verified}
    save(output,ready);return ready


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();track(args.suite,args.output)
