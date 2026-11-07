"""Điều phối R3 full CIFAR: cache F64 cố định,9 worker mới, oracle riêng và log offline."""
from pathlib import Path
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid


def prepare_fixture(checkpoint,reference,project_root,directory,device):
    """Trích đủ ảnh một lần, freeze/context/source phải đúng checkpoint R1 seed42."""
    import numpy as np
    import torch
    from src.cifar_data import load_cifar100,audit_data,make_protocol
    from src.cifar_smoke import seed_everything,validate_checkpoint,fingerprint,make_loader,extract_features
    from src.resnet import ResNet32,freeze_backbone
    from src.reporting import save_json,collect_provenance
    from scripts.r3_worker import sha
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    assert sha(checkpoint)==reference['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['checkpoint_format_version']==2 and saved['training_complete']
    assert saved['completed_epoch']==saved['target_epochs']==160 and saved['config']['seed']==42
    config={**reference['baseline_config'],'experiment':'ACIL-R3-C100-K5','purpose':'batch_analysis',
            'seed':42,'device':'cpu','expansion_dim':1024,'feature_expansion':True,'gamma':0.1,
            'torch_threads':4,'run_incremental_phases':5}
    seed_everything(42,4)
    arrays=load_cifar100(directory.parent/'cifar-data');protocol=make_protocol(42,5)
    audit=audit_data(arrays,protocol)
    validate_checkpoint(saved,config,protocol,audit['normalization'],arrays.source['revision'])
    assert arrays.source==reference['dataset_source']
    provenance=collect_provenance(project_root);assert provenance['source_sha256']==reference['source_sha256']
    model=ResNet32().to(device);model.load_state_dict(saved['model_state_dict']);freeze_backbone(model)
    before=fingerprint(model);assert before==reference['backbone_sha256']
    files={}
    def store(name,array):
        np.save(directory/name,array,allow_pickle=False);files[name]=sha(directory/name)
    started=time.perf_counter()
    for phase,classes in enumerate(protocol['phase_global_classes']):
        loader=make_loader(arrays,protocol,audit['normalization'],classes,'train',config,batch_size=256)
        raw,labels=extract_features(model,loader,device,f'R3 train phase {phase}')
        indices=np.flatnonzero(np.isin(np.asarray(protocol['original_to_global'])[arrays.train_labels],classes))
        assert np.array_equal(labels,np.asarray(protocol['original_to_global'])[arrays.train_labels[indices]])
        store(f'phase{phase}_features.npy',raw);store(f'phase{phase}_labels.npy',labels)
        store(f'phase{phase}_indices.npy',indices)
    loader=make_loader(arrays,protocol,audit['normalization'],range(100),'test',config,batch_size=256)
    raw,labels=extract_features(model,loader,device,'R3 test all100')
    store('test_features.npy',raw);store('test_labels.npy',labels)
    store('test_indices.npy',np.arange(10000,dtype=np.int64))
    after=fingerprint(model);assert before==after
    expansion=np.random.default_rng(1042).normal(size=(64,1024));store('expansion.npy',expansion)
    save_json(protocol,directory/'protocol.json');files['protocol.json']=sha(directory/'protocol.json')
    save_json(audit,directory/'data_audit.json');files['data_audit.json']=sha(directory/'data_audit.json')
    meta={'scope':'full_cifar100_r3','config':config,'train_counts':[25000,5000,5000,5000,5000,5000],
          'train_per_class':500,'test_per_class':100,'files':files,'reference':reference,'provenance':provenance,
          'class_order':protocol['class_order'],'normalization':audit['normalization'],
          'backbone_sha256_before':before,'backbone_sha256_after':after,
          'checkpoint_completed_epoch':160,'training_executed_this_run':False,
          'expansion_array_sha256':hashlib.sha256(expansion.tobytes()).hexdigest(),
          'feature_extraction_seconds_excluded':time.perf_counter()-started,
          'feature_extraction_environment':{'python':platform.python_version(),'numpy':np.__version__,
              'torch':str(torch.__version__),'cuda_device':torch.cuda.get_device_name(0),'device':str(device)},
          'fixture_note':'Fixed input for controlled study/oracle; not learner replay storage; train indices ascending inside each phase'}
    save_json(meta,directory/'fixture.json')
    del model,saved,arrays,raw,labels,expansion;gc.collect();torch.cuda.empty_cache()
    return meta


def run_suite(*,reference,checkpoint,project_root,working,test_proof):
    """Không complete nếu thiếu case, sai oracle hoặc chưa tạo đủ9 payload offline."""
    project_root,working=Path(project_root),Path(working);sys.path.insert(0,str(project_root))
    import torch
    import numpy as np
    import wandb
    from src.reporting import save_json
    from scripts.r3_worker import sha,compare_case
    suite_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:10]
    root=working/'r3'/suite_id;root.mkdir(parents=True,exist_ok=True);ready=working/'r3_ready.json'
    state={'status':'running','verification_passed':False,'suite_id':suite_id,'output_dir':str(root),
           'reference':reference,'tests':test_proof,'stage':'checkpoint_validation','cases':[],
           'execution_order':[[32,1],[128,1],[512,1],[128,2],[512,2],[32,2],[512,3],[32,3],[128,3]],
           'statistical_unit':'3 timing repetitions on one fixed seed42; not independent seeds'}
    save_json(state,ready)
    try:
        assert reference['seed']==42 and torch.cuda.is_available() and test_proof['cuda_resume_passed']
        assert test_proof['r3_driver_tests']['verification_passed'] and test_proof['r3_driver_tests']['tests_run']==5
        if checkpoint is None:
            matches=[p for p in sorted(Path('/kaggle/input').rglob('base_checkpoint.pt')) if sha(p)==reference['checkpoint_sha256']]
            if not matches:raise ValueError('Không có checkpoint seed42/160 đúng SHA; không train fallback')
            checkpoint=matches[0]
        if sha(checkpoint)!=reference['checkpoint_sha256']:raise ValueError('Checkpoint không đúng R1 seed42')
        state.update(stage='feature_fixture');save_json(state,ready)
        fixture=root/'fixture';meta=prepare_fixture(Path(checkpoint),reference,project_root,fixture,'cuda')
        digest=sha(fixture/'fixture.json');state['fixture_sha256']=digest;state['fixture_metadata']=meta
        script=project_root/'scripts/r3_worker.py'
        environment=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONPATH=str(project_root),
                         CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
        def launch(mode,output,extra=()):
            process=subprocess.run([sys.executable,'-B','-X','utf8',str(script),'--mode',mode,'--fixture',str(fixture),
                '--fixture-sha',digest,'--output',str(output),*extra],cwd=project_root,env=environment,
                capture_output=True,text=True,encoding='utf-8',timeout=1800)
            output.mkdir(exist_ok=True,parents=True)
            (output/'stdout.log').write_text(process.stdout,encoding='utf-8',newline='\n')
            (output/'stderr.log').write_text(process.stderr,encoding='utf-8',newline='\n')
            print(process.stdout,flush=True)
            if process.returncode:raise RuntimeError('R3 '+mode+' failed; giữ logs/kết quả, không bỏ case')
        for batch,repeat in state['execution_order']:
            key=f'b{batch}_r{repeat}';folder=root/'cases'/key
            state.update(stage='numeric_worker',current_case=key);save_json(state,ready)
            print('R3 START',key,'fresh CPU process,4 threads',flush=True)
            launch('worker',folder,('--batch',str(batch),'--repeat',str(repeat)))
            report=json.loads((folder/'results.json').read_bytes())
            assert report['status']=='complete' and report['verification_passed'] and report['fixture_sha256']==digest
            state['cases'].append({'key':key,'report':report});save_json(state,ready)
        assert len({case['report']['pid'] for case in state['cases']})==9
        state.update(stage='joint_oracle',current_case=None);save_json(state,ready)
        print('R3 ORACLE START: all9 measured workers have already ended',flush=True)
        oracle=root/'oracle';launch('oracle',oracle)
        state['oracle']=json.loads((oracle/'results.json').read_bytes())
        state.update(stage='oracle_comparison');save_json(state,ready)
        for case in state['cases']:
            folder=root/'cases'/case['key']
            proof=compare_case(fixture,digest,folder,oracle)
            case['comparison']=proof;save_json(state,ready)
            print('R3 VERIFIED',case['key'],'6 phases,0 prediction disagreements',flush=True)
        # Ghi nhận bitwise giữa repeat; điều kiện nghiệm thu vẫn là ngưỡng oracle và dự đoán.
        bitwise_equal=True
        for batch in (32,128,512):
            members=[case for case in state['cases'] if case['report']['batch_size']==batch]
            for phase in range(6):
                hashes=[]
                for case in members:
                    with np.load(root/'cases'/case['key']/f'phase{phase}_state.npz') as saved:
                        hashes.append([hashlib.sha256(saved[name].tobytes()).hexdigest() for name in ('inverse_gram','weights')])
                bitwise_equal = bitwise_equal and hashes[0]==hashes[1]==hashes[2]
                assert len({member['report']['results'][phase]['accuracy_pct'] for member in members})==1
        state.update(stage='offline_tracking');save_json(state,ready)
        for case in state['cases']:
            report,proof=case['report'],case['comparison']
            with wandb.init(entity='hsangg-2006',project='acil-reproduction',mode='offline',dir=str(working),
                    name='R3-seed42-'+case['key'],job_type='batch-analysis',tags=['R3','seed42','d1024',case['key'],'offline-to-sync'],
                    config={**report['config'],'r3_batch_size':report['batch_size'],'timing_repeat':report['repeat'],
                        'fixture_sha256':digest,'base_checkpoint_sha256':reference['checkpoint_sha256'],
                        'source_sha256':reference['source_sha256'],'worker_source_sha256':sha(script),
                        'baseline_r1_run_id':reference['wandb_run_id'],'statistical_unit':state['statistical_unit']}) as tracked:
                for row,comparison in zip(report['results'],proof['results']):
                    tracked.log({field:value for field,value in {**row,**comparison}.items() if isinstance(value,(int,float,bool))},step=row['phase'])
                tracked.summary.update({field:report[field] for field in ('base_seconds','incremental_seconds','learning_seconds',
                    'baseline_rss_bytes','peak_learning_rss_bytes','rss_increase_bytes','persistent_analytic_state_bytes')})
                tracked.summary.update({'verification_passed':True,'joint_comparison_passed':True,'training_executed_this_run':False,
                    'average_incremental_accuracy_pct':float(np.mean([row['accuracy_pct'] for row in report['results']])),
                    'final_accuracy_pct':report['results'][-1]['accuracy_pct'],
                    'max_relative_w_error':max(row['relative_w_error'] for row in proof['results']),
                    'max_relative_r_error':max(row['relative_r_error'] for row in proof['results']),
                    'max_abs_score_error':max(row['max_abs_score_error'] for row in proof['results']),
                    'prediction_disagreements':sum(row['prediction_disagreements'] for row in proof['results'])})
                offline_id,directory=tracked.id,str(Path(tracked.dir).parent)
            case['tracking']={'status':'offline_pending_sync','offline_run_id':offline_id,'offline_directory':directory}
            save_json(state,ready)
        state.update(status='complete',verification_passed=True,stage='complete',same_batch_repeats_bitwise_equal=bitwise_equal,
            worker_source_sha256=sha(script),runner_source_sha256=sha(project_root/'scripts/r3_runner.py'))
        save_json(state,ready);print('R3 SUITE COMPLETE:9 workers,54 phase comparisons,9 offline runs',flush=True)
        return state
    except BaseException as error:
        state.update(status='failed',verification_passed=False,error_type=type(error).__name__)
        save_json(state,ready);raise
