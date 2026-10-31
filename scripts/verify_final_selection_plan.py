"""Metadata/configuration checks only; no expression reader or model imports."""
import collections,copy,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'context/final-selection/v2'
EXPECTED_CORE={'GSE102971':100,'GSE8917':25,'GSE65292':20}
EXPECTED_AUX={'GSE85570':400,'GSE97000':12,'GSE20173':10}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def validate(splits,config,jobs):
    rows=splits['profiles'];lookup={r['profile_id']:r for r in rows}
    if len(rows)!=len(lookup) or len(rows)!=567:raise ValueError('Duplicate or wrong membership')
    core=splits['core_profiles'];aux=splits['auxiliary_profiles']
    if len(core)!=145 or len(aux)!=422 or len(set(core))!=145 or len(set(aux))!=422 or set(core)&set(aux) or set(core)|set(aux)!=set(lookup):raise ValueError('Invalid roles/membership')
    if dict(collections.Counter(lookup[s]['study'] for s in core))!=EXPECTED_CORE or dict(collections.Counter(lookup[s]['study'] for s in aux))!=EXPECTED_AUX:raise ValueError('Forbidden study or wrong counts')
    if any(lookup[s]['role']!='core_development' for s in core) or any(lookup[s]['role']=='core_development' for s in aux):raise ValueError('Role mismatch')
    for donor in {r['donor_group'] for r in rows}:
        if len({r['study'] for r in rows if r['donor_group']==donor})!=1:raise ValueError('Donor spans studies')
    if len({lookup[s]['donor_group'] for s in core})!=30 or len({lookup[s]['donor_group'] for s in aux})!=211:raise ValueError('Donor count changed')
    if len(splits['folds'])!=5:raise ValueError('Wrong fold count')
    held=[]
    for k,f in enumerate(splits['folds']):
        tr,va=f['training'],f['validation'];td={lookup[s]['donor_group'] for s in tr};vd={lookup[s]['donor_group'] for s in va}
        if len(tr)!=116 or len(va)!=29 or len(set(tr))!=116 or len(set(va))!=29 or set(tr)&set(va) or set(tr)|set(va)!=set(core) or td&vd:raise ValueError('Fold membership or donor leakage')
        if len(td)!=24 or len(vd)!=6 or f['auxiliary_training']!=aux or f['seed']!=20261002+k or f['fold_id']!=f'selection-{k}':raise ValueError('Fold contract changed')
        if dict(collections.Counter(lookup[s]['study'] for s in va))!={'GSE102971':20,'GSE8917':5,'GSE65292':4}:raise ValueError('Study balance changed')
        held+=va
    if collections.Counter(held)!=collections.Counter(core):raise ValueError('Repeated or missing validation profile')
    if splits['full_refit']['core_training']!=core or splits['full_refit']['auxiliary_training']!=aux:raise ValueError('Full-fit membership changed')
    if config['locked_test_included'] or config['cloud_execution_enabled'] or config['features']!=500:raise ValueError('Scope violation')
    if set(config['candidates'])!={'A','B','C','D','elastic_net','pls','gradient_boosting'}:raise ValueError('Unapproved family')
    for m,c in config['candidates'].items():
        if len(c)!=8 or [p['id'] for p in c]!=list(range(8)):raise ValueError('Candidate budget changed')
    expected={(f['fold_id'],m,p['id'],f['seed'],m in ('B','C','D')) for f in splits['folds'] for m,ps in config['candidates'].items() for p in ps}
    actual={(r['fold'],r['model'],r['candidate_id'],r['seed'],r['auxiliary_used']) for r in jobs}
    if len(jobs)!=280 or actual!=expected or len({r['job_id'] for r in jobs})!=280 or any(r['status']!='planned' for r in jobs):raise ValueError('Job map changed')

def main():
    freeze=read(OUT/'protocol-freeze.json');config=read(OUT/'config.json');splits=read(OUT/'splits.json');jobs=read(OUT/'planned-jobs.json');scope=read(OUT/'prediction-scope.json');stat=read(OUT/'statistical-plan.json');art=read(OUT/'expected-artifacts.json')
    for n,h in freeze['sha256'].items():assert sha(OUT/n)==h,n
    assert sha(ROOT/'scripts/prepare_final_selection.py')==freeze['preparation_source_sha256']
    evidence=read(OUT/'evidence-hashes.json')
    for n,h in evidence.items():assert sha(ROOT/n)==h,n
    neural=read(ROOT/'context/model-abcd/v1/config.json');classical=read(ROOT/'context/classical-comparison/v1/config.json')
    assert config['candidates']==neural['candidates']|classical['candidates'] and config['hidden']==[64,16]
    source=read(ROOT/config['release']/'metadata.json');expected={r['profile_id']:{k:r[k] for k in ['profile_id','study','donor_group','role']} for r in source['profiles']}
    assert {r['profile_id']:r for r in splits['profiles']}==expected
    validate(splits,config,jobs)
    checks=[]
    def rejected(name,s,c,j):
        try:validate(s,c,j)
        except ValueError:checks.append(name);return
        raise AssertionError('Unsafe fixture accepted: '+name)
    s=copy.deepcopy(splits);s['profiles'][0]['study']='GSE90909';rejected('reserved_study_rejected',s,config,jobs)
    s=copy.deepcopy(splits);f=s['folds'][0];a,b=f['training'][0],f['validation'][0];f['training'][0]=b;f['validation'][0]=a;rejected('split_donor_rejected',s,config,jobs)
    s=copy.deepcopy(splits);s['folds'][1]['validation']=s['folds'][0]['validation'];rejected('repeated_validation_rejected',s,config,jobs)
    c=copy.deepcopy(config);c['candidates']['A'].append(copy.deepcopy(c['candidates']['A'][0]));rejected('extra_trial_rejected',splits,c,jobs)
    j=copy.deepcopy(jobs);j[0]['auxiliary_used']=True;rejected('auxiliary_leak_into_A_rejected',splits,config,j)
    c=copy.deepcopy(config);c['cloud_execution_enabled']=True;rejected('cloud_execution_rejected',splits,c,jobs)
    trained=[a for a in art if a['model']!='median_null'];assert len(trained)==17 and len(art)==18
    for m in config['models']:
        r=[a for a in trained if a['model']==m];assert [a['seed'] for a in r]==([20261002,20261003,20261004] if m in ['A','B','C','D','gradient_boosting'] else [20261002]);assert all(a['candidate_id'] is None and a['status']=='not_trained' for a in r)
    assert scope['status']=='finalized_contract_not_deployed' and scope['defaults']['new_upload']=='analysis_only_pending_compatibility_review' and all(v is None for v in scope['numeric_thresholds'].values())
    assert stat['comparison_pairs']==[['A','B'],['A','C'],['A','D']] and stat['difference']=='A_minus_comparator' and stat['multiplicity']=='holm' and stat['family_size']==3 and stat['primary_test']=='paired_t_alternative_less'
    links=0
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith('http'):continue
            assert (p.parent/target).resolve().exists(),target;links+=1
    fold_summary=[dict(fold=f['fold_id'],train_profiles=len(f['training']),held_profiles=len(f['validation']),train_donors=len({expected[s]['donor_group'] for s in f['training']}),held_donors=len({expected[s]['donor_group'] for s in f['validation']}),seed=f['seed']) for f in splits['folds']]
    save('verification.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='PASS_FROZEN_SELECTION_PLAN_ONLY',core_profiles=145,auxiliary_profiles=422,folds=fold_summary,candidate_fits_planned=280,trained_refits_planned=17,median_artifacts_planned=1,contract_rejection_checks=checks,evidence_files_unchanged=len(evidence),local_links_checked=links,original_grids_exact=True,original_documents_preserved=True,training_executed=False,locked_expression_opened=False,hyperparameters_selected=False,app_enforcement_implemented=False,cloud_github_changed=False,verifier_sha256=sha(Path(__file__))))
    print('PASS: 5 donor-complete folds, 280 planned candidates, 6 invalid contracts rejected; no training/test access.')

if __name__=='__main__':main()
