"""Independent checks of membership, saved predictions, tuning and prior files."""
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; LOG=ROOT/'context/baseline-modeling/v1'


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def read(name): return json.loads((LOG/name).read_text())
def tsv(name): return list(csv.DictReader((LOG/name).open(),delimiter='\t'))


def main():
    s=read('summary.json'); checked={}
    def check(p,h):
        if sha(ROOT/p)!=h: raise ValueError('Changed file: '+p)
        checked[p]=h
    old=json.loads((ROOT/'context/core-reprocessing/v1/verification.json').read_text())
    for p,h in old['checked_sha256'].items(): check(p,h)
    for p,h in s['input_sha256'].items(): check(p,h)
    for p,h in s['code_sha256'].items(): check(p,h)
    for a in s['artifacts']:
        check(a['path'],a['sha256'])
        assert (ROOT/a['path']).stat().st_size==a['bytes']
    start=read('run-start.json')
    assert start['method_sha256']==sha(LOG/'METHOD.md')
    source=ROOT/'data/processed_pilot/core-reprocessing-v1/common_probe_background_corrected.npz'
    with np.load(source,allow_pickle=False) as z: data={k:z[k] for k in z.files}
    samples=data['sample_ids']; ix={v:i for i,v in enumerate(samples)}
    labels={r['gsm']:r for r in csv.DictReader((ROOT/'context/normalization-validation/v2/core-sample-labels-v2.tsv').open(),delimiter='\t')}
    outer=tsv('outer-predictions.tsv'); assert len(outer)==580
    assert len({(r['model'],r['gsm']) for r in outer})==580
    for r in outer:
        label=labels[r['gsm']]
        assert r['group']==label['split_group_id'] and r['held_study']==label['study']
        assert float(r['actual_gy'])==float(label['candidate_dose_gy'])
        assert np.isfinite(float(r['predicted_gy']))
        assert abs(float(r['error_gy'])-(float(r['predicted_gy'])-float(r['actual_gy'])))<1e-12
    independent=[]
    for m in s['metrics']:
        rr=[r for r in outer if r['model']==m['model'] and (m['study']=='pooled' or r['held_study']==m['study'])]
        groups=defaultdict(list)
        for r in rr: groups[r['group']].append(float(r['predicted_gy'])-float(r['actual_gy']))
        mae=sum(sum(abs(v) for v in e)/len(e) for e in groups.values())/len(groups)
        rmse=(sum(sum(v*v for v in e)/len(e) for e in groups.values())/len(groups))**.5
        bias=sum(sum(e)/len(e) for e in groups.values())/len(groups)
        assert all(abs(value-m[key])<1e-12 for key,value in [('mae',mae),('rmse',rmse),('bias',bias)])
        assert len(groups)==m['donors'] and len(rr)==m['samples']
        independent.append(dict(model=m['model'],study=m['study'],mae=mae,rmse=rmse,bias=bias))
    audits=read('fold-audits.json'); assert len(audits)==18
    audits_by_name={a['fold']:a for a in audits}
    all_core=set(samples)
    for a in audits:
        tr=set(a['training_sample_ids']); va=set(a['validation_sample_ids'])
        assert tr.isdisjoint(va) and tr|va<=all_core
        assert set(a['training_groups']).isdisjoint(a['validation_groups'])
        assert set(a['signature']['fit_sample_ids'])==tr
        assert a['nontraining_expression_mask_label_perturbation_unchanged']
        for membership,ids in [('training',tr),('validation',va)]:
            assert set(a[membership+'_groups'])=={labels[i]['split_group_id'] for i in ids}
        for k in (100,500):
            path=ROOT/f'data/models/core-baselines-v1/{a["fold"]}_genes{k}.npz'
            with np.load(path,allow_pickle=False) as z:
                assert set(z['training_ids'])==tr
                assert np.isfinite(z['medians']).all() and np.isfinite(z['scale']).all() and (z['scale']>0).all()
                assert len(z['selected'])<=k and len(set(z['selected']))==len(z['selected'])
        if a['fold'].endswith('_outer'):
            held=a['fold'].split('_')[1]
            assert tr|va==all_core
            assert all(labels[i]['study']==held for i in va) and all(labels[i]['study']!=held for i in tr)
    inner=tsv('inner-predictions.tsv'); trials=tsv('tuning-trials.tsv')
    assert len(trials)==96 and len({(t['held_study'],t['model'],t['trial']) for t in trials})==96
    for t in trials:
        rr=[r for r in inner if (r['held_study'],r['model'],r['trial'])==(t['held_study'],t['model'],t['trial'])]
        if t['status']=='valid':
            expected=set(audits_by_name[f'holdout_{t["held_study"]}_outer']['training_sample_ids'])
            assert len(rr)==len(expected) and {r['gsm'] for r in rr}==expected
            groups=defaultdict(list)
            for r in rr:
                assert r['gsm'] in audits_by_name[f'holdout_{t["held_study"]}_inner{r["inner_fold"]}']['validation_sample_ids']
                assert float(r['actual_gy'])==float(labels[r['gsm']]['candidate_dose_gy'])
                groups[r['group']].append(abs(float(r['predicted_gy'])-float(r['actual_gy'])))
            score=sum(sum(v)/len(v) for v in groups.values())/len(groups)
            assert abs(score-float(t['inner_donor_mae_gy']))<1e-12
    for selected in s['selected']:
        valid=[t for t in trials if t['held_study']==selected['held_study'] and t['model']==selected['model'] and t['status']=='valid']
        chosen=min(valid,key=lambda t:(float(t['inner_donor_mae_gy']),int(t['trial'])))
        assert json.loads(chosen['parameters'])==selected['parameters'] and int(chosen['trial'])==selected['trial']
    replay=[]
    for held in ('GSE102971','GSE8917','GSE65292'):
        tr=[i for i,sample in enumerate(samples) if labels[sample]['study']!=held]
        train_median=float(np.median([float(labels[samples[i]]['candidate_dose_gy']) for i in tr]))
        assert all(float(r['predicted_gy'])==train_median for r in outer if r['model']=='median_null' and r['held_study']==held)
        for name in ('elastic_net','pls','gradient_boosting'):
            rr=[r for r in outer if r['model']==name and r['held_study']==held]
            idx=[ix[r['gsm']] for r in rr]
            model=joblib.load(ROOT/f'data/models/core-baselines-v1/holdout_{held}_{name}.joblib')
            assert set(model.preparation.training_ids)=={samples[i] for i in tr}
            actual=model.predict(data['values'][:,idx],data['probe_ids'])
            delta=float(np.max(abs(actual-np.array([float(r['predicted_gy']) for r in rr]))))
            assert delta<=1e-8
            try: model.predict(data['values'][:,idx],data['probe_ids'][::-1])
            except ValueError: pass
            else: raise ValueError('Wrong probe order accepted')
            replay.append(dict(held_study=held,model=name,prediction_max_difference_gy=delta,wrong_probe_order_rejected=True))
    log=(LOG/'tests.log').read_text(); assert 'Ran 30 tests' in log and log.rstrip().endswith('OK')
    test_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tests').glob('test*.py'))}
    (LOG/'unit-test-result.json').write_text(json.dumps(dict(tests=30,success=True,test_hashes=test_hashes),indent=2)+'\n')
    staged=subprocess.run(['git','diff','--cached','--name-only'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
    assert not any(p.startswith('context/') or p.split('/')[-1]=='AGENTS.md' for p in staged)
    ignored=subprocess.run(['git','check-ignore','context/baseline-modeling/v1/REPORT.md','context/report-memory/PROJECT_REPORT_LOG.md','AGENTS.md','data/models/core-baselines-v1'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
    assert len(ignored)==4
    extra=[Path(__file__),ROOT/'scripts/report_core_baselines.py',*sorted((ROOT/'tests').glob('test*.py'))]
    for p in extra: checked[str(p.relative_to(ROOT))]=sha(p)
    result=dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),hash_checked_files=len(checked),
                previous_verified_files_unchanged=len(old['checked_sha256']),final_documents_unchanged=True,
                outer_prediction_rows=580,inner_prediction_rows=len(inner),independently_recomputed_metrics=independent,
                fold_audits=18,donor_group_separation=True,nontraining_perturbations_unchanged=True,
                tuning_trials_recomputed=len(trials),selected_hyperparameters_minimize_inner_mae=True,
                saved_predictors_replayed=replay,unit_tests=30,local_context_and_models_ignored=True,
                screenshots=dict(captured=False,reason='Scientific figures are saved. Prior browser local-file security restriction not bypassed.'),
                checked_sha256=checked,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(LOG.iterdir()) if p.is_file() and p.name!='verification.json'})
    (LOG/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checked_sha256','evidence_sha256','independently_recomputed_metrics','saved_predictors_replayed')},indent=2))


if __name__=='__main__': main()
