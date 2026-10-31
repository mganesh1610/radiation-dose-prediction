"""Independent arithmetic and deterministic replay of the saved one-time results."""
import csv,json,statistics,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy import stats
import torch
from audit_locked_metadata import ROOT,sha
from final_model_bundle import predict_panel

LOG=ROOT/'context/locked-evaluation/v1';MODELS=ROOT/'data/models/final-selection-v2'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open(),delimiter='\t'))
def close(a,b):
    if not np.allclose(a,b,rtol=0,atol=1e-11):raise ValueError(f'Numerical audit mismatch: {a} vs {b}')


def main():
    start=time.perf_counter();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for name in ['result-freeze.json','pre-execution-freeze.json']:
        for p,h in read(LOG/name)['sha256'].items():
            path=LOG/p if name=='result-freeze.json' else ROOT/p
            if sha(path)!=h:raise ValueError('Evaluation artifact changed: '+p)
    protected=read(LOG/'protected-before.json')
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('Prior artifact changed')
    cohort=read(ROOT/'context/locked-eligibility/v1/eligible-samples.json');lookup={r['gsm']:r for r in cohort}
    ids=list(lookup);groups=sorted({r['donor_group'] for r in cohort});preds=rows(LOG/'predictions.tsv')
    if len(preds)!=936:raise ValueError('Missing prediction rows')
    metrics={(r['model'],r['seed']):r for r in read(LOG/'model-metrics.json')}
    saved_donors={(r['model'],int(r['seed']),r['donor_group']):r for r in rows(LOG/'donor-metrics.tsv')}
    saved_strata={(r['model'],int(r['seed']),r['stratum']):r for r in rows(LOG/'stratum-metrics.tsv')}
    ix=np.load(LOG/'bootstrap-donor-indices.npy',allow_pickle=False)
    if not np.array_equal(ix,np.random.default_rng(20261003).integers(0,12,(10000,12))):raise ValueError('Bootstrap design changed')
    with np.load(ROOT/'data/processed_locked/GSE90909/v1/frozen-panels.npz',allow_pickle=False) as z:
        panels={k:z[k].copy() for k in ['log2','rank']}
        if list(z['sample_ids'])!=ids:raise ValueError('Panel order changed')
    index=read(MODELS/'artifact-index.json');donor_vectors={};replays=[];max_single=0.
    for artifact in index['artifacts']:
        name,seed=artifact['model'],artifact['seed'];selected=[r for r in preds if r['model']==name and int(r['seed'])==seed]
        if [r['profile_id'] for r in selected]!=ids:raise ValueError('Incomplete/reordered predictions')
        for r in selected:
            truth=lookup[r['profile_id']]
            if r['donor_group']!=truth['donor_group'] or float(r['actual_gy'])!=truth['dose_gy']:raise ValueError('Label mismatch')
        signed={g:[float(r['predicted_gy'])-float(r['actual_gy']) for r in selected if r['donor_group']==g] for g in groups}
        means=np.array([statistics.fmean(abs(e) for e in signed[g]) for g in groups]);donor_vectors[(name,seed)]=means
        bias=statistics.fmean(statistics.fmean(signed[g]) for g in groups)
        rmse=np.sqrt(statistics.fmean(statistics.fmean(e*e for e in signed[g]) for g in groups))
        reported=metrics[(name,seed)];close(means.mean(),reported['donor_mae']);close(bias,reported['donor_bias']);close(rmse,reported['donor_rmse'])
        residual=np.array([float(r['predicted_gy'])-float(r['actual_gy']) for r in selected])
        close([abs(residual).mean(),np.sqrt(np.mean(residual**2)),residual.mean()],
              [reported['pooled_mae'],reported['pooled_rmse'],reported['pooled_bias']])
        for i,g in enumerate(groups):
            row=saved_donors[(name,seed,g)];close(means[i],float(row['mae']))
            close(statistics.fmean(signed[g]),float(row['bias']))
            close(statistics.fmean(e*e for e in signed[g]),float(row['mse']))
            if int(row['profiles'])!=len(signed[g]):raise ValueError('Donor denominator changed')
        boot=np.array([statistics.fmean(means[i] for i in draw) for draw in ix])
        close(np.quantile(boot,[.025,.975]),[reported['mae_bootstrap_ci'][k] for k in ['low','high']])
        for title,predicate in [('control',lambda y:y==0),('above_2gy',lambda y:y>2)]+[(f'dose_{d:g}gy',lambda y,d=d:y==d) for d in sorted({r['dose_gy'] for r in cohort})]:
            subset=[r for r in selected if predicate(float(r['actual_gy']))];sg=sorted({r['donor_group'] for r in subset})
            e={g:[float(r['predicted_gy'])-float(r['actual_gy']) for r in subset if r['donor_group']==g] for g in sg}
            sr=saved_strata[(name,seed,title)]
            if int(sr['profiles'])!=len(subset) or int(sr['donors'])!=len(sg):raise ValueError('Stratum denominator mismatch')
            close(statistics.fmean(statistics.fmean(abs(x) for x in e[g]) for g in sg),float(sr['donor_mae']))
            close(statistics.fmean(statistics.fmean(e[g]) for g in sg),float(sr['donor_bias']))
        path=MODELS/artifact['path'];observed=np.array([float(r['predicted_gy']) for r in selected])
        if name=='median_null':actual=np.full(52,read(path/'model.json')['prediction_gy'])
        else:
            rep=read(path/'result.json')['params']['representation'];panel=panels[rep]
            actual=predict_panel(path,panel,index['identity'])
            if artifact['seed_role']=='primary':
                one=predict_panel(path,panel[[0]].copy(),index['identity'])[0]
                max_single=max(max_single,float(abs(one-observed[0])))
        diff=float(np.max(abs(actual-observed)));close(actual,observed)
        replays.append(dict(model=name,seed=seed,max_absolute_difference=diff))
    comparisons=read(LOG/'primary-comparisons.json');ps=[];difference_rows=rows(LOG/'paired-donor-differences.tsv')
    for name in 'BCD':
        d=donor_vectors[('A',20261002)]-donor_vectors[(name,20261002)];se=statistics.stdev(d)/np.sqrt(12)
        t=statistics.fmean(d)/se;p=float(stats.t.cdf(t,11));c=comparisons[name]
        close([d.mean(),t,p],[c['mean_difference'],c['t'],c['raw_p']]);ps.append(p)
        ci=d.mean()+np.array([-1,1])*stats.t.isf(.05/6,11)*se
        close(ci,[c['simultaneous_t_ci'][k] for k in ['low','high']])
        boot=np.array([statistics.fmean(d[i] for i in draw) for draw in ix])
        close(np.quantile(boot,[.025,.975]),[c['bootstrap_ci'][k] for k in ['low','high']])
        close(d,[float(r[f'A_minus_{name}']) for r in difference_rows])
    order=np.argsort(ps,kind='stable');adjusted=np.empty(3)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(np.array(ps)[order]*[3,2,1]))
    close(adjusted,[comparisons[n]['holm_adjusted_p'] for n in 'BCD'])
    if max_single>1e-5:raise ValueError('Single/batch model replay outside established tolerance')
    result=dict(status='PASS_INDEPENDENT_LOCKED_RESULT_AUDIT',utc=datetime.now(timezone.utc).isoformat(),
       prediction_rows=936,models=18,donor_rows=216,stratum_rows=162,primary_comparisons=3,
       bootstrap_resamples=10000,protected_files_unchanged=len(protected),all_saved_predictions_replayed=True,
       max_batch_replay_difference=max(r['max_absolute_difference'] for r in replays),
       max_single_batch_difference=max_single,seven_primary_single_input_replays=True,
       inference_reruns_for_deterministic_verification_only=True,no_new_models_fitted=True,
       seconds=time.perf_counter()-start,replays=replays,verifier_sha256=sha(Path(__file__)))
    (LOG/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='replays'},indent=2))


if __name__=='__main__':main()
