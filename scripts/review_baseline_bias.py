"""Descriptive bias review of saved nested development predictions; no refit."""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_auxiliary import sha, save_json, tsv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'context/bias-review/v1'
BASE = ROOT / 'context/baseline-modeling/v1'
LABELS = ROOT / 'context/normalization-validation/v2/core-sample-labels-v2.tsv'
STUDIES = ('GSE102971', 'GSE8917', 'GSE65292')
MODELS = ('median_null', 'elastic_net', 'pls', 'gradient_boosting')
COLORS = ('#89939D', '#087E8B', '#9D4EDD', '#C86222')


def balanced(records):
    if not records:
        raise ValueError('Empty metric slice')
    errors = defaultdict(list)
    for r in records:
        errors[r['group']].append(r['predicted_gy']-r['actual_gy'])
    return dict(profiles=len(records), donors=len(errors),
        mae_gy=float(np.mean([np.mean(np.abs(v)) for v in errors.values()])),
        rmse_gy=float(np.sqrt(np.mean([np.mean(np.square(v)) for v in errors.values()]))),
        bias_gy=float(np.mean([np.mean(v) for v in errors.values()])))


def support_slice(row, name):
    dose = row['actual_gy']
    return {'all':True, 'control':dose==0, 'positive_up_to_2Gy':0<dose<=2,
        'above_2Gy':dose>2, 'inside_training_range':row['inside_training_range'],
        'outside_training_range':not row['inside_training_range'],
        'exact_training_dose':row['exact_training_dose'],
        'unseen_training_dose_level':not row['exact_training_dose'],
        '0_or_2Gy':dose in (0,2)}[name]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prior=json.loads((BASE/'summary.json').read_text())
    expected={r['path']:r['sha256'] for r in prior['artifacts']}
    expected.update(prior['input_sha256'])
    verification=json.loads((BASE/'verification.json').read_text())
    expected.update(verification['checked_sha256'])
    expected.update(verification['evidence_sha256'])
    files=[BASE/'outer-predictions.tsv',LABELS,BASE/'metrics.tsv']
    for p in files:
        if sha(p)!=expected[str(p.relative_to(ROOT))]:
            raise ValueError('Prior scientific artifact hash changed')
    labels=list(csv.DictReader(LABELS.open(),delimiter='\t'))
    byid={r['gsm']:r for r in labels}
    if len(byid)!=145 or len(labels)!=145 or set(r['study'] for r in labels)!=set(STUDIES):
        raise ValueError('Unexpected core labels')
    schedules={st:sorted({float(r['candidate_dose_gy']) for r in labels if r['study']==st}) for st in STUDIES}
    raw=list(csv.DictReader((BASE/'outer-predictions.tsv').open(),delimiter='\t'))
    if len(raw)!=580 or len({(r['model'],r['gsm']) for r in raw})!=580:
        raise ValueError('Unexpected prediction rows')
    augmented=[]
    for r in raw:
        if r['model'] not in MODELS or r['held_study'] not in STUDIES:
            raise ValueError('Unexpected model/study')
        r=dict(r)
        for k in ('actual_gy','predicted_gy','error_gy'):r[k]=float(r[k])
        lab=byid[r['gsm']]
        if lab['study']!=r['held_study'] or lab['split_group_id']!=r['group'] or float(lab['candidate_dose_gy'])!=r['actual_gy']:
            raise ValueError('Prediction label or group mismatch')
        if abs((r['predicted_gy']-r['actual_gy'])-r['error_gy'])>1e-12:
            raise ValueError('Incorrect error field')
        training=[x for x in labels if x['study']!=r['held_study']]
        train_doses=np.array([float(x['candidate_dose_gy']) for x in training])
        r.update(training_min_gy=float(train_doses.min()),training_max_gy=float(train_doses.max()),
            inside_training_range=bool(train_doses.min()<=r['actual_gy']<=train_doses.max()),
            exact_training_dose=bool(np.isclose(train_doses,r['actual_gy'],rtol=0,atol=1e-10).any()),
            nearest_training_dose_distance_gy=float(abs(train_doses-r['actual_gy']).min()),
            prediction_outside_training_range=bool(r['predicted_gy']<train_doses.min() or r['predicted_gy']>train_doses.max()))
        augmented.append(r)
    slices=[]
    names=['all','control','positive_up_to_2Gy','above_2Gy','inside_training_range','outside_training_range','exact_training_dose','unseen_training_dose_level','0_or_2Gy']
    for model in MODELS:
        for study in (*STUDIES,'pooled'):
            rr=[r for r in augmented if r['model']==model and (study=='pooled' or r['held_study']==study)]
            for name in names:
                take=[r for r in rr if support_slice(r,name)]
                if take:slices.append(dict(model=model,study=study,slice=name,**balanced(take)))
    dose_rows=[]
    for model in MODELS:
        for study in STUDIES:
            for dose in schedules[study]:
                rr=[r for r in augmented if r['model']==model and r['held_study']==study and r['actual_gy']==dose]
                dose_rows.append(dict(model=model,study=study,dose_gy=dose,**balanced(rr),
                    mean_prediction_gy=float(np.mean([r['predicted_gy'] for r in rr])),
                    training_min_gy=rr[0]['training_min_gy'],training_max_gy=rr[0]['training_max_gy'],
                    exact_training_dose=rr[0]['exact_training_dose']))
    orig=list(csv.DictReader((BASE/'metrics.tsv').open(),delimiter='\t'))
    # The original table identifies its pooled row with study='pooled'.
    for r in slices:
        if r['slice']!='all':continue
        prior_row=next(x for x in orig if x['model']==r['model'] and x['study']==r['study'])
        for k in ['mae_gy','rmse_gy','bias_gy']:
            if not np.isclose(r[k],float(prior_row[k.removesuffix('_gy')]),atol=1e-10,rtol=0):raise ValueError('Metric replay mismatch')
    tsv(OUT/'prediction-support.tsv',augmented)
    tsv(OUT/'slice-metrics.tsv',slices)
    tsv(OUT/'dose-residuals.tsv',dose_rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,study in zip(axes,STUDIES):
        for model,color in zip(MODELS,COLORS):
            rr=[r for r in dose_rows if r['study']==study and r['model']==model]
            ax.plot([r['dose_gy'] for r in rr],[r['bias_gy'] for r in rr],marker='o',label=model.replace('_',' '),color=color)
        ax.axhline(0,color='#333333',lw=.8)
        ax.set(title=study,xlabel='Observed dose (Gy)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean prediction − observed dose (Gy)')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Saved nested-development predictions: residuals by held-out study and dose')
    fig.tight_layout();fig.savefig(OUT/'dose-residuals.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for st in STUDIES:
        y=[float(r['candidate_dose_gy']) for r in labels if r['study']==st]
        doses,counts=np.unique(y,return_counts=True)
        axes[0].plot(doses,counts,marker='o',label=st)
    axes[0].set(xlabel='Observed dose (Gy)',ylabel='Profiles',title='Different development dose schedules')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for j,sl in enumerate(['all','control','positive_up_to_2Gy','above_2Gy']):
        vals=[next(r['mae_gy'] for r in slices if r['model']==m and r['study']=='pooled' and r['slice']==sl) for m in MODELS]
        axes[1].bar(np.arange(4)+(j-1.5)*.19,vals,width=.19,label=sl.replace('_',' '))
    axes[1].set_xticks(range(4),['Median','Elastic Net','PLS','Boosting'])
    axes[1].set(ylabel='Donor-balanced MAE (Gy)',title='Descriptive slices; no models refitted',ylim=(0,4.7))
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'dose-support-and-slices.png',dpi=180);plt.close(fig)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),prediction_rows=len(augmented),
        prior_metrics_replayed=True,dose_schedules=schedules,
        exact_doses_shared_by_all_studies=sorted(set.intersection(*(set(v) for v in schedules.values()))),
        pooled={m:{r['slice']:{k:r[k] for k in ('profiles','donors','mae_gy','rmse_gy','bias_gy')}
                   for r in slices if r['model']==m and r['study']=='pooled'} for m in MODELS},
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},code_sha256=sha(__file__),
        models_refitted=False,calibration_applied=False,locked_test_read=False,
        interpretation='posthoc_descriptive_development_review_not_causal_attribution_or_new_validation')
    save_json(OUT/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
