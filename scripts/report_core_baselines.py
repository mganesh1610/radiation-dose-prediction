"""Render the measured baseline benchmark without modifying its results."""
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/baseline-modeling/v1'
NAMES={'median_null':'Training-median null','elastic_net':'Elastic Net','pls':'PLS','gradient_boosting':'Gradient boosting'}
COLORS={'GSE102971':'#087e8b','GSE8917':'#d67e22','GSE65292':'#8661ad'}


def main():
    s=json.loads((LOG/'summary.json').read_text())
    rows=list(csv.DictReader((LOG/'outer-predictions.tsv').open(), delimiter='\t'))
    metrics=s['metrics']; models=list(NAMES); studies=list(COLORS)
    pooled={m:next(r for r in metrics if r['study']=='pooled' and r['model']==m) for m in models}
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for ax,model in zip(axes,models[1:]):
        for study in studies:
            rr=[r for r in rows if r['model']==model and r['held_study']==study]
            ax.scatter([float(r['actual_gy']) for r in rr],[float(r['predicted_gy']) for r in rr],
                       label=f'{study} (n={len(rr)})',c=COLORS[study],s=24,alpha=.7,edgecolors='none')
        ax.axhline(0,color='#aaa',lw=.6); ax.plot([-2,9],[-2,9],'--',color='#555',lw=1)
        ax.set(title=f'{NAMES[model]}\nDonor MAE {pooled[model]["mae"]:.3f} Gy',xlabel='Observed dose (Gy)',ylabel='Study-held-out prediction (Gy)')
        ax.legend(fontsize=8)
    lo=min(-.5,min(float(r['predicted_gy']) for r in rows)-.25)
    hi=max(8.5,max(float(r['predicted_gy']) for r in rows)+.25)
    for ax in axes: ax.set_xlim(-.5,8.5); ax.set_ylim(lo,hi)
    fig.suptitle('Core baseline development | 145 profiles, 30 donor groups | final test remains sealed',fontsize=14)
    fig.savefig(LOG/'held-out-predictions.png',dpi=170); plt.close(fig)

    fig,axes=plt.subplots(1,4,figsize=(16,4.8),layout='constrained',sharey=True,sharex=True)
    for ax,study in zip(axes,studies+['pooled']):
        rr=[next(r for r in metrics if r['study']==study and r['model']==m) for m in models]
        vals=np.array([r['mae'] for r in rr]); low=np.array([r['mae_ci_low'] for r in rr]); high=np.array([r['mae_ci_high'] for r in rr])
        # Plot interval endpoints directly: a percentile interval need not bracket the point estimate.
        ax.hlines(np.arange(4),low,high,color='#526175',lw=2)
        ax.scatter(vals,np.arange(4),c=['#999','#147a7e','#997137','#71559c'],s=50,zorder=3)
        for i,v in enumerate(vals): ax.annotate(f'{v:.2f}',(v,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.set(xlabel='Donor-balanced MAE (Gy)',title=study+'\n'+('30 groups' if study=='pooled' else f'{rr[0]["donors"]} groups'))
        ax.set_yticks(range(4),[NAMES[m] for m in models]); ax.set_ylim(3.7,-.6)
        ax.set_xlim(0,max(r['mae_ci_high'] for r in metrics)*1.12); ax.grid(axis='x',alpha=.15)
    fig.suptitle('95% donor-cluster bootstrap intervals | conditional on these studies and fitted models',fontsize=14)
    fig.savefig(LOG/'donor-balanced-errors.png',dpi=170); plt.close(fig)

    table=['| Model | Donor MAE (Gy), 95% interval | RMSE (Gy) | Bias (Gy) | Equal-study MAE (Gy) |',
           '|---|---:|---:|---:|---:|']
    for model in models:
        r=pooled[model]
        table.append(f'| {NAMES[model]} | {r["mae"]:.3f} ({r["mae_ci_low"]:.3f}–{r["mae_ci_high"]:.3f}) | {r["rmse"]:.3f} | {r["bias"]:+.3f} | {s["equal_study_macro_mae"][model]:.3f} |')
    studytable=['| Held-out study | Profiles / donor groups | Median null MAE | Elastic Net MAE | PLS MAE | Boosting MAE |',
                '|---|---:|---:|---:|---:|---:|']
    for study in studies:
        rr=[next(r for r in metrics if r['study']==study and r['model']==m) for m in models]
        studytable.append(f'| {study} | {rr[0]["samples"]} / {rr[0]["donors"]} | '+' | '.join(f'{r["mae"]:.3f}' for r in rr)+' |')
    best=min(models[1:],key=lambda m:pooled[m]['mae'])
    gain=pooled['median_null']['mae']-pooled[best]['mae']
    beat_counts={m:sum(next(r['mae'] for r in metrics if r['model']==m and r['study']==st)<next(r['mae'] for r in metrics if r['model']=='median_null' and r['study']==st) for st in studies) for m in models[1:]}
    checks=s['prediction_checks']
    worst=max(max(c[k] for k in ('reload_max_difference_gy','reversed_max_difference_gy','single_max_difference_gy')) for c in checks)
    dose_rows=[]
    for model in models:
        for study in studies:
            rr=[r for r in rows if r['model']==model and r['held_study']==study]
            for dose in sorted({float(r['actual_gy']) for r in rr}):
                pp=np.array([float(r['predicted_gy']) for r in rr if float(r['actual_gy'])==dose])
                dose_rows.append(dict(model=model,study=study,actual_gy=dose,profiles=len(pp),
                    mean_prediction_gy=float(pp.mean()),mae_gy=float(abs(pp-dose).mean()),bias_gy=float((pp-dose).mean())))
    with (LOG/'dose-errors.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dose_rows[0]),delimiter='\t');w.writeheader();w.writerows(dose_rows)
    text=f'''# Core baseline modeling — measured development results

Completed: {s['completed_utc']}. These are nested cross-study development predictions, not final-test or clinical performance. GSE90909 expression was not read.

The lowest observed pooled MAE among the three learned baselines is **{NAMES[best]}: {pooled[best]['mae']:.3f} Gy**. Its MAE is {gain:.3f} Gy lower than the training-median null (negative would mean worse). This is an observed benchmark ranking, not proof that it is the best general model or a formal significance result. It beats the null in {beat_counts[best]} of three study holdouts.

**Transfer is still weak.** Elastic Net and boosting differ by only {abs(pooled['elastic_net']['mae']-pooled['gradient_boosting']['mae']):.3f} Gy in pooled MAE. Elastic Net bias is −1.901 Gy in GSE102971 and +1.625 Gy in GSE65292; high-dose GSE102971 profiles are systematically underpredicted. PLS is worse than the null on GSE102971 despite being best on GSE65292. These results call for development-only investigation of assay/dose-range differences and model calibration, not deployment or a declared winning strategy. Calibration must be fitted inside training folds, never on the held-out study's observed doses. See [errors at each observed dose](dose-errors.tsv).

## Pooled donor-balanced performance

{chr(10).join(table)}

Pooled results weight the 30 donors equally; 20 belong to GSE102971. The equal-study mean is secondary and weights the three study MAEs equally. Bias is predicted minus observed dose. Predictions are not clipped. Intervals use 2000 within-study donor-cluster resamples; they omit retraining/selection uncertainty and do not represent uncertainty across new studies.

The null's bootstrap interval collapses to one value because every donor within a study has the same observed dose schedule and receives the same prediction. This conditional interval does not establish certainty about its performance in a new study.

## Performance by held-out study

{chr(10).join(studytable)}

![Donor-balanced error comparison](donor-balanced-errors.png)

![Actual versus predicted doses](held-out-predictions.png)

## What was fitted

The [prespecified method](METHOD.md) fixes three outer study holdouts, five inner donor-grouped folds per holdout, the parameter grids and metrics. Within every fold, training samples alone determine probe eligibility, an equal-study frozen quantile target, gene medians, within-study dose-correlation ranking, selected gene panel and model scaling. Background correction was already independent per array. Feature centering within study is used for ranking only; prediction does not require a held-out cohort or known dose. No samples were removed.

Tuning evaluated {s['total_trials']} candidate configurations / {s['inner_fit_attempts']} inner fits. Valid configurations: {s['valid_trials']}; rejected configurations: {s['rejected_trials']}. Three parameter grids have different sizes; the proposal's equal-budget A–D neural comparison has not started. Full configurations and scores are in [selected configurations](selected-configurations.json), [all trials](tuning-trials.tsv) and [inner predictions](inner-predictions.tsv).

## Verification and reproducibility

All 18 training-fold preprocessing fits were unchanged after corrupting every nontraining expression value, detection flag and dose label. The [fold audit](fold-audits.json) records membership and fitted-state fingerprints. All nine learned outer predictors pass complete single-profile, reversed-batch and save/reload prediction checks; largest absolute difference is {worst:.3g} Gy (tolerance 1e-8). Thirty unit tests passed before fitting; [test log](tests.log). The independent [verification](verification.json) checks labels, group separation, tuning choices, prediction replay, metrics and hashes.

The [outer predictions](outer-predictions.tsv) contain 145 out-of-study predictions per model (580 rows). [Donor errors](donor-errors.tsv), [metrics with intervals](metrics.tsv), [prediction invariance/range checks](prediction-checks.json), [machine-readable summary](summary.json), [runtime](run-start.json), [environment](environment-freeze.txt) and [run log](run.log) retain evidence. Runtime: {s['runtime_seconds']:.1f} seconds, excluding package installation/tests/reporting. Saved model/preprocessing artifacts: {s['model_artifact_bytes']:,} bytes in local ignored `data/models/core-baselines-v1/`. The joblib models require the recorded runtime and should only be loaded from trusted locally generated files. These are fold-specific research models, not deployed predictors.

## Limits and next decision

There are only three core studies and just five inferred donor groups in each smaller study. Source-supported donor IDs are not independently proven identities across datasets. Assay, radiation and dose-range differences remain; GSE8917 includes 8 Gy while its outer training studies reach only 7 Gy. No preprocessing method guarantees removal of study effects. GSE65292 retains the documented 2.23 versus 2.2/2.25 Gy source-label discrepancy. The limited feature/parameter grid and equal-profile training loss are part of this benchmark, not optimized universal choices.

Review the by-study residuals and development-only sensitivities before freezing the final pipeline. A–D neural strategies require verified auxiliary processing and matched tuning. Keep GSE90909 sealed until compared pipelines are frozen; avoid iterating against its outcomes. No cloud resources, paid API requests, database uploads, production deployment or final-report claims were generated here. Figures are real plots of saved predictions, not UI screenshots. The prior browser security restriction was respected; no new screenshot was captured.

## Implementation references

[Training-only preprocessing and feature selection](https://scikit-learn.org/stable/common_pitfalls.html), [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html), [PLS](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html), [gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html). API documentation supports implementation choices, not biological validation.
'''
    (LOG/'REPORT.md').write_text(text)
    # Standalone local read-only report; Markdown remains the canonical narrative.
    head='<meta charset="utf-8"><title>Core baseline development</title><style>body{max-width:1200px;margin:36px auto;padding:0 28px;font:16px/1.5 system-ui;color:#1f2937}img{width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}</style>'
    body='<h1>Core baseline development results</h1><p>145 profiles • 30 donor groups • final test remains sealed</p><table><tr><th>Model</th><th>Donor MAE (Gy)</th><th>95% interval</th><th>RMSE</th><th>Bias</th></tr>'
    for model in models:
        r=pooled[model]; body+=f'<tr><td>{NAMES[model]}</td><td>{r["mae"]:.3f}</td><td>{r["mae_ci_low"]:.3f}–{r["mae_ci_high"]:.3f}</td><td>{r["rmse"]:.3f}</td><td>{r["bias"]:+.3f}</td></tr>'
    body+='</table><img src="donor-balanced-errors.png"><img src="held-out-predictions.png"><h2>Methods and limitations</h2><pre>'+html.escape(text)+'</pre>'
    (LOG/'report.html').write_text('<!doctype html>'+head+body)
    print('Report and two scientific figures saved.')


if __name__=='__main__': main()
