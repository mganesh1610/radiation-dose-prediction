"""Render immutable locked results; no fitting, prediction, selection or new testing."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'context/locked-evaluation/v1'
NAMES={'A':'A · Core neural','B':'B · Pretrain → fine-tune','C':'C · Multitask','D':'D · Weighted regression',
       'elastic_net':'Elastic Net','pls':'PLS','gradient_boosting':'Gradient boosting','median_null':'Training median'}
ORDER=['A','B','C','D','elastic_net','pls','gradient_boosting','median_null']
def read(name):return json.loads((OUT/name).read_text())
def tsv(name):return list(csv.DictReader((OUT/name).open(),delimiter='\t'))


def main():
    metrics=read('model-metrics.json');primary={x['model']:x for x in metrics if x['seed_role']!='sensitivity'}
    comparisons=read('primary-comparisons.json');predictions=tsv('predictions.tsv');diff=tsv('paired-donor-differences.tsv')
    strata=tsv('stratum-metrics.tsv');verification=read('verification.json');run=read('run-complete.json')
    plt.rcParams.update({'font.size':10,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#f5f7fb','axes.facecolor':'#f5f7fb','savefig.facecolor':'#f5f7fb'})
    colors=['#2563a9','#087e73','#6d5bb5','#be7141','#566778','#8493a6','#155f3d','#89909b']
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
    ax=axes[0]
    for i,m in enumerate(ORDER):
        x=primary[m];ci=x['mae_bootstrap_ci'];v=x['donor_mae']
        ax.errorbar(v,i,xerr=np.array([[v-ci['low']],[ci['high']-v]]),fmt='o',color=colors[i],capsize=4,ms=7)
        ax.text(ci['high']+.07,i,f'{v:.3f}',va='center',fontsize=9)
    ax.set_yticks(range(8),[NAMES[m] for m in ORDER]);ax.invert_yaxis();ax.set_xlim(0,4.35)
    ax.set_xlabel('Donor-balanced MAE (Gy) · lower is better');ax.set_title('Primary models + descriptive baselines',loc='left');ax.grid(axis='x',alpha=.15)
    ax=axes[1]
    for i,m in enumerate('BCD'):
        x=comparisons[m];ci=x['simultaneous_t_ci'];v=x['mean_difference']
        ax.scatter([float(r[f'A_minus_{m}']) for r in diff],np.full(12,i)-.14,s=15,color='#8d9eb6',alpha=.85)
        ax.errorbar(v,i+.07,xerr=[[v-ci['low']],[ci['high']-v]],fmt='D',color=colors[i+1],capsize=5,ms=7)
    ax.axvline(0,color='#364d69',ls='--',lw=1);ax.set_yticks(range(3),['A − B','A − C','A − D']);ax.invert_yaxis()
    ax.set_xlim(-.95,1.6);ax.set_xlabel('Paired donor MAE difference (Gy)');ax.set_title('Prespecified comparisons',loc='left');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Locked GSE90909 · 52 profiles / 12 donors · no retraining',fontsize=14,fontweight='bold')
    fig.supxlabel('Left: descriptive 95% paired-donor bootstrap intervals. Right: 98.333% simultaneous t intervals; dots are donors.\nNegative differences favor A. All three prespecified one-sided Holm-adjusted p values = 1.000.',fontsize=9)
    fig.savefig(OUT/'locked-results.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,m in zip(axes,'BCD'):
        d=np.array([float(r[f'A_minus_{m}']) for r in diff]);(q,ordered),(slope,intercept,_)=stats.probplot(d,dist='norm')
        ax.scatter(q,ordered,color='#2163a9',s=35);ax.plot(q,intercept+slope*q,color='#7d8da0',ls='--')
        for k in [0,len(d)-1]:
            donor=diff[int(np.argsort(d)[k])]['donor_group'].split()[-1]
            ax.annotate('Donor '+donor,(q[k],ordered[k]),xytext=(3,7 if k==0 else -12),textcoords='offset points',fontsize=8)
        ax.set_title('A − '+m);ax.set_xlabel('Theoretical normal quantile');ax.set_ylabel('Observed donor MAE difference (Gy)')
    fig.suptitle('Paired-t assumption diagnostic · all 12 donors retained',fontweight='bold')
    fig.supxlabel('Q–Q reference lines are descriptive. No normality-test switching or donor exclusion. Small n limits inference.',fontsize=9)
    fig.savefig(OUT/'paired-qq.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12,6.3),sharex=True,sharey=True,layout='constrained')
    for i,(ax,m) in enumerate(zip(axes.flat,ORDER)):
        rows=[r for r in predictions if r['model']==m and r['seed_role']!='sensitivity'];y=np.array([float(r['actual_gy']) for r in rows]);p=np.array([float(r['predicted_gy']) for r in rows])
        doses=sorted(set(y));ax.scatter(y,p,s=12,alpha=.5,color=colors[i]);ax.plot(doses,[p[y==d].mean() for d in doses],'-o',color=colors[i],ms=4)
        ax.plot([0,4],[0,4],color='#415776',ls='--',lw=1);ax.axhline(0,color='#999',lw=.5);ax.set_title(NAMES[m],fontsize=10)
        ax.set_xlim(-.15,4.15);ax.set_ylim(-3.9,8.3)
    fig.supxlabel('Actual dose (Gy)');fig.supylabel('Predicted dose (Gy) · unclipped')
    fig.suptitle('Locked dose response · dots are profiles; solid line is dose-group mean',fontweight='bold')
    fig.savefig(OUT/'dose-response.png',dpi=180);plt.close(fig)
    table='\n'.join(f"| {NAMES[m]} | {primary[m]['donor_mae']:.3f} | {primary[m]['mae_bootstrap_ci']['low']:.3f}–{primary[m]['mae_bootstrap_ci']['high']:.3f} | {primary[m]['donor_rmse']:.3f} | {primary[m]['donor_bias']:+.3f} |" for m in ORDER)
    pairtable='\n'.join(f"| A − {m} | {comparisons[m]['mean_difference']:+.3f} | [{comparisons[m]['simultaneous_t_ci']['low']:+.3f}, {comparisons[m]['simultaneous_t_ci']['high']:+.3f}] | {comparisons[m]['raw_p']:.6f} | {comparisons[m]['holm_adjusted_p']:.3f} | No |" for m in 'BCD')
    sens='\n'.join('| '+NAMES[m]+' | '+' | '.join(f"{next(r for r in metrics if r['model']==m and r['seed']==s)['donor_mae']:.3f}" for s in [20261002,20261003,20261004])+' |' for m in ['A','B','C','D','gradient_boosting'])
    subgroup='\n'.join('| '+NAMES[m]+' | '+' | '.join(f"{float(next(r for r in strata if r['model']==m and r['seed_role']!='sensitivity' and r['stratum']==st)['donor_mae']):.3f}" for st in ['control','above_2gy'])+' |' for m in ORDER)
    report=f'''# One-time locked evaluation: GSE90909

2026-10-03 · Team Defender · Frozen local research evaluation

**The prespecified hypothesis that core-only A has lower donor-balanced error than B, C and D is not supported.** All three A-minus-comparator differences are positive and all Holm-adjusted one-sided p values are 1.000. B has the lowest primary-seed error among A–D. Gradient boosting has the lowest observed error among the reported primary models/baselines, a descriptive result rather than a new confirmatory superiority test or an automatic production selection.

## Cohort, freeze and execution

Used the exact [previously audited cohort](../../locked-eligibility/v1/REPORT.md): 52 human whole-blood photon/control profiles, 12 explicit donor groups, 24-hour laboratory setting, 0–4 Gy, GPL13497. Forty neutron arrays were excluded by scope before raw processing; none of the 52 eligible profiles was removed during evaluation. Donor 7 has no deposited control; there are 11 controls and six profiles at 4 Gy. Donor-wise dose sets differ. No test-value filtering, clipping, imputation changes, calibration, tuning, new batch correction, best-seed choice or refitting occurred.

The current user authorized execution after the eligibility/QC gate. Historical plan-only flags remain unchanged as historical records. The [execution record](EXECUTION.md), [pre-execution code/test freeze](pre-execution-freeze.json), [run-start ledger](run-start.json), earlier [model freeze](../../final-training/v1/model-bundle-freeze.json), and [cohort/panel freeze](../../locked-eligibility/v1/evaluation-entry-freeze.json) establish the sequence. Seven synthetic engineering tests passed before inference. The one-shot runner refuses a second start; partial runs require explicit preserved diagnosis.

Evaluated all 18 fixed artifact versions: seven primary models, saved untuned training-median baseline, and ten sensitivity-seed versions. There are 936 prediction rows, 216 donor-metric rows and 162 descriptive stratum rows. Primary inference uses only seed 20261002. Seeds 20261003/20261004 supply initialization sensitivity, not more donors or extra primary tests. The median reference is the original ordinary core-profile median, 4.45 Gy; it was not reset to this test cohort's median.

## Primary results

All errors/biases are in Gy. MAE gives each of the 12 donors equal weight after averaging that donor's absolute profile errors. RMSE is the square root of the mean donor MSE; bias is the mean donor signed error (prediction minus actual). Bootstrap intervals are descriptive 95% paired donor-percentile intervals from 10,000 shared resamples, seed 20261003, conditional on the fixed trained models and observed dose design.

| Model | Donor MAE | 95% bootstrap interval | Donor RMSE | Donor bias |
|---|---:|---:|---:|---:|
{table}

![Frozen evaluation results](locked-results.png)

Pooled profile metrics are saved separately in [model metrics](model-metrics.json); they are not substituted for the primary donor-balanced estimand. Each donor's errors and profile count appear in [donor metrics](donor-metrics.tsv), and every signed/unclipped estimate appears in [predictions](predictions.tsv).

## Prespecified comparisons

Following the authoritative final PDF and [frozen statistical plan](../../final-selection/v2/STATISTICAL_PLAN.md), difference = donor MAE(A) minus donor MAE(comparator); negative favors A. H0: mean difference ≥0; H1: mean difference <0. Paired t tests use 12 donors (df11), with Holm across exactly three comparisons at .05. The supplied presentation's opposite wording remains documented; neither original was edited.

| Comparison | Mean difference | 98.333% simultaneous t interval | Raw one-sided p | Holm p | Reject for A lower? |
|---|---:|---:|---:|---:|---|
{pairtable}

There is no support for the directional claim that A outperforms B–D, and no equivalence conclusion follows. A–B's prespecified two-sided interval lies above zero, directionally favoring B under the t assumptions. We report that effect without reversing the preregistered one-sided alternative or adding another hypothesis test. The A–C and A–D simultaneous intervals span zero. Unadjusted descriptive bootstrap intervals need not match the simultaneous t intervals and do not replace the Holm decisions. [Complete comparison statistics](primary-comparisons.json), including t/df/standard errors and descriptive difference intervals; [all donor differences](paired-donor-differences.tsv).

![Paired donor Q–Q diagnostic](paired-qq.png)

Inference assumes the source's 12 distinct donor IDs represent independent volunteers and an adequate mean-difference t approximation. Public donor IDs cannot exclude shared people across development and testing, related donors or shared assay dependencies. There is only one test study. The Q–Q plots show an upper-tail departure for A–B (donor 12 has the largest difference, +1.393 Gy) and curvature for A–D. They are diagnostic, not an assumption pass/fail certificate; n12 limits assessment. No normality-based test switching or extreme-donor deletion was performed. Interpret nominal p values/intervals conditionally; the more direct finding is the observed donor-level effect pattern. These intervals are not uncertainty intervals for a new patient's dose.

## Bias, dose coverage and seed sensitivity

| Model | Controls MAE: 11 profiles / 11 donors | >2 Gy MAE: 6 profiles / 6 donors, all 4 Gy |
|---|---:|---:|
{subgroup}

A overpredicts all 52 primary-seed profiles; its donor bias equals its MAE (+1.537 Gy), and its 4 Gy mean signed error is +1.662 Gy. Earlier development high-dose underprediction does not carry over uniformly: the change in bias direction shows study-transfer instability. Gradient boosting's overall signed bias is +0.693 Gy despite the lowest aggregate MAE; controls still have 0.486 Gy MAE. Elastic Net's small overall bias (+0.098) masks control error of 1.095 Gy and 4 Gy underprediction. These are descriptive outcomes, not grounds to recalibrate on this test set.

![Unclipped dose-response results](dose-response.png)

Negative and above-range predictions are preserved, including negative estimates from B/C/D, Elastic Net, PLS and boosting. No clipping improves their metrics. The 0–4 Gy test cannot establish accuracy above 4 Gy, at other times/assays, in vivo, or clinically. A low aggregate row is not a validated threshold for acceptable performance.

| Model | Primary seed 20261002 | Sensitivity 20261003 | Sensitivity 20261004 |
|---|---:|---:|---:|
{sens}

B leads A–D at the primary and third seeds; C leads at the second. Do not replace the primary seed, create a post-hoc ensemble, or claim a seed-invariant neural winner. Classical EN/PLS each retain their single planned deterministic fit. Expanded exploratory models and clustering remain development results; they are not newly tested primary families here.

## Verification, reproducibility and limitations

[Independent audit](verification.json) recomputed membership/labels, all 18 metric sets, 216 donor rows, 162 stratum denominators/MAE/bias, all paired t/Holm/interval calculations, and the shared 10,000-draw bootstrap. All 936 saved predictions replayed exactly (max batch difference 0 Gy); seven primary single-input checks differed by at most {verification['max_single_batch_difference']:.3g} Gy. These repeated calculations verify persistence/integrity; they are not a second experiment or additional opportunity to select a result.

All {verification['protected_files_unchanged']:,} protected prior files—including raw sources, prepared test data, models, statistical plan and original PDF/PPTX—remain unchanged. Seven pre-inference tests passed. No execution failures occurred. Local inference/statistics with preflight verification took {run['seconds']:.2f} seconds on four CPU threads; the independent audit took {verification['seconds']:.2f} seconds. These exclude earlier training, raw/QC preparation, setup, reporting and screenshots. No cloud/API job, upload, application/IAM change or GitHub publication occurred. [Run completion](run-complete.json), [tests](tests.log), [result freeze](result-freeze.json).

The locked test is now unblinded and consumed for this frozen analysis. Preserve these results even when they oppose the original hypothesis. Future model changes based on this result are exploratory and require independent validation; do not reuse GSE90909 as a fresh untouched test. No clinical or general-purpose upload prediction claim is established.

Next prepare the model/result cards and wire these immutable comparisons into the reviewer dashboard after cloud-runtime parity and input-scope enforcement checks. No automatic model promotion or retraining follows this evaluation. Conditional test studies remain separately unaudited.
'''
    (OUT/'REPORT.md').write_text(report)
    tablerows=''.join(f"<tr><td>{NAMES[m]}</td><td>{primary[m]['donor_mae']:.3f}</td><td>{primary[m]['mae_bootstrap_ci']['low']:.3f}–{primary[m]['mae_bootstrap_ci']['high']:.3f}</td><td>{primary[m]['donor_bias']:+.3f}</td></tr>" for m in ORDER)
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Defender · One-time locked evaluation</title><style>*{box-sizing:border-box}body{background:#eef2f7;margin:0;color:#19314d;font:15px/1.55 system-ui,sans-serif}.page{max-width:1180px;margin:auto;padding:32px 26px}header{border-top:5px solid #255eab;padding:20px 0}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.13em;color:#526982;font-weight:700}h1{font-size:33px;line-height:1.15;margin:12px 0}h2{font-size:21px;margin:0 0 12px}p{margin:8px 0}.finding{background:#fff0d2;border-left:5px solid #b88118;border-radius:5px;padding:16px 19px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.card{background:white;border:1px solid #dce4ee;padding:15px;border-radius:7px}.big{font-size:27px;font-weight:750;color:#245b99}.small{font-size:12px;color:#556b82}.panel{background:white;border:1px solid #dce4ee;padding:20px;border-radius:8px;margin-top:16px}.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:25px}img{width:100%;display:block}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid #e5eaf1}th{font-size:12px;color:#526982}.foot{font-size:12px;color:#536b85;margin-top:18px;border-top:1px solid #ccd6e5;padding-top:12px}@media(max-width:780px){.grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}}</style></head><body><main class="page"><header><div class="eyebrow">Team Defender · 03 October 2026 · Frozen research evaluation</div><h1>One-time locked test: results</h1><p>GSE90909 · 52 profiles · 12 donor groups · laboratory photon/control · 0–4 Gy</p></header><div class="finding"><strong>The hypothesis that core-only A outperforms B, C and D is not supported.</strong><br>All three prespecified A-minus-comparator differences are positive; Holm-adjusted one-sided p = 1.000. B has the lowest primary-seed neural error. Gradient boosting has the lowest observed benchmark error.</div><div class="cards"><div class="card"><div class="big">52 / 52</div><div class="small">Eligible profiles evaluated; no removals</div></div><div class="card"><div class="big">18 versions</div><div class="small">Frozen models/baseline and planned sensitivity seeds</div></div><div class="card"><div class="big">0 refits</div><div class="small">No tuning, recalibration or clipping</div></div><div class="card"><div class="big">10,000</div><div class="small">Paired donor bootstrap resamples</div></div></div><section class="panel"><h2>Primary results and prespecified contrasts</h2><img src="locked-results.png" alt="Donor MAE and paired A-minus-comparator error differences; no comparison favors A."><p class="small">Donors are the inferential units. Benchmark ranking is descriptive; no new superiority test or automatic model promotion.</p></section><section class="panel grid"><div><h2>Donor-balanced error (Gy)</h2><table><tr><th>Model</th><th>MAE</th><th>95% bootstrap CI</th><th>Bias</th></tr>'''+tablerows+'''</table></div><div><h2>Limits that matter</h2><p>A overpredicts all 52 primary-seed profiles. Boosting still has +0.693 Gy overall bias and 0.486 Gy control MAE.</p><p>Only 12 donors, one study and 0–4 Gy are tested. Dose coverage differs across donors. Normal-reference inference remains conditional; Q–Q plots and every donor difference are in the full report.</p><p>The neural ranking changes with initialization: B leads at the primary/third seeds; C leads at the second. No best-seed substitution.</p><p><strong>This is a research result, not clinical validation.</strong> Negative estimates are retained in the metrics.</p></div></section><section class="panel"><h2>Results preserved and independently verified</h2><p>936 saved predictions replayed exactly. All 1,120 protected inputs/models remain unchanged. Seven pre-inference tests passed; all donor statistics, intervals and Holm calculations were independently checked.</p><p><strong>The locked test is now unblinded.</strong> Future changes based on it are exploratory and need independent validation. Cloud/app deployment remains a separate step.</p></section><footer class="foot">Source: frozen local model bundle and curated GSE90909 release. Machine-readable results and verification accompany this report. No cloud upload, application change or GitHub publication occurred.</footer></main></body></html>'''
    (OUT/'report.html').write_text(html)
    print('Rendered immutable result report and three scientific figures.')


if __name__=='__main__':main()
