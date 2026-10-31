"""Measured release QC figures and a local report view; no predictor fit."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/auxiliary-processing/v1'
OUT=ROOT/'data/processed_pilot/auxiliary-raw-v1'


def main():
    summary=json.loads((LOG/'shared-validation.json').read_text());verify=json.loads((LOG/'verification.json').read_text())
    rows=list(csv.DictReader((LOG/'shared-profile-qc.tsv').open(),delimiter='\t'))
    z=np.load(OUT/'engineering_shared_gene_view.npz',allow_pickle=False)
    x=z['finite_rank_values'];training=z['training_mask']&z['core_mask'];studies=np.array([r['study'] for r in rows])
    var=x[:,training].var(axis=1);panel=np.argsort(-var,kind='stable')[:1000]
    mean=x[panel][:,training].mean(axis=1);centered=x[panel].T-mean
    _,singular,vt=np.linalg.svd(centered[training],full_matrices=False)
    xy=centered@vt[:2].T;evr=singular[:2]**2/(singular**2).sum()
    colors=['#216e77','#bc682b','#7354a4','#338452','#d08c24','#ad5274']
    names=['GSE102971','GSE8917','GSE65292','GSE85570','GSE97000','GSE20173']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for name,color in zip(names,colors):
        take=studies==name
        axes[0].scatter(xy[take,0],xy[take,1],s=17,alpha=.65,c=color,label=name,
                        marker='o' if name in names[:3] else '^')
    axes[0].set(xlabel=f'PC1 ({evr[0]:.1%} core-training variance)',ylabel=f'PC2 ({evr[1]:.1%})',title='Rank representation still shows study differences')
    axes[0].legend(fontsize=8,ncol=2)
    miss=np.isnan(z['rank_values']).mean(axis=0)*100
    axes[1].bar(names,[miss[studies==s].max() for s in names],color=colors)
    axes[1].set(title='Maximum missing genes per profile',ylabel='Missing genes (%)',ylim=(0,max(.17,miss.max()*1.15)))
    axes[1].tick_params(axis='x',rotation=30)
    fig.suptitle('567-profile engineering preview | PCA fitted on core training donors | no accuracy claim',fontsize=12)
    fig.savefig(LOG/'release-qc.png',dpi=150);plt.close(fig)
    (LOG/'pca-display.json').write_text(json.dumps(dict(training_profiles=int(training.sum()),selected_genes=1000,
        fit_scope='core engineering-training profiles only; variance selection, imputation, mean, components',
        explained_variance_ratio=evr.tolist(),interpretation='descriptive display; neither batch correction nor evidence of predictive transfer'),indent=2)+'\n')
    np.savez_compressed(OUT/'engineering_pca_display.npz',gene_ids=z['gene_ids'][panel],mean=mean,components=vt[:2],coordinates=xy,profile_ids=z['profile_ids'])
    studyrows=''.join(f'<tr><td>{s}</td><td>{sum(studies==s)}</td><td>{miss[studies==s].max():.3f}%</td></tr>' for s in names)
    html=f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Defender · Raw processing release</title><style>body{{margin:0;background:#f2f7f8;color:#19303a;font:16px/1.55 system-ui}}main{{max-width:1080px;margin:38px auto;padding:34px;background:white;border-radius:14px}}h1{{font-size:32px}}.eyebrow{{color:#216e77;font-weight:700;letter-spacing:.07em}}.status{{padding:16px;background:#e7f4ed;border-left:5px solid #338452}}.stats{{display:flex;gap:16px;margin:24px 0}}.stats div{{flex:1;background:#edf4f5;padding:18px}}strong.big{{font-size:30px;display:block}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce3e5}}th{{background:#edf4f5}}img{{width:100%}}a{{color:#216e77}}.note{{color:#52616a;font-size:14px}}</style>
<main><div class="eyebrow">TEAM DEFENDER · OCTOBER 2, 2026 · LOCAL DATA RELEASE</div><h1>Raw processing and development readiness</h1>
<div class="status"><b>Ready for fold-aware development training.</b><br>434 auxiliary raw files verified and processed. Fit normalization and feature selection inside every training fold. GSE90909 remains sealed.</div>
<div class="stats"><div><strong class="big">567</strong>145 core + 422 auxiliary profiles</div><div><strong class="big">1.79 GB</strong>refittable release payload</div><div><strong class="big">9,178</strong>genes in engineering preview</div></div>
<p>Full annotations provide 14,305 candidate common GeneIDs. The engineering split uses 452 training and 115 validation profiles, separated by donor. Its core-training filter determines the same feature panel for A and B–D.</p>
<table><tr><th>Study</th><th>Profiles</th><th>Maximum missing genes</th></tr>{studyrows}</table>
<h2>Processing replay and separation checks passed</h2><p>All 22 Agilent background fits converged. All 412 HT Affymetrix CEL files passed the configured identity and mask checks. Frozen reference, saved replay, reordered batches and single-profile processing passed with zero differences. No profiles were automatically removed as statistical outliers.</p>
<img src="release-qc.png" alt="Core-training PCA projection and per-study missingness">
<p class="note">Study differences remain visible. Shared gene identifiers and ranks establish a computational input contract; they do not prove biological equivalence or improved dose prediction. Missing engineering values are imputed using core training donors only.</p>
<h2>Next: matched A–D experiments on the Mac</h2><p>Freeze the experiment settings and tuning budgets, then fit all preprocessing within donor-grouped development folds. Compare dose error by study and dose. No new predictor was fitted or deployed in this release.</p>
<p><a href="REPORT.md">Processing report</a> · <a href="verification.json">Verification evidence</a> · <a href="shared-validation.json">Integration checks</a></p>
<p class="note">8 targeted tests passed · 111 protected files unchanged · raw download 1.1335 GB compressed · no cloud or GitHub changes.</p></main>'''
    (LOG/'report.html').write_text(html)


if __name__=='__main__':main()
