"""Read-only replay/integrity audit; never invoke a dose predictor or alter the freeze."""
import json
import numpy as np
from audit_locked_metadata import ROOT, LOG, sha
from development_preprocessing import DevelopmentPreprocessor
from final_model_bundle import load_features


def main():
    entry=json.loads((LOG/'evaluation-entry-freeze.json').read_text())
    for p,h in entry['sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Locked entry artifact changed: '+p)
    for p,key in [('context/final-training/v1/model-bundle-freeze.json','model_bundle_freeze_sha256'),
                  ('context/final-selection/v2/STATISTICAL_PLAN.md','statistical_plan_sha256')]:
        if sha(ROOT/p)!=entry[key]:raise ValueError('Prior model/statistics freeze changed')
    for manifest,key in [('technical-qc-freeze.json','sha256'),('protected-hashes.json',None)]:
        record=json.loads((LOG/manifest).read_text());expected=record[key] if key else record
        for p,h in expected.items():
            if sha(ROOT/p)!=h:raise ValueError('Protected artifact changed: '+p)
    out=ROOT/'data/processed_locked/GSE90909/v1'
    with np.load(out/'corrected-probes.npz',allow_pickle=False) as z:core={k:z[k] for k in z.files}
    with np.load(out/'frozen-panels.npz',allow_pickle=False) as z:panels={k:z[k] for k in z.files}
    ids=[r['gsm'] for r in json.loads((LOG/'eligible-samples.json').read_text())]
    if list(core['sample_ids'])!=ids or list(panels['sample_ids'])!=ids:raise ValueError('Locked panel membership changed')
    full=ROOT/'data/models/final-selection-v2/folds/full-development'
    prep=DevelopmentPreprocessor.load(full/'preprocessing')
    data={'core':core,**{s:{'profile_ids':np.array([],dtype=str)} for s in prep.aux}}
    view=prep.transform(data,ids)
    audit=json.loads((LOG/'release-verification.json').read_text())
    sampleqc=json.loads((LOG/'sample-transformation-qc.json').read_text())
    for rep in ('log2','rank'):
        gp=load_features(full/f'{rep}-features.npz')
        values=view[f'{rep}_values'];expected=gp.transform(values.T,view['gene_ids'],True)
        missing=np.sum(~np.isfinite(values[gp.selected]),axis=0)
        if expected.shape!=(52,500) or not np.isfinite(expected).all() or not np.array_equal(expected,panels[rep]):
            raise ValueError('Generic frozen preprocessing replay mismatch')
        if not np.array_equal(missing,[r[f'{rep}_selected_genes_imputed'] for r in sampleqc]):
            raise ValueError('Imputation count mismatch')
        if int(missing.sum())!=audit['panels'][rep]['total_selected_gene_imputations']:
            raise ValueError('Recorded imputation total mismatch')
    print('PASS: 52-profile frozen membership; both saved 500-gene panels exact; protected freezes unchanged; no predictions.')


if __name__=='__main__':main()
