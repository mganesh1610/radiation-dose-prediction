"""Error-blinded technical QC and fixed transformations. Never load dose labels/models."""
import csv
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from audit_locked_metadata import ROOT, LOG, sha, write
from assay_qc import technical_valid
from check_core_compatibility import read_platform
from development_preprocessing import DevelopmentPreprocessor, sample_ranks
from final_model_bundle import load_features
from locked_agilent_adapter import assemble_corrected_probes
from prepare_core_background import parse, VALUES, FLAGS

OUT=ROOT/'data/processed_locked/GSE90909/v1'


def transform(prep, values, required):
    ids, logs, _=prep.core.transform(values,np.isfinite(values),required)
    lookup={g:i for i,g in enumerate(ids)}
    logs=logs[[lookup[g] for g in prep.gene_ids]]
    return {'log2':logs,'rank':sample_ranks(logs)}


def main():
    started=time.perf_counter()
    if OUT.exists():raise ValueError('Refuse to overwrite a locked transformation attempt')
    freeze=json.loads((LOG/'technical-qc-freeze.json').read_text())
    for p,h in freeze['sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Changed pre-QC contract: '+p)
    for p,h in json.loads((LOG/'protected-hashes.json').read_text()).items():
        if sha(ROOT/p)!=h:raise ValueError('Changed protected training artifact')
    inputs=json.loads((LOG/'blind-qc-inputs.json').read_text())
    if any(set(r)!={'gsm','path','sha256'} for r in inputs):raise ValueError('QC manifest must not contain labels')
    _,platform,_,_=read_platform('GSE65292')
    full=ROOT/'data/models/final-selection-v2/folds/full-development'
    prep=DevelopmentPreprocessor.load(full/'preprocessing');required=list(prep.core.probe_ids)
    OUT.mkdir(parents=True); x=[]; masks=[]; detected=[]; qc=[];feature_ids=None
    for raw in inputs:
        p=ROOT/raw['path']
        if sha(p)!=raw['sha256']:raise ValueError('Raw source changed')
        fi,pr,v,q,sequence,controls=parse(p,platform,'GPL13497')
        if feature_ids is not None and (not np.array_equal(feature_ids,fi) or not np.array_equal(probes,pr)):
            raise ValueError('Physical-feature axis changed')
        feature_ids,probes=fi,pr
        valid=technical_valid(q[:,None,:],FLAGS)[:,0]
        difference=v[:,VALUES.index('gMedianSignal')]-v[:,VALUES.index('gBGMedianSignal')]
        x.append(np.where(valid,difference,np.nan));masks.append(valid)
        detected.append(q[:,FLAGS.index('gIsWellAboveBG')].astype(bool))
        qc.append(dict(gsm=raw['gsm'],physical_noncontrol_features=len(fi),technical_usable=int(valid.sum()),
                       technical_masked=int((~valid).sum()),negative_fg_minus_bg_usable=int((difference[valid]<=0).sum()),
                       flag_counts={flag:int(q[:,i].sum()) for i,flag in enumerate(FLAGS)}))
    matrix=np.column_stack(x); matrix.T.astype('<f8').tofile(OUT/'background-input.bin')
    write('raw-technical-qc.json',dict(samples=qc,source='raw feature flags',dose_labels_used=False))
    command=['Rscript','scripts/locked_background_qc.R',str(matrix.shape[0]),str(matrix.shape[1]),
             str(OUT/'background-input.bin'),str(OUT/'background-output.bin'),
             str(LOG/'background-convergence.csv'),str(LOG/'R-sessionInfo.txt')]
    proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (LOG/'background-output.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError('Background correction failed; no cohort pruning or estimator substitution permitted')
    convergence=list(csv.DictReader((LOG/'background-convergence.csv').open()))
    if len(convergence)!=52 or any(r['convergence']!='0' or r['finite_parameters']!='TRUE' for r in convergence):
        raise ValueError('Incomplete/nonconvergent background correction')
    corrected=np.fromfile(OUT/'background-output.bin',dtype='<f8').reshape(52,matrix.shape[0]).T
    assembled=assemble_corrected_probes(corrected,probes,np.column_stack(masks),np.column_stack(detected),required)
    ids=[r['gsm'] for r in inputs]
    np.savez_compressed(OUT/'corrected-probes.npz',sample_ids=ids,**assembled)
    views=transform(prep,assembled['values'],required)
    per_sample=[];panels={};missing={};replay={}
    for rep in ('log2','rank'):
        gp=load_features(full/f'{rep}-features.npz')
        missing[rep]=np.sum(~np.isfinite(views[rep][gp.selected]),axis=0)
        if np.any(missing[rep]==len(gp.selected)):raise ValueError('All selected genes missing; halt without predictions')
        panels[rep]=gp.transform(views[rep].T,prep.gene_ids,True)
        if not np.isfinite(panels[rep]).all():raise ValueError('Nonfinite transformed model inputs')
        # Each input must be independent of other locked arrays/order; these calls never fit.
        rev=transform(prep,assembled['values'][:,::-1],required)[rep]
        reversed_panel=gp.transform(rev.T,prep.gene_ids,True)[::-1]
        diff=float(np.max(abs(panels[rep]-reversed_panel)))
        for j in (0,51):
            one=transform(prep,assembled['values'][:,j:j+1],required)[rep]
            diff=max(diff,float(np.max(abs(gp.transform(one.T,prep.gene_ids,True)[0]-panels[rep][j]))))
        if diff>1e-10:raise ValueError('Test cohort dependence in frozen transformation')
        replay[rep]=dict(max_single_or_reordered_difference=diff,features=len(gp.selected))
    for j,gsm in enumerate(ids):
        per_sample.append(dict(gsm=gsm,log2_selected_genes_imputed=int(missing['log2'][j]),
          rank_selected_genes_imputed=int(missing['rank'][j]),
          technically_missing_frozen_probes=int(np.isnan(assembled['values'][:,j]).sum())))
    np.savez_compressed(OUT/'frozen-panels.npz',sample_ids=ids,log2=panels['log2'],rank=panels['rank'])
    write('sample-transformation-qc.json',per_sample)
    protected=json.loads((LOG/'protected-hashes.json').read_text())
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('Training artifact changed during QC')
    result=dict(status='PASS_CURATED_ASSAY_TECHNICAL_COMPATIBILITY',utc=datetime.now(timezone.utc).isoformat(),
       profiles=52,background_fits_converged=52,background_method='normexp_saddle_offset_0_per_array',
       frozen_probes=len(required),replay=replay,
       technical_masked_fraction_range=[min(r['technical_masked']/r['physical_noncontrol_features'] for r in qc),
                                       max(r['technical_masked']/r['physical_noncontrol_features'] for r in qc)],
       selected_gene_imputation_range={rep:[int(missing[rep].min()),int(missing[rep].max())] for rep in missing},
       excluded_by_qc=0,preprocessing_reference_refitted=False,dose_labels_used_for_processing=False,
       prediction_models_loaded=False,predictions_generated=False,accuracy_estimated=False,
       protected_files_unchanged=len(protected),seconds=time.perf_counter()-started,
       limitations=['Technical pass does not establish predictive accuracy or biological harmonization.',
                    'No validated clinical missingness/OOD threshold is claimed.'],
       artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    write('technical-qc-result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
