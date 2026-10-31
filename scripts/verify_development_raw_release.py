"""Independent release checks, then enable local fold-aware development use."""
import csv
import gzip
import hashlib
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/auxiliary-processing/v1'
OUT=ROOT/'data/processed_pilot/auxiliary-raw-v1'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    protected=json.loads((LOG/'protected-before.json').read_text())
    for p,h in protected.items():
        if digest(ROOT/p)!=h:raise ValueError('Protected file changed: '+p)
    acq=json.loads((LOG/'downloads.json').read_text())
    selected=list(csv.DictReader((ROOT/'context/auxiliary-audit/v1/raw-selected-files.tsv').open(),delimiter='\t'))
    if {r['gsm'] for r in selected}!={r['gsm'] for r in acq['files']} or len(acq['files'])!=434:
        raise ValueError('Raw accession mismatch')
    if not acq['complete'] or acq['errors']:raise ValueError('Incomplete downloads')
    expanded=0
    for r in acq['files']:
        p=ROOT/r['path']
        if p.stat().st_size!=r['expected_bytes'] or digest(p)!=r['sha256']:raise ValueError('Raw integrity failure')
        total=0
        with gzip.open(p,'rb') as f:
            while b:=f.read(1024*1024):total+=len(b)
        if total!=r['expanded_bytes']:raise ValueError('Expanded size changed')
        expanded+=total
    directory=ROOT/json.loads((LOG/'release-location.json').read_text())['path']
    manifest=json.loads((directory/'release.json').read_text())
    for r in manifest['files']:
        p=directory/r['name']
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('Release payload checksum mismatch')
    for name,h in manifest['code_sha256'].items():
        if digest(ROOT/'scripts'/name)!=h:raise ValueError('Release code changed')
    meta=json.loads((directory/'metadata.json').read_text());rows=meta['profiles'];ids=[r['profile_id'] for r in rows]
    if len(ids)!=567 or len(set(ids))!=567:raise ValueError('Invalid profile count')
    counts=Counter(r['study'] for r in rows)
    if counts!={'GSE102971':100,'GSE8917':25,'GSE65292':20,'GSE85570':400,'GSE97000':12,'GSE20173':10}:raise ValueError('Wrong study counts')
    aux=[r for r in rows if r['role']=='auxiliary_development'];donors=defaultdict(list)
    for r in aux:donors[r['donor_group']].append(r)
    if len(donors)!=211 or any(len(rr)!=2 or {r['dose_gy'] for r in rr}!={0.,2.} for rr in donors.values()):
        raise ValueError('Auxiliary pair contract violated')
    array_ids=[a for r in aux for a in r['source_arrays']]
    if len(array_ids)!=434 or len(set(array_ids))!=434 or set(array_ids)!={r['gsm'] for r in selected}:raise ValueError('Repeat membership failure')
    rawqc=list(csv.DictReader((LOG/'affymetrix-raw-qc.tsv').open(),delimiter='\t'))
    if len(rawqc)!=412 or any(r['masks']!='0' or r['outliers']!='0' for r in rawqc):raise ValueError('CEL raw QC incomplete')
    params=list(csv.DictReader((LOG/'agilent-background-parameters.tsv').open(),delimiter='\t'))
    if len(params)!=22 or any(r['convergence']!='0' or r['warnings'] for r in params):raise ValueError('Agilent fit not accepted')
    checks=json.loads((LOG/'engineering-validation.json').read_text())
    if not checks['checks_pass'] or any(v!=0 for r in checks['studies'] for v in r['checks'].values()):raise ValueError('Auxiliary invariance failure')
    shared=json.loads((LOG/'shared-validation.json').read_text())
    if any(v!=0 for v in shared['checks'].values()):raise ValueError('Shared invariance failure')
    view=np.load(OUT/'engineering_shared_gene_view.npz',allow_pickle=False)
    if list(view['profile_ids'])!=ids or view['rank_values'].shape!=(shared['engineering_selected_genes'],567):raise ValueError('Preview axis mismatch')
    tr=view['training_mask'];core=view['core_mask'];x=view['rank_values']
    if set(r['donor_group'] for r,k in zip(rows,tr) if k)&set(r['donor_group'] for r,k in zip(rows,tr) if not k):raise ValueError('Donor leakage')
    median=np.nanmedian(x[:,tr&core],axis=1)
    np.testing.assert_array_equal(median,view['core_training_imputer'])
    np.testing.assert_array_equal(np.where(np.isfinite(x),x,median[:,None]),view['finite_rank_values'])
    if not np.isfinite(view['finite_rank_values']).all():raise ValueError('Nonfinite ready input')
    # Independently recompute average ranks for one profile per study.
    rank_delta=0
    for study in counts:
        j=next(j for j,r in enumerate(rows) if r['study']==study);v=view['log2_values'][:,j];ok=np.isfinite(v)
        unique,inverse,n=np.unique(v[ok],return_inverse=True,return_counts=True)
        ends=np.cumsum(n);starts=ends-n+1;expected=((starts+ends)[inverse]/2-.5)/n.sum()
        rank_delta=max(rank_delta,float(np.max(abs(expected-x[ok,j]))))
    if rank_delta!=0:raise ValueError('Rank implementation mismatch')
    for name,n in [('unit-tests.log',6),('integration-unit-tests.log',2)]:
        s=(LOG/name).read_text()
        if f'Ran {n} tests' not in s or not s.rstrip().endswith('OK'):raise ValueError('Targeted test failure')
    mp=json.loads((LOG/'median-polish-reference-check.json').read_text())
    if mp['R_stats_medpolish_max_difference']>=1e-12:raise ValueError('Median polish reference mismatch')
    summary=dict(verified_utc=datetime.now(timezone.utc).isoformat(),status='PASS_LOCAL_FOLD_AWARE_DEVELOPMENT_RELEASE',
        release_id=manifest['release_id'],protected_files_unchanged=len(protected),raw_files=434,
        compressed_raw_bytes=sum(r['bytes'] for r in acq['files']),expanded_raw_bytes=expanded,
        release_payload_bytes=manifest['total_bytes'],profiles=567,auxiliary_pairs=211,targeted_tests=8,
        independent_rank_difference=rank_delta,R_median_polish_difference=mp['R_stats_medpolish_max_difference'],
        fit_scope='engineering training donors only; every actual model fold must refit',
        locked_test_processed=False,new_prediction_models_fitted=False,cloud_changed=False,
        release_scope='raw-corrected inputs and validated refittable processing contract; predictive/domain transfer untested')
    manifest.update(training_release=True,status='ready_for_fold_aware_development',
                    predictive_transfer_validated=False,verification_file='context/auxiliary-processing/v1/verification.json')
    (directory/'release.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary['release_manifest_sha256']=digest(directory/'release.json')
    summary['code_sha256']={p.name:digest(p) for p in (ROOT/'scripts').glob('*auxiliary*') if p.is_file()}
    summary['code_sha256'].update({n:digest(ROOT/'scripts'/n) for n in ['development_preprocessing.py','frozen_affymetrix.py','package_development_raw.py','validate_development_release.py','verify_development_raw_release.py']})
    (LOG/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
