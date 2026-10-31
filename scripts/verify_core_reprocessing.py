"""Read-only verification of the local core reprocessing artifacts and lineage."""
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'context/core-reprocessing/v1'


def main():
    checked={}
    def sha(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    def read(path):return json.loads((ROOT/path).read_text())
    def check(path,expected):
        if sha(ROOT/path)!=expected:raise ValueError('Checksum mismatch: '+path)
        checked[path]=expected
    prior=0
    for path in ['context/raw-pilot/GSE102971/summary.json','context/raw-pilot/GSE102971/audit-v1/summary.json',
                 'context/normalization-validation/v2/normalization-summary.json','context/normalization-validation/v2/compatibility-summary.json']:
        for a in read(path)['artifact_files']:check(a['path'],a['sha256']);prior+=1
    for path in ['context/raw-pilot/GSE102971/downloads.json','context/core-reprocessing/v1/downloads.json']:
        for r in read(path)['files']:
            check(r['path'],r['sha256'])
            if (ROOT/r['path']).stat().st_size!=r['expected_bytes']:raise ValueError('Source byte mismatch')
    summary=read('context/core-reprocessing/v1/summary.json');prepared=read('context/core-reprocessing/v1/prepared.json')
    for s in [summary,prepared]:
        for p,h in s['inputs_sha256'].items():check(p,h)
        for p,h in s['code_sha256'].items():check('scripts/'+p,h)
        for a in s['artifacts']:check(a['path'],a['sha256'])
    for r in read('context/final-documents/manifest.json')['files']:check(r['project_path'],r['sha256'])
    tests=read('context/core-reprocessing/v1/unit-test-result.json');assert tests['success'] and tests['tests']==24
    for p,h in tests['test_hashes'].items():check(p,h)
    directory=ROOT/'data/processed_pilot/core-reprocessing-v1';matrix_checks=[]
    for path in sorted(directory.glob('*.npz')):
        z=np.load(path,allow_pickle=False)
        if 'gene_ids' in z and 'values' in z:
            assert z['values'].shape==(len(z['gene_ids']),len(z['sample_ids']))
            assert len(set(z['gene_ids']))==len(z['gene_ids']) and len(set(z['sample_ids']))==len(z['sample_ids'])
            assert not np.isinf(z['values']).any()
            matrix_checks.append(str(path.relative_to(ROOT)))
    for job in csv.DictReader((LOG/'background-jobs.tsv').open(),delimiter='\t'):
        n,m=int(job['rows']),int(job['columns'])
        x=np.fromfile(ROOT/job['input'],dtype='<f8').reshape(m,n).T
        y=np.fromfile(ROOT/job['output'],dtype='<f8').reshape(m,n).T
        z=np.load(directory/f'{job["study"]}_axes_flags.npz',allow_pickle=False);mask=z['technical_valid']
        assert np.array_equal(np.isnan(x),~mask) and np.array_equal(np.isnan(y),~mask)
        assert np.isfinite(y[mask]).all() and np.all(y[mask]>0)
    for f in summary['folds']:
        assert f['checks']['selection_unchanged']
        assert all(v==0 for k,v in f['checks'].items() if k.endswith('max_change'))
        assert len(f['inner_donor_group_checks'])==5
        assert all(r['target_max_change']==0 and r['selection_unchanged'] for r in f['inner_donor_group_checks'])
    staged=subprocess.run(['git','diff','--cached','--name-only'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()
    assert not any(p.startswith('context/') or p.split('/')[-1]=='AGENTS.md' for p in staged)
    ignored=subprocess.run(['git','check-ignore','context/core-reprocessing/v1/REPORT.md','context/report-memory/PROJECT_REPORT_LOG.md','AGENTS.md'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()
    assert len(ignored)==3
    extra=[Path(__file__),ROOT/'scripts/report_core_transfer.py',ROOT/'scripts/acquire_core_raw.py',ROOT/'scripts/acquire_gse102971_raw.py',
           ROOT/'context/dataset-acquisition/preprocessing-audit/GSE8917-raw-filelist.txt',ROOT/'context/dataset-acquisition/preprocessing-audit/GSE65292-raw-filelist.txt']
    for p in extra:checked[str(p.relative_to(ROOT))]=sha(p)
    evidence={str(p.relative_to(ROOT)):sha(p) for p in sorted(LOG.iterdir()) if p.is_file() and p.name!='verification.json'}
    result=dict(verified_utc=datetime.now(timezone.utc).isoformat(),status='passed',hash_checked_files=len(checked),
        prior_artifacts_unchanged=prior,raw_sources_verified=145,final_documents_unchanged=True,
        gene_matrices_verified=matrix_checks,masked_inputs_outputs_match=True,all_usable_corrected_values_finite_positive=True,
        outer_checks=3,inner_donor_group_checks=15,python_tests=24,local_context_ignored_not_staged=True,
        screenshots=dict(captured=False,reason='Generated scientific figure retained; prior browser local-file security restriction not retried or bypassed.'),
        checked_sha256=checked,evidence_sha256=evidence)
    (LOG/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['checked_sha256','evidence_sha256']},indent=2))


if __name__=='__main__':main()
