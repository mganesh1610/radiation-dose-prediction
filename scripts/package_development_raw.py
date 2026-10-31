"""Create a local content-addressed release of refittable development inputs."""
import csv
import json
import shutil
import hashlib
from pathlib import Path
from collections import Counter
from audit_auxiliary import sha,save_json

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/auxiliary-processing/v1'
OUT=ROOT/'data/processed_pilot/auxiliary-raw-v1'


def main():
    acquired=json.loads((LOG/'downloads.json').read_text())
    if not acquired['complete'] or acquired['errors']:raise ValueError('Incomplete acquisition')
    for r in acquired['files']:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Raw checksum changed')
    profiles=[]
    for r in csv.DictReader((ROOT/'context/normalization-validation/v2/core-sample-labels-v2.tsv').open(),delimiter='\t'):
        profiles.append(dict(profile_id=r['gsm'],study=r['study'],donor_group=r['split_group_id'],
                             dose_gy=float(r['candidate_dose_gy']),exposure=int(float(r['candidate_dose_gy'])>0),
                             time_hours=24,role='core_development',source_arrays=[r['gsm']],dose_status=r['dose_status']))
    arrays=list(csv.DictReader((ROOT/'context/auxiliary-audit/v1/array-manifest.tsv').open(),delimiter='\t'))
    grouped={}
    for r in arrays:
        if r['metadata_eligible_d']!='True':continue
        grouped.setdefault(r['profile_id'],[]).append(r)
    for p,rows in sorted(grouped.items()):
        r=rows[0]
        profiles.append(dict(profile_id=p,study=r['study'],donor_group=r['donor_group'],dose_gy=float(r['canonical_dose_gy']),
                             exposure=int(r['exposure']),time_hours=24,role='auxiliary_development',
                             source_arrays=sorted(a['gsm'] for a in rows),dose_status=r['dose_status'],
                             source_dose_text=r['source_dose_text'],source_conflict=r['source_conflict'],biological_domain=r['biological_domain']))
    if len(profiles)!=567 or len({r['profile_id'] for r in profiles})!=567:raise ValueError('Invalid profile count')
    mapping={s:{} for s in ['GSE85570','GSE97000','GSE20173']}
    for r in csv.DictReader((LOG/'full-platform-mapping.tsv').open(),delimiter='\t'):
        if r['gene_id']:mapping[r['study']][r['feature_id']]=r['gene_id']
    metadata=dict(profiles=profiles,mapping=mapping,
                  annotation_gene_ids=(LOG/'common-annotation-gene-ids.txt').read_text().splitlines(),
                  identity_limit='study-local donors; globally distinct people not independently proven')
    raw=json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()+b'\n'
    sources={'core.npz':ROOT/'data/processed_pilot/core-reprocessing-v1/common_probe_background_corrected.npz',
             'GSE85570.npz':OUT/'GSE85570_axes.npz','GSE85570-background.bin':OUT/'GSE85570_rma_background.bin',
             'GSE97000.npz':OUT/'GSE97000_probe_inputs.npz','GSE20173.npz':OUT/'GSE20173_probe_inputs.npz'}
    files=[dict(name=k,bytes=p.stat().st_size,sha256=sha(p)) for k,p in sources.items()]
    files.append(dict(name='metadata.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    release_id='development_raw_'+hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    directory=ROOT/'data/model_ready'/release_id;directory.mkdir(parents=True,exist_ok=True)
    for name,source in sources.items():
        target=directory/name
        if target.exists():
            if sha(target)!=sha(source):raise ValueError('Immutable release collision')
        else:shutil.copyfile(source,target)
    (directory/'metadata.json').write_bytes(raw)
    code_names=['development_preprocessing.py','frozen_affymetrix.py','masked_quantile.py','core_transfer.py','assay_qc.py']
    manifest=dict(schema_version=1,release_id=release_id,files=files,total_bytes=sum(r['bytes'] for r in files),
                  profiles=567,core_profiles=145,auxiliary_profiles=422,study_local_donors=len({r['donor_group'] for r in profiles}),
                  study_counts=dict(Counter(r['study'] for r in profiles)),annotation_candidate_genes=len(metadata['annotation_gene_ids']),
                  preprocessing='per-array background correction only; every between-array reference/selection must fit on training IDs',
                  release_kind='refittable_raw_development_inputs',locked_test_included=False,training_release=False,
                  status='awaiting_complete_engineering_verification',
                  code_sha256={n:sha(ROOT/'scripts'/n) for n in code_names},
                  model_note='No predictor or globally fitted training matrix. Use DevelopmentPreprocessor per fold.')
    save_json(directory/'release.json',manifest)
    save_json(LOG/'release-location.json',dict(path=str(directory.relative_to(ROOT)),release_id=release_id,bytes=manifest['total_bytes']))
    print(json.dumps(dict(path=str(directory),**{k:manifest[k] for k in ['release_id','profiles','study_local_donors','total_bytes']}),indent=2))


if __name__=='__main__':main()
