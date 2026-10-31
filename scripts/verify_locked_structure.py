"""Verify locked metadata/design only, and unchanged training artifacts. No inference."""
import json
import time
from collections import Counter
from datetime import datetime, timezone
from audit_locked_metadata import ROOT, LOG, sha, write
from check_core_compatibility import read_platform
from locked_agilent_adapter import inspect_design


def main():
    started = time.perf_counter()
    metadata_freeze = json.loads((LOG/'eligibility-freeze.json').read_text())
    for name, digest in metadata_freeze['sha256'].items():
        if sha(LOG/name) != digest: raise ValueError('Changed metadata freeze: '+name)
    if sha(ROOT/'scripts/audit_locked_metadata.py') != metadata_freeze['source_sha256']:
        raise ValueError('Changed membership implementation')
    eligible = json.loads((LOG/'eligible-samples.json').read_text())
    all_rows = json.loads((LOG/'all-sample-decisions.json').read_text())
    ids = {r['gsm'] for r in eligible}
    if len(ids) != 52 or len(all_rows) != 92: raise ValueError('Unexpected membership')
    if ids != {r['gsm'] for r in all_rows if r['eligible']}:
        raise ValueError('Inconsistent membership')
    if len({(r['donor_group'],r['dose_gy']) for r in eligible}) != 52:
        raise ValueError('Repeated donor-dose pair requires review')
    development = json.loads((ROOT/'data/model_ready/development_raw_f6022cfaaca6d1d9/metadata.json').read_text())
    source_ids = {s for p in development['profiles'] for s in p['source_arrays']}
    overlap = sorted(ids & source_ids)
    if overlap: raise ValueError('Locked/development raw accession overlap')
    integrity = json.loads((LOG/'raw-integrity.json').read_text())
    if not integrity['complete'] or integrity['errors'] or {r['gsm'] for r in integrity['files']} != ids:
        raise ValueError('Incomplete raw acquisition')
    annotation = json.loads((LOG/'annotation-compatibility.json').read_text())
    _, platform, _, platform_path = read_platform('GSE65292')
    if sha(platform_path) != annotation['platform_sha256']: raise ValueError('Annotation changed')
    required = [r['probe_id'] for r in annotation['probes']]
    design_records = []
    for raw in integrity['files']:
        path=ROOT/raw['path']
        if sha(path) != raw['sha256'] or path.stat().st_size != raw['bytes']:
            raise ValueError('Raw content changed')
        result=inspect_design(path,platform,required)
        design_records.append(dict(gsm=raw['gsm'],**result))
        print('DESIGN',len(design_records),raw['gsm'],flush=True)
    write('raw-design-compatibility.json',dict(files=design_records,expression_values_parsed=False,
          quality_flag_values_parsed=False,all_required_probes_present=True))
    protected={}
    full=ROOT/'data/models/final-selection-v2'
    bundle=json.loads((ROOT/'context/final-training/v1/model-bundle-freeze.json').read_text())
    for name,digest in bundle['files'].items():protected[str((full/name).relative_to(ROOT))]=digest
    run=json.loads((ROOT/'context/final-training/v1/run-start.json').read_text())
    protected.update(run['source_sha256'])
    protocol=ROOT/'context/final-selection/v2'
    pf=json.loads((protocol/'protocol-freeze.json').read_text())
    protected['context/final-selection/v2/protocol-freeze.json']=run['protocol_freeze_sha256']
    for name,digest in pf['sha256'].items():protected[str((protocol/name).relative_to(ROOT))]=digest
    protected.update(json.loads((protocol/'evidence-hashes.json').read_text()))
    originals=json.loads((ROOT/'context/final-documents/manifest.json').read_text())['files']
    for item in originals:protected[item['project_path']]=item['sha256']
    for name,digest in protected.items():
        if sha(ROOT/name) != digest:raise ValueError('Protected training/protocol/source/original changed: '+name)
    validation=json.loads((LOG/'development-adapter-validation.json').read_text())
    if validation['adapter_sha256'] != sha(ROOT/'scripts/locked_agilent_adapter.py'):
        raise ValueError('Adapter changed since development-only replay')
    result=dict(status='PASS_ELIGIBILITY_AND_STRUCTURAL_COMPATIBILITY',
      utc=datetime.now(timezone.utc).isoformat(),total_arrays=len(all_rows),selected_arrays=len(ids),
      excluded_neutron_arrays=len(all_rows)-len(ids),donors=len({r['donor_group'] for r in eligible}),
      dose_counts=dict(sorted(Counter(r['dose_gy'] for r in eligible).items())),
      development_profiles=len(development['profiles']),development_source_arrays=len(source_ids),
      exact_development_accession_overlap=overlap,cross_study_person_identity_verifiable=False,
      raw_files_verified=len(design_records),compressed_bytes=sum(r['bytes'] for r in integrity['files']),
      expanded_bytes_checked_not_saved=sum(r['expanded_bytes'] for r in integrity['files']),
      raw_designs=len({r['design_sha256'] for r in design_records}),
      noncontrol_physical_feature_counts=sorted({r['noncontrol_physical_features'] for r in design_records}),
      distinct_noncontrol_probe_counts=sorted({r['distinct_noncontrol_probes'] for r in design_records}),
      required_frozen_probes=len(required),protected_files_unchanged=len(protected),
      final_models_unchanged=bundle['final_models'],development_only_adapter_replay=validation['status'],
      test_intensities_parsed=False,test_quality_flags_parsed=False,test_normalized=False,
      predictions_generated=False,accuracy_estimated=False,
      next_gate='BLINDED_NUMERIC_RAW_QC_AND_FROZEN_TRANSFORMATION_NOT_YET_EXECUTED',
      seconds=time.perf_counter()-started)
    write('protected-hashes.json',protected)
    write('verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
