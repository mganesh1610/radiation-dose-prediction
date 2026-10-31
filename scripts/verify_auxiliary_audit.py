"""Independent consistency/immutability checks for local audit deliverables."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'context/auxiliary-audit/v1'
BIAS=ROOT/'context/bias-review/v1'


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream,delimiter='\t'))


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def main():
    arrays=read(OUT/'array-manifest.tsv');profiles=read(OUT/'profile-manifest.tsv')
    selected=[r for r in arrays if r['metadata_eligible_bc']=='True']
    candidate=[r for r in profiles if r['metadata_eligible_bc']=='True']
    require(len(arrays)==472 and len({r['gsm'] for r in arrays})==472,'Source membership')
    require(len(selected)==434 and len(candidate)==422,'Candidate count')
    require(all(r['training_ready']=='False' for r in arrays),'Unvalidated release admitted')
    require(Counter(r['canonical_dose_gy'] for r in candidate)=={'0.0':211,'2.0':211},'Dose balance')
    require(len({r['donor_group'] for r in candidate})==211,'Donor count')
    for donor in {r['donor_group'] for r in candidate}:
        rr=[r for r in candidate if r['donor_group']==donor]
        require(len(rr)==2 and {r['canonical_dose_gy'] for r in rr}=={'0.0','2.0'},'Incomplete donor pair')
    byid={r['gsm']:r for r in arrays};members=[]
    for r in profiles:
        ids=r['gsm_ids'].split(';');members+=ids
        require(len(ids)==int(r['array_count']),'Repeat count')
        require(all(byid[i]['profile_id']==r['profile_id'] and byid[i]['donor_group']==r['donor_group'] for i in ids),'Repeat identity')
    require(len(members)==len(set(members))==472,'Duplicated or absent profile member')
    core=read(ROOT/'context/normalization-validation/v2/core-sample-labels-v2.tsv')
    # Existing role metadata only; no reserved expression file is opened.
    old=read(ROOT/'context/dataset-acquisition/candidate-selection.tsv')
    reserved={r['gsm'] for r in old if r['role'] in ('locked_test','conditional_test')}
    require(not ({r['gsm'] for r in selected}&(reserved|{r['gsm'] for r in core})),'Accession overlap')
    raw=read(OUT/'raw-selected-files.tsv')
    require(len(raw)==434 and {r['gsm'] for r in raw}=={r['gsm'] for r in selected},'Raw plan membership')
    before=json.loads((OUT/'protected-before.json').read_text())['sha256']
    require(all(digest(ROOT/p)==v for p,v in before.items()),'Protected original changed')
    manifest=json.loads((OUT/'training-manifest.json').read_text())
    require(manifest['training_release'] is False and bool(manifest['blocked_gates']),'Training gate')
    require(set(manifest['candidate_auxiliary_profile_ids'])=={r['profile_id'] for r in candidate},'Manifest profile IDs')
    for key in ['array_manifest','profile_manifest','core_labels']:
        record=manifest[key];require(digest(ROOT/record['path'])==record['sha256'],'Manifest hash')
    downloads=json.loads((OUT/'sources/raw-format-downloads.json').read_text())
    for r in downloads:require(digest(ROOT/r['path'])==r['sha256'],'Raw format example hash')
    rows=read(BIAS/'prediction-support.tsv');slices=read(BIAS/'slice-metrics.tsv')
    require(len(rows)==580,'Prediction count')
    for s in slices:
        groups=defaultdict(list);count=0
        for r in rows:
            if r['model']!=s['model'] or (s['study']!='pooled' and r['held_study']!=s['study']):continue
            y=float(r['actual_gy']);lo=float(r['training_min_gy']);hi=float(r['training_max_gy'])
            take={'all':True,'control':y==0,'positive_up_to_2Gy':0<y<=2,'above_2Gy':y>2,
                  'inside_training_range':lo<=y<=hi,'outside_training_range':y<lo or y>hi,
                  'exact_training_dose':r['exact_training_dose']=='True',
                  'unseen_training_dose_level':r['exact_training_dose']=='False','0_or_2Gy':y in (0,2)}[s['slice']]
            if take:groups[r['group']].append(float(r['predicted_gy'])-y);count+=1
        n=len(groups)
        require(n==int(s['donors']) and count==int(s['profiles']),'Slice count')
        values={'mae_gy':sum(sum(abs(v) for v in vs)/len(vs) for vs in groups.values())/n,
                'rmse_gy':(sum(sum(v*v for v in vs)/len(vs) for vs in groups.values())/n)**.5,
                'bias_gy':sum(sum(vs)/len(vs) for vs in groups.values())/n}
        require(all(abs(v-float(s[k]))<1e-12 for k,v in values.items()),'Independent metric mismatch')
    result=dict(verified_utc=datetime.now(timezone.utc).isoformat(),status='PASS_AUDIT_NOT_TRAINING_RELEASE',
        arrays=472,candidate_arrays=434,candidate_profiles=422,candidate_donors=211,complete_donor_0_2_pairs=211,
        raw_selected_files=len(raw),selected_raw_bytes=sum(int(r['compressed_bytes']) for r in raw),
        raw_examples_downloaded=len(downloads),raw_example_bytes=sum(r['bytes'] for r in downloads),
        original_protected_files_unchanged=len(before),source_accession_overlap_with_core_or_reserved=0,
        global_person_identity_proven=False,independent_bias_slices_recomputed=len(slices),
        unit_test_evidence='context/auxiliary-audit/v1/tests.log',locked_expression_access=False,
        new_models_fitted=False,cloud_changes=False,
        artifact_sha256={str(f.relative_to(ROOT)):digest(f) for folder in [OUT,BIAS] for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='verification.json'},
        code_sha256={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'scripts/audit_auxiliary.py',ROOT/'scripts/review_baseline_bias.py',ROOT/'tests/test_auxiliary_audit.py']})
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_sha256','code_sha256')},indent=2))


if __name__=='__main__':main()
