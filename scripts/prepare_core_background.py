"""Validate core raw sources and prepare masked foreground-minus-background inputs.

Only core manifests are accepted. The locked-test cohort is never opened.
"""
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from assay_qc import technical_valid, platform_probe_id
from check_core_compatibility import read_platform, sha, tsv

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT/'context/core-reprocessing/v1'
OUT = ROOT/'data/processed_pilot/core-reprocessing-v1'
VALUES = ['gProcessedSignal','gMedianSignal','gBGMedianSignal','gBGSubSignal']
FLAGS = ['gIsSaturated','IsManualFlag','gIsFeatNonUnifOL','gIsFeatPopnOL',
         'gIsWellAboveBG','gIsBGNonUnifOL','gIsBGPopnOL','gIsFound','gIsPosAndSignif']


def parse(path, platform, gpl):
    ids, probes, values, flags = [], [], [], []
    sequence_available = False
    controls = Counter()
    with gzip.open(path,'rt') as f:
        for line in f:
            if line.startswith('FEATURES\t'):
                header=line.rstrip('\r\n').split('\t');break
        else:raise ValueError('Missing feature header')
        required=['FeatureNum','ProbeName','ControlType']+VALUES+FLAGS
        if len(set(header))!=len(header) or not set(required).issubset(header):
            raise ValueError('Missing/duplicate source columns')
        ix={k:header.index(k) for k in required}
        sequence_available='Sequence' in header
        for row in csv.reader(f,delimiter='\t'):
            if not row or row[0]!='DATA':continue
            if len(row)!=len(header):raise ValueError('Ragged feature row')
            controls[row[ix['ControlType']]]+=1
            if row[ix['ControlType']]!='0':continue
            feature,probe=row[ix['FeatureNum']],row[ix['ProbeName']]
            entry=platform.get(feature if gpl=='GPL1708' else probe)
            if entry is None or platform_probe_id(gpl,entry)!=probe:raise ValueError('Raw/platform identity mismatch')
            if sequence_available and row[header.index('Sequence')].upper()!=entry['SEQUENCE'].upper():
                raise ValueError('Raw/platform sequence mismatch')
            ids.append(feature);probes.append(probe)
            values.append([float(row[ix[k]]) for k in VALUES]);flags.append([int(row[ix[k]]) for k in FLAGS])
    if len(set(ids))!=len(ids):raise ValueError('Duplicate physical features')
    x=np.asarray(values,dtype=float);q=np.asarray(flags,dtype=np.uint8)
    if not np.isfinite(x).all() or not np.isin(q,[0,1]).all():raise ValueError('Invalid values/flags')
    return np.array(ids),np.array(probes),x,q,sequence_available,dict(controls)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    acquired=json.loads((LOG/'downloads.json').read_text())
    if not acquired['complete'] or acquired['errors']:raise ValueError('Incomplete core acquisition')
    for rel in ['context/raw-pilot/GSE102971/summary.json','context/raw-pilot/GSE102971/audit-v1/summary.json']:
        for a in json.loads((ROOT/rel).read_text())['artifact_files']:
            if sha(ROOT/a['path'])!=a['sha256']:raise ValueError('Prior artifact hash changed')
    arrays={};metrics=[];jobs=[];inputs={};studies=[]
    labels=list(csv.DictReader((ROOT/'context/normalization-validation/v2/core-sample-labels-v2.tsv').open(),delimiter='\t'))
    for study in ['GSE102971','GSE8917','GSE65292']:
        if study=='GSE102971':
            manifest=json.loads((ROOT/'context/raw-pilot/GSE102971/downloads.json').read_text())
            rows=manifest['files']
            cpath=ROOT/'data/processed_pilot/GSE102971/v1/canonical_source.npz'
            qpath=ROOT/'data/processed_pilot/GSE102971/audit-v1/quality_flags.npz'
            c=np.load(cpath,allow_pickle=False);q=np.load(qpath,allow_pickle=False)
            for key in ['feature_ids','sample_ids']:
                if not np.array_equal(c[key],q[key]):raise ValueError('Prior QC axes mismatch')
            ids,probes,x=c['feature_ids'],c['probe_names'],c['values']
            flags=q['flags'];flag_names=list(q['flag_columns']);value_names=list(c['value_columns'])
            samples=c['sample_ids'];sequence_available=True
            inputs[str(cpath.relative_to(ROOT))]=sha(cpath);inputs[str(qpath.relative_to(ROOT))]=sha(qpath)
        else:
            rows=[r for r in acquired['files'] if r['study']==study]
            gpl={'GSE8917':'GPL1708','GSE65292':'GPL13497'}[study]
            _,platform,_,platform_path=read_platform(study);inputs[str(platform_path.relative_to(ROOT))]=sha(platform_path)
            xx=[];qq=[];ids=None
            for r in rows:
                path=ROOT/r['path']
                if sha(path)!=r['sha256']:raise ValueError('Raw checksum changed')
                fi,pr,xxi,qqi,sequence_available,controls=parse(path,platform,gpl)
                if ids is not None and (not np.array_equal(ids,fi) or not np.array_equal(probes,pr)):
                    raise ValueError('Within-study feature order mismatch')
                ids,probes=fi,pr;xx.append(xxi);qq.append(qqi)
            x=np.stack(xx,axis=1);flags=np.stack(qq,axis=1)
            value_names=VALUES;flag_names=FLAGS;samples=np.array([r['gsm'] for r in rows])
            cpath=OUT/f'{study}_canonical_source.npz'
            np.savez_compressed(cpath,values=x,flags=flags,value_columns=value_names,flag_columns=flag_names,
                                sample_ids=samples,feature_ids=ids,probe_names=probes)
        if len(samples)!=len(rows) or set(samples)!={r['gsm'] for r in labels if r['study']==study}:
            raise ValueError('Sample selection mismatch')
        # Recheck every raw source including the cached GSE102971 source representation.
        for r in rows:
            if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Raw checksum changed')
            inputs[r['path']]=r['sha256']
        usable=technical_valid(flags,flag_names)
        fg=x[:,:,value_names.index('gMedianSignal')];bg=x[:,:,value_names.index('gBGMedianSignal')]
        processed=x[:,:,value_names.index('gProcessedSignal')]
        difference=np.where(usable,fg-bg,np.nan)
        if not np.isfinite(difference[usable]).all():raise ValueError('Invalid background input')
        input_path=OUT/f'{study}_background_input.bin'
        difference.T.astype('<f8').tofile(input_path)
        np.savez_compressed(OUT/f'{study}_axes_flags.npz',sample_ids=samples,feature_ids=ids,probe_names=probes,
                            technical_valid=usable,detected=flags[:,:,flag_names.index('gIsWellAboveBG')].astype(bool))
        job=dict(study=study,rows=len(ids),columns=len(samples),input=str(input_path.relative_to(ROOT)),
                 output=str((OUT/f'{study}_normexp.bin').relative_to(ROOT)))
        jobs.append(job)
        for j,sample in enumerate(samples):
            metrics.append(dict(study=study,gsm=sample,physical_features=len(ids),technical_excluded=int((~usable[:,j]).sum()),
                                nonpositive_published_processed_usable=int(((processed[:,j]<=0)&usable[:,j]).sum()),
                                nonpositive_median_fg_minus_bg_usable=int(((fg[:,j]-bg[:,j]<=0)&usable[:,j]).sum()),
                                raw_sequence_check='passed' if sequence_available else 'unavailable_column_absent'))
        studies.append(dict(study=study,samples=len(samples),physical_features=len(ids),distinct_probes=len(set(probes)),
                            raw_identity_check='passed',raw_sequence_check='passed_prior_audit' if study=='GSE102971' else ('passed' if sequence_available else 'unavailable_column_absent')))
        print(f'Prepared {study}: {x.shape[0]} features x {len(samples)} profiles',flush=True)
    tsv(LOG/'background-jobs.tsv',jobs);tsv(LOG/'sample-raw-qc.tsv',metrics)
    inputs.update({str(p.relative_to(ROOT)):sha(p) for p in [LOG/'downloads.json',ROOT/'context/raw-pilot/GSE102971/downloads.json',ROOT/'context/normalization-validation/v2/core-sample-labels-v2.tsv']})
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),studies=studies,inputs_sha256=inputs,
                code_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'scripts/assay_qc.py',ROOT/'scripts/check_core_compatibility.py']},
                artifacts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.glob('*')) if p.suffix in ['.bin','.npz']])
    (LOG/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
