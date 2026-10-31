"""Validate selected auxiliary Agilent arrays and prepare per-array limma inputs."""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from assay_qc import technical_valid
from audit_auxiliary import read_soft, sha, tsv, save_json

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/auxiliary-processing/v1'
OUT=ROOT/'data/processed_pilot/auxiliary-raw-v1'
FLAGS=['gIsSaturated','IsManualFlag','gIsFeatNonUnifOL','gIsFeatPopnOL','gIsWellAboveBG']


def parse(path,platform):
    features=[];probes=[];signal=[];flags=[];controls=Counter();sequence=False
    with gzip.open(path,'rt') as f:
        for line in f:
            if line.startswith('FEATURES\t'):
                header=line.rstrip('\r\n').split('\t');break
        else:raise ValueError('Feature header absent')
        required=['FeatureNum','ProbeName','ControlType','gMedianSignal','gBGMedianSignal']+FLAGS
        if len(set(header))!=len(header) or not set(required).issubset(header):
            raise ValueError('Missing or duplicate required columns')
        ix={k:header.index(k) for k in required};sequence='Sequence' in header
        for row in csv.reader(f,delimiter='\t'):
            if not row or row[0]!='DATA':continue
            if len(row)!=len(header):raise ValueError('Ragged array record')
            controls[row[ix['ControlType']]]+=1
            if row[ix['ControlType']]!='0':continue
            probe=row[ix['ProbeName']];entry=platform.get(probe)
            # GPL17077 SPOT_ID is heterogeneous (some are transcript accessions).
            # GPL ID is the explicit Agilent probe identifier on both platforms.
            if not entry or entry['ID']!=probe or entry['CONTROL_TYPE'] not in ('FALSE','0'):
                raise ValueError(f'Probe identity mismatch {probe}')
            if sequence and row[header.index('Sequence')].upper()!=entry['SEQUENCE'].upper():
                raise ValueError(f'Probe sequence mismatch {probe}')
            features.append(row[ix['FeatureNum']]);probes.append(probe)
            signal.append(float(row[ix['gMedianSignal']])-float(row[ix['gBGMedianSignal']]))
            flags.append([int(row[ix[k]]) for k in FLAGS])
    x=np.asarray(signal);q=np.asarray(flags)
    if len(set(features))!=len(features) or not np.isfinite(x).all() or not np.isin(q,[0,1]).all():
        raise ValueError('Invalid physical-feature values or identity')
    return np.array(features),np.array(probes),x,q,sequence,dict(controls)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((LOG/'downloads.json').read_text())
    if not manifest['complete'] or manifest['errors']:raise ValueError('Incomplete acquisition')
    reports=[];jobs=[];inputs={}
    for study in ['GSE97000','GSE20173']:
        _,_,platform,_=read_soft(study)
        rows=[r for r in manifest['files'] if r['study']==study];xs=[];qs=[];ids=None
        for row in rows:
            path=ROOT/row['path']
            if sha(path)!=row['sha256']:raise ValueError('Raw checksum changed')
            fi,pr,x,q,sequence,controls=parse(path,platform)
            if ids is not None and (not np.array_equal(ids,fi) or not np.array_equal(probes,pr)):
                raise ValueError('Within-study layout mismatch')
            ids,probes=fi,pr;xs.append(x);qs.append(q);inputs[row['path']]=row['sha256']
            valid=technical_valid(q[:,None,:],FLAGS)[:,0]
            reports.append(dict(study=study,gsm=row['gsm'],physical_features=len(fi),distinct_probes=len(set(pr)),
                                technical_excluded=int((~valid).sum()),controls=sum(controls.values())-len(fi),
                                sequence_check='passed' if sequence else 'column_absent_platform_identity_verified',
                                nonpositive_fg_minus_bg=int((x[valid]<=0).sum())))
        x=np.column_stack(xs);q=np.stack(qs,axis=1);valid=technical_valid(q,FLAGS)
        values=np.where(valid,x,np.nan)
        path=OUT/f'{study}_background_input.bin';values.T.astype('<f8').tofile(path)
        np.savez_compressed(OUT/f'{study}_axes_flags.npz',sample_ids=[r['gsm'] for r in rows],feature_ids=ids,
                            probe_names=probes,technical_valid=valid,detected=q[:,:,FLAGS.index('gIsWellAboveBG')].astype(bool))
        jobs.append(dict(study=study,rows=x.shape[0],columns=x.shape[1],input=str(path.relative_to(ROOT)),
                         output=str((OUT/f'{study}_normexp.bin').relative_to(ROOT))))
        print(study,x.shape,'identity validated',flush=True)
    tsv(LOG/'agilent-background-jobs.tsv',jobs);tsv(LOG/'agilent-raw-qc.tsv',reports)
    save_json(LOG/'agilent-prepared.json',dict(inputs_sha256=inputs,code_sha256=sha(Path(__file__)),arrays=len(reports)))


if __name__=='__main__':main()
