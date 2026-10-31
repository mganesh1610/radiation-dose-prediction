"""Verify the locked-assay adapter using existing core data, never test intensities."""
import csv
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from assay_qc import technical_valid
from audit_locked_metadata import ROOT, LOG, sha, write
from check_core_compatibility import read_platform
from core_transfer import CoreTransfer
from development_preprocessing import DevelopmentPreprocessor
from final_model_bundle import load_features
from locked_agilent_adapter import assemble_corrected_probes, inspect_design
from prepare_core_background import parse, VALUES, FLAGS


def main():
    start=time.perf_counter(); out=LOG/'development-only-fixtures';out.mkdir(exist_ok=True)
    # Read public annotation from the already-used core study, not locked expression.
    rows,platform,platform_meta,platform_path=read_platform('GSE65292')
    full=ROOT/'data/models/final-selection-v2/folds/full-development'
    prep=DevelopmentPreprocessor.load(full/'preprocessing')
    required=list(prep.core.probe_ids)
    if not set(required).issubset(platform):raise ValueError('Frozen probes absent from assay annotation')
    annotation=[]
    for probe,gene in zip(prep.core.probe_ids,prep.core.gene_keys):
        r=platform[probe]
        if r['GENE']!=gene or not r['SEQUENCE']:raise ValueError('Frozen gene/sequence mapping changed')
        annotation.append(dict(probe_id=probe,gene_id=gene,sequence_sha256=__import__('hashlib').sha256(r['SEQUENCE'].upper().encode()).hexdigest()))
    write('annotation-compatibility.json',dict(platform='GPL13497',platform_source=str(platform_path.relative_to(ROOT)),
          platform_sha256=sha(platform_path),required_probe_count=len(required),all_required_probes_present=True,
          frozen_input_gene_count=len(set(prep.core.gene_keys)),final_gene_universe=len(prep.gene_ids),
          selected_panels={rep:dict(genes=len(load_features(full/f'{rep}-features.npz').selected),
              complete_platform_annotation_coverage=True) for rep in ['log2','rank']},
          probes=annotation,scope='annotation coverage only; not observed test detection or missingness'))
    # Three raw development files cover control/medium/high dose in the same platform.
    acquisitions=json.loads((ROOT/'context/core-reprocessing/v1/downloads.json').read_text())['files']
    available={r['gsm']:r for r in acquisitions if r['study']=='GSE65292'}
    labels=list(csv.DictReader((ROOT/'context/normalization-validation/v2/core-sample-labels-v2.tsv').open(),delimiter='\t'))
    fixture=[]
    for dose in (0.,2.23,4.45):
        fixture.append(next(r['gsm'] for r in labels if r['study']=='GSE65292' and float(r['candidate_dose_gy'])==dose))
    canonical=np.load(ROOT/'data/processed_pilot/core-reprocessing-v1/GSE65292_canonical_source.npz',allow_pickle=False)
    axes=np.load(ROOT/'data/processed_pilot/core-reprocessing-v1/GSE65292_axes_flags.npz',allow_pickle=False)
    expected=np.fromfile(ROOT/'data/processed_pilot/core-reprocessing-v1/GSE65292_normexp.bin',dtype='<f8').reshape(len(axes['sample_ids']),len(axes['feature_ids'])).T
    x=[]; masks=[]; detects=[]; positions=[]; parsed=[]
    for gsm in fixture:
        record=available[gsm];p=ROOT/record['path']
        if sha(p)!=record['sha256']:raise ValueError('Development fixture raw changed')
        fi,pr,values,flags,sequence,controls=parse(p,platform,'GPL13497')
        j=list(canonical['sample_ids']).index(gsm);positions.append(j)
        if not np.array_equal(values,canonical['values'][:,j]) or not np.array_equal(flags,canonical['flags'][:,j]):
            raise ValueError('Existing raw parser replay mismatch')
        mask=technical_valid(flags[:,None,:],FLAGS)[:,0]
        x.append(np.where(mask,values[:,VALUES.index('gMedianSignal')]-values[:,VALUES.index('gBGMedianSignal')],np.nan))
        masks.append(mask);detects.append(flags[:,FLAGS.index('gIsWellAboveBG')].astype(bool))
        parsed.append(dict(gsm=gsm,study='GSE65292',raw_parser_exact_match=True,raw_sequence_check=sequence))
    matrix=np.column_stack(x); matrix.T.astype('<f8').tofile(out/'background-input.bin')
    r_source='''options(warn=2)
.libPaths(c(normalizePath("data/local_runtime/R-library"),.libPaths()))
library(limma)
args<-commandArgs(trailingOnly=TRUE)
n<-as.integer(args[1]);m<-as.integer(args[2])
x<-matrix(readBin(args[3],"double",n=n*m,size=8,endian="little"),nrow=n)
y<-matrix(NA_real_,nrow=n,ncol=m); records<-list()
for(j in seq_len(m)){
 fit<-normexp.fit(x[,j],method="saddle")
 stopifnot(fit$convergence==0,all(is.finite(fit$par)))
 y[,j]<-normexp.signal(fit$par,x[,j])
 records[[j]]<-data.frame(column=j,convergence=fit$convergence)
}
api<-backgroundCorrect(x,method="normexp",normexp.method="saddle",offset=0,verbose=FALSE)
stopifnot(identical(is.na(y),is.na(x)),max(abs(y-api),na.rm=TRUE)==0,all(y[is.finite(y)]>0))
writeBin(as.double(y),args[4],size=8,endian="little")
write.table(do.call(rbind,records),args[5],sep="\\t",quote=FALSE,row.names=FALSE)
capture.output(sessionInfo(),file=args[6])
'''
    (out/'background-fixture.R').write_text(r_source)
    command=['Rscript',str(out/'background-fixture.R'),str(matrix.shape[0]),str(matrix.shape[1]),str(out/'background-input.bin'),str(out/'background-output.bin'),str(out/'convergence.tsv'),str(out/'R-sessionInfo.txt')]
    proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (out/'R-output.txt').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError('Development fixture R processing failed; see preserved R-output')
    corrected=np.fromfile(out/'background-output.bin',dtype='<f8').reshape(3,matrix.shape[0]).T
    diff=float(np.nanmax(abs(corrected-expected[:,positions])))
    if diff>1e-10:raise ValueError('Per-array correction differs from trained method')
    assembled=assemble_corrected_probes(corrected,pr,np.column_stack(masks),np.column_stack(detects),required)
    with np.load(ROOT/'data/model_ready/development_raw_f6022cfaaca6d1d9/core.npz',allow_pickle=False) as z:
        indices=[list(z['sample_ids']).index(s) for s in fixture]
        expected_probe=z['values'][:,indices];expected_det=z['detected'][:,indices]
    pdiff=float(np.nanmax(abs(assembled['values']-expected_probe)))
    if pdiff>1e-10 or not np.array_equal(assembled['detected'],expected_det):raise ValueError('Probe assembly mismatch')
    gene_ids,logs,_=prep.core.transform(assembled['values'],np.isfinite(assembled['values']),required)
    from development_preprocessing import sample_ranks
    # DevelopmentPreprocessor restricts core output to the preserved annotation universe.
    idx={str(g):i for i,g in enumerate(gene_ids)};logs=logs[[idx[g] for g in prep.gene_ids]]
    views={'log2':logs,'rank':sample_ranks(logs)};panel_checks={}
    core_meta=json.loads((ROOT/'data/model_ready/development_raw_f6022cfaaca6d1d9/metadata.json').read_text())
    for rep in ('log2','rank'):
        gp=load_features(full/f'{rep}-features.npz');panel=gp.transform(views[rep].T,prep.gene_ids,True)
        with np.load(full/f'{rep}-panel.npz',allow_pickle=False) as z:
            target=z['train'][[list(z['training_ids']).index(s) for s in fixture]]
        d=float(np.max(abs(panel-target)))
        if d>1e-10:raise ValueError('Frozen panel replay mismatch')
        panel_checks[rep]=dict(features=500,max_absolute_difference=d)
    write('development-adapter-validation.json',dict(status='PASS_DEVELOPMENT_ONLY',fixtures=parsed,
          normexp_method='saddle',normexp_offset=0,background_max_difference=diff,probe_max_difference=pdiff,
          panels=panel_checks,test_expression_used=False,test_models_executed=False,
          seconds=time.perf_counter()-start,adapter_sha256=sha(ROOT/'scripts/locked_agilent_adapter.py')))
    print('Development-only raw → background → frozen 500-gene panels: PASS',flush=True)


if __name__=='__main__':main()
