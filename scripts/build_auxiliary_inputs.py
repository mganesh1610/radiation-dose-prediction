"""Build annotation-aligned, per-array corrected auxiliary inputs before fitting."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from assay_qc import aggregate_median
from audit_auxiliary import read_soft,gene_mapping,GENES,sha,save_json,tsv

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/auxiliary-processing/v1'
OUT=ROOT/'data/processed_pilot/auxiliary-raw-v1'


def main():
    labels={r['gsm']:r for r in csv.DictReader((ROOT/'context/auxiliary-audit/v1/array-manifest.tsv').open(),delimiter='\t')
            if r['metadata_eligible_d']=='True'}
    with gzip.open(GENES,'rt') as f: current={r['GeneID'] for r in csv.DictReader(f,delimiter='\t') if r['#tax_id']=='9606'}
    mappings=[];sets=[];sources=[]
    for study in ['GSE85570','GSE97000','GSE20173']:
        platform=read_soft(study)[2]
        mapping=gene_mapping(study,list(platform),platform,current);mappings.extend(mapping)
        maps={r['feature_id']:r['gene_id'] for r in mapping};sets.append({g for g in maps.values() if g})
        if study=='GSE85570':
            pm=list(csv.DictReader((LOG/'affymetrix-pm-map.tsv').open(),delimiter='\t'))
            jobs=list(csv.DictReader((LOG/'affymetrix-jobs.tsv').open(),delimiter='\t'))
            samples=[r['gsm'] for r in jobs]
            np.savez_compressed(OUT/f'{study}_axes.npz',probesets=[r['probeset_id'] for r in pm],
                                physical_indices=[int(float(r['physical_index_1based'])) for r in pm],sample_ids=samples,
                                profile_ids=[labels[s]['profile_id'] for s in samples],
                                donor_ids=[labels[s]['donor_group'] for s in samples])
            sources.append(dict(study=study,arrays=len(samples),features=len(pm),probe_sets=len(set(r['probeset_id'] for r in pm))))
        else:
            a=np.load(OUT/f'{study}_axes_flags.npz',allow_pickle=False)
            samples=a['sample_ids'];n=len(a['feature_ids']);m=len(samples)
            y=np.fromfile(OUT/f'{study}_normexp.bin',dtype='<f8').reshape((m,n)).T
            if not np.array_equal(np.isfinite(y),a['technical_valid']) or np.any(y[a['technical_valid']]<=0):
                raise ValueError('Invalid normexp result')
            probes,values,counts=aggregate_median(y,a['probe_names'])
            _,det,_=aggregate_median(a['detected'].astype(float),a['probe_names'],a['technical_valid'])
            np.savez_compressed(OUT/f'{study}_probe_inputs.npz',values=values,detected=np.isfinite(det)&(det>=.5),
                                probe_ids=probes,gene_keys=[maps.get(p,'') for p in probes],sample_ids=samples,
                                profile_ids=[labels[s]['profile_id'] for s in samples],donor_ids=[labels[s]['donor_group'] for s in samples],
                                contributing_spots=counts)
            sources.append(dict(study=study,arrays=len(samples),physical_features=n,distinct_probes=len(probes),
                                missing_probe_cells=int(np.isnan(values).sum()),mapped_gene_ids=len({maps.get(p,'') for p in probes}-{''})))
    core=list(csv.DictReader((ROOT/'context/core-reprocessing/v1/shared-probe-annotation.tsv').open(),delimiter='\t'))
    core_genes={r['GSE102971_gene_id'] for r in core if r['status']=='accepted'}
    common=sorted(set.intersection(core_genes,*sets))
    (LOG/'common-annotation-gene-ids.txt').write_text('\n'.join(common)+'\n')
    tsv(LOG/'full-platform-mapping.tsv',mappings)
    save_json(LOG/'inputs-prepared.json',dict(studies=sources,common_annotation_genes=len(common),
              membership='full external GPL/CDF annotation; no deposited expression feature filtering',
              raw_arrays=434,profiles=422,locked_test_access=False))
    print(json.dumps(dict(studies=sources,common_annotation_genes=len(common)),indent=2))


if __name__=='__main__':main()
