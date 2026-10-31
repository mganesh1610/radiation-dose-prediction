"""Read-only audit of three explicit auxiliary sources; never fits a predictor.

Writes local evidence and a gated metadata manifest, not a training release.
"""
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'context/auxiliary-audit/v1'
STUDIES = ('GSE85570', 'GSE97000', 'GSE20173')
GENES = ROOT / 'context/raw-pilot/GSE102971/audit-v1/sources/Homo_sapiens.gene_info.gz'
PAPER20173 = 'https://doi.org/10.1371/journal.pone.0031293'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def tsv(path, records):
    if not records:
        raise ValueError('Empty evidence table')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(records)


def source_path(study, kind):
    if study not in STUDIES or kind not in ('family.soft', 'series_matrix.txt'):
        raise ValueError('Source outside auxiliary audit allowlist')
    return ROOT / 'data/public_radiation/auxiliary_candidate' / study / f'{study}_{kind}.gz'


def read_soft(study):
    series = defaultdict(list)
    samples = {}
    platforms = []
    current = None
    table = False
    header = None
    values = set()
    with gzip.open(source_path(study, 'family.soft'), 'rt') as stream:
        for line in stream:
            if line.startswith('^SAMPLE = '):
                gsm = line.strip().split(' = ', 1)[1]
                if gsm in samples:
                    raise ValueError('Duplicate sample')
                current = defaultdict(list)
                samples[gsm] = current
            if line.startswith('!platform_table_begin'):
                table = True
                header = None
            elif line.startswith('!platform_table_end'):
                table = False
            elif table:
                fields = next(csv.reader([line], delimiter='\t'))
                if header is None:
                    header = fields
                else:
                    # GEO omits trailing empty columns in some platform rows.
                    if len(fields) > len(header):
                        raise ValueError('Overlong platform row')
                    platforms.append(dict(zip(header, fields + [''] * (len(header)-len(fields)))))
            elif line.startswith('!Series_') and ' = ' in line:
                key, val = line.rstrip().split(' = ', 1)
                if 'contact' not in key and key != '!Series_sample_id':
                    series[key].append(val)
            elif current is not None and line.startswith('!Sample_') and ' = ' in line:
                key, val = line.rstrip().split(' = ', 1)
                if 'contact' not in key:
                    current[key].append(val)
            elif line.startswith('#VALUE = '):
                values.add(line.strip().split(' = ', 1)[1])
    byid = {r['ID']: r for r in platforms}
    if len(byid) != len(platforms):
        raise ValueError('Duplicate platform feature ID')
    return dict(series), samples, byid, sorted(values)


def traits(sample):
    result = {}
    for text in sample['!Sample_characteristics_ch1']:
        key, sep, value = text.partition(': ')
        if not sep or key in result:
            raise ValueError('Missing or duplicated characteristic')
        result[key] = value
    return result


def label_record(study, gsm, sample):
    t = traits(sample)
    title = sample['!Sample_title'][0]
    if sample['!Sample_organism_ch1'] != ['Homo sapiens'] or sample['!Sample_channel_count'] != ['1']:
        raise ValueError('Unexpected species/channel')
    conflict = ''
    source_dose = ''
    note = ''
    if study == 'GSE85570':
        donor = t['patient_id']
        dose = float(re.fullmatch(r'(0|2)Gy', t['irradiation'])[1])
        if f'_{int(dose)}Gy_' not in title:
            raise ValueError('Conflicting GSE85570 title dose')
        cohort = t['study']
        if cohort not in ('prospective', 'retrospective (old)', 'retrospective (new)'):
            raise ValueError('Unknown cohort')
        include = cohort == 'prospective'
        reason = 'prospective_pretreatment' if include else 'prior_clinical_treatment_separate_sensitivity_only'
        group = f'{study}:{donor}'
        profile = f'{group}:{cohort}:{dose:g}Gy'
        batch, plate = t['batch'], t['array_plate']
        source_dose = t['irradiation']
        dose_status = 'explicit_sample_irradiation'
        evidence = 'Sample irradiation characteristic; Series overall design; extract protocol 24h'
        if not any('24h' in v for v in sample['!Sample_extract_protocol_ch1']):
            raise ValueError('Harvest time not confirmed')
        domain = 'PHA_stimulated_14_days_isolated_lymphocytes_prostate_cancer'
        scale = 'published_log2_RMA'
        raw_gate = 'Affymetrix_CEL_QC_and_training_only_RMA_summary_required'
    elif study == 'GSE97000':
        donor = re.fullmatch(r'Person (\d+)', t['individual'])[1]
        dose = float(re.search(r'_(0|2)Gy_', title)[1])
        if (dose == 0 and t['treatment'] != 'control') or (dose == 2 and t['treatment'] != '2 Gy g-rays'):
            raise ValueError('Conflicting GSE97000 treatment')
        if '_24hr_' not in title:
            raise ValueError('Not 24h')
        cohort = 'HighH2AX' if '_HighH2AX_' in title else 'LowH2AX'
        group = f'{study}:{donor}'
        profile = f'{group}:{dose:g}Gy'
        include, reason, batch, plate = True, 'paired_24h_donor', '', ''
        source_dose = t['treatment']
        dose_status = 'title_and_treatment_agree'
        evidence = 'Series overall design; sample title/individual/treatment'
        domain = 'whole_blood_irradiated_then_PHA_culture_and_lymphocyte_isolation'
        scale = 'published_normalized_VALUE_scale_not_explicit_paper_reports_log2'
        raw_gate = 'Agilent_raw_QC_background_and_frozen_training_reference_required'
        note = 'Title rep suffix is not a donor key; use individual across high/low groups'
    elif study == 'GSE20173':
        donor = t.get('donor', t.get('individual', '').removeprefix('donor '))
        if donor not in ('A', 'C', 'G', 'H', 'P'):
            raise ValueError('Unknown GSE20173 donor')
        control = title.startswith('CTR_')
        gravity = '1g' if ('-1g-' in title or 'CTR_1g_' in title) else 'mmg'
        dose = 0.0 if control else 2.0
        include = gravity == '1g'
        reason = 'normal_gravity_24h' if include else 'modeled_microgravity_excluded'
        cohort = gravity
        group = f'{study}:{donor}'
        profile = f'{group}:{gravity}:{dose:g}Gy'
        batch = plate = ''
        source_dose = t['treatment']
        if not control:
            if '20 Gy' not in source_dose or not any('2.0 Gy' in v for v in sample['!Sample_treatment_protocol_ch1']):
                raise ValueError('Source dose conflict changed; review required')
            conflict = 'GEO_characteristic_20Gy_vs_protocol_2Gy'
            dose_status = '2Gy_adjudicated_from_primary_mRNA_methods_preserving_20Gy_conflict'
        else:
            dose_status = 'explicit_nonirradiated_control'
        evidence = PAPER20173 + '; Methods: Cells/irradiation and Gene expression profiling; GEO design/protocol'
        domain = 'unstimulated_quiescent_isolated_PBL_normal_gravity' if include else 'modeled_microgravity_PBL'
        scale = 'published_log2_quantile_normalized_filtered_across_20_arrays'
        raw_gate = 'Agilent_raw_QC_background_and_training_only_filter_reference_required'
    else:
        raise ValueError('Study outside audit allowlist')
    return dict(study=study, gsm=gsm, title=title, platform=sample['!Sample_platform_id'][0],
                donor_group=group, profile_id=profile, cohort=cohort, time_hours=24,
                exposure=int(dose > 0), source_dose_text=source_dose, canonical_dose_gy=dose,
                dose_status=dose_status, source_conflict=conflict, evidence=evidence,
                metadata_eligible_bc=include, metadata_eligible_d=include,
                inclusion_reason=reason, biological_domain=domain, batch=batch, array_plate=plate,
                source_representation=scale, processing_gate=raw_gate, training_ready=False,
                group_scope='study_local_source_id_global_identity_unverified', notes=note)


def read_matrix(study):
    meta = defaultdict(list)
    with gzip.open(source_path(study, 'series_matrix.txt'), 'rt') as stream:
        for line in stream:
            if line.startswith('!Sample_'):
                fields = next(csv.reader([line], delimiter='\t'))
                meta[fields[0]].append(fields[1:])
            if line.startswith('!series_matrix_table_begin'):
                break
        reader = csv.reader(stream, delimiter='\t')
        header = next(reader)
        ids = header[1:]
        features, data = [], []
        for row in reader:
            if row[0].startswith('!series_matrix_table_end'):
                break
            if len(row) != len(header):
                raise ValueError('Ragged expression matrix')
            features.append(row[0])
            data.append([np.nan if v in ('', 'NA', 'null', 'NaN') else float(v) for v in row[1:]])
    if len(ids) != len(set(ids)) or len(features) != len(set(features)) or ids != meta['!Sample_geo_accession'][0]:
        raise ValueError('Matrix identity mismatch')
    x = np.array(data, dtype=float)
    if np.isinf(x).any():
        raise ValueError('Infinite expression value')
    return ids, features, x, meta


def gene_mapping(study, features, platform, current):
    key = {'GSE85570': 'ENTREZ_GENE_ID', 'GSE97000': 'LOCUSLINK_ID', 'GSE20173': 'GENE'}[study]
    records = []
    for feature in features:
        row = platform.get(feature)
        if row is None:
            records.append(dict(study=study, feature_id=feature, source_gene_ids='', gene_id='', status='missing_platform_row'))
            continue
        source = row.get(key, '').strip()
        control = feature.startswith('AFFX-') if study == 'GSE85570' else row['CONTROL_TYPE'] not in ('0', 'FALSE')
        ids = set(re.split(r'\s*///\s*|\s*//\s*|\s*;\s*', source)) - {'', '---'}
        gene = ''
        if control:
            status = 'control'
        elif not ids:
            status = 'unmapped'
        elif len(ids) != 1 or not next(iter(ids)).isdigit():
            status = 'ambiguous'
        elif next(iter(ids)) not in current:
            status = 'absent_from_pinned_human_gene_snapshot'
        else:
            gene = next(iter(ids))
            status = 'unambiguous_current_gene_id'
        records.append(dict(study=study, feature_id=feature, source_gene_ids=source, gene_id=gene, status=status))
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = {r['path']: r['sha256'] for r in json.loads((ROOT/'context/dataset-acquisition/downloads.json').read_text())}
    inputs = {}
    for study in STUDIES:
        for kind in ('family.soft', 'series_matrix.txt'):
            path = source_path(study, kind)
            rel = str(path.relative_to(ROOT))
            inputs[rel] = sha(path)
            if inputs[rel] != expected[rel]:
                raise ValueError('Source differs from original downloaded checksum')
    inputs[str(GENES.relative_to(ROOT))] = sha(GENES)
    with gzip.open(GENES, 'rt') as stream:
        current = {r['GeneID'] for r in csv.DictReader(stream, delimiter='\t') if r['#tax_id'] == '9606'}
    core_annotation = ROOT/'context/core-reprocessing/v1/shared-probe-annotation.tsv'
    inputs[str(core_annotation.relative_to(ROOT))] = sha(core_annotation)
    core_genes = {r['GSE102971_gene_id'] for r in csv.DictReader(core_annotation.open(), delimiter='\t') if r['status'] == 'accepted'}
    all_arrays, all_profiles, all_qc, all_mapping, repeat_pairs, raw_plan, batch_counts = [], [], [], [], [], [], []
    summary, metadata_evidence, sets = [], {}, {}
    for study in STUDIES:
        series, samples, platform, values = read_soft(study)
        ids, features, x, matrix_meta = read_matrix(study)
        if set(ids) != set(samples):
            raise ValueError('SOFT/matrix membership differs')
        records = [label_record(study, gsm, samples[gsm]) for gsm in ids]
        for i, r in enumerate(records):
            if matrix_meta['!Sample_title'][0][i] != r['title']:
                raise ValueError('SOFT/matrix title disagreement')
        mapping = gene_mapping(study, features, platform, current)
        sets[study] = {r['gene_id'] for r in mapping if r['gene_id']}
        all_mapping.extend(mapping)
        grouped = defaultdict(list)
        for r in records:
            grouped[r['profile_id']].append(r)
        study_repeats = []
        for profile, rr in sorted(grouped.items()):
            if len({(r['donor_group'], r['canonical_dose_gy'], r['metadata_eligible_bc']) for r in rr}) != 1:
                raise ValueError('Inconsistent profile collapse')
            item = rr[0]
            all_profiles.append(dict(study=study, profile_id=profile, donor_group=item['donor_group'],
                gsm_ids=';'.join(r['gsm'] for r in rr), array_count=len(rr), exposure=item['exposure'],
                canonical_dose_gy=item['canonical_dose_gy'], time_hours=24, cohort=item['cohort'],
                metadata_eligible_bc=item['metadata_eligible_bc'], metadata_eligible_d=item['metadata_eligible_d'],
                aggregation_plan='median_log2_within_donor_dose_after_raw_QC' if len(rr)>1 else 'single_array',
                split_rule='all_doses_and_repeats_of_donor_stay_together', training_ready=False))
            if len(rr)>1:
                for a in range(len(rr)):
                    for b in range(a+1, len(rr)):
                        ia, ib = ids.index(rr[a]['gsm']), ids.index(rr[b]['gsm'])
                        take = np.isfinite(x[:,ia]) & np.isfinite(x[:,ib])
                        rec = dict(study=study, profile_id=profile, gsm_a=ids[ia], gsm_b=ids[ib],
                                   common_finite_features=int(take.sum()),
                                   pearson_processed=float(np.corrcoef(x[take,ia], x[take,ib])[0,1]),
                                   median_absolute_difference=float(np.median(abs(x[take,ia]-x[take,ib]))),
                                   same_batch=rr[a]['batch']==rr[b]['batch'],
                                   same_plate=rr[a]['array_plate']==rr[b]['array_plate'])
                        repeat_pairs.append(rec); study_repeats.append(rec)
        hashes = defaultdict(list)
        for i, gsm in enumerate(ids):
            v=x[:,i];finite=v[np.isfinite(v)]
            hashes[hashlib.sha256(np.nan_to_num(v, nan=9.87654321e99).tobytes()).hexdigest()].append(gsm)
            quant=np.quantile(finite,[0,.01,.25,.5,.75,.99,1])
            all_qc.append(dict(study=study,gsm=gsm,features=len(features),finite=int(len(finite)),missing=int(np.isnan(v).sum()),
                nonpositive=int((finite<=0).sum()),minimum=float(quant[0]),q01=float(quant[1]),q25=float(quant[2]),
                median=float(quant[3]),q75=float(quant[4]),q99=float(quant[5]),maximum=float(quant[6]),
                action='descriptive_only_no_sample_excluded'))
            urls=samples[gsm].get('!Sample_supplementary_file',[])
            for url in urls:
                if url.startswith('ftp://ftp.ncbi.nlm.nih.gov/'):
                    raw_plan.append(dict(study=study,gsm=gsm,url=url.replace('ftp://','https://',1),
                        metadata_eligible=records[i]['metadata_eligible_bc'], downloaded=False))
        for (batch,plate,dose,cohort),n in sorted(Counter((r['batch'],r['array_plate'],r['canonical_dose_gy'],r['cohort']) for r in records).items()):
            batch_counts.append(dict(study=study,batch=batch,plate=plate,dose_gy=dose,cohort=cohort,arrays=n))
        included=[r for r in records if r['metadata_eligible_bc']]
        finite=x[np.isfinite(x)]
        summary.append(dict(study=study, source_arrays=len(ids), features=len(features), expression_positions=int(x.size),
            missing_cells=int(np.isnan(x).sum()), minimum=float(finite.min()), maximum=float(finite.max()),
            nonpositive_cells=int((finite<=0).sum()),exact_duplicate_columns=[g for g in hashes.values() if len(g)>1],
            source_donors=len({r['donor_group'] for r in records}),candidate_arrays=len(included),
            candidate_profiles=len({r['profile_id'] for r in included}),candidate_donors=len({r['donor_group'] for r in included}),
            excluded_arrays=len(records)-len(included),candidate_dose_counts=dict(Counter(str(r['canonical_dose_gy']) for r in included)),
            mapping_status=dict(Counter(r['status'] for r in mapping)),mapped_genes=len(sets[study]),
            mapped_genes_overlapping_core=len(sets[study]&core_genes),
            duplicate_profile_groups=sum(len(rr)>1 for rr in grouped.values()),
            candidate_extra_technical_arrays=len(included)-len({r['profile_id'] for r in included}),
            repeat_pearson_range=[min(r['pearson_processed'] for r in study_repeats),max(r['pearson_processed'] for r in study_repeats)] if study_repeats else None,
            raw_files_listed=sum(r['study']==study for r in raw_plan),training_ready=False))
        metadata_evidence[study]=dict(series=series,value_descriptions=values,
            unique_processing=sorted({v for s in samples.values() for v in s.get('!Sample_data_processing',[])}),
            unique_extract_protocol=sorted({v for s in samples.values() for v in s.get('!Sample_extract_protocol_ch1',[])}))
        all_arrays.extend(records)
    common = core_genes.intersection(*sets.values())
    tsv(OUT/'array-manifest.tsv', all_arrays)
    tsv(OUT/'profile-manifest.tsv', all_profiles)
    tsv(OUT/'sample-qc.tsv', all_qc)
    tsv(OUT/'feature-mapping.tsv', all_mapping)
    tsv(OUT/'technical-replicate-pairs.tsv', repeat_pairs)
    tsv(OUT/'raw-acquisition-plan.tsv', raw_plan)
    tsv(OUT/'batch-dose-counts.tsv', batch_counts)
    (OUT/'candidate-common-gene-ids.txt').write_text('\n'.join(sorted(common,key=int))+'\n')
    save_json(OUT/'metadata-evidence.json',metadata_evidence)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),scope='metadata_eligibility_and_processed_data_audit_not_training_release',
        studies=summary,core_annotation_gene_count=len(core_genes),candidate_shared_gene_count=len(common),
        all_source_arrays=len(all_arrays),candidate_arrays=sum(r['metadata_eligible_bc'] for r in all_arrays),
        candidate_profiles=sum(r['metadata_eligible_bc'] for r in all_profiles),
        source_files_sha256=inputs,code_sha256=sha(__file__),locked_test_expression_read=False,
        processing_completed=False,models_trained=False,cloud_mutations=False)
    save_json(OUT/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
