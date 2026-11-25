"""Package one previously evaluated GSE90909 profile per dose for frozen prediction."""
import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from frozen_inference import ADAPTER, HELD_OUT_FOLDER, HELD_OUT_MANIFEST_SHA256, METHODS, FrozenInference, digest, encode


def read_json(path):
    return json.loads(path.read_bytes())


def verify_file(path, expected):
    if digest(path.read_bytes()) != expected:
        raise ValueError(f'Frozen source digest mismatch for {path.name}')


def build(root, output):
    frozen = root / 'data/cloud_prediction/frozen-v1'
    engine = FrozenInference(frozen)
    models = root / 'data/models/final-selection-v2'
    model_ledger = root / 'context/final-training/v1/model-bundle-freeze.json'
    verify_file(model_ledger, engine.manifest['source_freeze_sha256'])
    model_files = read_json(model_ledger)['files']
    training_ids = set()
    for method in METHODS:
        if method == 'median_null':
            relative = 'final/median_null/model.json'
        else:
            relative = f"final/{method}/{engine.models[method]['seed']}/result.json"
        verify_file(models / relative, model_files[relative])
        fitted = read_json(models / relative)
        training_ids.update(fitted['training_ids'])
        training_ids.update(fitted.get('auxiliary_training_ids', []))

    eligibility = root / 'context/locked-eligibility/v1'
    frozen_files = read_json(eligibility / 'evaluation-entry-freeze.json')['sha256']
    labels = eligibility / 'eligible-samples.json'
    matrix = root / 'data/processed_locked/GSE90909/v1/corrected-probes.npz'
    for path in (labels, matrix):
        verify_file(path, frozen_files[path.relative_to(root).as_posix()])
    rows = sorted(read_json(labels), key=lambda row: row['gsm'])
    if len({row['gsm'] for row in rows}) != len(rows) or training_ids.intersection(row['gsm'] for row in rows):
        raise ValueError('Held-out membership conflicts with the training or evaluation inventory')
    # Choose by recorded dose and study-local donor before reading model outputs.
    chosen = []
    donors = set()
    for dose in sorted({row['dose_gy'] for row in rows}):
        row = next(row for row in rows if row['dose_gy'] == dose and row['donor_group'] not in donors)
        donors.add(row['donor_group'])
        chosen.append(row)

    evaluation = root / 'context/locked-evaluation/v1'
    archived = evaluation / 'predictions.tsv'
    verify_file(archived, read_json(evaluation / 'result-freeze.json')['sha256'][archived.name])
    with archived.open() as stream:
        expected = {(r['profile_id'], r['model']): r for r in csv.DictReader(stream, delimiter='\t')
                    if r['seed_role'] != 'sensitivity'}
    tolerance = read_json(frozen / 'replay.json')['tolerance_gy']
    payloads = {}
    examples = []
    differences = []
    with np.load(matrix, allow_pickle=False) as corrected:
        ids = corrected['sample_ids'].tolist()
        probe_ids = corrected['probe_ids'].tolist()
        values = corrected['values']
        if set(ids) != {r['gsm'] for r in rows} or len(ids) != len(rows) or probe_ids != list(engine.core.probe_ids):
            raise ValueError('Expression axes differ from the audited assay')
        for row in chosen:
            sample, dose = row['gsm'], row['dose_gy']
            if (row['study'] != 'GSE90909' or not row['eligible'] or row['tissue'] != 'peripheral_whole_blood'
                    or row['time_hours'] != 24 or row['exposure'] != ('control' if dose == 0 else 'photon')):
                raise ValueError(f'Unexpected exposure metadata for {sample}')
            metadata = dict(study=row['study'], organism='human', tissue='whole_blood', time_hours=24,
                            exposure='matched_control' if dose == 0 else 'ex_vivo_xray',
                            units='positive_linear_intensity', normalization='background_corrected_not_between_array_normalized',
                            technical_flags='masked_before_normalization',
                            source_url='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + sample,
                            provenance='Audited GPL13497 Agilent profile from the completed GSE90909 evaluation. '
                                       'Per-array background correction and technical masking precede frozen preprocessing.')
            column = values[:, ids.index(sample)]
            body = dict(schema_version=1, adapter=ADAPTER, sample_id=sample, metadata=metadata,
                        probe_ids=probe_ids, values=[float(x) if np.isfinite(x) else None for x in column])
            x, qc = engine.validate(body)
            if qc['blockers']:
                raise ValueError(f'Adapter compatibility failed for {sample}')
            panels, _ = engine.panels(x)
            for method in METHODS:
                reference = expected[sample, method]
                estimate = float(engine.predict_values(method, panels)[0])
                difference = abs(estimate - float(reference['predicted_gy']))
                if float(reference['actual_gy']) != dose or not np.isfinite(difference) or difference > tolerance:
                    raise ValueError(f'Frozen evaluation mismatch for {sample} with {method}')
                differences.append(difference)
            name = sample + '.json'
            raw = encode(body)
            payloads[name] = raw
            examples.append(dict(id=sample, file=name, study=row['study'], observed_dose_gy=dose,
                                 training_overlap=False, purpose='Previously evaluated held-out study profile',
                                 bytes=len(raw), sha256=digest(raw)))
    payloads['manifest.json'] = encode(dict(schema_version=1, release_sha256=engine.release_sha, examples=examples))
    if digest(payloads['manifest.json']) != HELD_OUT_MANIFEST_SHA256:
        raise ValueError('Generated profiles differ from the reviewed example catalog')
    if output.exists():
        if {p.name for p in output.iterdir()} != set(payloads) or any((output / n).read_bytes() != v for n, v in payloads.items()):
            raise ValueError('The existing example bundle has different contents')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as temp:
            stage = Path(temp) / 'bundle'
            stage.mkdir()
            for name, raw in payloads.items():
                (stage / name).write_bytes(raw)
            stage.rename(output)
    return dict(manifest_sha256=digest(payloads['manifest.json']), release_sha256=engine.release_sha,
                sample_ids=[r['gsm'] for r in chosen], doses_gy=[r['dose_gy'] for r in chosen],
                distinct_study_donors=len(donors), excluded_training_ids=len(training_ids),
                archived_predictions_checked=len(differences), maximum_difference_gy=max(differences),
                tolerance_gy=tolerance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output or root / 'data/cloud_prediction' / HELD_OUT_FOLDER
    print(json.dumps(build(root, output), indent=2))
