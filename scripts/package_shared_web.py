"""Package the reviewed public explorer and the server-side study-profile predictor."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from public_site_assets import FILES, GUEST_EXTRAS, RECORDS, encode, share_template
from frozen_inference import FrozenInference

ROOT = Path(__file__).resolve().parents[1]


def published_predictions(model_dir):
    engine = FrozenInference(model_dir)
    catalog = engine.catalog(prediction_picker=True)
    fields = ('id', 'study', 'training_overlap', 'observed_dose_gy')
    choices = {'curated_only': True, 'saved_estimates': True, 'methods': catalog['methods'], 'uploads': [],
               'examples': [{k: row[k] for k in fields if k in row} for row in catalog['examples']]}
    records = {'catalog': choices, 'profiles': {}}
    for row in catalog['examples']:
        sample = row['id']
        raw = json.loads(engine.example_bytes(sample))
        if raw['sample_id'] != sample or raw['metadata']['study'] != row['study']:
            raise ValueError('Published study and input mismatch')
        results = {}
        for model in catalog['methods']:
            result = engine.predict(raw, model)
            if engine.curated[result['input_sha256']]['id'] != sample:
                raise ValueError('Published input hash mismatch')
            overlap = row.get('training_overlap', True)
            observed = row['observed_dose_gy']
            # Publish the reproducible output and its release identity, never a raw profile or estimator.
            results[model] = {k: result[k] for k in (
                'model','sample_id','estimate_gy','seed','model_sha256','release','release_sha256',
                'outside_training_dose_range','individual_prediction_interval','limitations')}
            results[model].update(status='completed', training_overlap=overlap, computation='published_frozen_estimate',
                dose_comparison=dict(study=row['study'], observed_dose_gy=observed,
                                     estimate_minus_observed_gy=result['estimate_gy'] - observed))
        records['profiles'][sample] = results
    return records


def package(public_dir, manifest_path, destination, model_dir):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Choose an empty build directory')
    manifest = json.loads(manifest_path.read_bytes())
    publication = json.loads((public_dir / 'data/publication.json').read_bytes())
    data = publication['data_path'].removeprefix('/')
    records = RECORDS
    expected = set(FILES) | {'data/publication.json', 'robots.txt'} | {f'{data}/{name}.json' for name in records}
    if set(manifest) != expected:
        raise ValueError('The public build allowlist does not match the expected files')
    index = {name + '.json': manifest[f'{data}/{name}.json']['sha256'] for name in records}
    if data != 'data/' + hashlib.sha256(encode(index)).hexdigest()[:20]:
        raise ValueError('Published data version mismatch')
    site = destination
    for name in sorted(expected):
        p = public_dir / name
        if p.is_symlink() or not p.resolve().is_relative_to(public_dir.resolve()):
            raise ValueError('Unexpected public release path')
        raw = p.read_bytes()
        if len(raw) != manifest[name]['bytes'] or hashlib.sha256(raw).hexdigest() != manifest[name]['sha256']:
            raise ValueError('Published file integrity failure: ' + name)
        if name == 'vercel.json':
            continue
        if name == 'index.html':
            raw = share_template(raw.decode(), (ROOT / 'app/static/index.html').read_text()).encode()
        target = site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for name, source in GUEST_EXTRAS.items():
        target = site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    records = published_predictions(model_dir)
    record_file = site / 'published' / 'estimates.json'
    record_file.parent.mkdir()
    record_file.write_bytes(encode(records))
    return {'version': data, 'files': sum(p.is_file() for p in destination.rglob('*')),
            'eligible_development_profiles': publication['eligible_development_profiles'],
            'published_prediction_profiles': len(records['profiles'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-dir', type=Path, default=ROOT/'data/public_web')
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/public-build-manifest.json')
    parser.add_argument('--model-dir', type=Path, default=ROOT/'data/cloud_prediction/frozen-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.public_dir, args.manifest, args.output, args.model_dir)))
