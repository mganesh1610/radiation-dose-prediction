"""Export the verified frozen whole-blood + WBC boosting model for numeric-only inference."""
import argparse
import io
import json
from pathlib import Path

import joblib
import numpy as np

from frozen_inference import WBC_BOOSTING, WBC_FILE, WBC_SHA256, FrozenInference, digest, encode
from portable_model import estimator_spec, numeric_predict

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = '136be9fd50c29e253aeca801a06c4f15a9f68a67436320470eb787774b7f9dda'
BASE_SHA = '9189a9cc87866315cf1f90f90c0e7796e7ff2607c74c18aa45808072ea00d191'
ARTIFACT = 'models/combined-20261002.joblib'


def checked(path, expected):
    payload = path.read_bytes()
    if digest(payload) != expected:
        raise ValueError(f'Frozen source changed: {path.name}')
    return payload


def export(root, destination):
    run = root / 'context/retrospective-gb-wbc-2026-11-14'
    freeze = json.loads(checked(run/'model-freeze.json', FREEZE_SHA))
    receipt = json.loads(checked(run/'training-complete.json',freeze['training_complete_sha256']))
    protocol = json.loads(checked(run/'protocol.json',freeze['protocol_sha256']))
    model_record = next(r for r in receipt['models'] if r['role']=='primary')
    if (model_record['model_file'] != ARTIFACT or model_record['seed'] != protocol['primary_seed']
            or model_record['training_profiles'] != 567 or protocol['estimator']['representation'] != 'log2'):
        raise ValueError('Unexpected combined model')
    frozen = root / 'data/cloud_prediction/frozen-v1'
    checked(frozen/'manifest.json', BASE_SHA)
    engine = FrozenInference(frozen)
    artifact = run/ARTIFACT
    source_sha = freeze['sha256'][ARTIFACT]
    estimator = joblib.load(io.BytesIO(checked(artifact,source_sha)))
    mean,scale = model_record['target_mean'],model_record['target_scale']
    spec = estimator_spec('gradient_boosting',estimator,None)
    # Only the original development examples enter the runtime self-check.
    replay = {}
    for sample in engine.manifest['examples']:
        name = sample['id']
        x,qc = engine.validate(json.loads(engine.example_bytes(name)))
        if qc['blockers']:
            raise ValueError('Invalid development replay input')
        panels,_ = engine.panels(x)
        original = estimator.predict(panels['log2'])*scale+mean
        np.testing.assert_allclose(numeric_predict(spec,panels['log2'])*scale+mean,original,rtol=0,atol=1e-11)
        replay[name] = float(original[0])
    bundle = dict(schema_version=1, release='frozen-gb-wbc-v1', base_release_sha256=BASE_SHA,
                  model=WBC_BOOSTING, training_profiles=567, whole_blood_profiles=145,
                  wbc_profiles=422, replay_gy=replay,
                  spec=dict(estimator=spec,representation='log2',seed=model_record['seed'],source_sha256=source_sha,
                            target_mean=mean,target_scale=scale))
    raw = encode(bundle)
    if digest(raw) != WBC_SHA256:
        raise ValueError('Generated WBC model differs from the reviewed numeric export')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.read_bytes()!=raw:
        raise ValueError('Existing WBC model export differs')
    destination.write_bytes(raw)
    return dict(sha256=digest(raw),bytes=len(raw),source_sha256=source_sha)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/cloud_prediction'/WBC_FILE)
    print(json.dumps(export(ROOT,parser.parse_args().output)))
