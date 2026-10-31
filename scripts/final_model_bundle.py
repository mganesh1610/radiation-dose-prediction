"""Local research replay of an integrity-checked artifact; not an upload approval API."""
import json
from pathlib import Path
import joblib
import numpy as np
from abcd_models import load_model, predict
from core_baseline import GenePreparation
from development_preprocessing import DevelopmentPreprocessor
from run_abcd_experiments import completed


def load_features(path):
    with np.load(path, allow_pickle=False) as z:
        return GenePreparation(tuple(z['gene_ids']), z['medians'], z['selected'], z['mean'],
                               z['scale'], z['selection_scores'], tuple(z['training_ids']))


def predict_panel(directory, panel, identity):
    directory = Path(directory)
    if not completed(directory, identity):
        raise ValueError('Incomplete model artifact')
    result = json.loads((directory / 'result.json').read_text())
    if result['model'] in ('A', 'B', 'C', 'D'):
        model, saved = load_model(directory / 'model.pt')
        values = predict(model, panel, saved['target_mean'], saved['target_scale'])
    else:
        # Load only files validated against this locally trusted run's completion hash.
        model = joblib.load(directory / 'model.joblib')
        values = np.asarray(model.predict(panel)).ravel() * result['target_scale'] + result['target_mean']
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite saved-model prediction')
    return values


def raw_panel(root, directory, data, profile_ids, identity):
    root, directory = Path(root), Path(directory)
    if not completed(directory, identity):
        raise ValueError('Incomplete model')
    result = json.loads((directory / 'result.json').read_text())
    prep_dir = (root / result['preprocessing']).resolve()
    if not prep_dir.is_relative_to(root.resolve()) or not completed(prep_dir, identity):
        raise ValueError('Invalid preprocessing artifact')
    prep = DevelopmentPreprocessor.load(prep_dir / 'preprocessing')
    features = load_features(prep_dir / f"{result['params']['representation']}-features.npz")
    if set(prep.core_training) != set(result['training_ids']) or list(features.training_ids) != result['training_ids']:
        raise ValueError('Model/preprocessing membership mismatch')
    view = prep.transform(data, profile_ids)
    return features.transform(view[f"{result['params']['representation']}_values"].T, view['gene_ids'], True)


def predict_development(root, directory, data, profile_ids, identity):
    return predict_panel(directory, raw_panel(root, directory, data, profile_ids, identity), identity)
