from artifact_support import require_artifacts
import csv
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_development_explorer import build_payload


def rows(path):
    return list(csv.DictReader(io.StringIO((ROOT / path).read_text()), delimiter='\t'))


class DevelopmentExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_artifacts(ROOT/'context/clustering/v1/artifact-manifest.json')
        cls.payload, cls.inputs = build_payload()

    def test_export_is_curated_and_final_test_is_separate(self):
        p = self.payload
        self.assertEqual(len(self.inputs), 10)
        self.assertEqual(p, json.loads((ROOT / 'app/static/development-results.json').read_text()))
        self.assertEqual(len(p['views']), 4)
        self.assertEqual(len(p['regression']), 20)
        self.assertEqual(len(p['classification']), 12)
        self.assertFalse(p['final_test_included'])
        text = json.dumps(p, allow_nan=False)
        for absent in ('GSE90909', 'context/', '/Users/', '@asu.edu', 'gs://', 'donor_group', 'joblib'):
            self.assertNotIn(absent, text)

    def test_every_partition_matches_saved_coordinates_and_independent_metrics(self):
        for v in self.payload['views']:
            with np.load(ROOT / f"data/models/clustering-v1/diagnostics/{v['scope']}-{v['representation']}/view/panel.npz", allow_pickle=False) as panel:
                self.assertEqual([p['id'] for p in v['points']], panel['ids'].tolist())
                np.testing.assert_array_equal([[p['x'], p['y']] for p in v['points']], panel['z'][:, :2])
                for fit in v['fits']:
                    mask = np.array([c is not None for c in fit['labels']])
                    labels = np.array([c for c in fit['labels'] if c is not None])
                    pts = [p for p, keep in zip(v['points'], mask) if keep]
                    self.assertEqual(len(pts), (30 if v['scope'] == 'core' else 241) if fit['controls_only'] else len(v['points']))
                    self.assertEqual(set(labels), set(range(fit['k'])))
                    if fit['controls_only']:
                        self.assertTrue(all(p['dose_gy'] == 0 for p in pts))
                    self.assertAlmostEqual(adjusted_rand_score([p['study'] for p in pts], labels), fit['study_ari'])
                    self.assertAlmostEqual(silhouette_score(panel['z'][mask], labels), fit['silhouette'])
        core = self.payload['views'][0]
        self.assertEqual(next(f['study_ari'] for f in core['fits'] if f['method'] == 'gmm' and not f['controls_only']), 1)

    def test_twenty_regression_rows_recompute_from_original_primary_predictions(self):
        predictions = {}
        for path in ('context/model-abcd/v1/predictions.tsv', 'context/classical-comparison/v1/predictions.tsv', 'context/extended-models/v1/predictions.tsv', 'context/clustering/v1/predictions.tsv'):
            for r in rows(path):
                if r['seed_index'] != '0' or r.get('task', 'regression') != 'regression':
                    continue
                model = r.get('model', r.get('strategy'))
                predictions.setdefault(model, {})[r['profile_id']] = r
        def metric(rs):
            groups = {}
            for r in rs:
                groups.setdefault(r['donor_group'], []).append(float(r.get('output', r.get('predicted_gy'))) - float(r['actual_gy']))
            return (np.mean([np.mean(np.abs(v)) for v in groups.values()]),
                    np.sqrt(np.mean([np.mean(np.square(v)) for v in groups.values()])))
        for result in self.payload['regression']:
            rs = list(predictions[result['model']].values())
            self.assertEqual(len(rs), 145, result['model'])
            self.assertEqual(len({r['donor_group'] for r in rs}), 30)
            mae, rmse = metric(rs)
            self.assertAlmostEqual(mae, result['mae_gy'], places=12)
            self.assertAlmostEqual(rmse, result['rmse_gy'], places=12)
            worst = max(metric([r for r in rs if r['study'] == study])[0] for study in {r['study'] for r in rs})
            self.assertAlmostEqual(worst, result['worst_study_mae_gy'], places=12)

    def test_classification_keeps_donor_balancing_and_study_specificity(self):
        predictions = rows('context/extended-models/v1/predictions.tsv')
        for result in self.payload['classification']:
            rs = [r for r in predictions if r['seed_index'] == '0' and r['model'] == result['model'] and (result['study'] == 'pooled' or r['study'] == result['study'])]
            recalls = []
            self.assertEqual(len(rs), result['profiles'])
            for cls in (False, True):
                grouped = {}
                for r in rs:
                    if (float(r['actual_gy']) > 0) == cls:
                        grouped.setdefault(r['donor_group'], []).append((float(r['output']) >= .5) == cls)
                recalls.append(np.mean([np.mean(v) for v in grouped.values()]))
            self.assertAlmostEqual(result['specificity'], recalls[0])
            self.assertAlmostEqual(result['sensitivity'], recalls[1])
            self.assertAlmostEqual(result['balanced_accuracy'], np.mean(recalls))
            if result['study'] == 'GSE8917':
                self.assertEqual(result['specificity'], .4)


if __name__ == '__main__':
    unittest.main()
