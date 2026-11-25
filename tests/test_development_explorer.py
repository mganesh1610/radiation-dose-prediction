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
        require_artifacts(ROOT/'context/clustering/v1/artifact-manifest.json', ROOT/'data/experiments/combined-development-v2/summary.json', ROOT/'data/experiments/combined-567-gse90909-v1/results.json')
        cls.payload, cls.inputs = build_payload()

    def test_export_is_curated_and_final_test_is_separate(self):
        p = self.payload
        self.assertIn('data/experiments/combined-development-v2/summary.json', self.inputs)
        self.assertEqual((p['eligible_profiles'],p['wbc_profiles'],p['held_out_profiles']),(567,422,145))
        self.assertEqual(p, json.loads((ROOT / 'app/static/development-results.json').read_text()))
        self.assertEqual(len(p['views']), 4)
        self.assertEqual(len(p['regression']), 20)
        self.assertEqual(len(p['classification']), 12)
        pair = p['matched_gradient_boosting']
        self.assertEqual((pair['held_out_profiles'], pair['donor_groups']), (145, 30))
        self.assertEqual([[round(r[k], 3) for k in ('mae_gy', 'rmse_gy', 'worst_study_mae_gy')] for r in pair['rows']],
                         [[2.016, 2.467, 2.383], [2.669, 3.232, 2.981]])
        self.assertFalse(p['final_test_included'])
        text = json.dumps(p, allow_nan=False)
        retrospective = p['retrospective_evaluation']
        self.assertEqual((retrospective['study'], retrospective['profiles'], retrospective['donors']), ('GSE90909', 52, 12))
        self.assertTrue(retrospective['retrospective'])
        self.assertEqual(set(retrospective['models']), {r['model'] for r in p['regression']})
        self.assertNotIn('GSE90909', json.dumps(p['regression']))
        combined = 'gradient_boosting_wb_wbc'
        self.assertAlmostEqual(retrospective['models'][combined]['mae_gy'], 0.7765355840564675)
        self.assertEqual(round(retrospective['fixed_whole_blood_control']['mae_gy'], 3), 0.795)
        self.assertFalse(set(retrospective['models']) & {'gradient_boosting', 'A'})
        for absent in ('context/', '/Users/', '@asu.edu', 'gs://', '"donor_group"', 'joblib'):
            self.assertNotIn(absent, text)

    def test_every_partition_matches_saved_coordinates_and_independent_metrics(self):
        for v in self.payload['views']:
            with np.load(ROOT / f"data/models/clustering-v1/diagnostics/{v['scope']}-{v['representation']}/view/panel.npz", allow_pickle=False) as panel:
                self.assertEqual([p['id'] for p in v['points']], panel['ids'].tolist())
                np.testing.assert_array_equal([[p['x'], p['y'], p['z']] for p in v['points']], panel['z'][:, :3])
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

    def held_out_rows(self, model):
        name = {'gradient_boosting_wb_wbc':'gradient_boosting','pooled_neural_regression':'A'}.get(model,model)
        meta = json.loads((ROOT/'data/model_ready/development_raw_f6022cfaaca6d1d9/metadata.json').read_text())
        labels = {r['profile_id']:r for r in meta['profiles']}
        rows = []
        for study, fold in self.payload['folds'].items():
            source = ROOT/'data/experiments/combined-development-v2/outer'/study/(name+'.json')
            fit = json.loads(source.read_text())
            self.assertEqual(fit['training_profiles'],fold['training'])
            self.assertEqual(fold['wbc_training'],422)
            self.assertEqual(fold['training']+fold['validation'],567)
            for pid, value in zip(fit['validation_ids'],fit['predictions'][0],strict=True):
                record=labels[pid]
                self.assertEqual(record['study'],study)
                self.assertEqual(record['role'],'core_development')
                rows.append(dict(profile_id=pid,study=study,donor_group=record['donor_group'],actual_gy=record['dose_gy'],output=value))
        self.assertEqual(len({r['profile_id'] for r in rows}),145)
        return rows

    def test_twenty_regression_rows_recompute_from_primary_predictions(self):
        def metric(rs):
            groups = {}
            for r in rs:
                groups.setdefault(r['donor_group'], []).append(float(r.get('output', r.get('predicted_gy'))) - float(r['actual_gy']))
            return (np.mean([np.mean(np.abs(v)) for v in groups.values()]),
                    np.sqrt(np.mean([np.mean(np.square(v)) for v in groups.values()])))
        for result in self.payload['regression']:
            rs = self.held_out_rows(result['model'])
            self.assertEqual(len(rs), 145, result['model'])
            self.assertEqual(len({r['donor_group'] for r in rs}), 30)
            mae, rmse = metric(rs)
            self.assertAlmostEqual(mae, result['mae_gy'], places=12)
            self.assertAlmostEqual(rmse, result['rmse_gy'], places=12)
            errors = [np.mean([r['output']-r['actual_gy'] for r in rs if r['donor_group']==g]) for g in {r['donor_group'] for r in rs}]
            self.assertAlmostEqual(np.mean(errors),result['bias_gy'], places=12)
            self.assertAlmostEqual(np.mean([metric([r for r in rs if r['study']==st])[0] for st in {r['study'] for r in rs}]),result['equal_study_mae_gy'],places=12)
            worst = max(metric([r for r in rs if r['study'] == study])[0] for study in {r['study'] for r in rs})
            self.assertAlmostEqual(worst, result['worst_study_mae_gy'], places=12)

    def test_classification_keeps_donor_balancing_and_study_specificity(self):
        for result in self.payload['classification']:
            rs = [r for r in self.held_out_rows(result['model']) if result['study'] == 'pooled' or r['study'] == result['study']]
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


if __name__ == '__main__':
    unittest.main()
