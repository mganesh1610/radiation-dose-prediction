from artifact_support import require_artifacts
import hashlib
import json
import sys
import unittest
from html.parser import HTMLParser
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from build_public import DEST, DATASET, LocalRelease, Explorer, RESULT, FILES


class PublicExportTests(unittest.TestCase):
    def test_analysis_tabs_have_real_panels_in_both_sites(self):
        class Elements(HTMLParser):
            def __init__(self):
                super().__init__(); self.ids=set(); self.panels=[]
            def handle_starttag(self, tag, attrs):
                a=dict(attrs)
                if 'id' in a:self.ids.add(a['id'])
                if a.get('role')=='tab':self.panels.append(a['aria-controls'])
        for path in (ROOT/'app/static/index.html', DEST/'index.html'):
            html=path.read_text();page=Elements();page.feed(html)
            self.assertTrue(set(page.panels).issubset(page.ids), str(path))
            self.assertEqual(page.panels,['analysis-panel'] * 4 + ['clustering-explorer'])
            self.assertNotIn('statistical-explorer',page.ids)
            self.assertNotIn('/static/statistical-explorer.js',html)

    @classmethod
    def setUpClass(cls):
        require_artifacts(DEST/'data/publication.json', ROOT/'data/reviewer_release/research-results.json')
        cls.publication = json.loads((DEST / 'data/publication.json').read_text())
        cls.folder = DEST / cls.publication['data_path'].lstrip('/')

    def test_only_reviewed_static_files_are_packaged(self):
        manifest = json.loads((ROOT / 'data/public-build-manifest.json').read_text())
        actual = {str(p.relative_to(DEST)): p for p in DEST.rglob('*') if p.is_file()}
        self.assertEqual(set(manifest), set(actual))
        # Thirty immutable result JSONs, publication metadata and robots, plus reviewed assets.
        self.assertEqual(len(actual), 32 + len(FILES))
        for name in ('analysis-workspace.js','analysis-workspace.css','analysis-scene.mjs','spatial-model.mjs',
                     'vendor/three/three.module.js','vendor/three/three.core.js','vendor/three/OrbitControls.js','vendor/three/LICENSE'):
            self.assertIn('static/' + name, actual)
        for name, path in actual.items():
            self.assertFalse(path.is_symlink())
            self.assertNotIn('context', name)
            self.assertNotIn('AGENTS', name)
            self.assertNotIn('/api/', '/' + name)
            self.assertNotIn(path.suffix, ('.py', '.npz', '.pkl', '.env'))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest[name]['sha256'])
        for path in (DEST / 'data').rglob('*.json'):
            text = path.read_text()
            for private in ('@asu.edu', 'gserviceaccount.com', 'gs://', 'training/v1/',
                            '/Users/', 'web_202610_', 'model_manifest', 'job_id', 'worker'):
                self.assertNotIn(private, text, str(path))
        for path in self.folder.glob('analysis-*.json'):
            self.assertNotIn('GSE90909',path.read_text())
        payload=json.loads((self.folder/'research-results.json').read_text())
        self.assertEqual(payload,json.loads((ROOT/'data/reviewer_release/research-results.json').read_text()))
        self.assertFalse(payload['live_prediction_enabled'])
        self.assertEqual(len(payload['model_cards']),8)
        html = (DEST / 'index.html').read_text()
        for absent in ('train-form', 'id="train"', 'id="worker"', 'id="jobs"', '/static/app.js'):
            self.assertNotIn(absent, html)

    def test_all_public_variants_preserve_scientific_values_and_boundaries(self):
        service = Explorer(DATASET)
        store = LocalRelease()
        for study, representation, genes in product(('all', 'GSE102971', 'GSE8917', 'GSE65292'),
                                                    ('log2', 'quantile'), (100, 500, 2000)):
            options = dict(study=study, representation=representation, genes=genes)
            published = json.loads((self.folder / f'analysis-{study}-{representation}-{genes}.json').read_text())
            reference = service.analyze(store, options)
            for key in ('options', 'samples', 'pca', 'heatmap', 'correlation', 'qc', 'selected_gene_ids', 'provenance'):
                self.assertEqual(published[key], reference[key], (options, key))
            self.assertFalse(published['locked_test_included'])
            self.assertEqual(published['qc']['automatic_exclusions'], 0)
            self.assertEqual(set(published['source']), {'sha256', 'release_id'})

    def test_results_match_reviewed_run_and_independent_donor_balanced_metrics(self):
        public = json.loads((self.folder / 'experiments.json').read_text())['runs'][0]['result']
        original = json.loads(RESULT.read_text())
        for k in ('metrics', 'method', 'holdout_study', 'purpose'):
            self.assertEqual(public[k], original[k])
        self.assertEqual(len(public['predictions']), len(original['predictions']))
        for exported, saved in zip(public['predictions'], original['predictions']):
            for key in ('sample_id', 'group_id', 'observed_gy', 'predicted_gy'):
                self.assertEqual(exported[key], saved[key])
        rows = public['predictions']
        catalog = {r['sample_id']: r for r in json.loads((self.folder / 'catalog.json').read_text())['samples']}
        grouped = {}
        for r in rows:
            self.assertEqual(r['observed_gy'], catalog[r['sample_id']]['dose_gy'])
            self.assertEqual(catalog[r['sample_id']]['study_id'], 'GSE8917')
            grouped.setdefault(r['group_id'], []).append(r['predicted_gy'] - r['observed_gy'])
        self.assertAlmostEqual(public['metrics']['mae'], np.mean([np.mean(np.abs(e)) for e in grouped.values()]))
        self.assertAlmostEqual(public['metrics']['rmse'], np.sqrt(np.mean([np.mean(np.square(e)) for e in grouped.values()])))
        self.assertAlmostEqual(public['metrics']['bias'], np.mean([np.mean(e) for e in grouped.values()]))

    def test_network_policy_and_routes_cannot_reach_private_api(self):
        config = json.loads((DEST / 'vercel.json').read_text())
        self.assertNotIn('rewrites', config)
        self.assertNotIn('functions', config)
        headers = {h['key']: h['value'] for h in config['headers'][0]['headers']}
        self.assertIn("connect-src 'self'", headers['Content-Security-Policy'])
        self.assertIn("form-action 'none'", headers['Content-Security-Policy'])
        self.assertEqual(config['buildCommand'], '')
        self.assertEqual(config['installCommand'], '')


if __name__ == '__main__':
    unittest.main()
