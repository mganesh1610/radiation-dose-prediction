from artifact_support import require_artifacts
"""Publication checks for frozen aggregate scientific results, without inference."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_reviewer_results import build_payload, encode, verify_freeze, EVAL, ORDER


class ReviewerResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_artifacts(EVAL/'result-freeze.json')
        cls.payload, cls.checked = build_payload()

    def test_published_errors_match_independent_donor_arithmetic(self):
        with (EVAL/'donor-metrics.tsv').open() as f:
            rows=list(csv.DictReader(f,delimiter='\t'))
        for card in self.payload['model_cards']:
            for score in card['scores']:
                donors=[r for r in rows if r['model']==card['id'] and int(r['seed'])==score['seed']]
                self.assertEqual(len(donors),12)
                for source,target in [('mae','donor_mae'),('bias','donor_bias')]:
                    self.assertAlmostEqual(statistics.fmean(float(r[source]) for r in donors),score[target],places=12)
                self.assertAlmostEqual(statistics.fmean(float(r['mse']) for r in donors)**.5,score['donor_rmse'],places=12)

    def test_all_frozen_versions_and_correct_primary_roles(self):
        cards=self.payload['model_cards']
        self.assertEqual([c['id'] for c in cards],list(ORDER))
        self.assertEqual(sum(len(c['scores']) for c in cards),18)
        for card in cards:
            first=card['scores'][0]
            self.assertEqual(first['seed'],card['primary_seed'])
            self.assertEqual(first['seed_role'],'untuned' if card['id']=='median_null' else 'primary')
        self.assertFalse(self.payload['live_prediction_enabled'])
        self.assertEqual(self.payload['status'],'completed_consumed')
        self.assertTrue(all(not c['reject'] and c['holm_adjusted_p']==1.0 for c in self.payload['comparisons']))

    def test_export_contains_no_private_records_or_executable_artifacts(self):
        encoded=encode(self.payload).decode()
        for forbidden in ['/Users/', 'context/', 'gs://', '@asu.edu', '@gmail.com', 'GSM',
                          'training_ids', 'donor_group', 'private_key', 'access_token',
                          'prediction_gy', '.joblib', '.pt', 'run-start', 'protected-before']:
            self.assertNotIn(forbidden,encoded)
        self.assertLess(len(encoded.encode()),50000)
        second,_=build_payload()
        self.assertEqual(encode(second),encode(self.payload))

    def test_modified_source_and_freeze_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'metric.json'; source.write_text('{"mae":1}')
            freeze=root/'freeze.json';freeze.write_text(json.dumps({'files':{'metric.json':hashlib.sha256(source.read_bytes()).hexdigest()}}))
            pinned=hashlib.sha256(freeze.read_bytes()).hexdigest()
            self.assertEqual(verify_freeze(freeze,pinned,root,'files'),1)
            source.write_text('{"mae":0}')
            with self.assertRaisesRegex(ValueError,'integrity'):
                verify_freeze(freeze,pinned,root,'files')
            freeze.write_text('{"files":{}}')
            with self.assertRaisesRegex(ValueError,'freeze changed'):
                verify_freeze(freeze,pinned,root,'files')


if __name__=='__main__':unittest.main()
