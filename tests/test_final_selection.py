import copy
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_final_selection import candidate_score, choose_candidate
from run_abcd_experiments import completed, finish
from final_model_bundle import predict_panel


class FinalSelectionTests(unittest.TestCase):
    def test_donors_not_profiles_receive_equal_metric_weight(self):
        labels = {s: dict(dose_gy=0, donor_group='many' if s != 'd' else 'one') for s in 'abcd'}
        score = candidate_score([dict(prediction_ids=list('abcd'), prediction_gy=[0, 0, 0, 4])], list('abcd'), labels)
        self.assertEqual(score['mae'], 2)

    def test_repeated_or_missing_validation_rejected(self):
        labels = {s: dict(dose_gy=0, donor_group=s) for s in 'ab'}
        for ids in (['a', 'a'], ['a']):
            with self.assertRaises(ValueError):
                candidate_score([dict(prediction_ids=ids, prediction_gy=[0] * len(ids))], ['a', 'b'], labels)

    def test_original_id_breaks_exact_tie_and_no_partial_search(self):
        candidates = [dict(params=dict(id=i), metrics=dict(mae=1.0)) for i in reversed(range(8))]
        self.assertEqual(choose_candidate(candidates)['params']['id'], 0)
        with self.assertRaises(ValueError):
            choose_candidate(candidates[:-1])
        bad = copy.deepcopy(candidates); bad[0]['metrics']['mae'] = float('nan')
        with self.assertRaises(ValueError):
            choose_candidate(bad)

    def test_corrupt_or_other_run_artifact_cannot_resume_or_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p / 'model.joblib').write_bytes(b'not executable')
            finish(p, 'trusted')
            self.assertTrue(completed(p, 'trusted'))
            with self.assertRaises(ValueError):
                predict_panel(p, [], 'different')
            (p / 'model.joblib').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                predict_panel(p, [], 'trusted')


if __name__ == '__main__':
    unittest.main()
