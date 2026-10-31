import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from investigate_control_bias import donor_contrasts, linear_contributions


class ControlBiasDiagnosticsTests(unittest.TestCase):
    def records(self, offset=3., slope=.4, donor='s:d1'):
        return [dict(model='m', seed_index=0, study='s', donor_group=donor,
                     profile_id=f'{donor}:{y}', actual_gy=y, predicted_gy=offset+slope*y)
                for y in [0., .5, 2., 5., 8.]]

    def test_separates_offset_from_compressed_response(self):
        detail, slopes = donor_contrasts(self.records())
        self.assertAlmostEqual(slopes[0]['slope_gy_per_gy'], .4)
        self.assertAlmostEqual(slopes[0]['top_predicted_increment_gy'], 1.2)
        for r in detail:
            self.assertAlmostEqual(r['residual_gy'], 3.-.6*r['dose_gy'])
            self.assertAlmostEqual(r['increment_error_gy'], -.6*r['dose_gy'])
            self.assertAlmostEqual(r['control_prediction_gy']+r['increment_error_gy'], r['residual_gy'])
        _, flat = donor_contrasts(self.records(slope=0.))
        self.assertEqual(flat[0]['slope_gy_per_gy'], 0.)
        self.assertIsNone(flat[0]['spearman'])  # Correlation is undefined for a constant predictor.

    def test_matches_each_donor_control_despite_input_order(self):
        data = self.records() + self.records(-2., 1.3, 's:d2')
        detail, slopes = donor_contrasts(list(reversed(data)))
        self.assertEqual(len(slopes), 2)
        for r in detail:
            self.assertEqual(r['control_prediction_gy'], 3. if r['donor_group']=='s:d1' else -2.)

    def test_missing_and_duplicate_controls_or_doses_rejected(self):
        r = self.records()
        for bad in [r[1:], r+[r[0]], r+[r[2]]]:
            with self.assertRaises(ValueError):
                donor_contrasts(bad)

    def test_contributions_equal_independent_linear_score_difference(self):
        coef = np.array([.5, -1., 2.]); a = np.array([1., 3., 8.]); b = np.array([2., 7., 4.])
        value = linear_contributions(coef, a, b, 2.5)
        def prediction(x):
            return (coef@x + 1.7)*2.5 + 4.
        self.assertAlmostEqual(value.sum(), prediction(b)-prediction(a))
        np.testing.assert_allclose(value, [1.25, -10., -20.])
        with self.assertRaises(ValueError):
            linear_contributions(coef, a[:2], b, 1.)


if __name__ == '__main__':
    unittest.main()
