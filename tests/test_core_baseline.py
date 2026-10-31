import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from core_baseline import GenePreparation, donor_metrics, inner_splits, bootstrap_intervals, fit_estimator


class BaselineTests(unittest.TestCase):
    def test_donor_balancing_not_sample_balancing(self):
        m, _ = donor_metrics(np.zeros(4), np.array([0., 0., 0., 4.]), ['a', 'a', 'a', 'b'])
        self.assertEqual(m['mae'], 2.)
        self.assertEqual(m['bias'], 2.)
        self.assertAlmostEqual(m['rmse'], np.sqrt(8.))

    def test_all_donor_profiles_stay_together_and_validate_once(self):
        groups = np.repeat([f'{s}:{d}' for s in ('a', 'b') for d in range(5)], 3)
        studies = np.array([g[0] for g in groups])
        count = np.zeros(len(groups), int)
        for tr, va in inner_splits(studies, groups):
            self.assertFalse(set(groups[tr]) & set(groups[va]))
            self.assertEqual(set(studies[va]), {'a', 'b'})
            count[va] += 1
        np.testing.assert_array_equal(count, 1)
        with self.assertRaises(ValueError):
            list(inner_splits(['a', 'b'], ['same', 'same']))

    def test_ranking_rejects_study_only_signal_and_empty_gene(self):
        y = np.array([0., 1., 2., 0., 1., 2.])
        st = np.array(['a'] * 3 + ['b'] * 3)
        x = np.column_stack((y, [0., 0., 0., 100., 100., 100.], np.full(6, np.nan)))
        p = GenePreparation.fit(x, ['dose', 'batch', 'empty'], y, st, list('abcdef'), 3)
        np.testing.assert_array_equal(p.selected, [0])
        before = p.fingerprint()
        p.transform(np.array([[np.nan, 1e9, 50.]]), p.gene_ids, True)
        self.assertEqual(before, p.fingerprint())
        with self.assertRaises(ValueError):
            p.transform(x, p.gene_ids[::-1], True)

    def test_imputation_scaling_only_use_training_statistics(self):
        x = np.array([[1., 7.], [3., 8.], [np.nan, 9.]])
        p = GenePreparation.fit(x, ['1', '2'], np.arange(3.), ['s'] * 3, list('abc'), 2)
        np.testing.assert_array_equal(p.medians, [2., 8.])
        transformed = p.transform(np.array([[np.nan, np.nan]]), p.gene_ids, False)
        np.testing.assert_array_equal(transformed, p.medians[p.selected][None, :])
        self.assertTrue(np.isfinite(p.transform(x, p.gene_ids, True)).all())

    def test_bootstrap_reproducible_and_cluster_degenerate_case(self):
        y = np.zeros(4); pred = np.full(4, 2.); groups = ['a', 'a', 'b', 'b']; st = ['s'] * 4
        ci = bootstrap_intervals(y, pred, groups, st, 100)
        self.assertEqual(ci, {'mae': [2., 2.], 'rmse': [2., 2.], 'bias': [2., 2.]})
        self.assertEqual(ci, bootstrap_intervals(y, pred, groups, st, 100))

    def test_invalid_outcomes_fail_and_estimators_predict_finitely(self):
        with self.assertRaises(ValueError): donor_metrics([1.], [np.nan], ['a'])
        rng = np.random.default_rng(8); x = rng.normal(size=(15, 6)); y = x[:, 0] * 2
        for name, params in [('elastic_net', dict(alpha=.1, l1_ratio=.5)),
                             ('pls', dict(n_components=2)),
                             ('gradient_boosting', dict(max_depth=1, n_estimators=10))]:
            model = fit_estimator(name, params, x, y)
            self.assertTrue(np.isfinite(model.predict(x)).all())


if __name__ == '__main__': unittest.main()
