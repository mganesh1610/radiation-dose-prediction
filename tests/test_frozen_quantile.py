import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from frozen_quantile import FrozenQuantile


class FrozenQuantileTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1., 4., 5.], [3., 8., 5.], [2., 2., 10.]])
        self.features = ('a', 'b', 'c')
        self.samples = ('train1', 'train2', 'validation')
        self.fit = FrozenQuantile.fit(self.x, self.features, self.samples, self.samples[:2])

    def test_known_target_and_tie_average(self):
        np.testing.assert_array_equal(self.fit.target, [1.5, 3., 5.5])
        np.testing.assert_array_equal(self.fit.transform(self.x[:, 2:], self.features)[:, 0], [2.25, 2.25, 5.5])

    def test_validation_changes_do_not_affect_fit_even_when_invalid(self):
        changed = self.x.copy(); changed[:, 2] = np.nan
        second = FrozenQuantile.fit(changed, self.features, self.samples, self.samples[:2])
        np.testing.assert_array_equal(second.target, self.fit.target)

    def test_single_batch_and_batch_order_invariance(self):
        together = self.fit.transform(self.x, self.features)
        np.testing.assert_array_equal(together[:, 2:], self.fit.transform(self.x[:, 2:], self.features))
        np.testing.assert_array_equal(together[:, ::-1], self.fit.transform(self.x[:, ::-1], self.features))

    def test_artifact_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'reference.npz'; self.fit.save(p)
            loaded = FrozenQuantile.load(p)
            self.assertEqual(loaded.fit_sample_ids, self.samples[:2])
            np.testing.assert_array_equal(loaded.transform(self.x, self.features), self.fit.transform(self.x, self.features))

    def test_invalid_values_and_feature_order_rejected(self):
        with self.assertRaises(ValueError): self.fit.transform(self.x[::-1], self.features[::-1])
        for value in (0., -1., np.nan, np.inf):
            x = self.x.copy(); x[0, 0] = value
            with self.assertRaises(ValueError): self.fit.transform(x, self.features)

    def test_duplicate_or_absent_training_ids_rejected(self):
        for ids in [('train1', 'train1'), ('absent',), ()]:
            with self.assertRaises(ValueError): FrozenQuantile.fit(self.x, self.features, self.samples, ids)


if __name__ == '__main__':
    unittest.main()
