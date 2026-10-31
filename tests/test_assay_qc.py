import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from assay_qc import aggregate_median, fit_probe_filter, resolve_gene, technical_valid, platform_probe_id


class AssayQCTests(unittest.TestCase):
    def test_platform_probe_column_semantics(self):
        row={'ID':'3','NAME':'NM_001003689','SPOT_ID':'A_23_P80353'}
        self.assertEqual(platform_probe_id('GPL1708',row),'A_23_P80353')
        self.assertEqual(platform_probe_id('GPL10332',{'NAME':'A_23_P80353'}),'A_23_P80353')
        self.assertEqual(platform_probe_id('GPL13497',{'ID':'A_23_P80353'}),'A_23_P80353')
        with self.assertRaises(ValueError):platform_probe_id('unknown',row)

    def test_technical_flags_do_not_confuse_low_detection_with_defects(self):
        columns = ['gIsSaturated', 'IsManualFlag', 'gIsFeatNonUnifOL', 'gIsFeatPopnOL', 'gIsFound', 'gIsWellAboveBG']
        flags = np.zeros((2, 2, 6), dtype=np.uint8)
        flags[:, :, 4] = 1
        flags[0, 0, 4] = 0  # not-found alone does not imply technical corruption
        flags[1, 1, 3] = 1
        np.testing.assert_array_equal(technical_valid(flags, columns), [[True, True], [True, False]])
        with self.assertRaises(ValueError):
            technical_valid(flags, columns[:-1])

    def test_repeated_spots_do_not_overweight_a_probe_at_gene_level(self):
        p, v, n = aggregate_median(np.array([[1.], [1.], [9.]]), ['p1', 'p1', 'p2'])
        np.testing.assert_array_equal(v[:, 0], [1, 9])
        _, g, _ = aggregate_median(v, ['gene', 'gene'])
        self.assertEqual(g[0, 0], 5)
        self.assertEqual(n[0, 0], 2)

    def test_masked_values_stay_missing_and_batch_independent(self):
        x = np.array([[1., 1000.], [9., 3.], [4., 5.]])
        m = np.array([[False, False], [False, True], [True, True]])
        _, full, n = aggregate_median(x, ['p', 'p', 'q'], m)
        self.assertTrue(np.isnan(full[0, 0]))
        self.assertEqual(n[0, 0], 0)
        _, single, _ = aggregate_median(x[:, [1]], ['p', 'p', 'q'], m[:, [1]])
        np.testing.assert_array_equal(single, full[:, [1]])

    def test_filter_is_training_only(self):
        x = np.ones((3, 4)); x[0, 0] = np.nan
        d = np.ones_like(x, dtype=bool); d[1, :2] = False
        train = np.array([True, True, False, False])
        before = fit_probe_filter(x, d, train)[0]
        x[:, 2:] = np.nan; d[:, 2:] = False
        np.testing.assert_array_equal(before, fit_probe_filter(x, d, train)[0])
        np.testing.assert_array_equal(before, [False, False, True])

    def test_mapping_rejects_ambiguous_old_and_missing_ids(self):
        current = {'1': {}, '2': {}}
        self.assertEqual(resolve_gene(['1', '1'], current), ('1', 'current_exact_gene_id'))
        for ids in [['1', '2'], ['1', ''], ['1 /// 2'], ['9'], ['']]:
            self.assertEqual(resolve_gene(ids, current)[0], '')

    def test_invalid_mask_and_empty_training_fail(self):
        with self.assertRaises(ValueError):
            aggregate_median(np.ones((2, 2)), ['p', 'q'], np.ones((2, 2)))
        with self.assertRaises(ValueError):
            fit_probe_filter(np.ones((2, 2)), np.ones((2, 2), bool), np.zeros(2, bool))


if __name__ == '__main__':
    unittest.main()
