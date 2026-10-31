import sys,unittest,tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from scipy.stats import t
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from locked_evaluation_stats import donor_errors,paired_comparison,bootstrap_interval,holm_three


class LockedEvaluationTests(unittest.TestCase):
    def test_unequal_repeats_receive_equal_donor_weight(self):
        rows,m=donor_errors([0,0,0,0],[0,6,6,6],['a','b','b','b'])
        self.assertEqual(m['donor_mae'],3);self.assertEqual(m['pooled_mae'],4.5)
        self.assertAlmostEqual(m['donor_rmse'],np.sqrt(18));self.assertEqual(len(rows),2)

    def test_invalid_rows_cannot_be_silently_removed(self):
        with self.assertRaises(ValueError):donor_errors([0,1],[0,np.nan],['a','b'])
        with self.assertRaises(ValueError):donor_errors([0,1],[0],['a','b'])

    def test_direction_and_bonferroni_interval_match_hand_formula(self):
        d=np.array([-1.,-2.,-4.,1.]);ix=np.tile(np.arange(4),(10,1))
        result=paired_comparison(d+5,np.repeat(5.,4),ix)
        statistic=d.mean()/(d.std(ddof=1)/2)
        self.assertAlmostEqual(result['t'],statistic)
        self.assertAlmostEqual(result['raw_p'],t.cdf(statistic,3))
        self.assertAlmostEqual(result['simultaneous_t_ci']['high'],d.mean()+t.ppf(1-.05/6,3)*d.std(ddof=1)/2)
        self.assertLess(result['raw_p'],.5)

    def test_holm_includes_entire_fixed_family(self):
        c={k:dict(bookkeeping_p=p,estimable=True) for k,p in zip('BCD',[.01,.04,.03])}
        r=holm_three(c)
        np.testing.assert_allclose([r[k]['holm_adjusted_p'] for k in 'BCD'],[.03,.06,.06])
        self.assertTrue(r['B']['holm_reject']);self.assertFalse(r['D']['holm_reject'])
        with self.assertRaises(ValueError):holm_three({'B':c['B']})

    def test_nonestimable_remains_in_family(self):
        ix=np.tile(np.arange(4),(10,1));c=paired_comparison(np.ones(4),np.zeros(4),ix)
        self.assertFalse(c['estimable']);self.assertIsNone(c['raw_p']);self.assertEqual(c['bookkeeping_p'],1.)
        self.assertFalse(paired_comparison([1,np.nan],[0,0],np.array([[0,1]]))['estimable'])
        self.assertFalse(paired_comparison([1,2],[0,0],np.array([[0,1]]),False)['estimable'])

    def test_paired_bootstrap_preserves_constant_difference(self):
        a=np.arange(12.);b=a+.5;ix=np.random.default_rng(20261003).integers(0,12,(10000,12))
        ci=bootstrap_interval(a-b,ix)
        self.assertEqual(ci['low'],-.5);self.assertEqual(ci['high'],-.5);self.assertTrue(ci['degenerate'])

    def test_one_time_ledger_rejects_restart_before_any_inference(self):
        import run_locked_evaluation as runner
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);(directory/'run-start.json').write_text('{}')
            with patch.object(runner,'LOG',directory),patch.object(runner,'predict_panel') as predictor:
                with self.assertRaisesRegex(ValueError,'already started'):runner.run()
                predictor.assert_not_called()


if __name__=='__main__':unittest.main()
