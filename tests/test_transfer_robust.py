import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from transfer_robust import fit_stable, panel_transform, load_features, design_score, candidate_score


class TransferRobustTests(unittest.TestCase):
    def fixture(self):
        y=np.tile([0.,2.,5.],4); studies=np.repeat(['a','b'],6)
        donors=np.repeat(['a:1','a:2','b:1','b:2'],3); ids=[f'p{i}' for i in range(12)]
        good=np.column_stack([y*(i+1)+i for i in range(5)])
        reverse=np.where(studies=='a',y,-y)
        shifted=y+np.where(studies=='a',0,100.)
        donor_only=np.repeat([0.,20.,40.,60.],3)
        x=np.column_stack([good,reverse,shifted,donor_only,np.ones(12),np.full(12,np.nan)])
        return x,[f'g{i}' for i in range(10)],y,studies,donors,ids
    def test_filters_reversal_source_offset_and_donor_confounder(self):
        g,info,e=fit_stable(*self.fixture(),width=12)
        self.assertEqual(set(g.selected),set(range(5)))
        self.assertEqual(info['selected_genes'],5);self.assertFalse(info['single_source_fallback'])
        self.assertFalse(e['eligible'][5]);self.assertFalse(e['eligible'][6]);self.assertFalse(e['eligible'][7])
    def test_single_source_fallback_uses_only_its_controls(self):
        x,genes,y,s,d,ids=self.fixture()
        g,info,e=fit_stable(x[:6],genes,y[:6],s[:6],d[:6],ids[:6],width=12)
        self.assertTrue(info['single_source_fallback']);self.assertIn(6,g.selected)
        self.assertTrue(np.all(e['control_shift_sd'][e['eligible']]==0))
    def test_zero_padding_reload_and_transform_do_not_refit(self):
        x,genes,y,s,d,ids=self.fixture();g,_,_=fit_stable(x,genes,y,s,d,ids,width=12)
        before=g.fingerprint();a=panel_transform(g,x,genes,12)
        self.assertEqual(a.shape,(12,12));np.testing.assert_array_equal(a[:,5:],0.)
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'panel.npz';g.save(f);loaded=load_features(f)
            np.testing.assert_array_equal(panel_transform(loaded,x,genes,12),a)
        panel_transform(g,x*1000,genes,12);self.assertEqual(before,g.fingerprint())
    def test_training_membership_permutation_preserves_selected_genes(self):
        x,genes,y,s,d,ids=self.fixture();g,_,_=fit_stable(x,genes,y,s,d,ids,width=12)
        order=np.arange(12)[::-1]
        h,_,_=fit_stable(x[order],genes,y[order],s[order],d[order],np.array(ids)[order],width=12)
        self.assertEqual(set(g.selected),set(h.selected))
        np.testing.assert_allclose(panel_transform(g,x,genes,12),panel_transform(h,x,genes,12),atol=1e-12)
    def test_insufficient_stable_genes_fails_without_relaxation(self):
        x,genes,y,s,d,ids=self.fixture()
        with self.assertRaises(ValueError):fit_stable(x[:,5:],genes[5:],y,s,d,ids,width=12)
    def test_invalid_donor_membership_and_control_rejected(self):
        x,genes,y,s,d,ids=self.fixture();d=d.copy();d[6:9]='a:1'
        with self.assertRaises(ValueError):fit_stable(x,genes,y,s,d,ids,width=12)
        x,genes,y,s,d,ids=self.fixture();y=y+1
        with self.assertRaises(ValueError):fit_stable(x,genes,y,s,d,ids,width=12)
    def records(self,offset_a=1.,offset_b=3.):
        rr=[]
        for study,nd,offset in [('a',4,offset_a),('b',1,offset_b)]:
            for donor in range(nd):
                for dose in [0.,2.,5.]:rr.append(dict(profile_id=f'{study}:{donor}:{dose}',study=study,donor_group=f'{study}:{donor}',actual_gy=dose,predicted_gy=dose+offset))
        return rr
    def test_equal_study_score_not_dominated_by_larger_source(self):
        score,detail=design_score(self.records())
        self.assertAlmostEqual(score,2.)
        result=candidate_score(self.records(),self.records(3.,5.))
        self.assertAlmostEqual(result['score'],3.)
    def test_scoring_rejects_duplicate_and_missing_strata_or_members(self):
        rr=self.records()
        for bad in [rr+[rr[0]],[r for r in rr if r['actual_gy']!=0]]:
            with self.assertRaises(ValueError):design_score(bad)
        with self.assertRaises(ValueError):candidate_score(rr,rr[:-1])


if __name__=='__main__':unittest.main()
