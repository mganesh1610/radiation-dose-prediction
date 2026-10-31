import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from frozen_affymetrix import FrozenAffymetrix,median_polish_batch,quantile_apply


class FrozenAffymetrixTests(unittest.TestCase):
    def data(self):
        x=np.array([[2,3,9,10],[5,6,20,21],[10,12,30,33],[3,5,13,15],[8,9,22,24]],dtype=float)
        return x,np.array(['a','a','a','b','b']),np.arange(5),np.array(['s1','s2','s3','s4']),np.array(['p1','p1','p2','p3'])

    def test_heldout_values_never_enter_fit(self):
        x,ps,phys,samples,profiles=self.data()
        a=FrozenAffymetrix.fit(x,ps,phys,samples,profiles,['p1','p2'])
        x[:,3]=np.nan
        b=FrozenAffymetrix.fit(x,ps,phys,samples,profiles,['p1','p2'])
        np.testing.assert_array_equal(a.target,b.target);np.testing.assert_array_equal(a.effects,b.effects)
        self.assertEqual(a.fit_sample_ids,('s1','s2','s3'))

    def test_repeated_array_does_not_reweight_profile_reference(self):
        x,ps,phys,samples,profiles=self.data();x[:,1]=x[:,0]
        a=FrozenAffymetrix.fit(x,ps,phys,samples,profiles,['p1','p2'])
        ix=[0,2,3];b=FrozenAffymetrix.fit(x[:,ix],ps,phys,samples[ix],profiles[ix],['p1','p2'])
        np.testing.assert_array_equal(a.target,b.target);np.testing.assert_array_equal(a.effects,b.effects)

    def test_replay_order_and_single_sample(self):
        x,ps,phys,samples,profiles=self.data();a=FrozenAffymetrix.fit(x,ps,phys,samples,profiles,['p1','p2'])
        y=a.transform(x,ps,phys)
        np.testing.assert_array_equal(y[:,::-1],a.transform(x[:,::-1],ps,phys))
        np.testing.assert_array_equal(y[:,3:4],a.transform(x[:,3:4],ps,phys))
        with TemporaryDirectory() as d:
            p=Path(d)/'a.npz';a.save(p);b=FrozenAffymetrix.load(p)
            np.testing.assert_array_equal(y,b.transform(x,ps,phys))
        with self.assertRaises(ValueError):a.transform(x,ps,phys[::-1])

    def test_ties_and_invalid_signal(self):
        y=quantile_apply(np.array([2,1,1,4]),np.array([1,3,5,7]))
        np.testing.assert_array_equal(y,[5,2,2,7])
        with self.assertRaises(ValueError):quantile_apply(np.array([0,1]),np.array([2,3]))

    def test_additive_model_recovered(self):
        x=np.array([[[1,3,7],[2,4,8],[4,6,10]]],dtype=float)
        r,t,c,it=median_polish_batch(x)
        np.testing.assert_allclose(t[:,None,None]+r[:,:,None]+c[:,None,:],x)
        np.testing.assert_allclose(np.median(x-r[:,:,None],axis=1),t[:,None]+c)

    def test_duplicate_identity_and_unknown_training_rejected(self):
        x,ps,phys,samples,profiles=self.data()
        with self.assertRaises(ValueError):FrozenAffymetrix.fit(x,ps,phys,samples,profiles,['unknown'])
        samples[1]=samples[0]
        with self.assertRaises(ValueError):FrozenAffymetrix.fit(x,ps,phys,samples,profiles,['p1'])


if __name__=='__main__':unittest.main()
