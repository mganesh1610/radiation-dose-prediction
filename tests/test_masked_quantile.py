import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from frozen_quantile import FrozenQuantile
from masked_quantile import MaskedFrozenQuantile


class MaskedQuantileTests(unittest.TestCase):
    def test_hand_calculated_missing_fixture(self):
        # Column 1 interpolates [2, 6] to [2, 4, 6]; col 2 is [4, 8, 12].
        x = np.array([[2., 4.], [999., 8.], [6., 12.]])
        m = np.array([[True, True], [False, True], [True, True]])
        q = MaskedFrozenQuantile.fit(x, m, ['a','b','c'], ['s1','s2'], ['s1','s2'])
        np.testing.assert_array_equal(q.target, [3,6,9])
        np.testing.assert_array_equal(q.transform(x,m,['a','b','c']), [[3,3],[np.nan,6],[9,9]])

    def test_complete_case_equivalence_including_ties(self):
        x = np.array([[1.,4.], [1.,8.], [1.,16.], [7.,32.]])
        f = ['a','b','c','d']; s = ['s1','s2']; m = np.ones_like(x,dtype=bool)
        old = FrozenQuantile.fit(x,f,s,s); new = MaskedFrozenQuantile.fit(x,m,f,s,s)
        np.testing.assert_array_equal(old.target, new.target)
        np.testing.assert_array_equal(old.transform(x,f), new.transform(x,m,f))

    def test_masked_values_cannot_influence_reference_or_result(self):
        x=np.array([[1.,2.],[3.,4.],[5.,6.]]);m=np.array([[True,True],[False,True],[True,False]])
        f=['a','b','c'];s=['x','y']; q=MaskedFrozenQuantile.fit(x,m,f,s,s)
        changed=x.copy();changed[~m]=np.nan
        q2=MaskedFrozenQuantile.fit(changed,m,f,s,s)
        np.testing.assert_array_equal(q.target,q2.target)
        np.testing.assert_array_equal(q.transform(x,m,f),q2.transform(changed,m,f))

    def test_validation_values_and_masks_cannot_influence_fit(self):
        x=np.array([[1.,2.,5.],[3.,4.,6.],[5.,6.,7.]]);m=np.ones_like(x,dtype=bool)
        f=['a','b','c'];s=['x','y','z'];q=MaskedFrozenQuantile.fit(x,m,f,s,['x','y'])
        x[:,2]=np.nan;m[:,2]=False
        np.testing.assert_array_equal(q.target,MaskedFrozenQuantile.fit(x,m,f,s,['x','y']).target)

    def test_single_batch_order_and_artifact(self):
        x=np.array([[1.,8.,5.],[3.,4.,6.],[5.,2.,7.]]);m=np.ones_like(x,dtype=bool);m[1,1]=False
        f=['a','b','c'];s=['x','y','z'];q=MaskedFrozenQuantile.fit(x,m,f,s,['x','y'])
        result=q.transform(x,m,f)
        for j in range(3):np.testing.assert_array_equal(result[:,[j]],q.transform(x[:,[j]],m[:,[j]],f))
        np.testing.assert_array_equal(result[:,::-1],q.transform(x[:,::-1],m[:,::-1],f))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fit.npz';q.save(p)
            np.testing.assert_array_equal(result,MaskedFrozenQuantile.load(p).transform(x,m,f))

    def test_invalid_contracts_fail(self):
        x=np.ones((3,2));m=np.ones_like(x,dtype=bool);f=['a','b','c'];s=['x','y']
        for train in [[],['x','x'],['absent']]:
            with self.assertRaises(ValueError):MaskedFrozenQuantile.fit(x,m,f,s,train)
        m[:,0]=False
        with self.assertRaises(ValueError):MaskedFrozenQuantile.fit(x,m,f,s,s)
        m[:]=True;x[0,0]=0
        with self.assertRaises(ValueError):MaskedFrozenQuantile.fit(x,m,f,s,s)
        x[:]=1;q=MaskedFrozenQuantile.fit(x,m,f,s,s)
        with self.assertRaises(ValueError):q.transform(x,m,f[::-1])


if __name__=='__main__':unittest.main()
