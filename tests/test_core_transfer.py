import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from core_transfer import CoreTransfer


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[1,1,9,200],[3,3,11,400],[5,5,13,800]],float)
        self.ids=['p1','p2','p3'];self.genes=['1','2','2']
        self.samples=['a','b','c','held'];self.studies=['large','large','small','held']
        self.mask=np.ones_like(self.x,dtype=bool);self.detect=self.mask.copy()
    def fit(self,x=None,mask=None,det=None):
        return CoreTransfer.fit(self.x if x is None else x,self.mask if mask is None else mask,
                                self.detect if det is None else det,self.ids,self.genes,self.samples,self.studies,self.samples[:3])
    def test_equal_study_not_equal_sample_weight(self):
        q=self.fit();np.testing.assert_array_equal(q.reference.target,[5,7,9])
    def test_heldout_values_mask_detection_ignored(self):
        q=self.fit();x=self.x.copy();m=self.mask.copy();d=self.detect.copy()
        x[:,-1]=np.nan;m[:,-1]=False;d[:,-1]=False
        r=self.fit(x,m,d)
        np.testing.assert_array_equal(q.keep,r.keep);np.testing.assert_array_equal(q.reference.target,r.reference.target)
    def test_small_study_filter_is_not_overwhelmed(self):
        d=self.detect.copy();d[0,2]=False;q=self.fit(det=d)
        np.testing.assert_array_equal(q.keep,[False,True,True])
    def test_single_order_roundtrip_and_missingness(self):
        q=self.fit();m=self.mask.copy();m[0,-1]=False
        g,a,_=q.transform(self.x,m,self.ids)
        _,b,_=q.transform(self.x[:,-1:],m[:,-1:],self.ids)
        np.testing.assert_allclose(a[:,-1:],b,equal_nan=True);self.assertTrue(np.isnan(a[0,-1]))
        _,rev,_=q.transform(self.x[:,::-1],m[:,::-1],self.ids)
        np.testing.assert_allclose(a[:,::-1],rev,equal_nan=True)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ref.npz';q.save(p);r=CoreTransfer.load(p)
            np.testing.assert_allclose(a,r.transform(self.x,m,self.ids)[1],equal_nan=True)
    def test_unknown_membership_and_reordered_probe_rejected(self):
        with self.assertRaises(ValueError):CoreTransfer.fit(self.x,self.mask,self.detect,self.ids,self.genes,self.samples,self.studies,['absent'])
        with self.assertRaises(ValueError):self.fit().transform(self.x,self.mask,self.ids[::-1])


if __name__=='__main__':unittest.main()
