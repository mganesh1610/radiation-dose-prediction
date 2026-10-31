import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from development_preprocessing import check_training_ids,sample_ranks,CORE,AUX
from prepare_auxiliary_agilent import parse


class DevelopmentProcessingTests(unittest.TestCase):
    def test_donor_complete_training_membership(self):
        metadata=[dict(profile_id='a0',donor_group='a',study='GSE97000'),dict(profile_id='a2',donor_group='a',study='GSE97000'),
                  dict(profile_id='b0',donor_group='b',study='GSE20173'),dict(profile_id='b2',donor_group='b',study='GSE20173')]
        check_training_ids(metadata,['a0','a2'],AUX)
        with self.assertRaises(ValueError):check_training_ids(metadata,['a0'],AUX)
        with self.assertRaises(ValueError):check_training_ids(metadata,['a0','a2'],CORE)
        with self.assertRaises(ValueError):check_training_ids(metadata,['GSE90909'],AUX)

    def test_sample_rank_monotonic_invariance_and_missingness(self):
        x=np.array([[1,7],[2,np.nan],[2,3],[9,1]],dtype=float)
        a=sample_ranks(x);b=sample_ranks(3*x+8)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,0:1],sample_ranks(x[:,0:1]))
        self.assertTrue(np.isnan(a[1,1]));self.assertEqual(a[1,0],a[2,0])
        self.assertTrue(np.nanmin(a)>0);self.assertTrue(np.nanmax(a)<1)


if __name__=='__main__':unittest.main()
