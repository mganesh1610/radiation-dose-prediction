import sys,unittest
from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import PLSRegression
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from matched_classical import integer_weight_counts,fit_classical
from abcd_models import balanced_weights

class MatchedClassicalTests(unittest.TestCase):
    def test_exact_study_donor_weights_and_moments(self):
        studies=['a']*7+['b']*4;donors=['a:1']*4+['a:2']*3+['b:1']*2+['b:2']*2
        w=balanced_weights(studies,donors);c=integer_weight_counts(w)
        np.testing.assert_allclose(c/c.sum(),w,atol=1e-15,rtol=0)
        x=np.random.default_rng(12).normal(size=(11,5));mx=np.average(x,weights=w,axis=0)
        rx=np.repeat(x,c,axis=0);np.testing.assert_allclose(rx.mean(0),mx,atol=1e-14)
        np.testing.assert_allclose((rx-mx).T@(rx-mx)/len(rx),(x-mx).T@(w[:,None]*(x-mx)),atol=1e-14)
        self.assertAlmostEqual(w[:7].sum(),.5);self.assertAlmostEqual(w[7:].sum(),.5)
    def test_uniform_pls_matches_native(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(35,8));y=x[:,0]*2+x[:,1]+rng.normal(size=35)*.1
        model,_=fit_classical(dict(model='pls',n_components=3),x,y,np.ones(35),1)
        expected=PLSRegression(n_components=3,scale=False).fit(x,y)
        np.testing.assert_allclose(model.predict(x),expected.predict(x),atol=1e-12)
    def test_weight_scale_invariance_and_permutation(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(15,9));y=rng.normal(size=15);w=np.array([1]*10+[2]*5)
        a,_=fit_classical(dict(model='pls',n_components=3),x,y,w,1)
        ix=rng.permutation(15);b,_=fit_classical(dict(model='pls',n_components=3),x[ix],y[ix],w[ix]*3,1)
        np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-10)
    def test_weighted_pls_matches_independent_explicit_replication(self):
        rng=np.random.default_rng(33);x=rng.normal(size=(18,7));y=rng.normal(size=18)
        counts=np.array([1]*12+[3]*6)
        model,_=fit_classical(dict(model='pls',n_components=3),x,y,counts/counts.sum(),1)
        native=PLSRegression(n_components=3,scale=False).fit(np.repeat(x,counts,axis=0),np.repeat(y,counts))
        uniform=PLSRegression(n_components=3,scale=False).fit(x,y)
        np.testing.assert_allclose(model.predict(x),native.predict(x),atol=1e-12)
        self.assertGreater(np.max(abs(model.predict(x)-uniform.predict(x))),1e-3)
    def test_invalid_and_excessive_replication_rejected(self):
        for w in [[0,1],[-1,2],[np.nan,1],[1/100003,1-1/100003]]:
            with self.assertRaises(ValueError):integer_weight_counts(w)
    def test_classical_fit_weights_change_solution_and_deterministic_replay(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(40,6));y=2*x[:,0]+rng.normal(size=40)
        params=[dict(model='elastic_net',alpha=.01,l1_ratio=.5),dict(model='gradient_boosting',max_depth=2,n_estimators=20)]
        for p in params:
            a,_=fit_classical(p,x,y,np.ones(40),42);b,_=fit_classical(p,x,y,np.r_[np.ones(20),np.ones(20)*9],42)
            c,_=fit_classical(p,x,y,np.ones(40),42)
            self.assertGreater(np.max(abs(a.predict(x)-b.predict(x))),1e-5)
            np.testing.assert_array_equal(a.predict(x),c.predict(x))
    def test_pls_dimension_guard_uses_original_profiles(self):
        with self.assertRaises(ValueError):fit_classical(dict(model='pls',n_components=4),np.ones((4,10)),np.arange(4),np.array([1,2,3,4]),1)

if __name__=='__main__':unittest.main()
