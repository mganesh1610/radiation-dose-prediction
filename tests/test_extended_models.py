import sys,unittest,tempfile
from pathlib import Path
import numpy as np
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from extended_models import fit_estimator,binary_metrics,WeightedPolynomialRidge

class ExtendedTests(unittest.TestCase):
    def test_polynomial_training_only_reload(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(25,12));y=x[:,0]**2+.1*x[:,1];w=np.arange(1,26)
        m=WeightedPolynomialRidge(5,1).fit(x,y,w);state=m.axes_.copy();center=m.center_.copy()
        v=rng.normal(size=(4,12));p=m.predict(v);m.predict(v+999)
        np.testing.assert_array_equal(state,m.axes_);np.testing.assert_array_equal(center,m.center_)
        np.testing.assert_allclose(center,np.average(x,axis=0,weights=w))
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'m.joblib';joblib.dump(m,f);np.testing.assert_allclose(joblib.load(f).predict(v),p)
        np.testing.assert_allclose(np.concatenate([m.predict(a[None]) for a in v]),p,atol=1e-10)
        self.assertEqual(m.poly_.n_output_features_,20)
    def test_class_metric_equal_donor(self):
        # One donor has more repeated positive samples; their influence remains equal.
        v=binary_metrics([0,1,0,1,1,1],[.1,.9,.8,.2,.2,.2],['a','a','b','b','b','b'])
        self.assertEqual(v['balanced_accuracy'],.5);self.assertEqual(v['specificity'],.5)
    def test_class_labels_invalid(self):
        with self.assertRaises(ValueError): binary_metrics([1,1],[.3,.4],['a','b'])
        with self.assertRaises(ValueError):binary_metrics([0,1],[-.1,.5],['a','b'])
    def test_estimator_smoke_weight_invariance(self):
        rng=np.random.default_rng(6);x=rng.normal(size=(24,8));y=x[:,0]+x[:,1]*.3;w=np.arange(1,25)
        params=[dict(model='ridge',alpha=1),dict(model='lasso',alpha=.1),dict(model='svr_linear',C=1,epsilon=.1),dict(model='svr_rbf',C=1,gamma=.01),dict(model='random_forest',max_depth=3,min_samples_leaf=3),dict(model='adaboost',max_depth=1,learning_rate=.1),dict(model='xgboost',max_depth=1,reg_lambda=1),dict(model='polynomial_ridge',components=5,alpha=1)]
        for p in params:
            a=fit_estimator(p,x,y,w,7);b=fit_estimator(p,x,y,w*100,7)
            np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-6)
    def test_classifiers_and_probabilities(self):
        x=np.random.default_rng(9).normal(size=(40,10));y=(x[:,0]>0).astype(int)
        for p in [dict(model='logistic_l1',C=.1),dict(model='logistic_l2',C=.1),dict(model='gaussian_nb',var_smoothing=1e-5)]:
            m=fit_estimator(p,x,y,np.ones(40),3);self.assertEqual(list(m.classes_),[0,1]);np.testing.assert_allclose(m.predict_proba(x).sum(1),1.)
    def test_fit_rejects_invalid(self):
        with self.assertRaises(ValueError):fit_estimator(dict(model='ridge',alpha=1),np.ones((2,3)),[1,2],[1,0],3)

if __name__=='__main__':unittest.main()
