import sys, tempfile, unittest
from pathlib import Path
import joblib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from clustering_models import WeightedPCA, VariableGeneView, ClusterRegressor, weighted_median


class ClusteringTests(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(74)
        self.x = r.normal(size=(40,12)); self.x[:20,0] += 3
        self.y = r.normal(size=40); self.w = np.linspace(1,3,40)

    def test_weighted_pca_and_weight_scale(self):
        a = WeightedPCA(5).fit(self.x,self.w)
        b = WeightedPCA(5).fit(self.x,100*self.w)
        np.testing.assert_allclose(a.center_,np.average(self.x,axis=0,weights=self.w))
        np.testing.assert_allclose(a.axes_.T@a.axes_,b.axes_.T@b.axes_,atol=1e-12)
        np.testing.assert_allclose(np.average(a.transform(self.x),axis=0,weights=self.w),0,atol=1e-12)

    def test_view_missing_and_gene_order(self):
        x = self.x.copy(); x[1,1] = np.nan
        genes = list(map(str,range(12)))
        v = VariableGeneView(8,5).fit(x,genes,self.w)
        original = v.transform(x,genes)
        extreme = np.ones((1,12))*1e8; v.transform(extreme,genes)
        np.testing.assert_array_equal(v.transform(x,genes),original)
        with self.assertRaises(ValueError): v.transform(x,genes[::-1])

    def test_clusters_do_not_fit_targets(self):
        for model in ['kmeans_ridge','gmm_ridge','cluster_median']:
            a = ClusterRegressor(model,5,2,1.,17).fit(self.x,self.y,self.w)
            b = ClusterRegressor(model,5,2,1.,17).fit(self.x,self.y[::-1],self.w)
            np.testing.assert_array_equal(a.pca_.axes_,b.pca_.axes_)
            np.testing.assert_array_equal(a.cluster_.predict(a.pca_.transform(self.x)),b.cluster_.predict(b.pca_.transform(self.x)))

    def test_persistence_and_single_batch(self):
        for model in ['pca_ridge','kmeans_ridge','gmm_ridge','cluster_median']:
            a = ClusterRegressor(model,5,2,1.,17).fit(self.x,self.y,self.w)
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'m.joblib';joblib.dump(a,p);b=joblib.load(p)
                np.testing.assert_allclose(a.predict(self.x),b.predict(self.x),atol=1e-12)
                np.testing.assert_allclose(a.predict(self.x),[a.predict(x[None])[0] for x in self.x],atol=1e-10)

    def test_median_and_invalid(self):
        self.assertEqual(weighted_median(np.array([0.,2.,99.]),np.array([1.,4.,1.])),2.)
        with self.assertRaises(ValueError): WeightedPCA(50).fit(self.x,self.w)
        with self.assertRaises(ValueError): ClusterRegressor('pca_ridge').fit(self.x,self.y,np.zeros(40))
        with self.assertRaises(ValueError): ClusterRegressor('gmm_ridge',5,80).fit(self.x,self.y,self.w)


if __name__ == '__main__': unittest.main()
