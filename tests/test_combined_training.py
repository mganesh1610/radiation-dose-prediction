import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from training_contract import COMBINED_DATASET, job_request, validate_job
from weighted_pls import WeightedPLS


class CombinedTrainingTests(unittest.TestCase):
    def test_dataset_and_request_versions_must_agree(self):
        job = job_request(dict(method='pls',job_id='combined_test',holdout_study='GSE8917'),COMBINED_DATASET)
        self.assertEqual(job['schema_version'],2)
        validate_job(job)
        with self.assertRaises(ValueError):
            validate_job(dict(job,schema_version=1))

    def test_weighted_pls_matches_exact_replication(self):
        rng=np.random.default_rng(567)
        x,y,w=rng.normal(size=(40,8)),rng.normal(size=40),rng.integers(1,6,size=40)
        v=rng.normal(size=(10,8))
        expected=PLSRegression(n_components=5,scale=False).fit(np.repeat(x,w,axis=0),np.repeat(y,w))
        np.testing.assert_allclose(WeightedPLS(5).fit(x,y,w).predict(v),expected.predict(v).ravel(),atol=1e-10)

    def test_regularized_solver_preserves_penalty_in_original_units(self):
        try:
            from weighted_elastic_net import fit_regularized
        except ModuleNotFoundError as exc:
            if exc.name!='osqp': raise
            self.skipTest('Development experiment solver is optional for the web app')
        rng=np.random.default_rng(13)
        x,y,w=rng.normal(size=(50,9)),rng.normal(size=50),rng.integers(1,5,size=50)
        v=rng.normal(size=(8,9))
        for ratio in (.1,.9,1.):
            b,a=fit_regularized(x,y,w,.01,ratio)
            reference=ElasticNet(alpha=.01,l1_ratio=ratio,tol=1e-10,max_iter=100000).fit(x,y,sample_weight=w)
            np.testing.assert_allclose(v@b+a,reference.predict(v),atol=1e-6)
