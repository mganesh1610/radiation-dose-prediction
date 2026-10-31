import io
import json
import os
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from analysis_service import Explorer, validate_options, pca, pairwise_correlation
from app.server import DATASET, create_app
from test_training_integration import MemoryStore

RELEASE = ROOT / 'data/cloud_exports' / ('core_' + DATASET['sha256'][:16]) / 'core.npz'
def needs_release(test):
    if os.environ.get('DEFENDER_REQUIRE_DATA_TESTS') == '1' or RELEASE.is_file():
        return test
    return unittest.skip('Requires the separately provisioned verified core release')(test)


class ReleaseStore(MemoryStore):
    def __init__(self):
        super().__init__();self.reads=[]
    def read(self,name,generation,expected_sha=None,limit=None):
        self.reads.append(name)
        assert name==DATASET['object'] and generation==DATASET['generation']
        return RELEASE.read_bytes()


class AnalysisTests(unittest.TestCase):
    def test_options_reject_locked_test_arbitrary_fields_and_unbounded_size(self):
        base=dict(study='all',representation='quantile',genes=500)
        for change in [dict(study='GSE90909'),dict(study=[]),dict(genes=True),dict(genes=999999),
                       dict(representation='clinical'),dict(command='read_file')]:
            with self.assertRaises(ValueError):validate_options(dict(base,**change))

    def test_pca_matches_independent_svd_with_missing_values(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(12,25));x[2,4]=np.nan
        scores,ratios=pca(x)
        filled=np.where(np.isfinite(x),x,np.nanmedian(x,axis=0));filled-=filled.mean(axis=0)
        u,s,v=np.linalg.svd(filled,full_matrices=False)
        self.assertTrue(np.allclose(ratios,s[:2]**2/np.sum(s**2),atol=1e-12))
        self.assertTrue(np.allclose(scores@scores.T,(u[:,:2]*s[:2])@(u[:,:2]*s[:2]).T,atol=1e-10))

    def test_pairwise_missing_constant_and_small_sample_contract(self):
        x=np.array([[1,2,3,np.nan],[3,2,1,0],[1,1,1,1],[1,np.nan,np.nan,2]])
        values,counts=pairwise_correlation(x)
        self.assertAlmostEqual(values[0][1],-1);self.assertEqual(counts[0][1],3)
        self.assertIsNone(values[2][2]);self.assertIsNone(values[0][3])

    @needs_release
    def test_catalog_and_live_analysis_are_pinned_read_only_and_repeatable(self):
        store=ReleaseStore();service=Explorer(DATASET);catalog=service.catalog(store)
        self.assertEqual([s['profiles'] for s in catalog['studies']],[100,25,20])
        self.assertEqual(catalog['groups'],30);self.assertEqual(catalog['probes'],16237)
        self.assertFalse(catalog['locked_test_included'])
        before=service.data['values'].copy()
        first=service.analyze(store,dict(study='GSE8917',representation='quantile',genes=100))
        self.assertEqual(len(first['samples']),25);self.assertEqual(first['genes_used'],100)
        self.assertTrue(all(s['study_id']=='GSE8917' for s in first['samples']))
        self.assertEqual(first['qc']['automatic_exclusions'],0)
        self.assertEqual(np.array(first['heatmap']['log2']).shape,(20,25))
        self.assertTrue(np.array_equal(before,service.data['values'],equal_nan=True))
        second=service.analyze(store,dict(study='GSE8917',representation='quantile',genes=100))
        self.assertEqual(first,second);self.assertEqual(len(store.reads),1)
        self.assertEqual(store.objects,{})
        json.dumps(first,allow_nan=False)

    def test_checksum_failure_never_enters_cached_state(self):
        class Bad(ReleaseStore):
            def read(self,*a,**kw):return b'wrong'
        service=Explorer(DATASET)
        with self.assertRaises(ValueError):service.catalog(Bad())
        self.assertIsNone(service.data)

    @needs_release
    def test_new_routes_require_auth_and_validate_before_loading(self):
        store=ReleaseStore();app=create_app(lambda **k:store,lambda h:'owner@example.test');client=app.test_client()
        r=client.post('/api/analysis',json=dict(study='GSE90909',representation='log2',genes=100),
            headers={'X-Defender-Request':'1','Origin':'https://localhost'})
        self.assertEqual(r.status_code,400);self.assertEqual(store.reads,[])
        self.assertEqual(client.post('/api/analysis',json={}).status_code,403)
        self.assertEqual(client.get('/api/catalog').status_code,200)
        for path in ['/api/catalog','/api/analysis','/static/assets/cell-hero.png']:
            denied=create_app(lambda **k:self.fail('Unauthorized cloud call')).test_client()
            self.assertEqual(denied.get(path).status_code,403)


if __name__=='__main__':unittest.main()
