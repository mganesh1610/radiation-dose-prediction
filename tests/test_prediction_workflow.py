from artifact_support import require_artifacts
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
from frozen_inference import FrozenInference,ranks
from prediction_service import PredictionService,BASE
from test_training_integration import MemoryStore as BaseMemoryStore
from cloud_io import Conflict
class MemoryStore(BaseMemoryStore):
 def put(self,name,content,generation=0,content_type="application/octet-stream"):
  return self.state(name,json.loads(content),generation)
 def immutable(self,name,content):
  value=json.loads(content)
  try:self.state(name,value,0)
  except Conflict:
   if self.read_json(name)[0]!=value:raise

from app.server import create_app
from scipy.stats import rankdata,ttest_1samp
from build_statistical_explorer import bh,paired

class WorkflowTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  require_artifacts(ROOT/'data/cloud_prediction/frozen-v1/manifest.json')
  cls.engine=FrozenInference(ROOT/'data/cloud_prediction/frozen-v1');cls.example=json.loads((cls.engine.folder/cls.engine.manifest['examples'][0]['file']).read_text())
 def setUp(self):
  self.store=MemoryStore();self.s=PredictionService(self.store,self.engine,'reviewer@example.test',True)
 def test_curated_upload_retries_exact_and_prediction_persists(self):
  r=self.s.upload(self.example);self.assertEqual(r['status'],'approved');self.assertTrue(r['training_overlap'])
  self.assertEqual(self.s.upload(self.example),r);a=self.s.predict(r['upload_id'],'A');self.assertEqual(a,self.s.predict(r['upload_id'],'A'));self.assertIsNone(a['individual_prediction_interval'])
 def test_new_input_cannot_self_approve_or_borrow_curated_identity(self):
  x=copy.deepcopy(self.example);x['values'][0]*=1.01;r=self.s.upload(x);self.assertEqual(r['status'],'pending_compatibility_review');self.assertIsNone(self.s.predict(r['upload_id'],'A')['estimate_gy'])
  x['approved']=True
  with self.assertRaises(ValueError):self.s.upload(x)
 def test_owner_isolation_and_reviewer_cannot_review(self):
  r=self.s.upload(self.example);other=PredictionService(self.store,self.engine,'other@example.test',True)
  with self.assertRaises(FileNotFoundError):other.predict(r['upload_id'],'A')
  with self.assertRaises(PermissionError):self.s.review(r['upload_id'],{})
 def test_researcher_can_revoke_cached_prediction(self):
  r=self.s.upload(self.example);self.s.predict(r['upload_id'],'A');owner=PredictionService(self.store,self.engine,'owner@example.test',False)
  body=dict(decision='rejected',evidence='Test revocation: compatibility decision withdrawn by researcher.',training_overlap='yes',checks={k:False for k in ['assay','gene_mapping','normalization','technical_flags','tissue_exposure_time','missingness']})
  owner.review(r['upload_id'],body);self.assertIsNone(self.s.predict(r['upload_id'],'A')['estimate_gy'])
 def test_unsupported_inputs_no_dose_and_cannot_be_approved(self):
  x=copy.deepcopy(self.example);x['metadata']['normalization']='log2';r=self.s.upload(x);self.assertEqual(r['status'],'incompatible');self.assertIsNone(self.s.predict(r['upload_id'],'A')['estimate_gy'])
  owner=PredictionService(self.store,self.engine,'owner@example.test',False)
  body=dict(decision='approved',evidence='Attempt to approve invalid units despite a documented blocker.',training_overlap='unknown',checks={k:True for k in ['assay','gene_mapping','normalization','technical_flags','tissue_exposure_time','missingness']})
  with self.assertRaises(ValueError):owner.review(r['upload_id'],body)
 def test_nan_infinity_duplicate_probes_and_string_values(self):
  for v in [float('inf'),float('nan'),'123',True,{}]:
   x=copy.deepcopy(self.example);x['values'][0]=v
   with self.assertRaises((ValueError,TypeError)):self.s.upload(x)
  x=copy.deepcopy(self.example);x['probe_ids'][0]=x['probe_ids'][1]
  with self.assertRaises(ValueError):self.s.upload(x)
 def test_rank_ties_match_scipy(self):
  x=np.array([[2.,1.],[1.,1.],[2.,np.nan],[4.,5.]])
  for j in range(2):
   take=np.isfinite(x[:,j]);np.testing.assert_allclose(ranks(x)[take,j],(rankdata(x[take,j])-.5)/take.sum())
 def test_lifetime_upload_bound(self):
  self.store.state(BASE+'registry.json',{'schema_version':1,'uploads':{str(i):{'owner':'x'} for i in range(40)}},0)
  from dispatch_control import BudgetExceeded
  with self.assertRaises(BudgetExceeded):self.s.upload(self.example)
 def test_http_csrf_auth_and_upload_to_prediction(self):
  with patch.dict('os.environ',{'REVIEWER_EMAILS':'reviewer@example.test'}):
   c=create_app(lambda **k:self.store,lambda h:'reviewer@example.test',self.engine).test_client();headers={'X-Defender-Request':'1','Origin':'https://localhost'}
   self.assertEqual(c.post('/api/uploads',json=self.example).status_code,403)
   r=c.post('/api/uploads',json=self.example,headers=headers);self.assertEqual(r.status_code,201)
   result=c.post('/api/predict',json={'upload_id':r.json['upload_id'],'model':'A'},headers=headers);self.assertEqual(result.status_code,200);self.assertEqual(result.json['status'],'completed')
   self.assertEqual(c.get('/api/prediction-runtime-check').status_code,403)
 def test_statistics_against_reference_and_zero_variance(self):
  x=np.array([[1,2,4,2,5],[0,0,0,0,0],[1,np.nan,2,4,5]],float);effect,n,p,q=paired(x)
  self.assertAlmostEqual(p[0],ttest_1samp(x[0],0).pvalue,14);self.assertTrue(np.isnan(p[1]));self.assertEqual(n[2],4)
  np.testing.assert_allclose(bh([.01,.04,.03]),[.03,.04,.04]);self.assertTrue(np.isnan(bh([np.nan])[0]))
if __name__=='__main__':unittest.main()
