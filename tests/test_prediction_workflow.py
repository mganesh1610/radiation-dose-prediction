from artifact_support import require_artifacts
import copy,json,sys,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
from frozen_inference import WBC_BOOSTING, WBC_FILE, WBC_SHA256, HELD_OUT_FOLDER, FrozenInference, digest, encode, ranks
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
 def test_wbc_picker_uses_separate_approved_artifact_and_cache(self):
  require_artifacts(self.engine.folder.parent/WBC_FILE,self.engine.folder.parent/HELD_OUT_FOLDER/'manifest.json')
  client=create_app(lambda **kw:self.store,lambda h:self.s.owner,self.engine).test_client()
  catalog=client.get('/api/prediction-catalog').get_json()
  legacy=client.get('/api/prediction-chat-catalog').get_json()['methods']
  self.assertEqual(catalog['methods'],[WBC_BOOSTING if m=='gradient_boosting' else m for m in legacy])
  self.assertEqual(catalog['model_release_overrides'],{WBC_BOOSTING:WBC_SHA256})
  body=json.loads(self.engine.example_bytes('GSM2417396'))
  uid=self.s.upload(body)['upload_id']
  original=self.s.predict(uid,'gradient_boosting')
  self.assertAlmostEqual(original['estimate_gy'],.09576921097496305,places=11)
  self.assertIsNone(self.s.predict(uid,WBC_BOOSTING)['estimate_gy'])
  h={'Origin':'https://localhost','X-Defender-Request':'1'}
  checked=client.post('/api/uploads/'+uid+'/check?model='+WBC_BOOSTING,json={},headers=h)
  self.assertEqual(checked.status_code,200)
  self.assertEqual(checked.json['compatibility']['release_sha256'],WBC_SHA256)
  result=client.post('/api/predict',json={'model':WBC_BOOSTING,'upload_id':uid},headers=h).get_json()
  self.assertEqual((result['model'],result['release_sha256'],result['release']),(WBC_BOOSTING,WBC_SHA256,'frozen-gb-wbc-v1'))
  self.assertNotEqual(result['model_sha256'],original['model_sha256'])
  self.assertAlmostEqual(result['estimate_gy'],.37188545512437177,places=11)
  self.assertEqual(self.s.predict(uid,WBC_BOOSTING),result)
  self.assertEqual(self.s.predict(uid,'gradient_boosting'),original)
  self.assertFalse(result['training_overlap'])
  self.assertEqual(result['dose_comparison']['observed_dose_gy'],.1)

 def test_tampered_wbc_supplement_is_rejected(self):
  require_artifacts(self.engine.folder.parent/WBC_FILE)
  with TemporaryDirectory() as temp:
   parent=Path(temp);base=parent/'frozen';base.symlink_to(self.engine.folder,target_is_directory=True)
   with self.assertRaises(RuntimeError): FrozenInference(base).catalog(prediction_picker=True)
   (parent/WBC_FILE).write_bytes(b'{}')
   with self.assertRaisesRegex(ValueError,'WBC model integrity'): FrozenInference(base)

 def test_previous_manual_review_does_not_approve_a_new_model_release(self):
  require_artifacts(self.engine.folder.parent/WBC_FILE)
  body=copy.deepcopy(self.example);body['values'][0]*=1.01
  uid=self.s.upload(body)['upload_id'];owner=PredictionService(self.store,self.engine,self.s.owner,False)
  decision={'decision':'approved','training_overlap':'unknown','evidence':'Audited assay, gene mapping, masking and units match the declared input release.',
            'checks':dict.fromkeys(('assay','gene_mapping','normalization','technical_flags','tissue_exposure_time','missingness'),True)}
  owner.review(uid,decision)
  self.assertEqual(owner.check_stored(uid,model=WBC_BOOSTING)['status'],'pending_compatibility_review')
  self.assertIsNone(owner.predict(uid,WBC_BOOSTING)['estimate_gy'])
  owner.review(uid,decision,model=WBC_BOOSTING)
  self.assertEqual(owner.predict(uid,WBC_BOOSTING)['status'],'completed')
  owner.review(uid,{**decision,'decision':'rejected'},model=WBC_BOOSTING)
  self.assertIsNone(owner.predict(uid,WBC_BOOSTING)['estimate_gy'])
  self.assertEqual(owner.get(uid)['status'],'approved')

 def test_new_input_cannot_self_approve_or_borrow_curated_identity(self):
  x=copy.deepcopy(self.example);x['values'][0]*=1.01;r=self.s.upload(x);self.assertEqual(r['status'],'pending_compatibility_review');self.assertIsNone(self.s.predict(r['upload_id'],'A')['estimate_gy'])
  x['approved']=True
  with self.assertRaises(ValueError):self.s.upload(x)
 def test_recorded_dose_is_returned_for_fresh_and_cached_estimates(self):
  example = next(e for e in self.engine.manifest['examples'] if e['id'] == 'GSM226009')
  body = json.loads((self.engine.folder / example['file']).read_text())
  uid = self.s.upload(body)['upload_id']
  result = self.s.predict(uid, 'gradient_boosting')
  self.assertEqual(result['dose_comparison'], {
      'study': 'GSE8917', 'source': 'frozen_release_example', 'observed_dose_gy': 0,
      'estimate_minus_observed_gy': result['estimate_gy']})
  with patch.object(self.engine, 'predict', side_effect=AssertionError('Unexpected refit or inference')):
   self.assertEqual(self.s.predict(uid, 'gradient_boosting'), result)
 def test_approved_upload_with_same_accession_cannot_borrow_reference_dose(self):
  body = copy.deepcopy(self.example)
  body['values'][0] *= 1.01
  uid = self.s.upload(body)['upload_id']
  reviewer = PredictionService(self.store, self.engine, 'researcher@example.test', False)
  reviewer.review(uid, dict(decision='approved', training_overlap='unknown',
      evidence='Test review of a compatible profile with different measurements.',
      checks=dict.fromkeys(('assay','gene_mapping','normalization','technical_flags','tissue_exposure_time','missingness'), True)))
  result = self.s.predict(uid, 'gradient_boosting')
  self.assertEqual(result['status'], 'completed')
  self.assertIsNone(result['dose_comparison'])
 def test_held_out_picker_downloads_and_frozen_prediction(self):
  require_artifacts(self.engine.folder.parent / HELD_OUT_FOLDER / 'manifest.json')
  client = create_app(lambda **kwargs: self.store, lambda headers: self.s.owner, self.engine).test_client()
  catalog = client.get('/api/prediction-catalog').get_json()
  examples = [row for row in catalog['examples'] if row['training_overlap'] is False]
  self.assertEqual([row['observed_dose_gy'] for row in examples], [0, .1, .3, .5, 1, 2, 4])
  self.assertEqual(catalog['release_sha256'], digest(encode(self.engine.manifest)))
  for row in examples:
   response = client.get('/api/prediction-examples/' + row['id'])
   self.assertEqual(response.status_code, 200)
   self.assertEqual(digest(response.data), row['sha256'])
   self.assertEqual(response.json['sample_id'], row['id'])
   self.assertNotIn('observed_dose_gy', response.json['metadata'])
  self.assertEqual(client.get('/api/prediction-examples/unlisted').status_code, 404)
  body = json.loads(self.engine.example_bytes(examples[-1]['id']))
  r = self.s.upload(body)
  self.assertEqual(r['status'], 'approved')
  self.assertIs(r['training_overlap'], False)
  result = self.s.predict(r['upload_id'], 'gradient_boosting')
  self.assertEqual(result['dose_comparison']['source'], 'held_out_evaluation_example')
  self.assertEqual(result['dose_comparison']['observed_dose_gy'], 4)
  self.assertEqual(result['dose_comparison']['estimate_minus_observed_gy'], result['estimate_gy'] - 4)
  self.assertIs(result['training_overlap'], False)
  self.assertEqual(self.s.upload(body), r)
  self.assertEqual(self.s.predict(r['upload_id'], 'gradient_boosting'), result)
  body['values'][next(i for i,v in enumerate(body['values']) if v is not None)] *= 1.01
  altered = self.s.upload(body)
  self.assertFalse(altered['curated'])
  self.assertEqual(altered['status'], 'pending_compatibility_review')
  self.assertEqual(altered['training_overlap'], 'unknown')
 def test_new_example_catalog_preserves_existing_upload_and_review(self):
  require_artifacts(self.engine.folder.parent / HELD_OUT_FOLDER / 'manifest.json')
  candidate = next(r for r in self.engine.catalog()['examples'] if r['training_overlap'] is False)
  body = json.loads(self.engine.example_bytes(candidate['id']))
  with patch.dict(self.engine.curated):
   self.engine.curated.pop(candidate['sha256'])
   old = self.s.upload(body)
   self.assertEqual(old['status'], 'pending_compatibility_review')
  current = self.s.upload(body)
  self.assertEqual(current['upload_id'], old['upload_id'])
  self.assertTrue(current['curated'])
  self.assertIs(current['training_overlap'], False)
  self.assertEqual(current['status'], 'approved')
  reviewer = PredictionService(self.store, self.engine, 'researcher@example.test', False)
  reviewer.review(old['upload_id'], dict(decision='rejected', training_overlap='no',
      evidence='Test researcher decision for an existing compatibility record.',
      checks=dict.fromkeys(('assay','gene_mapping','normalization','technical_flags','tissue_exposure_time','missingness'), False)))
  self.assertEqual(self.s.upload(body)['status'], 'rejected')
 def test_base_release_and_untrusted_example_catalog(self):
  with TemporaryDirectory() as folder:
   frozen = Path(folder) / 'frozen'
   frozen.symlink_to(self.engine.folder.resolve(), target_is_directory=True)
   original = FrozenInference(frozen)
   self.assertEqual(original.release_sha, self.engine.release_sha)
   self.assertEqual(original.catalog()['examples'], original.manifest['examples'])
   extra = Path(folder) / HELD_OUT_FOLDER
   extra.mkdir()
   (extra / 'manifest.json').write_bytes(encode({'schema_version':1, 'release_sha256':original.release_sha, 'examples':[]}))
   with self.assertRaisesRegex(ValueError, 'manifest integrity'):
    FrozenInference(frozen)
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
