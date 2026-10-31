import copy
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.server import create_app, verify_identity
from dispatch_control import DispatchControl, BudgetExceeded, month_key
from job_queue import Queue, job_name
from mac_worker_loop import process_entry
from test_training_integration import MemoryStore, request


class WebTests(unittest.TestCase):
    def setUp(self):
        self.store=MemoryStore()
        self.app=create_app(lambda **kwargs:self.store,lambda headers:'owner@example.test')
        self.client=self.app.test_client()
        self.body={'job_id':'web_'+month_key().replace('-','')+'_'+'a'*32,
                   'method':'elastic_net','holdout_study':'GSE8917'}
        self.headers={'X-Defender-Request':'1','Origin':'https://localhost'}

    def post(self, body=None):
        return self.client.post('/api/jobs',json=body or self.body,headers=self.headers)

    def test_auth_denies_missing_forged_or_wrong_identity_before_cloud(self):
        app=create_app(lambda **k: self.fail('Cloud called without authentication'))
        self.assertEqual(app.test_client().get('/api/bootstrap').status_code,403)
        self.assertEqual(app.test_client().get('/api/bootstrap',headers={'X-Goog-Authenticated-User-Email':'owner@example.test'}).status_code,403)
        with patch('google.oauth2.id_token.verify_token',return_value={'iss':'bad','sub':'1','email':'owner@example.test'}):
            with self.assertRaises(PermissionError):verify_identity({'X-Goog-IAP-JWT-Assertion':'fake'})
        with patch('google.oauth2.id_token.verify_token',return_value={'iss':'https://cloud.google.com/iap','sub':'1','email':'intruder@example.test'}):
            with self.assertRaises(PermissionError):verify_identity({'X-Goog-IAP-JWT-Assertion':'fake'})

    def test_csrf_content_and_size_guards(self):
        self.assertEqual(self.client.post('/api/jobs',json=self.body).status_code,403)
        self.assertEqual(self.client.post('/api/jobs',json=self.body,headers={**self.headers,'Origin':'https://evil.test'}).status_code,403)
        self.assertEqual(self.client.post('/api/jobs',data='x'*5000,content_type='application/json',headers=self.headers).status_code,413)

    def test_duplicate_submit_once_and_conflicting_request_rejected(self):
        self.assertEqual(self.post().status_code,202)
        self.assertEqual(self.post().status_code,202)
        self.assertEqual(len(DispatchControl(self.store).read()[0]['jobs']),1)
        self.assertEqual(self.post(dict(self.body,method='pls')).status_code,400)
        self.assertEqual(self.store.read_json(job_name(self.body['job_id']))[0]['attempt'],0)

    def test_owned_jobs_and_locked_test(self):
        self.assertEqual(self.post(dict(self.body,holdout_study='GSE90909')).status_code,400)
        self.assertEqual(self.post(dict(self.body,method=[])).status_code,400)
        self.assertEqual(self.post(dict(self.body,holdout_study={})).status_code,400)
        self.post()
        other=create_app(lambda **k:self.store,lambda h:'someone@example.test').test_client()
        self.assertEqual(other.get('/api/jobs/'+self.body['job_id']).status_code,404)
        self.assertEqual(self.client.post('/api/predict',json={},headers=self.headers).status_code,400)

    def test_cancellation_fences_running_publish_and_worker(self):
        self.post();q=Queue(self.store);state,g=q.claim(self.body['job_id'])
        result=self.client.post('/api/jobs/'+self.body['job_id']+'/cancel',json={},headers=self.headers)
        self.assertEqual(result.json['status'],'CANCELLED')
        from cloud_io import Conflict
        with self.assertRaises(Conflict):q.update(state,g,'completed',{'example':True})
        entry=DispatchControl(self.store).read()[0]['jobs'][self.body['job_id']]
        self.assertEqual(process_entry(self.store,entry,lambda _:self.fail('Cancelled job ran')),'CANCELLED')

    def test_admitted_job_repaired_after_api_crash(self):
        from app.server import DATASET
        from training_contract import PRESETS
        job=dict(self.body,schema_version=1,purpose='engineering_smoke',dataset=DATASET,parameters=PRESETS['elastic_net'])
        control=DispatchControl(self.store);control.admit(job,'owner@example.test')
        seen=[];entry=control.read()[0]['jobs'][self.body['job_id']]
        process_entry(self.store,entry,seen.append)
        self.assertEqual(seen,[self.body['job_id']])
        self.assertEqual(self.store.read_json(job_name(self.body['job_id']))[0]['status'],'QUEUED')

    def test_expired_exhausted_worker_job_becomes_terminal(self):
        self.post()
        entry=DispatchControl(self.store).read()[0]['jobs'][self.body['job_id']]
        state,generation=self.store.read_json(job_name(self.body['job_id']))
        state.update(status='RUNNING',attempt=3,lease={'id':'old','expires_at':0})
        self.store.state(job_name(self.body['job_id']),state,generation)
        self.assertEqual(process_entry(self.store,entry,lambda _:self.fail('Exhausted job ran')),'FAILED')
        self.assertEqual(self.store.read_json(job_name(self.body['job_id']))[0]['status'],'FAILED')

    def test_budget_survives_new_api_instances(self):
        control=DispatchControl(self.store)
        control.mutate(lambda s:s.update(requests=600))
        self.assertEqual(self.client.get('/api/bootstrap').status_code,429)
        with self.assertRaises(BudgetExceeded):DispatchControl(self.store).reserve_request()
        control.mutate(lambda s:s.update(requests=0,jobs={str(n):{} for n in range(12)}))
        self.assertEqual(self.post().status_code,429)

    @patch.dict('os.environ', {'DEFENDER_MONTHLY_API_LIMIT':'603'})
    def test_finite_extension_preserves_usage_and_reports_applied_limit(self):
        DispatchControl(self.store).mutate(lambda s:s.update(requests=600))
        first=self.client.get('/api/bootstrap')
        self.assertEqual(first.json['requests_used'],601)
        self.assertEqual(first.json['limits']['api_requests'],603)
        self.assertEqual(first.json['access']['request_limit'],603)
        self.client.get('/api/bootstrap')
        self.assertEqual(self.client.get('/api/bootstrap').status_code,200)
        self.assertEqual(self.client.get('/api/bootstrap').status_code,429)
        self.assertEqual(DispatchControl(self.store).read()[0]['requests'],603)

    def test_invalid_allowance_fails_before_mutating_counters(self):
        for value in ['0','-1','5001','unlimited','3.5','', '６００']:
            with patch.dict('os.environ', {'DEFENDER_MONTHLY_API_LIMIT':value}):
                with self.assertRaises(ValueError):DispatchControl(self.store)
        self.assertEqual(DispatchControl(self.store).read()[0]['requests'],0)

    @patch.dict('os.environ', {'REVIEWER_EMAILS':'owner@example.test,second@example.test'})
    def test_reviewer_admission_is_per_account_and_idempotent(self):
        first=self.body.copy()
        self.assertEqual(self.post().status_code,202)
        self.body['job_id']=self.body['job_id'][:-32]+'b'*32
        self.assertEqual(self.post().status_code,202)
        self.assertEqual(self.post().status_code,202) # same admission survives the cap
        self.client.post('/api/jobs/'+first['job_id']+'/cancel',json={},headers=self.headers)
        self.body['job_id']=self.body['job_id'][:-32]+'c'*32
        self.assertEqual(self.post().status_code,429) # cancelling does not replenish allowance
        other=create_app(lambda **k:self.store,lambda h:'second@example.test').test_client()
        self.assertEqual(other.post('/api/jobs',json=self.body,headers=self.headers).status_code,202)
        bootstrap=self.client.get('/api/bootstrap').json
        self.assertEqual(bootstrap['access']['role'],'reviewer')
        self.assertEqual(bootstrap['access']['job_limit'],2)
        self.assertEqual(len(bootstrap['jobs']),2)
        self.assertEqual(bootstrap['jobs_used'],3)

    @patch.dict('os.environ', {'REVIEWER_EMAILS':'owner@example.test'})
    def test_reviewer_request_cap_persists_without_blocking_another_account(self):
        control=DispatchControl(self.store)
        control.mutate(lambda s:s.update(requests=150,requests_by_owner={'owner@example.test':150}))
        self.assertEqual(self.client.get('/api/bootstrap').status_code,429)
        other=create_app(lambda **k:self.store,lambda h:'other@example.test').test_client()
        self.assertEqual(other.get('/api/bootstrap').status_code,200)
        self.assertEqual(control.read()[0]['requests'],151)

    def test_training_pause_preserves_results_and_cancellation(self):
        self.assertEqual(self.post().status_code,202)
        with patch.dict('os.environ',{'TRAINING_ENABLED':'false'}):
            self.assertEqual(self.post().status_code,403)
            self.assertFalse(self.client.get('/api/bootstrap').json['training_enabled'])
            self.assertEqual(self.client.get('/api/jobs/'+self.body['job_id']).status_code,200)
            self.assertEqual(self.client.post('/api/jobs/'+self.body['job_id']+'/cancel',json={},headers=self.headers).status_code,200)

    def test_verified_reviewer_identity_is_allowlisted(self):
        with patch.dict('os.environ',{'ALLOWED_EMAILS':'researcher@example.test,collaborator@example.test'}), patch('app.server.AUDIENCE','/projects/123/locations/us-west1/services/example'):
            for email in ('researcher@example.test','collaborator@example.test'):
                with patch('google.oauth2.id_token.verify_token',return_value={'iss':'https://cloud.google.com/iap','sub':'1','email':email}):
                    self.assertEqual(verify_identity({'X-Goog-IAP-JWT-Assertion':'fake'}),email)

    def test_demo_polling_expires_and_local_timer_skips_network_poll(self):
        from mac_worker_loop import polling_policy
        now=10000
        self.assertEqual(polling_policy({'last_poll':now-65},now,{}),(600,False,False))
        self.assertEqual(polling_policy({'last_poll':now-65},now,{'expires_at':now+3600}),(60,True,True))
        self.assertEqual(polling_policy({'last_poll':now-65},now,{'expires_at':now}),(600,False,False))
        self.assertEqual(polling_policy({'last_poll':now-601},now,{'expires_at':now+7201}),(600,False,True))
        for malformed in ([],None,{'expires_at':'later'}):
            self.assertEqual(polling_policy({'last_poll':now-601},now,malformed),(600,False,True))

    def test_security_headers_and_no_training_imports_in_server_bundle(self):
        r=self.client.get('/')
        self.assertIn("frame-ancestors 'none'",r.headers['Content-Security-Policy'])
        self.assertEqual(r.headers['Cache-Control'],'no-store')
        r.close()
        import subprocess
        result=subprocess.run([sys.executable,'-c',"import app.server,sys; assert 'numpy' not in sys.modules; assert 'training_worker' not in sys.modules"],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
