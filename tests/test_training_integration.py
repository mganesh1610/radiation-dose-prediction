import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import http.client
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cloud_io import Conflict, encode, digest
from training_contract import validate_job, validate_rows, PRESETS
from training_worker import Queue, train
from portable_model import estimator_spec, numeric_predict
from core_baseline import fit_estimator


def request(method='elastic_net'):
    return {'schema_version':1,'job_id':'test_1', 'method':method,'parameters':PRESETS[method],
        'dataset':{'object':'training/v1/releases/core_'+'a'*16+'/core.npz','sha256':'a'*64,'generation':'1','bytes':100},
        'holdout_study':'GSE8917','purpose':'engineering_smoke'}


class MemoryStore:
    def __init__(self): self.objects={};self.lock=threading.Lock()
    def read_json(self,name):
        with self.lock:
            if name not in self.objects: raise FileNotFoundError(name)
            return copy.deepcopy(self.objects[name])
    def state(self,name,body,generation):
        with self.lock:
            actual=self.objects.get(name,({},'0'))[1]
            if str(generation)!=actual: raise Conflict('CAS failed')
            new=str(int(actual)+1);self.objects[name]=(copy.deepcopy(body),new);return new


class IntegrationTests(unittest.TestCase):
    def test_submit_idempotent_and_conflicting_job_rejected(self):
        q=Queue(MemoryStore());q.submit(request());q.submit(request())
        other=request('pls')
        with self.assertRaises(ValueError):q.submit(other)

    def test_only_one_worker_claim_and_stale_completion_fenced(self):
        store=MemoryStore();clock=[1000.];q=Queue(store,lambda:clock[0]);q.submit(request())
        state,g=q.claim('test_1')
        with self.assertRaises(Conflict):q.claim('test_1')
        clock[0]+=901
        next_state,next_gen=q.claim('test_1')
        with self.assertRaises(Conflict):q.update(state,g,'completed',{'x':1})
        done,_=q.update(next_state,next_gen,'completed',{'x':2})
        self.assertEqual(done['attempt'],2);self.assertIsNone(q.claim('test_1'))

    def test_cas_race(self):
        s=MemoryStore();q=Queue(s);q.submit(request());a,g=s.read_json('training/v1/jobs/test_1.json')
        s.state('training/v1/jobs/test_1.json',a,g)
        with self.assertRaises(Conflict):s.state('training/v1/jobs/test_1.json',a,g)

    def test_bounded_retries(self):
        clock=[0.];q=Queue(MemoryStore(),lambda:clock[0]);q.submit(request())
        for _ in range(3):q.claim('test_1');clock[0]+=901
        with self.assertRaises(ValueError):q.claim('test_1')

    def test_immutable_upload_recovers_ambiguous_commit_and_bounds_retries(self):
        from training_worker import WorkerCloud
        from cloud_io import Cloud
        cloud=WorkerCloud()
        metadata={'generation':'7','size':'3'}
        with patch.object(cloud,'put',side_effect=[http.client.RemoteDisconnected(),Conflict('already created')]) as put, patch.object(cloud,'metadata',return_value=metadata), patch.object(cloud,'read',return_value=b'abc') as read, patch('training_worker.time.sleep'):
            result=cloud.immutable('training/v1/artifacts/example',b'abc')
            self.assertEqual(result['generation'],'7')
            self.assertEqual(put.call_count,2)
            self.assertEqual(read.call_args.args[2],digest(b'abc'))
        with patch.object(Cloud,'immutable',side_effect=TimeoutError()) as write, patch('training_worker.time.sleep'):
            with self.assertRaises(TimeoutError):cloud.immutable('training/v1/artifacts/example',b'abc')
            self.assertEqual(write.call_count,3)
        with patch.object(Cloud,'immutable',side_effect=ValueError('bad checksum')) as write:
            with self.assertRaises(ValueError):cloud.immutable('training/v1/artifacts/example',b'abc')
            self.assertEqual(write.call_count,1)

    def test_failed_worker_releases_only_its_own_attempt(self):
        from training_worker import run
        from job_queue import job_name
        store=MemoryStore();q=Queue(store);q.submit(request())
        with patch('training_worker.run_claimed',side_effect=TimeoutError()):
            with self.assertRaises(TimeoutError):run(store,'test_1')
        state,_=store.read_json(job_name('test_1'))
        self.assertEqual(state['status'],'RETRYABLE')
        def cancelled(cloud,job_id,claimed):
            Queue(cloud).cancel(job_id)
            raise TimeoutError()
        with patch('training_worker.run_claimed',side_effect=cancelled):
            with self.assertRaises(TimeoutError):run(store,'test_1')
        self.assertEqual(store.read_json(job_name('test_1'))[0]['status'],'CANCELLED')

    def test_unknown_commands_locked_data_and_bad_hash_rejected(self):
        for key,value in [('purpose','final_test'),('method','shell'),('holdout_study','GSE90909'),('job_id','../../bad')]:
            job=request();job[key]=value
            with self.assertRaises(ValueError):validate_job(job)
        job=request();job['dataset']['sha256']='b'*64
        with self.assertRaises(ValueError):validate_job(job)
        job=request();job['parameters']={**job['parameters'],'max_iter':10**12}
        with self.assertRaises(ValueError):validate_job(job)

    def test_schema_duplicate_nan_and_extra_fields_fail(self):
        row={'release_id':'core_1','sample_id':'GSM1','study_id':'GSE1','group_id':'GSE1:A'}
        validate_rows('samples',[row])
        with self.assertRaises(ValueError):validate_rows('samples',[row,row])
        with self.assertRaises(ValueError):validate_rows('samples',[dict(row,extra=1)])
        with self.assertRaises(ValueError):validate_rows('model_metrics',[dict(job_id='a',metric='mae',value=float('nan'),unit='Gy',evaluation_scope='dev')])

    def test_all_numeric_estimators_match_on_unseen_synthetic_data(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(60,8));y=x[:,0]*2+x[:,3]**2
        v=rng.normal(size=(19,8))
        for method in PRESETS:
            model=None if method=='median_null' else fit_estimator(method,PRESETS[method],x,y)
            expected=np.full(len(v),np.median(y)) if model is None else model.predict(v).reshape(-1)
            actual=numeric_predict(estimator_spec(method,model,y),v)
            np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)


if __name__=='__main__':unittest.main()
