"""Atomic cloud job state; no numerical training dependencies."""
import json
import time
import uuid
from cloud_io import Conflict, PREFIX, digest, encode
from training_contract import identifier, validate_job

def job_name(job_id):
    return PREFIX+'jobs/'+identifier(job_id)+'.json'


class Queue:
    def __init__(self,cloud,clock=time.time): self.cloud=cloud;self.clock=clock

    def submit(self,job):
        validate_job(job)
        state={'request':job,'request_sha256':digest(encode(job)), 'status':'QUEUED',
               'attempt':0,'created_at':self.clock(),'lease':None,'result':None}
        try: self.cloud.state(job_name(job['job_id']),state,0)
        except Conflict:
            old,_=self.cloud.read_json(job_name(job['job_id']))
            if old['request_sha256']!=state['request_sha256']: raise ValueError('Job ID already used for different request')
        return job['job_id']

    def claim(self,job_id):
        state,generation=self.cloud.read_json(job_name(job_id));validate_job(state['request'])
        if state['request']['job_id']!=job_id: raise ValueError('Job identity mismatch')
        if digest(encode(state['request']))!=state['request_sha256']: raise ValueError('Request integrity failed')
        if state['status']=='SUCCEEDED': return None
        if state['status']=='RUNNING' and state['lease']['expires_at']>self.clock(): raise Conflict('Worker already owns job')
        if state['status'] not in ('QUEUED','RUNNING','RETRYABLE'): raise ValueError('Job is terminal')
        if state['attempt']>=3: raise ValueError('Attempt limit reached; explicit review required')
        state['attempt']+=1;state['status']='RUNNING'
        state['lease']={'id':str(uuid.uuid4()),'expires_at':self.clock()+900}
        state['stage']='download'
        generation=self.cloud.state(job_name(job_id),state,generation)
        return state,generation

    def update(self,state,generation,stage,result=None):
        if state['lease']['expires_at']<=self.clock(): raise Conflict('Lease expired; completion fenced')
        new=json.loads(json.dumps(state));new['stage']=stage
        new['lease']['expires_at']=self.clock()+900
        if result is not None: new.update(status='SUCCEEDED',result=result,completed_at=self.clock())
        generation=self.cloud.state(job_name(state['request']['job_id']),new,generation)
        return new,generation


    def cancel(self, job_id):
        state, generation = self.cloud.read_json(job_name(job_id))
        if state['status'] in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
            return state
        state.update(status='CANCELLED', stage='cancelled', cancelled_at=self.clock())
        self.cloud.state(job_name(job_id), state, generation)
        return state

    def fail(self, job_id, lease_id, message):
        state, generation = self.cloud.read_json(job_name(job_id))
        if state['status'] != 'RUNNING' or state.get('lease', {}).get('id') != lease_id:
            return
        state.update(status='FAILED' if state['attempt'] >= 3 else 'RETRYABLE',
                     stage='failed', error=message[:200])
        self.cloud.state(job_name(job_id), state, generation)
