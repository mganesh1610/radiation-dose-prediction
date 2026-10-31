"""UI-neutral application operations; all authoritative state is in Google Cloud.

The authenticated web layer can call these operations without importing training
code in its request handler. This module does not host an HTTP service itself.
"""
import json
from cloud_io import digest, encode, PREFIX
from training_contract import METHODS, PRESETS, CORE, validate_job, identifier


class ApplicationService:
    def __init__(self,cloud,queue,release):
        self.cloud=cloud;self.queue=queue;self.release=release

    def methods(self):
        labels={'median_null':'Median predictor','elastic_net':'Elastic Net',
                'pls':'Partial least squares','gradient_boosting':'Gradient boosting'}
        return {'methods':[{'id':m,'label':labels[m],'enabled':True,'purpose':'engineering_smoke',
                            'parameters':PRESETS[m]} for m in METHODS],
                'neural_strategies':{'A':'frozen_research_model_separate_from_engineering_jobs', 'B':'frozen_research_model_separate_from_engineering_jobs',
                                     'C':'frozen_research_model_separate_from_engineering_jobs','D':'frozen_research_model_separate_from_engineering_jobs'},
                'prediction_available':False}

    def create_job(self,request):
        if set(request)!={'job_id','method','holdout_study'}: raise ValueError('Unexpected request fields')
        if request['method'] not in METHODS: raise ValueError('Unsupported method')
        job={**request,'schema_version':1,'purpose':'engineering_smoke','dataset':self.release['dataset'],
             'parameters':PRESETS[request['method']]}
        validate_job(job)
        job_id=self.queue.submit(job)
        return {'job_id':job_id,'worker':'local_mac','status':self.get_job(job_id)['status']}

    def get_job(self,job_id):
        identifier(job_id)
        state,_=self.cloud.read_json(PREFIX+'jobs/'+job_id+'.json')
        if state['request']['dataset']!=self.release['dataset']: raise ValueError('Job belongs to another release')
        return {'job_id':job_id,'status':state['status'],'stage':state.get('stage'),
                'method':state['request']['method'],'attempt':state['attempt'],
                'purpose':state['request']['purpose'],'prediction_available':False}

    def results(self,job_id):
        status=self.get_job(job_id)
        if status['status']!='SUCCEEDED': return {**status,'results':None}
        state,_=self.cloud.read_json(PREFIX+'jobs/'+job_id+'.json');ref=state['result']
        result=json.loads(self.cloud.read(ref['object'],ref['generation'],ref['sha256']))
        if result['request_sha256']!=digest(encode(state['request'])):
            raise ValueError('Result/request mismatch')
        return {**status,'results':result['result'], 'model_manifest':ref,
                'notice':'Engineering integration test; not a tuned comparison or a published prediction model.'}

    def predict(self,*args,**kwargs):
        raise ValueError('No scientifically selected model is published for prediction')
