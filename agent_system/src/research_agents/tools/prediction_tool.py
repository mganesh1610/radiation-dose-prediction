"""Read owned uploads, reuse frozen inference; no upload, review or training tools."""
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import threading

from ..persistence.control_store import Conflict
from .query_sql import digest

BASE='training/v1/control/inference/'
METHODS=('A','B','C','D','elastic_net','pls','gradient_boosting','median_null')


class PredictionStorage:
    """Read-only legacy namespace; result writes stay in the private copilot store."""
    def __init__(self,source,store):self.source,self.store=source,store

    def cache_key(self,name):
        match=re.fullmatch(re.escape(BASE)+r'uploads/([a-f0-9]{64})/result-([a-f0-9]{64})-(A|B|C|D|elastic_net|pls|gradient_boosting|median_null)\.json',name)
        if not match:raise ValueError('Unsupported prediction result key')
        return f'prediction-results/{match[1]}/{match[2]}-{match[3].lower()}.json'

    def read_json(self,name):
        if '/result-' in name:
            row,generation=self.store.read(self.cache_key(name))
            if row is None:raise FileNotFoundError()
            if digest(row['result'])!=row['sha256']:raise RuntimeError('Prediction cache integrity failure')
            return row['result'],generation
        if not (name==BASE+'registry.json' or re.fullmatch(re.escape(BASE)+r'uploads/[a-f0-9]{64}/(?:input|record|decision-[a-f0-9]{64})\.json',name)):
            raise PermissionError('Prediction source path is not admitted')
        return self.source.read_json(name)

    def immutable(self,name,payload):
        if len(payload)>30000:raise ValueError('Prediction result too large')
        result=json.loads(payload);row={'result':result,'sha256':digest(result)};key=self.cache_key(name)
        try:self.store.write(key,row,'0')
        except Conflict:
            previous,_=self.store.read(key)
            if previous!=row:raise RuntimeError('Immutable prediction collision')


class PredictionTool:
    def __init__(self,engine,service_factory,guard,owner_lookup,*,code_hash):
        self.engine,self.service_factory,self.guard,self.owner_lookup=engine,service_factory,guard,owner_lookup
        self.code_hash=code_hash
        self.slots=threading.BoundedSemaphore(2)

    async def run(self,upload_id,model=None):
        scope=await self.guard.authorize()
        await asyncio.to_thread(self.guard.ledger.reserve,scope.principal.user_id,scope.run_id,scope.fence,'tool')
        contract=scope.contract
        if (not re.fullmatch(r'[a-f0-9]{64}',upload_id or '') or contract.get('upload_ids')!=[upload_id]):
            return {'status':'clarification_required','reason':'Select one of your uploaded profiles using Add expression profile, or provide its exact upload ID in this question.'}
        if model is not None and (model not in METHODS or not contract.get('prediction_requested') or contract.get('prediction_models')!=[model]):
            return {'status':'clarification_required','reason':'Explicitly request prediction for one upload and one frozen model (A, B, C, D, elastic_net, pls, gradient_boosting or median_null).'}
        if model is None and contract.get('prediction_requested'):
            return {'status':'clarification_required','reason':'Select a specific frozen model before requesting a prediction. No model was chosen for you.'}
        owner=await asyncio.to_thread(self.owner_lookup,scope.principal)
        if not isinstance(owner,str) or not owner:raise PermissionError('No verified upload-owner binding')
        def execute():
            if not self.slots.acquire(blocking=False):raise RuntimeError('Prediction concurrency limit')
            try:
                service=self.service_factory(owner)
                try:
                    record=service.get(upload_id)  # always strict owner access
                    result=service.predict(upload_id,model) if model is not None else None
                except FileNotFoundError:
                    return None,None
                # Current access and approval are rechecked before receipt publication.
                current=service.get(upload_id)
                if current!=record:raise RuntimeError('Upload or compatibility changed during request')
                return record,result
            finally:self.slots.release()
        record,result=await asyncio.to_thread(execute)
        await self.guard.authorize()
        if await asyncio.to_thread(self.owner_lookup,scope.principal)!=owner:raise PermissionError('Owner binding changed')
        if record is None:
            return {'status':'not_found','reason':'No accessible upload with this ID. Select a profile from your own uploads.'}
        if result is None or result.get('status')!='completed':
            answer={'status':record['status'],'upload_id':upload_id,'estimate_gy':None,'qc':record['qc'],
                    'curated':record['curated'],'training_overlap':record['training_overlap'],
                    'release_sha256':self.engine.release_sha,
                    'explanation':('Compatibility approved; explicitly choose a model to run a research estimate.' if record['status']=='approved' else
                        'No dose estimate. A researcher must resolve compatibility in the Predictions workspace; the agent cannot approve inputs.')}
        else:
            answer={k:result[k] for k in ('status','upload_id','estimate_gy','model','seed','model_sha256','release',
                'release_sha256','input_sha256','qc','training_overlap',
                'individual_prediction_interval','limitations')}
            answer['dose_kind']='model_estimate_not_recorded_dose'
            answer['estimate_outside_training_dose_range']=result['outside_training_dose_range']
            answer['range_interpretation']='Only the numeric estimate is compared with 0–8 Gy. This is not evidence that the input profile is outside the training distribution; no recorded input dose is inferred.'
        # User-supplied sample names, provenance prose and review identities stay
        # in the owned receipt; do not place raw uploads or metadata in LLM context.
        verified={'answer':answer,'sample_id':record['sample_id'],'input_provenance':record['metadata'],
                  'compatibility':record.get('compatibility'),'input_sha256':record['input_sha256']}
        receipt={'tool':'prediction','verified_result':verified,'release_manifest_sha256':self.engine.release_sha,
                 'procedure_version':'owned-frozen-prediction-v1','code_sha256':self.code_hash,
                 'result_sha256':digest(verified)}
        receipt['evidence_id']='ev-'+digest(receipt)[:24]
        await asyncio.to_thread(self.guard.ledger.progress,scope.principal.user_id,scope.run_id,scope.fence,'verified_prediction',receipt)
        return {'evidence_id':receipt['evidence_id'],'answer':answer}


def create_prediction_tool(settings,guard,memberships,store):
    """Load a pinned copy of the existing numeric runtime; no executable upload input."""
    directory=Path(settings['prediction_code_dir']).resolve()
    expected=settings['prediction_source_hashes']
    required={'prediction_service.py','frozen_inference.py','core_transfer.py','portable_model.py',
              'masked_quantile.py','assay_qc.py','cloud_io.py','dispatch_control.py','training_contract.py'}
    if set(expected)!=required:raise ValueError('Incomplete frozen runtime source manifest')
    for name,sha in expected.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=sha:raise ValueError('Prediction code integrity failure')
    sys.path.insert(0,str(directory))
    runtime=importlib.import_module('frozen_inference');services=importlib.import_module('prediction_service')
    cloud=importlib.import_module('cloud_io')
    for module in (runtime,services,cloud):
        if Path(module.__file__).resolve().parent!=directory:raise ValueError('Unexpected prediction module origin')
    engine=runtime.FrozenInference(settings['prediction_model_dir'])
    if engine.release_sha!=settings['prediction_release_sha256']:raise ValueError('Frozen prediction release mismatch')
    def owner(principal):
        row,_=store.read('identities/'+principal.user_id+'.json')
        if (not row or row.get('user_id')!=principal.user_id or row.get('email') not in memberships.policy.emails()):
            raise PermissionError('Verified upload owner unavailable')
        return row['email']
    def service(email):
        # Reviewer=True deliberately removes researcher cross-owner review powers.
        return services.PredictionService(PredictionStorage(cloud.Cloud(max_requests=42,max_upload_bytes=0),store),engine,email,True)
    return PredictionTool(engine,service,guard,owner,code_hash=digest(expected))
