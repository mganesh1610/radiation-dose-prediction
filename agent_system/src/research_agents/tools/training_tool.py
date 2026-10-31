"""Read-only bridge evidence. Approval, submission and cancellation remain in UI."""
import asyncio
import hashlib
import importlib
from pathlib import Path
import re
from .query_sql import digest


class TrainingTool:
    def __init__(self,bridge_factory,guard,owner_lookup):
        self.bridge_factory,self.guard,self.owner_lookup=bridge_factory,guard,owner_lookup

    async def run(self,job_id):
        scope=await self.guard.authorize()
        await asyncio.to_thread(self.guard.ledger.reserve,scope.principal.user_id,scope.run_id,scope.fence,'tool')
        if scope.contract.get('training_job_ids')!=[job_id]:
            return {'status':'clarification_required','reason':'Select a job in Training or give its exact web_ job ID in this question.'}
        owner=await asyncio.to_thread(self.owner_lookup,scope.principal)
        try:answer=await asyncio.to_thread(self.bridge_factory(owner).status,job_id)
        except FileNotFoundError:return {'status':'not_found','reason':'No accessible training job with this ID.'}
        await self.guard.authorize()
        if await asyncio.to_thread(self.owner_lookup,scope.principal)!=owner:raise PermissionError('Owner binding changed')
        receipt={'tool':'training_status','verified_result':answer,'result_sha256':digest(answer),
                 'procedure_version':'approved-engineering-v1'}
        receipt['evidence_id']='ev-'+digest(receipt)[:24]
        await asyncio.to_thread(self.guard.ledger.progress,scope.principal.user_id,scope.run_id,scope.fence,'verified_training_status',receipt)
        return {'evidence_id':receipt['evidence_id'],'answer':answer}


def create_training_tool(settings,guard,memberships,store):
    directory=Path(settings['prediction_code_dir']).resolve()
    hashes=settings['training_bridge_source_hashes']
    if set(hashes)!={'training_bridge.py','job_queue.py'}:raise ValueError('Incomplete bridge runtime manifest')
    for name,sha in hashes.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=sha:raise ValueError('Bridge code integrity failure')
    module=importlib.import_module('training_bridge');cloud=importlib.import_module('cloud_io')
    if Path(module.__file__).resolve().parent!=directory:raise ValueError('Unexpected bridge module origin')
    def owner(principal):
        row,_=store.read('identities/'+principal.user_id+'.json')
        if not row or row.get('user_id')!=principal.user_id or row.get('email') not in memberships.policy.emails():
            raise PermissionError('Verified training owner unavailable')
        return row['email']
    class ReadOnly(cloud.Cloud):
        def put(self,*a,**k):raise PermissionError('Training evidence is read-only')
        def name(self,name):
            # Monthly admissions carry ownership. No raw datasets or input arrays.
            if not re.fullmatch(r'training/v1/(?:control/(?:[0-9]{4}-[0-9]{2}\.json|worker\.json|bridge/web_[0-9]{6}_[a-f0-9]{32}\.json)|jobs/web_[0-9]{6}_[a-f0-9]{32}\.json|artifacts/web_[0-9]{6}_[a-f0-9]{32}/[a-f0-9-]{36}/result\.json)',name):
                raise PermissionError('Training source path is not admitted')
            return super().name(name)
    return TrainingTool(lambda email:module.TrainingBridge(ReadOnly(max_requests=28,max_upload_bytes=0),email,
        settings['training_dataset'],settings['training_worker_hashes'],reviewer=True),guard,owner)
