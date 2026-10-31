"""Immutable private artifacts, delivered only through owned unexpired chat runs."""
import asyncio
import json
import re
from .control_store import Conflict
from ..tools.query_sql import digest


class AnalysisArtifacts:
    def __init__(self,store):self.store=store

    def save(self,scope,payload):
        artifact_id='plot-'+digest([scope.run_id,payload])[:32]
        record={'owner':scope.principal.user_id,'chat_id':scope.chat_id,'run_id':scope.run_id,'payload':payload}
        if len(json.dumps(record).encode())>450_000:raise ValueError('Analysis artifact too large')
        key=f'artifacts/{scope.run_id}/{artifact_id}.json'
        try:self.store.write(key,record,'0')
        except Conflict:
            previous,_=self.store.read(key)
            if previous!=record:raise RuntimeError('Immutable artifact collision')
        return {'artifact_id':artifact_id,'sha256':digest(record),'kind':'expression_by_recorded_dose',
                'png_sha256':payload['png_sha256'],'csv_sha256':payload['csv_sha256']}

    async def get(self,service,principal,run_id,artifact_id):
        if not re.fullmatch(r'run-[a-f0-9]{32}',run_id) or not re.fullmatch(r'plot-[a-f0-9]{32}',artifact_id):
            raise FileNotFoundError()
        run=await service.get_run(principal,run_id)
        await asyncio.to_thread(service.ledger.chat,principal.user_id,run['chat_id'])
        if run['status']!='completed':raise FileNotFoundError()
        ref=next((r['artifact'] for r in run['receipts'] if r.get('tool')=='analysis' and r.get('artifact',{}).get('artifact_id')==artifact_id),None)
        if ref is None:raise FileNotFoundError()
        record,_=await asyncio.to_thread(self.store.read,f'artifacts/{run_id}/{artifact_id}.json')
        if record is None:raise FileNotFoundError()
        if (record.get('owner')!=principal.user_id or record.get('run_id')!=run_id
                or record.get('chat_id')!=run['chat_id'] or digest(record)!=ref['sha256']):
            raise RuntimeError('Artifact integrity failure')
        await asyncio.to_thread(service.authorize,principal)
        await asyncio.to_thread(service.ledger.chat,principal.user_id,run['chat_id'])
        return record['payload']
