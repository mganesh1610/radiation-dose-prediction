import asyncio
import time
from types import SimpleNamespace
from fastapi.testclient import TestClient
from test_chat import setup,create,key,headers
from research_agents.services.chat_api import create_app
from research_agents.policies.access import Principal
from research_agents.persistence.analysis_artifacts import AnalysisArtifacts


def test_private_artifact_route_and_chat_lifecycle():
    original,service,_,enabled=setup()
    artifacts=AnalysisArtifacts(service.ledger.store)
    class Auth:
        def authenticate(self,h):
            user=h.get('test-identity')
            if user not in enabled:raise PermissionError()
            return Principal(user,time.time()+600)
    client=TestClient(create_app(service,Auth(),artifacts=artifacts));chat=create(client)
    async def execute(scope,session):
        artifact={'schema_version':1,'png_sha256':'a'*64,'csv_sha256':'b'*64,'csv':'test'}
        ref=artifacts.save(scope,artifact)
        service.ledger.progress(scope.principal.user_id,scope.run_id,scope.fence,'verified_analysis',{'tool':'analysis','artifact':ref})
    service.executor=execute
    result=client.post(f'/api/copilot/chats/{chat["chat_id"]}/messages',json={'request_id':key(),'text':'Plot Gene ID 7157 in GSE1'},headers=headers()).json()
    run=service.ledger.run('alice',result['run_id']);ref=run['receipts'][0]['artifact']
    url=f'/api/copilot/runs/{result["run_id"]}/artifacts/{ref["artifact_id"]}'
    answer=client.get(url,headers=headers());assert answer.status_code==200 and answer.json()['csv']=='test'
    assert answer.headers['Cache-Control']=='no-store'
    assert client.get(url,headers=headers('bob')).status_code==404
    assert client.get(url).status_code==403
    # Always change the ID: a fixed replacement sometimes equals its last digit.
    wrong_url=url[:-1]+('0' if url[-1]!='0' else '1')
    assert client.get(wrong_url,headers=headers()).status_code==404
    service.ledger.mark_deleted('alice',chat['chat_id'])
    assert client.get(url,headers=headers()).status_code==404
