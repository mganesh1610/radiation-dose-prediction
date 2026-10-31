"""Authenticated website boundaries using synthetic uploads, no cloud calls."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'agent_system/tests/tools')]
from test_prediction_tool import Source,Engine
from prediction_service import PredictionService
from app.server import create_app


def test_chat_catalog_only_exposes_owned_uploads_and_uploads_need_same_origin(monkeypatch):
    source=Source();body={'sample_id':'synthetic','metadata':{},'values':[1.,3.]};engine=Engine(body)
    engine.catalog=lambda:dict(methods=['A'],examples=[])
    mine=PredictionService(source,engine,'alice',True).upload(body)
    other=PredictionService(source,engine,'bob',True).upload(body)
    def identity(headers):
        if headers.get('X-Test-User') not in ('alice','bob'):raise PermissionError('No identity')
        return headers['X-Test-User']
    monkeypatch.setenv('REVIEWER_EMAILS','bob');monkeypatch.setenv('DEFENDER_REVIEWER_MONTHLY_API_LIMIT','180')
    client=create_app(lambda **kw:source,identity,engine).test_client()
    assert client.get('/api/prediction-chat-catalog').status_code==403
    for user,expected in [('alice',mine),('bob',other)]:
        r=client.get('/api/prediction-chat-catalog',headers={'X-Test-User':user})
        assert r.status_code==200 and r.json['uploads']==[{'upload_id':expected['upload_id']}]
        assert r.headers['Cache-Control']=='no-store'
    headers={'X-Test-User':'alice','X-Defender-Request':'1','Origin':'https://localhost'}
    assert client.post('/api/uploads',json=body,headers={'X-Test-User':'alice'}).status_code==403
    assert client.post('/api/uploads',json=body,headers=dict(headers,Origin='https://attacker.test')).status_code==403
    assert client.post('/api/uploads',json=body,headers=headers).status_code==201
    assert client.post('/api/predict',json={'upload_id':other['upload_id'],'model':'A'},headers=headers).status_code==404
    assert client.post('/api/predict',json={'upload_id':mine['upload_id'],'model':'A'},headers=headers).json['estimate_gy']==2
    assert client.post('/api/uploads/'+other['upload_id']+'/review',json={},headers=dict(headers,**{'X-Test-User':'bob'})).status_code==403


@pytest.mark.parametrize('limit',['0','751','-1','NaN','1.5',''])
def test_reviewer_allowance_invalid_config_fails_closed(limit):
    with patch.dict('os.environ',{'DEFENDER_REVIEWER_MONTHLY_API_LIMIT':limit}):
        with pytest.raises(ValueError):create_app()
