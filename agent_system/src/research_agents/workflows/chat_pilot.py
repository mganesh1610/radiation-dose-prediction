"""Explicit operator-only pilot: Gemini, real A2A, managed Sessions and verified data.

Uses an operator principal, never claims a browser IAP/reviewer sign-in test.
Creates at most two new turns and removes their managed conversation afterwards.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import time
import uuid
import uvicorn
from ..policies.access import Principal
from ..services.bootstrap import Components


async def run(config,output):
    os.environ['COPILOT_LOOPBACK_TOKEN']=secrets.token_urlsafe(32)
    c=Components(config);e=Components(config)
    principal=Principal('operator_chat_probe',time.time()+900)
    c.memberships.bind(principal,os.environ['COPILOT_OPERATOR_EMAIL'])
    server=uvicorn.Server(uvicorn.Config(e.evidence_app(),host='127.0.0.1',port=8766,log_level='warning'))
    task=asyncio.create_task(server.serve());chat=None;results=[];errors=[]
    service=c.chat_service();original=service.executor
    async def execute(scope,session):
        try:await original(scope,session)
        except Exception as exc:
            import traceback
            traceback.print_exc();errors.append(type(exc).__name__);raise
    service.executor=execute
    try:
        for _ in range(200):
            if server.started:break
            await asyncio.sleep(.01)
        if not server.started:raise RuntimeError('Evidence service did not start')
        chat=await service.create(principal,uuid.uuid4().hex)
        for question in ['How many profiles in the approved core release have a recorded dose of 2 Gy? Give the evidence ID.',
                'For sample GSM2751831, retrieve NCBI Gene ID 7157 expression values. Preserve separate probes and state the scale and normalization.']:
            key=uuid.uuid4().hex;start=time.perf_counter()
            turn=await service.send(principal,chat['chat_id'],key,question)
            elapsed=time.perf_counter()-start
            receipt=await service.get_run(principal,turn['run_id'])
            history=await service.messages(principal,chat['chat_id'])
            duplicate=await service.send(principal,chat['chat_id'],key,question)
            results.append({'question':question,'status':turn['status'],'elapsed_seconds':elapsed,
                'answer':history['messages'][-1]['text'],'duplicate_not_reexecuted':duplicate==turn,
                'receipt':{k:receipt[k] for k in ('run_id','model_calls','tool_calls','prompt_tokens','output_tokens','receipts')}})
            Path(output).write_text(json.dumps({'results':results,'errors':errors,'reviewer_sign_in_tested':False},indent=2)+'\n')
            if turn['status']!='completed' or receipt['tool_calls']!=1 or not receipt['receipts']:break
        restored=await Components(config).chat_service().messages(principal,chat['chat_id'])
        assert restored['messages']==history['messages']
    finally:
        if chat:
            await service.delete(principal,chat['chat_id'])
        server.should_exit=True;await task
    record={'results':results,'errors':errors,'reviewer_sign_in_tested':False,
        'history_survived_new_service':True,'chat_deleted':chat is not None,
        'status':'passed' if len(results)==2 and all(r['status']=='completed' and r['receipt']['tool_calls']==1
            and r['receipt']['receipts'] for r in results) else 'failed'}
    Path(output).write_text(json.dumps(record,indent=2)+'\n');print(record['status'])
    if record['status']!='passed':raise SystemExit(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    asyncio.run(run(a.config,a.output))


if __name__=='__main__':main()
