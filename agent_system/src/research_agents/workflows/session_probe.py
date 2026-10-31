"""Opt-in managed persistence probe; never calls a model or scientific tool."""
import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from google.adk.events import Event, EventActions
from google.genai import types
from ..persistence.managed_sessions import ManagedSessions


async def probe(project, location, engine_id):
    def client(): return ManagedSessions(project=project, location=location, engine_id=engine_id)
    service=client(); sid='probe-'+uuid.uuid4().hex
    owner='engineering-session-probe'; app='copilot_probe'; checks={}
    session=await service.create_session(app_name=app,user_id=owner,session_id=sid,
        state={'selected_release':'core-test'},ttl='86400s')
    try:
        event=Event(id='event-'+uuid.uuid4().hex,invocation_id='probe',author='user',
            content=types.Content(role='user',parts=[types.Part(text='Engineering persistence check')]),
            actions=EventActions(state_delta={'probe_value':2,'temp:discard':True}),timestamp=time.time())
        await service.append_event(session,event)
        fresh=client()
        restored=await fresh.get_session(app_name=app,user_id=owner,session_id=sid)
        checks['event_survives_new_client']=any(e.id==event.id for e in restored.events)
        checks['state_persisted']=restored.state.get('probe_value')==2
        checks['temporary_state_not_persisted']='temp:discard' not in restored.state
        listed=await fresh.list_sessions(app_name=app,user_id=owner)
        checks['list_owned_session']=any(s.id==sid for s in listed.sessions)
        try:
            wrong=await fresh.get_session(app_name=app,user_id='different-owner',session_id=sid)
            checks['wrong_owner_denied']=wrong is None
        except ValueError:
            checks['wrong_owner_denied']=True
        assert all(checks.values()),checks
    finally:
        await service.delete_session(app_name=app,user_id=owner,session_id=sid)
    checks['deletion_verified']=await service.get_session(app_name=app,user_id=owner,session_id=sid) is None
    assert all(checks.values()),checks
    return {'status':'managed_session_lifecycle_passed','checks':checks,'model_calls':0,
        'reviewer_sign_in_tested':False,'engine_id':engine_id}


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--location',required=True)
    p.add_argument('--engine-id',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=asyncio.run(probe(a.project,a.location,a.engine_id))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(result['status'])


if __name__=='__main__': main()
