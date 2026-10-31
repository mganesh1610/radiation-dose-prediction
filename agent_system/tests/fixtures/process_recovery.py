"""Disposable subprocess harness. SQLite is a test store, never production fallback."""
import asyncio,json,socket,sqlite3,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
# Child interpreters do not inherit the parent suite's socket monkeypatch.
def blocked(*args,**kwargs):raise AssertionError('No network in process-recovery fixture')
socket.socket.connect=blocked;socket.socket.connect_ex=blocked
from google.adk.sessions import Session
from google.adk.events import Event
from google.genai import types
from research_agents.persistence.control_store import Conflict
from research_agents.persistence.ledger import Ledger
from research_agents.services.chat_service import ChatService
from research_agents.policies.runtime import ChatConfig
from research_agents.policies.access import AccessPolicy,Grant,Principal

class DurableFixtureStore:
 def __init__(self,path):
  self.path=str(path)
  with sqlite3.connect(self.path) as db:db.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY,value TEXT,generation INTEGER)')
 def read(self,key):
  with sqlite3.connect(self.path) as db:row=db.execute('SELECT value,generation FROM records WHERE key=?',(key,)).fetchone()
  return (json.loads(row[0]),str(row[1])) if row else (None,'0')
 def write(self,key,value,generation):
  with sqlite3.connect(self.path) as db:
   db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT generation FROM records WHERE key=?',(key,)).fetchone()
   if int(generation)!=(row[0] if row else 0):raise Conflict('Competing fixture write')
   db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?)',(key,json.dumps(value),int(generation)+1))
class Sessions:
 """Durable session double using ADK Session/Event schemas and SQLite commits."""
 def __init__(self,path):self.store=DurableFixtureStore(path)
 async def create_session(self,*,app_name,user_id,session_id,state,**kwargs):
  session=Session(id=session_id,app_name=app_name,user_id=user_id,state=state,last_update_time=time.time())
  self.store.write(session_id,session.model_dump(mode='json'),'0');return session
 async def get_session(self,*,app_name,user_id,session_id):
  data,_=self.store.read(session_id)
  return Session.model_validate(data) if data and data['user_id']==user_id and data['app_name']==app_name else None
 async def append_event(self,session,event):
  data,generation=self.store.read(session.id);fresh=Session.model_validate(data)
  fresh.events.append(event);fresh.state.update(event.actions.state_delta);fresh.last_update_time=time.time()
  self.store.write(session.id,fresh.model_dump(mode='json'),generation)
  session.events=fresh.events;session.state=fresh.state;return event
 async def close(self):pass

async def main():
 mode=sys.argv[1];p=Path(sys.argv[2]);p.mkdir(exist_ok=True)
 store=DurableFixtureStore(p/'control.sqlite');ledger=Ledger(store,'fixture')
 sessions=Sessions(p/'sessions.sqlite')
 principal=Principal('alice',time.time()+600)
 access=AccessPolicy(lambda u,w:Grant(u,w,frozenset({'core_test'}),'r1') if u=='alice' and w=='fixture' else None)
 config=ChatConfig('fixture','core_test','copilot','https://fixture.test','http://127.0.0.1:9999')
 async def execute(scope,session):
  with (p/'calls.jsonl').open('a') as f:f.write(json.dumps({'run_id':scope.run_id,'mode':mode})+'\n')
  if mode=='start':
   # Shorten only the isolated fixture deadline, keeping production's 180 seconds intact.
   ledger.change(lambda s:s['runs'][scope.run_id].update(deadline=time.time()+3))
   (p/'entered.json').write_text(json.dumps({'run_id':scope.run_id,'at':time.time()}))
   await asyncio.Event().wait()
  if mode=='fail':raise RuntimeError('Synthetic provider failure')
  await sessions.append_event(session,Event(author='research_coordinator',invocation_id=scope.run_id,content=types.Content(role='model',parts=[types.Part(text='Synthetic recovery response')]),timestamp=time.time()))
 service=ChatService(config,ledger,sessions,access,execute)
 if mode in ('start','fail'):
  chat=await service.create(principal,uuid.uuid4().hex);request=uuid.uuid4().hex
  (p/'request.json').write_text(json.dumps({'chat_id':chat['chat_id'],'request_id':request,'text':'Synthetic process recovery question'}))
  result=await service.send(principal,chat['chat_id'],request,'Synthetic process recovery question')
  (p/'initial-result.json').write_text(json.dumps(result))
 else:
  request=json.loads((p/'request.json').read_text());old=next(iter(ledger.read()['runs'].values()))
  initial=await service.get_run(principal,old['run_id']);replay=await service.send(principal,**request)
  before=await service.messages(principal,request['chat_id'])
  healthy=await service.send(principal,request['chat_id'],uuid.uuid4().hex,'Fresh synthetic question')
  after=await service.messages(principal,request['chat_id'])
  (p/'recovery.json').write_text(json.dumps({'old':initial,'replay':replay,'before':before,'healthy':healthy,'after':after,'ledger':ledger.read()}))
 await sessions.close()
asyncio.run(main())
