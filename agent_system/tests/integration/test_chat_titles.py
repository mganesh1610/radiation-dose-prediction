"""Persistent sidebar labels without model calls or per-chat history fanout."""
import asyncio
import time
from google.adk.events import Event
from google.genai import types
from research_agents.policies.access import Principal
from research_agents.policies.run_scope import session_user
from research_agents.services.chat_service import ChatService
from research_agents.services.chat_titles import TITLE_KEY, normalize_title, first_question_title
from test_chat import setup, create, headers, key


def send(client, chat, text, user='alice'):
    r=client.post('/api/copilot/chats/'+chat['chat_id']+'/messages',
                  json={'request_id':key(),'text':text},headers=headers(user))
    assert r.status_code==200 and r.json()['status']=='completed'
    return r.json()


def test_title_survives_service_restart_and_does_not_change_on_followup():
    c,s,calls,_=setup();chat=create(c)
    send(c,chat,'  What is the\n recorded dose for GSM2751831? ')
    send(c,chat,'A different follow-up')
    restarted=ChatService(s.config,s.ledger,s.sessions,s.access,s.executor)
    rows=asyncio.run(restarted.list(Principal('alice',time.time()+60)))
    assert rows[0]['title']=='What is the recorded dose for GSM2751831?'
    assert len(calls)==2
    assert TITLE_KEY not in s.ledger.chat('alice',chat['chat_id'])


def test_title_list_never_fetches_individual_history_or_leaks_other_users(monkeypatch):
    c,s,_,_=setup();a=create(c);b=create(c,'bob')
    send(c,a,'Alice question');send(c,b,'Bob private question','bob')
    async def no_history(**kwargs):raise AssertionError('Sidebar must not fetch every conversation history')
    monkeypatch.setattr(s.sessions,'get_session',no_history)
    assert [x['title'] for x in c.get('/api/copilot/chats',headers=headers()).json()['chats']]==['Alice question']
    assert [x['title'] for x in c.get('/api/copilot/chats',headers=headers('bob')).json()['chats']]==['Bob private question']


def test_title_scope_rechecked_after_session_list(monkeypatch):
    c,s,_,enabled=setup();a=create(c);send(c,a,'Question')
    original=s.sessions.list_sessions
    async def revoke(**kwargs):
        result=await original(**kwargs);enabled.remove('alice');return result
    monkeypatch.setattr(s.sessions,'list_sessions',revoke)
    assert c.get('/api/copilot/chats',headers=headers()).status_code==403


def test_deleted_expired_and_unowned_sessions_do_not_show_titles():
    c,s,_,_=setup();a=create(c);b=create(c)
    send(c,a,'Deleted question');send(c,b,'Expired question')
    assert c.request('DELETE','/api/copilot/chats/'+a['chat_id'],json={},headers=headers()).status_code==200
    s.ledger.clock=lambda:time.time()+31*86400
    assert c.get('/api/copilot/chats',headers=headers()).json()=={'chats':[]}


def test_legacy_title_comes_from_first_real_user_event():
    c,s,_,_=setup();chat=create(c)
    async def legacy():
        _,session=await s.session(Principal('alice',time.time()+60),chat['chat_id'])
        await s.sessions.append_event(session,Event(author='research_coordinator',
            content=types.Content(parts=[types.Part(text='Not a user question')]),timestamp=time.time()))
        await s.sessions.append_event(session,Event(author='user',
            content=types.Content(parts=[types.Part(text='Legacy question')]),timestamp=time.time()))
        assert first_question_title(session)=='Legacy question'
    asyncio.run(legacy())
    response=c.get('/api/copilot/chats/'+chat['chat_id']+'/messages',headers=headers())
    assert response.json()['chat']['title']=='Legacy question'
    send(c,chat,'Later question')
    assert c.get('/api/copilot/chats',headers=headers()).json()['chats'][0]['title']=='Legacy question'


def test_empty_and_unrelated_evidence_sessions_have_no_sidebar_title():
    c,s,_,_=setup();a=create(c)
    asyncio.run(s.sessions.create_session(app_name=s.config.app_name,user_id=session_user('workspace','alice'),
                session_id='evidence-unlinked',state={TITLE_KEY:'Hidden evidence'}))
    rows=c.get('/api/copilot/chats',headers=headers()).json()['chats']
    assert len(rows)==1 and rows[0]['chat_id']==a['chat_id'] and rows[0]['title'] is None


def test_title_normalization_bounds_control_characters_without_html_execution():
    assert normalize_title(None) is None
    assert normalize_title(' \n\t ') is None
    assert normalize_title('Dose\u202e\x00\nquestion')=='Dose question'
    assert normalize_title('<b>question</b>')=='<b>question</b>'  # Frontend uses textContent.
    assert len(normalize_title('a'*200))==72
