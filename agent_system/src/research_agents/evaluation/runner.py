"""Opt-in authenticated HTTP collector. Sends only questions; never reference answers."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

from .references import sha


def check_allowance(preflight, count, chats=None):
    chats = count if chats is None else chats
    checked = datetime.fromisoformat(preflight['checked_at_utc'])
    age = (datetime.now(timezone.utc) - checked).total_seconds()
    if not 0 <= age <= 86400: raise ValueError('Refresh the read-only allowance preflight')
    if count > preflight['max_runs'] - preflight['runs_used'] or chats > preflight['max_chats'] - preflight['chats_used']:
        raise ValueError(f'Remaining collection needs up to {count} turns and {chats} fresh chats; current allowance is insufficient')
    if not preflight.get('billing_review_completed'):
        raise ValueError('Actual usage/credits review must be recorded before a paid acceptance run')


def safe_json(response):
    if response.is_redirect: raise RuntimeError('Authentication redirect; sign-in required')
    response.raise_for_status()
    if len(response.content) > 2_000_000: raise ValueError('Response exceeds evaluation bound')
    return response.json()


def write(path, data):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def pending_requirements(questions, output, *, resume=False):
    """Conservatively reserve unknown outcomes; known run/chat IDs need no new slot."""
    output = Path(output)
    ids = [c['id'] for c in questions['cases']]
    if len(ids) != 38 or set(ids) != {f'{prefix}{i:02}' for prefix, n in [('F',30),('B',8)] for i in range(1,n+1)}:
        raise ValueError('Expected the complete 38-case question inventory')
    if any(set(c) != {'id','question'} for c in questions['cases']):
        raise ValueError('Collector accepts questions only')
    if output.exists() and not resume:
        raise FileExistsError('Use a new output directory or explicit resume')
    signature = sha(json.dumps(questions,sort_keys=True).encode())
    if output.exists():
        saved = output/'questions.sha256'
        if not saved.exists() or saved.read_text().strip() != signature:
            raise ValueError('Resume questions changed or signature is missing')
    turns = chats = pending = 0
    for case in questions['cases']:
        result_path = output/(case['id']+'.json')
        journal_path = output/(case['id']+'.journal.json')
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('case_id') != case['id'] or result.get('question') != case['question'] or result.get('reference_sha256') != questions['reference_sha256']:
                raise ValueError('Existing result does not match the frozen question')
            if result.get('status') == 'completed':
                if not result.get('run_id') or result['run_id'] != journal.get('run_id'):
                    raise ValueError('Completed result has no matching run journal')
                continue
        pending += 1
        turns += not bool(journal.get('run_id'))
        chats += not bool(journal.get('chat_id'))
    return {'turns':turns,'chats':chats,'pending_cases':pending}


def collect(questions, preflight, client, output, *, resume=False):
    """Server still enforces authoritative ownership/quotas. Preflight is an operator check."""
    cases = questions['cases']; output = Path(output)
    needed = pending_requirements(questions,output,resume=resume)
    check_allowance(preflight,needed['turns'],needed['chats'])
    if not needed['pending_cases']:
        return {'status':'collected','cases':len(cases),'answer_review_required':True}
    output.mkdir(parents=True,exist_ok=True)
    signature = sha(json.dumps(questions,sort_keys=True).encode())
    signature_path = output/'questions.sha256'
    signature_path.write_text(signature+'\n')
    # An authenticated read must succeed before the first chat mutation.
    status = safe_json(client.get('/api/copilot/status'))
    if status.get('mode') != 'adk_a2a' or not status.get('enabled'): raise ValueError('Live ADK service required')
    for case in cases:
        destination = output/(case['id']+'.json')
        if destination.exists() and json.loads(destination.read_text()).get('status') == 'completed': continue
        journal_path = output/(case['id']+'.journal.json')
        if journal_path.exists(): journal = json.loads(journal_path.read_text())
        else:
            journal = {'create_request_id':uuid.uuid4().hex,'message_request_id':uuid.uuid4().hex,'chat_id':None,'run_id':None,'attempts':0}
        journal['attempts'] += 1; write(journal_path,journal)
        record = {'case_id':case['id'],'question':case['question'],'reference_sha256':questions['reference_sha256'],
                  'status':'outcome_unknown','answer':'','receipt':{},'started_at_utc':datetime.now(timezone.utc).isoformat(),
                  'endpoint':str(client.base_url),'attempt':journal['attempts']}
        start=time.perf_counter()
        try:
            if not journal['chat_id']:
                chat=safe_json(client.post('/api/copilot/chats',json={'request_id':journal['create_request_id']}))
                journal['chat_id']=chat['chat_id'];write(journal_path,journal)
            # Reuse keys across unknown network outcomes; no new paid retry is invented.
            turn=safe_json(client.post('/api/copilot/chats/'+journal['chat_id']+'/messages',json={'request_id':journal['message_request_id'],'text':case['question']}))
            journal['run_id']=turn['run_id'];write(journal_path,journal)
            deadline=time.monotonic()+190
            while turn['status'] not in ('completed','failed','interrupted','cancelled'):
                if time.monotonic()>=deadline: raise TimeoutError('Run remains unresolved')
                time.sleep(2)
                turn=safe_json(client.get('/api/copilot/runs/'+journal['run_id']))
            record.update(run_id=journal['run_id'],chat_id=journal['chat_id'],status=turn['status'])
            record['receipt']=safe_json(client.get('/api/copilot/runs/'+journal['run_id']+'/receipt'))
            history=safe_json(client.get('/api/copilot/chats/'+journal['chat_id']+'/messages'))
            answers=[m['text'] for m in history['messages'] if m['role']=='assistant' and m['invocation_id']==journal['run_id']]
            record['answer']='\n\n'.join(answers)
        except Exception as exc:
            # Never serialize credential headers, raw provider response or exception strings.
            record['error_type']=type(exc).__name__
        finally:
            attempt_seconds=time.perf_counter()-start
            journal.setdefault('attempt_seconds',[]).append(attempt_seconds);write(journal_path,journal)
            record['elapsed_seconds']=sum(journal['attempt_seconds'])
            record['attempt_seconds']=journal['attempt_seconds'];write(destination,record)
        if record['status']!='completed':
            return {'status':'stopped','case_id':case['id'],'reason':'Unsuccessful/unknown run retained; review before explicit resume'}
    return {'status':'collected','cases':len(cases),'answer_review_required':True}


def http_client(base_url, headers):
    import httpx
    parsed=urlparse(base_url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('','/'):
        raise ValueError('Use an explicit HTTPS origin without credentials or path')
    protected={'content-type':'application/json','x-defender-request':'1','origin':base_url.rstrip('/')}
    return httpx.Client(base_url=base_url.rstrip('/'),headers={**headers,**protected},timeout=200,follow_redirects=False)
