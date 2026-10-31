"""Bounded, outbound worker dispatcher with a local one-minute launchd timer.

One local lock prevents overlap; persisted local counters bound polling/attempts
across restarts. GCS CAS leases still fence competing machines and cancellation.
"""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cloud_io import Cloud, Conflict, PREFIX, encode
from dispatch_control import DispatchControl
from job_queue import Queue, job_name

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'data/worker_service'


def polling_policy(saved, now, window):
    """Fast mode expires automatically; all actual polls still use the monthly guard."""
    until = window.get('expires_at', 0) if isinstance(window,dict) else 0
    # Ignore malformed or unbounded windows, including after clock rollback.
    fast = isinstance(until, (int,float)) and now < until <= now + 7200
    interval = 60 if fast else 600
    return interval, fast, now - saved.get('last_poll', 0) >= interval


def process_entry(cloud, entry, execute):
    if entry.get('bridge_approval_sha256'):
        from training_bridge import validate_approval,worker_hashes
        validate_approval(cloud,entry,worker_hashes(ROOT/'scripts'))
    job = entry['request']; q = Queue(cloud); q.submit(job)
    state, generation = cloud.read_json(job_name(job['job_id']))
    if state['status'] in ('CANCELLED','FAILED'):
        return state['status']
    if state['status'] == 'RUNNING' and state['lease']['expires_at'] > time.time():
        return 'BUSY'
    if state['status'] != 'SUCCEEDED' and state['attempt'] >= 3:
        state.update(status='FAILED', stage='attempts_exhausted')
        cloud.state(job_name(job['job_id']),state,generation)
        return 'FAILED'
    execute(job['job_id'])
    return 'PUBLISHED'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true',required=True)
    args=parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'worker.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return
        path=STATE/'state.json'
        saved=json.loads(path.read_text()) if path.exists() else {'months':{},'done':[], 'heartbeat':0}
        try: window=json.loads((STATE/'demo-window.json').read_text())
        except (FileNotFoundError,ValueError): window={}
        interval,fast,due=polling_policy(saved,time.time(),window)
        if not due:return
        now=datetime.now(timezone.utc);month=now.strftime('%Y-%m')
        counts=saved['months'].setdefault(month,{'polls':0,'runs':0})
        if counts['polls'] >= 4500:
            print('Monthly polling guard reached; dispatcher paused',flush=True);return
        counts['polls']+=1
        saved['last_poll']=time.time()
        def persist():
            tmp=path.with_suffix('.tmp');tmp.write_bytes(encode(saved));tmp.replace(path)
        persist() # reserve before network activity, even if credentials/network fail
        cloud=Cloud(max_requests=150,max_upload_bytes=1_000_000)
        completed=[]
        def execute(job_id):
            if counts['runs']>=36: raise RuntimeError('Monthly job-attempt guard reached')
            counts['runs']+=1;persist()
            log=STATE/(job_id+'.log')
            with log.open('ab') as output:
                proc=subprocess.run([sys.executable,str(ROOT/'scripts/training_worker.py'),'run','--job-id',job_id],
                    cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=720,check=False)
            if proc.returncode:
                # Sanitize cloud error text; detailed diagnostics remain local.
                # Leave the owned lease to expire; never mutate another worker's lease.
                raise RuntimeError('Training/publication failed')
        try:
            # Include the previous month so a job queued at rollover can still finish.
            prior=(now.replace(day=1)-timedelta(days=1)).strftime('%Y-%m')
            for key in (prior,month):
                control,_=DispatchControl(cloud,key).read()
                for job_id,entry in control['jobs'].items():
                    if job_id in saved['done']:continue
                    if len(completed)>=2:break
                    status=process_entry(cloud,entry,execute)
                    if status in ('PUBLISHED','CANCELLED','FAILED','ATTEMPTS_EXHAUSTED'):
                        saved['done'].append(job_id);persist();completed.append(job_id)
            state='waiting'
        except Exception as error:
            state='attention_required'
            print(type(error).__name__+': '+str(error)[:180],flush=True)
        # Hourly status writes avoid consuming the Class A allowance every poll.
        if (time.time()-saved['heartbeat']>=(900 if fast else 3600) or completed
                or saved.get('fast_mode') != fast
                or (state=='attention_required' and saved.get('last_status')!=state)):
            try:
                _,generation=cloud.read_json(PREFIX+'control/worker.json')
            except FileNotFoundError:generation=0
            cloud.state(PREFIX+'control/worker.json',{'status':state,'checked_at':time.time(),
                'poll_interval_seconds':interval,'demo_until':window.get('expires_at') if fast else None,
                'month':month,'polls_used':counts['polls'],
                'attempts_used':counts['runs'],'training_location':'local_mac'},generation)
            saved['heartbeat']=time.time();saved['last_status']=state;saved['fast_mode']=fast;persist()
        print(json.dumps({'status':state,'completed':completed,'requests':cloud.requests}),flush=True)


if __name__=='__main__': main()
