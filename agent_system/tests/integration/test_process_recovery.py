"""Actual process termination with durable test state; no Cloud Run crash claim."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
import pytest
FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/process_recovery.py'
def command(mode,path):return [sys.executable,str(FIXTURE),mode,str(path)]
def run(mode,path):
 r=subprocess.run(command(mode,path),capture_output=True,text=True,timeout=35)
 assert r.returncode==0,r.stderr

def verify(path,status):
 r=json.loads((path/'recovery.json').read_text());calls=[json.loads(x) for x in (path/'calls.jsonl').read_text().splitlines()]
 assert r['old']['status']==r['replay']['status']==status
 assert r['old']['run_id']==r['replay']['run_id']
 assert len(calls)==2 and len({x['run_id'] for x in calls})==2
 assert [m['role'] for m in r['before']['messages']]==['user']
 assert r['healthy']['status']=='completed'
 assert [m['role'] for m in r['after']['messages']]==['user','user','assistant']
 assert len(r['ledger']['runs'])==2
 assert all(c['active_run'] is None for c in r['ledger']['chats'].values())
 assert all('Synthetic provider failure' not in m['text'] for m in r['after']['messages'])

def test_sigkill_restarted_process_does_not_redispatch_claimed_turn(tmp_path):
 with (tmp_path/'child.log').open('w') as log:
  child=subprocess.Popen(command('start',tmp_path),stdout=log,stderr=log)
  try:
   deadline=time.monotonic()+25
   while not (tmp_path/'entered.json').exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.05)
   assert (tmp_path/'entered.json').exists(),(tmp_path/'child.log').read_text()
   child.kill();assert child.wait(timeout=5)==-signal.SIGKILL
   entered=json.loads((tmp_path/'entered.json').read_text())
   time.sleep(max(0,entered['at']+3.2-time.time()))
   run('recover',tmp_path);verify(tmp_path,'interrupted')
  finally:
   if child.poll() is None:child.kill();child.wait(timeout=5)

def test_provider_failure_remains_terminal_after_process_restart(tmp_path):
 run('fail',tmp_path);run('recover',tmp_path);verify(tmp_path,'failed')
