"""Install the project worker as a per-user launchd agent, or remove that agent."""
import argparse
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import json
import time

ROOT=Path(__file__).resolve().parents[1]
LABEL='edu.defender.radiation-training'
TARGET=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
STATE=ROOT/'data/worker_service'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['install','remove','demo'])
    parser.add_argument('--minutes',type=int,default=120);args=parser.parse_args()
    domain=f'gui/{os.getuid()}'
    if args.action=='demo':
        if not 1 <= args.minutes <= 120:raise ValueError('Demo window must be 1–120 minutes')
        if not TARGET.exists() or plistlib.loads(TARGET.read_bytes()).get('StartInterval') != 60:
            raise ValueError('Install the updated worker first')
        STATE.mkdir(parents=True,exist_ok=True)
        target=STATE/'demo-window.json';tmp=target.with_suffix('.tmp')
        tmp.write_text(json.dumps({'expires_at':time.time()+60*args.minutes}));tmp.replace(target)
        subprocess.run(['launchctl','kickstart',domain+'/'+LABEL],check=True)
        print(f'One-minute pickup enabled for {args.minutes} minutes; then returns to ten-minute polling')
        return
    if args.action=='remove':
        subprocess.run(['launchctl','bootout',domain+'/'+LABEL],capture_output=True)
        TARGET.unlink(missing_ok=True);print('Worker agent removed; cloud jobs and local results preserved');return
    STATE.mkdir(parents=True,exist_ok=True);TARGET.parent.mkdir(parents=True,exist_ok=True)
    spec={'Label':LABEL,'ProgramArguments':[str(ROOT/'data/local_runtime/mac-training-v1/bin/python'),
            str(ROOT/'scripts/mac_worker_loop.py'),'--once'],
          'WorkingDirectory':str(ROOT),'RunAtLoad':True,'StartInterval':60,
          'ProcessType':'Background','LowPriorityIO':True,
          'EnvironmentVariables':{'PATH':'/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin',
            'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','VECLIB_MAXIMUM_THREADS':'4',
            'PYTHONUNBUFFERED':'1'},
          'StandardOutPath':str(STATE/'launchd.log'),'StandardErrorPath':str(STATE/'launchd-error.log')}
    if TARGET.exists():
        old=plistlib.loads(TARGET.read_bytes())
        if old.get('ProgramArguments')!=spec['ProgramArguments']:
            raise ValueError('Existing launch agent points elsewhere; refusing to replace')
        subprocess.run(['launchctl','bootout',domain+'/'+LABEL],capture_output=True)
    TARGET.write_bytes(plistlib.dumps(spec));TARGET.chmod(0o600)
    subprocess.run(['launchctl','bootstrap',domain,str(TARGET)],check=True)
    print('Installed '+str(TARGET)+'; local timer every minute, cloud polls every ten minutes unless a bounded demo is active')


if __name__=='__main__': main()
