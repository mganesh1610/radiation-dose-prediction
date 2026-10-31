"""Shared protected bootstrap validation; no web framework or storage SDK import."""
import hashlib
import json
import re
from pathlib import Path
from ..policies.runtime import ChatConfig, PolicyFile


def read_settings(path):
    p=json.loads(Path(path).read_text());config=ChatConfig(**p['chat'])
    policy=PolicyFile(p['policy_file']);rules=policy.read()
    if rules['workspace']!=config.workspace or config.release_id not in rules['releases']:
        raise ValueError('Protected policy does not admit this deployment')
    raw=Path(p['admission_file']).read_bytes()
    if len(raw)>2_000_000 or hashlib.sha256(raw).hexdigest()!=p['admission_sha256']:
        raise ValueError('Admission hash mismatch')
    for key in ['memory_enabled','protocol_rag_enabled','analysis_enabled','prediction_enabled','training_enabled']:
        if type(p.get(key,False)) is not bool:raise ValueError('Feature flags must be boolean')
    users=p.get('phase5_user_ids',[])
    if p.get('memory_enabled') or p.get('protocol_rag_enabled'):
        if not isinstance(users,list) or not 1<=len(users)<=3 or not all(isinstance(u,str) and re.fullmatch(r'usr_[a-f0-9]{64}',u) for u in users):
            raise ValueError('Protected feature allowlist required')
    if not re.fullmatch(r'[a-z0-9_-]+',config.workspace):raise ValueError('Invalid control workspace')
    return p,config,policy,users
