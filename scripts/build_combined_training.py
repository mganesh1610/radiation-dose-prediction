"""Package only the engineering panels, labels, membership and portable refit reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from combined_training import SCOPES, verify_bundle


def build(root, destination):
    meta=json.loads((root/'data/model_ready/development_raw_f6022cfaaca6d1d9/metadata.json').read_text())['profiles']
    meta.sort(key=lambda r:r['profile_id'])
    ids=[r['profile_id'] for r in meta]
    index={pid:i for i,pid in enumerate(ids)}
    data=dict(profile_ids=np.array(ids),studies=np.array([r['study'] for r in meta]),groups=np.array([r['donor_group'] for r in meta]),
              dose=np.array([r['dose_gy'] for r in meta],dtype=float))
    aux={r['profile_id'] for r in meta if r['role']=='auxiliary_development'}
    for scope in SCOPES:
        folder=root/('data/models/final-selection-v2/folds/full-development' if scope=='full' else 'data/models/abcd-v1/folds/'+scope+'/outer')
        hashes=json.loads((folder/'complete.json').read_text())['files']
        for name in ('log2-features.npz','log2-panel.npz','fit.json','preprocessing/core.npz'):
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=hashes[name]:
                raise ValueError('Prepared source changed: '+name)
        fit=json.loads((folder/'fit.json').read_text())
        with np.load(folder/'log2-panel.npz',allow_pickle=False) as p, np.load(folder/'log2-features.npz',allow_pickle=False) as f:
            train, test, auxiliary = p['training_ids'].tolist(),p['validation_ids'].tolist(),p['auxiliary_ids'].tolist()
            if (fit['training']!=train or fit['validation']!=test or not fit['nontraining_perturbation_invariant']
                    or set(auxiliary)!=aux or set(f['training_ids'])!=set(train)):
                raise ValueError('Invalid input provenance')
            selected=f['selected'][:100]
            chunk=dict(train=np.array([index[i] for i in train+auxiliary],dtype=int),test=np.array([index[i] for i in test],dtype=int),
                       x=np.vstack([p['train'][:,:100],p['aux'][:,:100]]),v=p['validation'][:,:100],
                       genes=f['gene_ids'][selected],medians=f['medians'][selected],mean=f['mean'][:100],scale=f['scale'][:100])
            data.update({scope+'_'+k:v for k,v in chunk.items()})
        if scope=='full':
            data['full_reference']=np.frombuffer((folder/'preprocessing/core.npz').read_bytes(),dtype=np.uint8)
    verify_bundle(data)
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**data)
    return {'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'bytes':destination.stat().st_size}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path)
    args=p.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1],args.destination)))
