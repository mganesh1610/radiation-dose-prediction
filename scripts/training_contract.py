"""Versioned, strict contracts for the first core-only training integration."""
import re
import math

CORE = {'GSE102971':100, 'GSE8917':25, 'GSE65292':20}
METHODS = ('median_null', 'elastic_net', 'pls', 'gradient_boosting')
PRESETS = {'median_null':{}, 'elastic_net':{'k':100,'alpha':0.1,'l1_ratio':0.5},
           'pls':{'k':100,'n_components':5},
           'gradient_boosting':{'k':100,'max_depth':1,'n_estimators':100}}


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', value):
        raise ValueError('Invalid identifier')
    return value


def validate_data(data):
    import numpy as np
    required = {'values','detected','probe_ids','gene_keys','sample_ids','study_ids','split_group_ids','y'}
    if set(data) != required:
        raise ValueError('Unexpected core data fields')
    samples, studies, groups = data['sample_ids'], data['study_ids'], data['split_group_ids']
    p,n = data['values'].shape
    if (n != 145 or len(samples) != n or len(set(samples)) != n or p != len(data['probe_ids'])
            or len(set(data['probe_ids'])) != p or len(data['gene_keys']) != p
            or data['detected'].shape != (p,n) or data['detected'].dtype != np.bool_
            or studies.shape != (n,) or groups.shape != (n,) or data['y'].shape != (n,)
            or len(set(groups)) != 30 or not np.isfinite(data['y']).all() or np.any(data['y'] < 0)
            or np.isinf(data['values']).any() or np.any(data['values'][np.isfinite(data['values'])] <= 0)):
        raise ValueError('Invalid core dimensions, identities or values')
    if {s:int(np.sum(studies==s)) for s in set(studies)} != CORE:
        raise ValueError('Only the verified core membership is supported')
    for s,g in zip(studies,groups):
        if not g.startswith(s+':'): raise ValueError('Group is not study-scoped')
    for g in set(groups):
        if len(set(studies[groups==g])) != 1: raise ValueError('Group crosses studies')


def validate_job(job):
    if set(job) != {'schema_version','job_id','dataset','method','holdout_study','purpose','parameters'}:
        raise ValueError('Unexpected job fields')
    identifier(job['job_id'])
    if (job['schema_version'] != 1 or job['purpose'] != 'engineering_smoke'
            or job['method'] not in METHODS or job['holdout_study'] not in CORE
            or job['parameters'] != PRESETS[job['method']]):
        raise ValueError('Unsupported method, parameters or scientific run; only bounded smoke jobs enabled')
    ref=job['dataset']
    if set(ref) != {'object','generation','sha256','bytes'}:
        raise ValueError('Invalid dataset reference')
    if not re.fullmatch(r'training/v1/releases/core_[a-f0-9]{16}/core.npz',ref['object']):
        raise ValueError('Dataset outside core release path')
    if not str(ref['generation']).isdigit() or not re.fullmatch('[a-f0-9]{64}',ref['sha256']):
        raise ValueError('Dataset generation/hash required')
    if ref['object'].split('/')[3] != 'core_'+ref['sha256'][:16]:
        raise ValueError('Release ID does not match content hash')
    if not isinstance(ref['bytes'],int) or not 0 < ref['bytes'] < 80_000_000:
        raise ValueError('Invalid dataset size')


def fields(spec):
    return [{'name':n,'type':t,'mode':'REQUIRED' if required else 'NULLABLE'} for n,t,required in spec]


SCHEMAS = {
    'studies':fields([('release_id','STRING',1),('study_id','STRING',1),('source_url','STRING',1),('role','STRING',1)]),
    'sample_groups':fields([('release_id','STRING',1),('group_id','STRING',1),('study_id','STRING',1),('identity_status','STRING',1)]),
    'samples':fields([('release_id','STRING',1),('sample_id','STRING',1),('study_id','STRING',1),('group_id','STRING',1)]),
    'sample_annotations':fields([('release_id','STRING',1),('sample_id','STRING',1),('dose_gy','FLOAT',1),('dose_status','STRING',1),('time_after_exposure_hours','FLOAT',1),('organism','STRING',1),('tissue','STRING',1)]),
    'features':fields([('release_id','STRING',1),('probe_id','STRING',1),('gene_id','STRING',1),('gene_namespace','STRING',1)]),
    'expression_sets':fields([('release_id','STRING',1),('expression_set_id','STRING',1),('representation','STRING',1),('scale','STRING',1),('processing_scope','STRING',1),('normalization_status','STRING',1),('samples','INTEGER',1),('probes','INTEGER',1),('object','STRING',1),('generation','STRING',1),('sha256','STRING',1)]),
    'split_members':fields([('release_id','STRING',1),('split_id','STRING',1),('sample_id','STRING',1),('partition','STRING',1)]),
    'model_runs':fields([('job_id','STRING',1),('release_id','STRING',1),('method','STRING',1),('holdout_study','STRING',1),('purpose','STRING',1),('status','STRING',1),('deployment_allowed','BOOLEAN',1),('train_samples','INTEGER',1),('validation_samples','INTEGER',1),('training_seconds','FLOAT',1),('artifact_manifest','STRING',1)]),
    'model_metrics':fields([('job_id','STRING',1),('metric','STRING',1),('value','FLOAT',1),('unit','STRING',1),('evaluation_scope','STRING',1)]),
    'predictions':fields([('job_id','STRING',1),('sample_id','STRING',1),('group_id','STRING',1),('observed_gy','FLOAT',1),('predicted_gy','FLOAT',1),('evaluation_scope','STRING',1)]),
}


def validate_rows(table, rows):
    schema = SCHEMAS[table]
    allowed = {s['name'] for s in schema}
    keys = {'studies':('release_id','study_id'), 'sample_groups':('release_id','group_id'),
            'samples':('release_id','sample_id'),'sample_annotations':('release_id','sample_id'),
            'features':('release_id','probe_id'),'expression_sets':('expression_set_id',),
            'split_members':('release_id','split_id','sample_id'), 'model_runs':('job_id',),
            'model_metrics':('job_id','metric'),'predictions':('job_id','sample_id')}[table]
    seen=set()
    for row in rows:
        if set(row) != allowed: raise ValueError('Row fields differ from schema: '+table)
        for f in schema:
            value=row[f['name']]
            if value is None and f['mode']=='NULLABLE': continue
            valid = {'STRING':lambda:isinstance(value,str) and bool(value),
                     'FLOAT':lambda:type(value) in (int,float) and math.isfinite(value),
                     'INTEGER':lambda:type(value) is int,
                     'BOOLEAN':lambda:type(value) is bool}[f['type']]()
            if not valid: raise ValueError('Invalid field '+f['name'])
        key=tuple(row[k] for k in keys)
        if key in seen: raise ValueError('Duplicate key in '+table)
        seen.add(key)
