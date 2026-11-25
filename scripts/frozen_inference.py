"""Numeric-only replay of frozen research models. No fitting or executable uploads."""
import hashlib
import json
from pathlib import Path
import numpy as np
from core_transfer import CoreTransfer
from portable_model import numeric_predict

METHODS = ('A','B','C','D','elastic_net','pls','gradient_boosting','median_null')
WBC_BOOSTING = 'gradient_boosting_wb_wbc'
WBC_FILE = 'frozen-gb-wbc-v1.json'
WBC_SHA256 = '1c6ee67740a712f4964f6be8576f2c521ed6f3e676c91cca39abc98a5e1d08c0'
MAX_INPUT_BYTES = 800_000
ADAPTER = 'core-agilent-background-v1'
HELD_OUT_FOLDER = 'held-out-examples-v1'
HELD_OUT_MANIFEST_SHA256 = 'caf11a070d9a03ed822dda78be9544a144dd25fd335e55be0ea6bc6d8d5eebad'
LIMITATIONS = [
    'Research use only; not for clinical decisions.',
    'Study-associated bias and high-dose underprediction remain unresolved.',
    'No calibrated individual prediction interval or validated clinical error threshold.',
    'The final test is consumed. This operation does not refit, calibrate or select a model.',
    'An estimate may fall outside 0–8 Gy; estimates are not clipped.',
]

def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(payload): return hashlib.sha256(payload).hexdigest()


def ranks(x):
    result=np.full_like(x,np.nan)
    for j in range(x.shape[1]):
        finite=np.flatnonzero(np.isfinite(x[:,j]));n=len(finite)
        if n<2:raise ValueError('Too few usable genes')
        order=finite[np.argsort(x[finite,j],kind='stable')]
        starts=np.r_[0,np.flatnonzero(np.diff(x[order,j])!=0)+1]
        lengths=np.diff(np.r_[starts,n])
        result[order,j]=np.repeat((starts+(lengths-1)/2+.5)/n,lengths)
    return result


class FrozenInference:
    def __init__(self,folder):
        self.folder=Path(folder)
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        if self.manifest['schema_version']!=1:raise ValueError('Unsupported frozen release')
        for name,sha in self.manifest['files'].items():
            path=(self.folder/name).resolve()
            if not path.is_relative_to(self.folder.resolve()) or digest(path.read_bytes())!=sha:
                raise ValueError('Frozen release integrity failure')
        self.release_sha=digest(encode(self.manifest))
        self.core=CoreTransfer.load(self.folder/'core.npz')
        self.features={}
        for rep in ('log2','rank'):
            with np.load(self.folder/f'{rep}-features.npz',allow_pickle=False) as z:
                self.features[rep]={k:z[k] for k in z.files}
        self.models=json.loads((self.folder/'models.json').read_text())
        self.wbc_bundle = None
        wbc_path = self.folder.parent / WBC_FILE
        if wbc_path.exists():
            raw = wbc_path.read_bytes()
            if digest(raw) != WBC_SHA256:
                raise ValueError('Frozen WBC model integrity failure')
            bundle = json.loads(raw)
            if (bundle['base_release_sha256'] != self.release_sha or bundle['model'] != WBC_BOOSTING
                    or bundle['schema_version'] != 1):
                raise ValueError('WBC model and preprocessing release mismatch')
            self.wbc_bundle = bundle
            self.models[WBC_BOOSTING] = bundle['spec']
        # Examples can be added without changing the model identity used by saved approvals and estimates.
        self.example_index = {row['id']: (self.folder, row) for row in self.manifest['examples']}
        extra = self.folder.parent / HELD_OUT_FOLDER
        if extra.exists():
            raw = (extra / 'manifest.json').read_bytes()
            if digest(raw) != HELD_OUT_MANIFEST_SHA256:
                raise ValueError('Held-out example manifest integrity failure')
            supplement = json.loads(raw)
            if supplement['schema_version'] != 1 or supplement['release_sha256'] != self.release_sha:
                raise ValueError('Held-out examples belong to a different model release')
            for row in supplement['examples']:
                path = (extra / row['file']).resolve()
                if (path.parent != extra.resolve() or row['id'] in self.example_index
                        or row['training_overlap'] is not False or digest(path.read_bytes()) != row['sha256']):
                    raise ValueError('Held-out example integrity failure')
                self.example_index[row['id']] = (extra, row)
        self.curated = {row['sha256']: row for _, row in self.example_index.values()}

    def example_bytes(self, sample):
        if sample not in self.example_index:
            raise FileNotFoundError()
        directory, row = self.example_index[sample]
        payload = (directory / row['file']).read_bytes()
        if digest(payload) != row['sha256']:
            raise ValueError('Example profile integrity failure')
        return payload

    def model_release(self, method=None):
        if method is not None and (not isinstance(method,str) or method not in self.models):
            raise ValueError('Unsupported frozen model')
        if method == WBC_BOOSTING:
            return self.wbc_bundle['release'], WBC_SHA256
        return self.manifest['release'], self.release_sha

    def catalog(self, *, prediction_picker=False):
        methods = list(METHODS)
        scopes = {}
        if prediction_picker:
            if self.wbc_bundle is None:
                raise RuntimeError('Frozen WBC model is unavailable')
            methods[methods.index('gradient_boosting')] = WBC_BOOSTING
            scopes[WBC_BOOSTING] = WBC_SHA256
        return dict(release=self.manifest['release'],release_sha256=self.release_sha,
                    methods=methods,model_release_overrides=scopes,adapter=ADAPTER,max_input_bytes=MAX_INPUT_BYTES,
                    examples=list(self.curated.values()),limitations=LIMITATIONS,
                    input_status='New inputs require a server-side compatibility review.',
                    supported_scope='Human whole blood, ex vivo X-ray/gamma or matched controls, audited approximately 24 hours; reviewed Agilent adapter only.')

    def validate(self,body):
        if not isinstance(body,dict) or set(body)!={'schema_version','adapter','sample_id','metadata','probe_ids','values'}:
            raise ValueError('Expected the documented single-profile JSON adapter')
        if type(body['schema_version']) is not int or body['schema_version']!=1:
            raise ValueError('Unsupported input schema')
        if len(encode(body))>MAX_INPUT_BYTES:raise ValueError('Input exceeds 800 kB limit')
        if not isinstance(body['sample_id'],str) or not 1<=len(body['sample_id'])<=80:
            raise ValueError('Expected a short pseudonymous sample identifier')
        md=body['metadata']
        fields={'study','organism','tissue','exposure','time_hours','units','normalization','technical_flags','source_url','provenance'}
        if not isinstance(md,dict) or set(md)!=fields:raise ValueError('Required provenance metadata missing')
        if any(not isinstance(md[k],str) or not 1<=len(md[k])<=2000 for k in fields-{'time_hours'}):
            raise ValueError('Invalid provenance metadata')
        if type(md['time_hours']) not in (int,float) or not np.isfinite(md['time_hours']):
            raise ValueError('Time must be recorded in hours')
        ids,values=body['probe_ids'],body['values']
        if not isinstance(ids,list) or not isinstance(values,list) or len(ids)!=len(self.core.probe_ids) or len(values)!=len(ids):
            raise ValueError('Expected the complete 16,237-probe adapter universe')
        if any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids) or set(ids)!=set(self.core.probe_ids):
            raise ValueError('Unknown, duplicate or missing probe identifiers')
        if any(v is not None and (type(v) not in (int,float) or not np.isfinite(v)) for v in values):
            raise ValueError('Values must be finite numbers or null for masked measurements')
        lookup=dict(zip(ids,values));x=np.array([lookup[k] for k in self.core.probe_ids],dtype=float)[:,None]
        finite=x[np.isfinite(x)]
        blockers=[]
        for valid,message in [
            (body['adapter']==ADAPTER,'Unsupported assay adapter'),
            (md['organism']=='human','Unsupported organism'),
            (md['tissue']=='whole_blood','Unsupported tissue'),
            (md['exposure'] in ('ex_vivo_xray','ex_vivo_gamma','matched_control'),'Unsupported exposure context'),
            (md['time_hours']==24,'Timepoint requires a separately audited adapter'),
            (md['units']=='positive_linear_intensity','Unsupported measurement units; counts/log values are not accepted'),
            (md['normalization']=='background_corrected_not_between_array_normalized','Already transformed or undocumented normalization'),
            (md['technical_flags']=='masked_before_normalization','Technical-flag handling is not documented'),
            (len(finite)>=2 and np.all(finite>0) and np.ptp(finite)>0,'Insufficient variable positive measurements')]:
            if not valid:blockers.append(message)
        qc=dict(probes=len(ids),usable_probes=len(finite),masked_probes=len(ids)-len(finite),
                selected_gene_missing={},blockers=blockers,interpretation='QC descriptions are not validated clinical thresholds.')
        if not blockers:
            panels,q=self.panels(x);qc.update(q)
        return x,qc

    def panels(self,x):
        genes,y,_=self.core.transform(x,np.isfinite(x),self.core.probe_ids)
        lookup={str(g):i for i,g in enumerate(genes)}
        universe=self.features['log2']['gene_ids']
        y=np.stack([y[lookup[str(g)]] if str(g) in lookup else np.full(x.shape[1],np.nan) for g in universe])
        views={'log2':y,'rank':ranks(y)};panels={};missing={}
        for rep,f in self.features.items():
            if not np.array_equal(f['gene_ids'],universe):raise ValueError('Frozen gene universe mismatch')
            a=views[rep].T[:,f['selected']]
            missing[rep]=int(np.isnan(a).sum())
            a=np.where(np.isfinite(a),a,f['medians'][f['selected']])
            panels[rep]=(a-f['mean'])/f['scale']
        return panels,dict(selected_gene_missing=missing,preprocessing='Frozen equal-study masked quantile reference → log2 → median probe-to-GeneID mapping → fixed 500-gene panel, training medians/scaling; rank representation for D/PLS.')

    def predict_values(self,method,panels):
        if not isinstance(method,str) or method not in self.models:raise ValueError('Unsupported frozen model')
        m=self.models[method]
        if method=='median_null':return np.full(len(panels['log2']),m['value'])
        x=panels[m['representation']];spec=m['estimator']
        if spec['kind']=='neural':
            a=x.astype(np.float32)
            for layer in spec['layers']:
                a=a@np.asarray(layer['weight'],dtype=np.float32).T+np.asarray(layer['bias'],dtype=np.float32)
                if layer['relu']:a=np.maximum(a,0)
            result=a.ravel().astype(float)
        else:result=numeric_predict(spec,x)
        result=result*m['target_scale']+m['target_mean']
        if not np.isfinite(result).all():raise ValueError('Nonfinite frozen prediction')
        return result

    def predict(self,body,method):
        x,qc=self.validate(body)
        if qc['blockers']:raise ValueError('Input is incompatible; no dose estimate')
        panels,_=self.panels(x)
        value=float(self.predict_values(method,panels)[0]);m=self.models[method]
        release, release_sha = self.model_release(method)
        return dict(estimate_gy=value,model=method,seed=m.get('seed'),model_sha256=m['source_sha256'],
                    release=release,release_sha256=release_sha,
                    input_sha256=digest(encode(body)),sample_id=body['sample_id'],qc=qc,
                    outside_training_dose_range=not 0<=value<=8,individual_prediction_interval=None,
                    limitations=LIMITATIONS)
