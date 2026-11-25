"""Owner-isolated, bounded upload records and model-bound compatibility decisions."""
import json,re,time,math
from cloud_io import PREFIX,Conflict
from dispatch_control import BudgetExceeded
from frozen_inference import encode,digest

BASE=PREFIX+'control/inference/'
MAX_UPLOADS=40  # Lifetime bound for this release, not replenished monthly.

class PredictionService:
    def __init__(self,cloud,engine,owner,reviewer):
        self.cloud,self.engine,self.owner,self.reviewer=cloud,engine,owner,reviewer

    def key(self,upload_id):
        if not isinstance(upload_id,str) or not re.fullmatch('[a-f0-9]{64}',upload_id):raise ValueError('Invalid upload ID')
        return BASE+'uploads/'+upload_id+'/'

    def registry(self):
        try:return self.cloud.read_json(BASE+'registry.json')
        except FileNotFoundError:return {'schema_version':1,'uploads':{}},0

    def admit(self,uid):
        for _ in range(4):
            state,generation=self.registry()
            if uid in state['uploads']:
                if state['uploads'][uid]['owner']!=self.owner:raise PermissionError('Upload is not owned by this account')
                return
            if len(state['uploads'])>=MAX_UPLOADS:raise BudgetExceeded('This release has reached its 40-upload storage allowance')
            state['uploads'][uid]=dict(owner=self.owner,created_at=time.time())
            try:self.cloud.state(BASE+'registry.json',state,generation);return
            except Conflict:continue
        raise Conflict('Concurrent upload admission')

    def owned(self,uid,review=False):
        self.key(uid)
        state,_=self.registry();entry=state['uploads'].get(uid)
        if not entry or (entry['owner']!=self.owner and not (review and not self.reviewer)):raise FileNotFoundError()
        return entry

    def list(self,owned_only=False):
        state,_=self.registry()
        return [{'upload_id':k,**v} for k,v in state['uploads'].items() if v['owner']==self.owner or (not owned_only and not self.reviewer)]

    def upload(self,body,*,model=None):
        _, release_sha = self.engine.model_release(model)
        _,qc=self.engine.validate(body);payload=encode(body);sha=digest(payload)
        uid=digest((self.owner+'\0'+sha).encode());base=self.key(uid);self.admit(uid)
        # Payload is immutable; a lost response can safely be retried with identical input.
        self.cloud.immutable(base+'input.json',payload)
        curated=self.engine.curated.get(sha)
        overlap = curated.get('training_overlap', True) if curated else 'unknown'
        record=dict(schema_version=1,upload_id=uid,input_sha256=sha,sample_id=body['sample_id'],
                    metadata=body['metadata'],qc=qc,curated=bool(curated),
                    training_overlap=overlap,
                    status='incompatible' if qc['blockers'] else 'pending_compatibility_review')
        try:
            self.cloud.put(base+'record.json', encode(record), generation=0, content_type='application/json')
        except Conflict:
            stored, _ = self.cloud.read_json(base+'record.json')
            # A previously submitted profile may now be in the authenticated example catalog.
            if any(stored.get(k) != v for k, v in record.items() if k not in ('curated', 'training_overlap')):
                raise RuntimeError('Stored upload record integrity failure')
        if curated and not qc['blockers']:
            self._approve_curated(uid,sha,curated,release_sha)
        return self.get(uid,model=model)

    def _approve_curated(self,uid,sha,example,release_sha):
        overlap = example.get('training_overlap',True)
        kind = 'development' if overlap else 'held-out'
        decision = dict(input_sha256=sha,release_sha256=release_sha,decision='approved',
                        reviewed_by='curated-release-audit',evidence=f"Exact hash match to audited {kind} profile {example['id']}",
                        scope='Research functionality demo; overlaps training' if overlap else example['purpose'],
                        training_overlap=overlap)
        name = self.key(uid)+'decision-'+release_sha+'.json'
        try: self.cloud.read_json(name)
        except FileNotFoundError:
            try: self.cloud.put(name,encode(decision),generation=0,content_type='application/json')
            except Conflict: pass

    def check_stored(self,uid,*,model,review=False):
        record = self.get(uid,review,model=model)
        example = self.engine.curated.get(record['input_sha256'])
        if example and not record['qc']['blockers'] and 'compatibility' not in record:
            _, release_sha = self.engine.model_release(model)
            self._approve_curated(uid,record['input_sha256'],example,release_sha)
            return self.get(uid,review,model=model)
        return record

    def get(self,uid,review=False,*,model=None):
        _, release_sha = self.engine.model_release(model)
        self.owned(uid,review);record,_=self.cloud.read_json(self.key(uid)+'record.json')
        curated = self.engine.curated.get(record['input_sha256'])
        if curated:
            record = dict(record, curated=True, training_overlap=curated.get('training_overlap', True))
        try:decision,_=self.cloud.read_json(self.key(uid)+'decision-'+release_sha+'.json')
        except FileNotFoundError:decision=None
        if decision:
            if decision['input_sha256']!=record['input_sha256'] or decision['release_sha256']!=release_sha:raise RuntimeError('Decision binding mismatch')
            record=dict(record,status=decision['decision'],compatibility=decision,training_overlap=decision['training_overlap'])
        return record

    def review(self,uid,body,*,model=None):
        _, release_sha = self.engine.model_release(model)
        if self.reviewer:raise PermissionError('Researcher compatibility review required')
        record=self.get(uid,review=True,model=model)
        if not isinstance(body,dict) or set(body)!={'decision','evidence','training_overlap','checks'}:raise ValueError('Expected a documented compatibility decision')
        if body['decision'] not in ('approved','rejected'):raise ValueError('Invalid decision')
        if not isinstance(body['evidence'],str) or not 40<=len(body['evidence'])<=2000:raise ValueError('Record the reviewed source, assay adapter and QC rationale (40–2000 characters)')
        if body['training_overlap'] not in ('yes','no','unknown'):raise ValueError('Record training overlap')
        checks=body['checks']
        expected={'assay','gene_mapping','normalization','technical_flags','tissue_exposure_time','missingness'}
        if not isinstance(checks,dict) or set(checks)!=expected or any(type(v)!=bool for v in checks.values()):raise ValueError('All compatibility checks must be explicitly recorded')
        if body['decision']=='approved' and (not all(checks.values()) or record['qc']['blockers']):raise ValueError('Unresolved compatibility checks prevent approval')
        decision=dict(input_sha256=record['input_sha256'],release_sha256=release_sha,
                      reviewed_by=self.owner,**body)
        name=self.key(uid)+'decision-'+release_sha+'.json'
        try:previous,generation=self.cloud.read_json(name)
        except FileNotFoundError:previous,generation=None,0
        if previous!=decision:
            self.cloud.immutable(self.key(uid)+'review-audit-'+digest(encode(decision))+'.json',encode(decision))
            self.cloud.state(name,decision,generation)
        return self.get(uid,review=True,model=model)

    def predict(self,uid,method):
        _, release_sha = self.engine.model_release(method)
        if method is None: raise ValueError('A frozen model is required')
        record=self.get(uid,model=method)
        if record['status']!='approved':return dict(status=record['status'],record=record,estimate_gy=None,explanation='No dose estimate: compatibility approval for this input and frozen release is required.')
        result_name=self.key(uid)+'result-'+release_sha+'-'+method+'.json'
        try:result=self.cloud.read_json(result_name)[0]
        except FileNotFoundError:
            body,_=self.cloud.read_json(self.key(uid)+'input.json')
            if digest(encode(body))!=record['input_sha256']:raise RuntimeError('Stored input integrity failure')
            result=self.engine.predict(body,method)
            result.update(status='completed',upload_id=uid,training_overlap=record['training_overlap'],
                          compatibility=record['compatibility'],input_provenance=record['metadata'])
            # Recheck the decision before publishing a new immutable result.
            self._same_approval(uid,record,method)
            self.cloud.immutable(result_name,encode(result))
        expected={'status':'completed','upload_id':uid,'model':method,
                  'input_sha256':record['input_sha256'],'sample_id':record['sample_id'],
                  'release_sha256':release_sha,
                  'model_sha256':self.engine.models[method]['source_sha256']}
        if (any(result.get(k)!=v for k,v in expected.items()) or
                type(result.get('estimate_gy')) not in (int,float) or not math.isfinite(result['estimate_gy'])):
            raise RuntimeError('Stored prediction binding/integrity failure')
        self._same_approval(uid,record,method)
        # A previously cached numeric result can be reused after a new valid
        # approval, but the current compatibility decision must travel with it.
        comparison = None
        example = self.engine.curated.get(record['input_sha256'])
        if (example and example['id'] == record['sample_id'] and example.get('study')
                and example['study'] == record['metadata'].get('study')):
            observed = example.get('observed_dose_gy')
            if type(observed) in (int, float) and math.isfinite(observed) and observed >= 0:
                source = 'held_out_evaluation_example' if example.get('training_overlap') is False else 'frozen_release_example'
                comparison = dict(study=example['study'], source=source,
                                  observed_dose_gy=observed,
                                  estimate_minus_observed_gy=result['estimate_gy'] - observed)
        return dict(result, compatibility=record['compatibility'], training_overlap=record['training_overlap'],
                    dose_comparison=comparison)

    def _same_approval(self,uid,record,method):
        current=self.get(uid,model=method)
        if current['status']!='approved' or current.get('compatibility')!=record.get('compatibility'):
            raise RuntimeError('Compatibility decision changed; reload before prediction')
