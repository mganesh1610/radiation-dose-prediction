"""Small fabricated failure fixtures; actual-source parity is a separate saved pilot."""
import asyncio
import base64
import copy
import hashlib
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from research_agents.contracts.analysis import AnalysisPlan,bind_analysis
from research_agents.policies.request_contract import compile_contract
from research_agents.tools.scientific_analysis import summarize,render_plot,validate_inputs,csv_data,AnalysisTool
from research_agents.tools.sample_queries import validate_answer,IntegrityError
from research_agents.contracts.queries import QueryPlan
from research_agents.tools.query_sql import ApprovedRelease,digest
from research_agents.persistence.analysis_artifacts import AnalysisArtifacts
from research_agents.persistence.control_store import MemoryControlStore


def rows():
    return [{'release_id':'core_test','sample_id':'GSM'+str(i+1),'study_id':'GSE1','group_id':g,'dose_gy':d,
             'dose_status':'source_label','time_after_exposure_hours':24.,'organism':'Homo sapiens',
             'tissue':'whole_blood','gene_id':'7157','probe_id':'probe1','expression':v,'detected':flag}
        for i,(g,d,v,flag) in enumerate([('d1',0.,1.,True),('d1',0.,3.,False),('d2',0.,10.,True),
                                       ('d2',2.,None,False),('d3',None,9.,True)])]


def fixture():
    rr=rows();core=ApprovedRelease('core_test','a'*64,tuple((r['sample_id'],r['study_id'],r['group_id']) for r in rr),
        (('samples','s1'),('sample_annotations','a1')))
    a=SimpleNamespace(core=core,features=[('probe1','7157')],sha256='b'*64,mapping_sha256='c'*64,fingerprint='d'*64,
        expression_set_id='core_test_linear',representation='core_common_probes_background_corrected',scale='linear_intensity',
        normalization_status='between_array_normalization_deferred_to_training_fold',processing_scope='per_sample')
    meta={'answer':{'records':[{k:v for k,v in r.items() if k not in ('gene_id','probe_id','expression','detected')} for r in rr]},
          'receipt':{'release_manifest_sha256':core.manifest_sha256}}
    answer={k:getattr(a,k) for k in ('expression_set_id','representation','scale','normalization_status','processing_scope')}
    answer.update(release_id='core_test',aggregation='none',axis_order='probe_by_sample',features=[{'probe_id':'probe1','gene_id':'7157'}],
        samples=[{k:r[k] for k in ('sample_id','study_id','group_id')} for r in rr],values=[[r['expression'] for r in rr]],detected=[[r['detected'] for r in rr]])
    slices=[{'answer':answer,'receipt':{'matrix_sha256':a.sha256,'mapping_sha256':a.mapping_sha256,
             'admission_sha256':a.fingerprint,'release_manifest_sha256':core.manifest_sha256}}]
    return a,meta,slices


def test_means_weight_donors_not_repeats_and_do_not_fill_missing():
    result=summarize(rows());zero=result['by_dose'][0]
    assert zero['donor_balanced_mean']==6. # mean(mean(1,3),10), not 14/3
    assert zero['undetected_finite_profiles']==1 and zero['usable_donor_groups']==2
    assert result['profiles']==5 and result['study_local_donor_groups']==3
    assert result['plotted_cells']==3 and result['missing_expression_cells']==1 and result['missing_dose_profiles']==1
    assert result['by_dose'][1]['donor_balanced_mean'] is None
    second=[dict(r,probe_id='probe2',expression=None if r['expression'] is None else 2*r['expression']) for r in rows()]
    summary=summarize(rows()+second)
    assert summary['probe_count']==2 and summary['profiles']==5
    assert summary['by_dose'][2]['donor_balanced_mean']==12.


def test_valid_join_and_repeatable_png():
    a,m,s=fixture();plan=AnalysisPlan(study_id='GSE1',gene_id='7157')
    assert validate_inputs(plan,a,m,s)==rows()
    raw=render_plot(plan,rows())
    assert raw==render_plot(plan,rows())
    image=Image.open(io.BytesIO(raw));assert image.format=='PNG' and image.width>900


@pytest.mark.parametrize('mutation',['duplicate','missing','group','scale','matrix','mapping','manifest','gene','shape','nan','detection'])
def test_join_corruption_stops_plot(mutation):
    a,m,s=fixture();x=s[0]['answer'];r=s[0]['receipt']
    if mutation=='duplicate':x['samples'][1]=x['samples'][0]
    elif mutation=='missing':m['answer']['records'].pop()
    elif mutation=='group':x['samples'][0]['group_id']='other'
    elif mutation=='scale':x['scale']='log2'
    elif mutation=='matrix':r['matrix_sha256']='e'*64
    elif mutation=='mapping':r['mapping_sha256']='e'*64
    elif mutation=='manifest':m['receipt']['release_manifest_sha256']='e'*64
    elif mutation=='gene':x['features'][0]['gene_id']='999'
    elif mutation=='shape':x['values'][0].pop()
    elif mutation=='nan':x['values'][0][0]=float('nan')
    elif mutation=='detection':x['detected'][0][0]=1
    with pytest.raises(IntegrityError):validate_inputs(AnalysisPlan(study_id='GSE1',gene_id='7157'),a,m,s)


@pytest.mark.parametrize('text',[
    'Plot TP53 expression by dose in GSE1','Plot Gene ID 7157 in GSE1 and GSE2',
    'Plot Gene ID 7157 in GSE1 for GSM1', 'Plot normalized Gene ID 7157 in GSE1',
    'Plot Gene ID 7157 in GSE1 only at 0 Gy','Plot log2 Gene ID 7157 in GSE1',
    'Predict dose using Gene ID 7157 in GSE1','Plot Gene ID 7157 in GSE1 excluding controls',
    'How many samples in GSE1 have Gene ID 7157?'])
def test_current_question_cannot_be_silently_broadened(text):
    assert bind_analysis(AnalysisPlan(study_id='GSE1',gene_id='7157'),compile_contract(text),{'GSE1'})


def test_supported_question_and_outside_study():
    p=AnalysisPlan(study_id='GSE1',gene_id='7157');c=compile_contract('Plot NCBI Gene ID 7157 expression by recorded dose in GSE1.')
    assert bind_analysis(p,c,{'GSE1'}) is None
    assert bind_analysis(p,c,{'GSE2'})['status']=='unsupported'


def test_csv_formula_guard_and_missing_cell():
    r=rows();r[0]['dose_status']='=bad()';data=csv_data(r)
    assert "'=bad()" in data and ',None,' not in data


def test_study_template_requires_exact_population():
    a,m,_=fixture();answer=m['answer']['records'];n=len(answer)
    envelope={k:n for k in ('sample_rows','unique_samples','annotation_rows','unique_annotations','joined_rows')}
    import json
    envelope.update(unknown_samples=0,unknown_annotations=0,answer_json=json.dumps(answer))
    p=QueryPlan(version=1,metric='study_records',population='approved_core_release_v1',release_id='core_test',dose_kind='recorded',filters={'study_id':'GSE1'})
    assert validate_answer(envelope,p,a.core)['records']==answer
    envelope['answer_json']=json.dumps(answer[:-1]+answer[:1])
    with pytest.raises(IntegrityError):validate_answer(envelope,p,a.core)


def test_artifacts_are_immutable_owned_complete_and_expiring():
    store=MemoryControlStore();artifacts=AnalysisArtifacts(store)
    scope=SimpleNamespace(principal=SimpleNamespace(user_id='alice'),run_id='run-'+'a'*32,chat_id='chat-'+'b'*32)
    payload={'png_sha256':'a'*64,'csv_sha256':'b'*64};ref=artifacts.save(scope,payload)
    assert artifacts.save(scope,payload)==ref
    run={'status':'completed','chat_id':scope.chat_id,'receipts':[{'tool':'analysis','artifact':ref}]}
    class Service:
        def __init__(self):self.ledger=self;self.expired=False;self.revoked=False
        async def get_run(self,p,r):
            if p.user_id!='alice' or self.expired:raise FileNotFoundError()
            return run
        def chat(self,u,c):
            if self.expired:raise FileNotFoundError()
        def authorize(self,p):
            if self.revoked:raise PermissionError()
    service=Service()
    def get(who='alice'):
        return asyncio.run(artifacts.get(service,SimpleNamespace(user_id=who),scope.run_id,ref['artifact_id']))
    assert get()==payload
    with pytest.raises(FileNotFoundError):get('bob')
    run['status']='failed'
    with pytest.raises(FileNotFoundError):get()
    run['status']='completed';service.expired=True
    with pytest.raises(FileNotFoundError):get()
    service.expired=False;service.revoked=True
    with pytest.raises(PermissionError):get()
    service.revoked=False
    key=f'artifacts/{scope.run_id}/{ref["artifact_id"]}.json'
    record,generation=store.read(key);record['payload']['csv_sha256']='bad';store.write(key,record,generation)
    with pytest.raises(RuntimeError):get()
