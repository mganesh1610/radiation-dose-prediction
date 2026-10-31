"""Descriptive expression-by-dose workflow over authorized immutable source reads."""
import asyncio
import base64
import csv
import hashlib
import io
import math
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ..contracts.analysis import AnalysisPlan, bind_analysis
from .query_sql import digest
from .sample_queries import IntegrityError

VERSION = 'expression-by-recorded-dose-v1'
LIMITATIONS = ('Exploratory source-scale description, not a dose prediction or causal/differential-expression test. '
    'Background-corrected linear intensities; between-array normalization is deferred to training folds. '
    'Profiles from the same study-local donor group are repeated observations. '
    'Probes remain separate. Finite undetected values are retained and marked; missing expression or dose is excluded from plotted summaries without imputation.')
STEPS = ('authorize_and_bind_current_question', 'read_verified_study_records', 'read_verified_expression_batches',
         'validate_exact_join_and_representation', 'parallel_summary_and_plot', 'reauthorize', 'persist_immutable_artifact', 'record_receipt')
PROCEDURE_HASH = digest([VERSION, STEPS])
PLOT_LOCK = threading.Lock()


def validate_inputs(plan, admission, metadata, slices):
    records = metadata['answer']['records']
    expected = sorted(m for m in admission.core.members if m[1] == plan.study_id)
    if (not 1 <= len(expected) <= 145 or [(r['sample_id'],r['study_id'],r['group_id']) for r in records] != expected
            or metadata['receipt']['release_manifest_sha256'] != admission.core.manifest_sha256):
        raise IntegrityError('Study population or release mismatch')
    probes = [(p,g) for p,g in admission.features if g == plan.gene_id]
    if not 1 <= len(probes) <= 4 or len(probes)*len(records) > 580:
        raise ValueError('Select a mapped gene with at most four probes')
    if any(r['release_id'] != admission.core.release_id or r['organism'] != 'Homo sapiens'
           or not r['tissue'] or r['dose_gy'] is not None and (not math.isfinite(r['dose_gy']) or r['dose_gy'] < 0)
           for r in records):
        raise IntegrityError('Inconsistent metadata')
    by_id = {r['sample_id']:r for r in records}
    seen = set(); rows = []
    for result in slices:
        a, receipt = result['answer'],result['receipt']
        if (receipt['matrix_sha256'] != admission.sha256 or receipt['mapping_sha256'] != admission.mapping_sha256
                or receipt['release_manifest_sha256'] != admission.core.manifest_sha256
                or receipt['admission_sha256'] != admission.fingerprint
                or a['release_id'] != admission.core.release_id or a['expression_set_id'] != admission.expression_set_id
                or [(f['probe_id'],f['gene_id']) for f in a['features']] != probes
                or any(a[k] != getattr(admission,k) for k in ('representation','scale','normalization_status','processing_scope'))
                or a['aggregation'] != 'none' or a['axis_order'] != 'probe_by_sample'):
            raise IntegrityError('Expression version, scale or probe mismatch')
        samples = a['samples']; n = len(samples)
        if (len(a['values']) != len(probes) or len(a['detected']) != len(probes)
                or any(len(v) != n for v in a['values']+a['detected'])):
            raise IntegrityError('Expression axes mismatch')
        for j,s in enumerate(samples):
            sid=s['sample_id']; record=by_id.get(sid)
            if sid in seen or record is None or (s['study_id'],s['group_id']) != (record['study_id'],record['group_id']):
                raise IntegrityError('Duplicate, missing or inconsistent profile join')
            seen.add(sid)
            for i,(probe,gene) in enumerate(probes):
                value=a['values'][i][j]; detected=a['detected'][i][j]
                if type(detected) is not bool or value is not None and (type(value) not in (float,int) or not math.isfinite(value) or value < 0):
                    raise IntegrityError('Invalid expression or detection value')
                rows.append({**record, 'gene_id':gene,'probe_id':probe,'expression':value,'detected':detected})
    if seen != set(by_id):raise IntegrityError('Incomplete expression population')
    return sorted(rows,key=lambda r:(r['probe_id'],r['sample_id']))


def donor_values(rows):
    grouped=defaultdict(list)
    for r in rows:
        if r['expression'] is not None and r['dose_gy'] is not None:
            grouped[(r['probe_id'],r['dose_gy'],r['group_id'])].append(r['expression'])
    return {k:math.fsum(v)/len(v) for k,v in grouped.items()}


def summarize(rows):
    donors=donor_values(rows); summaries=[]
    for probe in sorted({r['probe_id'] for r in rows}):
        for dose in sorted({r['dose_gy'] for r in rows if r['dose_gy'] is not None}):
            selected=[r for r in rows if r['probe_id']==probe and r['dose_gy']==dose]
            values=[v for (p,d,g),v in donors.items() if (p,d)==(probe,dose)]
            summaries.append({'probe_id':probe,'recorded_dose_gy':dose,'profiles':len(selected),
                'study_local_donor_groups':len({r['group_id'] for r in selected}),
                'usable_profiles':sum(r['expression'] is not None for r in selected),
                'usable_donor_groups':len(values),
                'undetected_finite_profiles':sum(r['expression'] is not None and not r['detected'] for r in selected),
                'donor_balanced_mean':math.fsum(values)/len(values) if values else None})
    return {'profiles':len({r['sample_id'] for r in rows}),
        'study_local_donor_groups':len({r['group_id'] for r in rows}),
        'probe_count':len({r['probe_id'] for r in rows}), 'measurement_cells':len(rows),
        'missing_expression_cells':sum(r['expression'] is None for r in rows),
        'missing_dose_profiles':len({r['sample_id'] for r in rows if r['dose_gy'] is None}),
        'plotted_cells':sum(r['expression'] is not None and r['dose_gy'] is not None for r in rows),
        'tissues':sorted({r['tissue'] for r in rows}),
        'time_after_exposure_hours':sorted({r['time_after_exposure_hours'] for r in rows if r['time_after_exposure_hours'] is not None}),
        'dose_statuses':sorted({r['dose_status'] for r in rows}), 'by_dose':summaries}


def render_plot(plan, rows):
    # No pyplot/global GUI, random jitter, fitted curves or statistical intervals.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    probes=sorted({r['probe_id'] for r in rows}); donors=donor_values(rows)
    with PLOT_LOCK:
        figure=Figure(figsize=(8.8,3.7*len(probes)+1),dpi=120,facecolor='#f8faf8')
        FigureCanvasAgg(figure)
        for index,probe in enumerate(probes):
            ax=figure.add_subplot(len(probes),1,index+1);ax.set_facecolor('#f8faf8')
            data=[r for r in rows if r['probe_id']==probe and r['dose_gy'] is not None and r['expression'] is not None]
            for flag,marker,color,label in [(True,'o','#9ab8ab','Detected profile'),(False,'x','#b4652a','Undetected finite profile')]:
                group=[r for r in data if r['detected']==flag]
                if group:ax.scatter([r['dose_gy'] for r in group],[r['expression'] for r in group],s=18,marker=marker,color=color,alpha=.65,label=label)
            points=sorted((d,v) for (p,d,g),v in donors.items() if p==probe)
            if points:ax.scatter([d for d,v in points],[v for d,v in points],s=24,facecolors='none',edgecolors='#265e4a',label='Donor-group mean')
            means=[(d,math.fsum(v for dose,v in points if dose==d)/sum(dose==d for dose,v in points)) for d in sorted({d for d,v in points})]
            if means:ax.scatter([d for d,v in means],[v for d,v in means],marker='D',s=44,color='#102e24',label='Donor-balanced mean')
            if not data:ax.text(.5,.5,'No complete expression–dose pairs',transform=ax.transAxes,ha='center')
            ax.set_title(f'{plan.study_id} · NCBI Gene {plan.gene_id} · {probe}',loc='left',fontsize=12,pad=14)
            ax.set_xlabel('Recorded laboratory dose (Gy)');ax.set_ylabel('Background-corrected\nlinear intensity')
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16)
            if data:ax.legend(fontsize=7,loc='best')
        figure.subplots_adjust(left=.12,right=.97,top=.88,bottom=.18,hspace=.65)
        figure.text(.12,.04,'Exploratory source scale · between-array normalization deferred\nRepeated profiles are not independent donors · missing pairs excluded · research use only',fontsize=8,color='#4a5e54')
        output=io.BytesIO();figure.savefig(output,format='png',metadata={'Software':'Biodesign '+VERSION})
        raw=output.getvalue()
    if len(raw)>220_000:raise ValueError('Plot exceeds artifact limit')
    return raw


def csv_data(rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader()
    # Guard spreadsheet formula interpretation of textual source annotations.
    writer.writerows({k: ("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v) for k,v in row.items()} for row in rows)
    return stream.getvalue()


class AnalysisTool:
    def __init__(self, queries_factory, expression_reader, admission, guard, artifacts):
        self.queries_factory,self.reader,self.admission,self.guard,self.artifacts=queries_factory,expression_reader,admission,guard,artifacts
        self.slots=threading.BoundedSemaphore(2)

    def prepare(self, principal, workspace, plan):
        a=self.admission
        query={'version':1,'release_id':a.core.release_id,'population':'approved_core_release_v1',
               'dose_kind':'recorded','metric':'study_records','filters':{'study_id':plan.study_id}}
        metadata=self.queries_factory().run(principal,workspace,query)
        ids=[r['sample_id'] for r in metadata['answer']['records']]
        slices=[]
        for start in range(0,len(ids),50):
            slices.append(self.reader.run(principal,workspace,{'version':1,'release_id':a.core.release_id,
                'expression_set_id':a.expression_set_id,'sample_ids':ids[start:start+50],
                'gene_namespace':'NCBI_GeneID','gene_ids':[plan.gene_id],'aggregation':'none'}))
        rows=validate_inputs(plan,a,metadata,slices)
        # Both independent branches read the same validated snapshot; no shared writes.
        with ThreadPoolExecutor(max_workers=2) as pool:
            summary_future=pool.submit(summarize,rows);plot_future=pool.submit(render_plot,plan,rows)
            summary,plot=summary_future.result(),plot_future.result()
        csv_text=csv_data(rows)
        parameters={**plan.model_dump(),'release_id':a.core.release_id,'expression_set_id':a.expression_set_id,
            'population':'all_approved_profiles_in_selected_study','scale':a.scale,'probe_aggregation':'none',
            'summary':'mean_within_donor_and_dose_then_equal_weight_mean_across_donors'}
        provenance={'procedure_id':VERSION,'procedure_sha256':PROCEDURE_HASH,'parameters':parameters,
            'analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parameters_sha256':digest(parameters),'input_rows_sha256':digest(rows),'release_manifest_sha256':a.core.manifest_sha256,
            'matrix_sha256':a.sha256,'mapping_sha256':a.mapping_sha256,
            'metadata_receipt':metadata['receipt'],'expression_receipts':[s['receipt'] for s in slices]}
        artifact={'schema_version':1,'png_base64':base64.b64encode(plot).decode(), 'csv':csv_text,
            'png_sha256':hashlib.sha256(plot).hexdigest(),'csv_sha256':hashlib.sha256(csv_text.encode()).hexdigest(),
            'summary':summary,'provenance':provenance,'limitations':LIMITATIONS}
        return artifact

    async def run(self,study_id,gene_id):
        plan=AnalysisPlan(study_id=study_id,gene_id=gene_id);scope=await self.guard.authorize()
        await asyncio.to_thread(self.guard.ledger.reserve,scope.principal.user_id,scope.run_id,scope.fence,'tool')
        refusal=bind_analysis(plan,scope.contract,{m[1] for m in self.admission.core.members})
        if refusal:return refusal
        probes=sum(g==gene_id for p,g in self.admission.features)
        if not 1<=probes<=4:
            return {'status':'unsupported','reason':'The numeric gene ID must map to one to four approved probes; no guessed mapping or probe pooling is performed.'}
        if not self.slots.acquire(blocking=False):raise RuntimeError('Analysis is busy')
        try:
            artifact=await asyncio.to_thread(self.prepare,scope.principal,scope.workspace,plan)
            await self.guard.authorize()
            ref=await asyncio.to_thread(self.artifacts.save,scope,artifact)
            await self.guard.authorize()
            answer={'status':'ok','study_id':study_id,'gene_id':gene_id,'gene_namespace':'NCBI_GeneID',
                'dose_kind':'recorded','dose_unit':'Gy','scale':self.admission.scale,
                **artifact['summary'],'artifact':ref,'limitations':LIMITATIONS}
            receipt={**artifact['provenance'],'tool':'analysis','verified_result':answer,
                'artifact':ref,'result_sha256':digest(answer),'completed_at_utc':datetime.now(timezone.utc).isoformat()}
            evidence_id='ev-'+digest(receipt)[:24];receipt['evidence_id']=evidence_id
            await asyncio.to_thread(self.guard.ledger.progress,scope.principal.user_id,scope.run_id,scope.fence,'verified_analysis',receipt)
            return {'evidence_id':evidence_id,'answer':answer}
        finally:self.slots.release()
