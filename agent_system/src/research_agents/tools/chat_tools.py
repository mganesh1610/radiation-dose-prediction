"""Bind the existing verified facades to trusted invocation context."""
import asyncio
from ..policies.run_scope import current_run
from ..tools.query_sql import digest
from ..policies.request_contract import bind_plan
from ..tools.sample_queries import IntegrityError


class ChatTools:
    def __init__(self, queries_factory, expression_reader, guard, admission):
        self.queries_factory,self.expression_reader,self.guard,self.admission = queries_factory,expression_reader,guard,admission

    async def execute(self, kind, plan):
        s=await self.guard.authorize()
        await asyncio.to_thread(self.guard.ledger.reserve,s.principal.user_id,s.run_id,s.fence,"tool")
        allowed_studies={member[1] for member in self.admission.core.members}
        plan, refusal=bind_plan(kind,plan,s.contract,allowed_studies)
        if refusal:return refusal
        if kind=="expression":
            genes=set(plan["gene_ids"])
            if not set(plan['sample_ids']).issubset({m[0] for m in self.admission.core.members}):
                return {'status':'unsupported','reason':'One or more sample IDs are outside the connected core expression set. No expression values were retrieved.'}
            if not genes.issubset({g for _,g in self.admission.features}):
                return {'status':'unsupported','reason':'One or more NCBI Gene IDs have no approved mapping in this expression set. No aliases or substitute genes were used.'}
            probes=sum(g in genes for p,g in self.admission.features)
            if probes*len(plan["sample_ids"])>200:
                return {"status":"unsupported","reason":"Chat supports at most 200 expression cells; select fewer genes or profiles."}
            call=lambda:self.expression_reader.run(s.principal,s.workspace,plan)
        else:
            call=lambda:self.queries_factory().run(s.principal,s.workspace,plan)
        result=await asyncio.to_thread(call)
        await self.guard.authorize()
        if kind=='query' and plan['metric'] in {'sample_count','study_group_count'}:
            expected='profiles' if plan['metric']=='sample_count' else 'study_local_donor_groups'
            if result['answer'].get('unit')!=expected:
                raise IntegrityError('Returned count unit differs from the requested metric')
        evidence_id="ev-"+digest(result)[:24]
        receipt=dict(result["receipt"],evidence_id=evidence_id,tool=kind,verified_result=result['answer'])
        await asyncio.to_thread(self.guard.ledger.progress,s.principal.user_id,s.run_id,s.fence,"verified_"+kind,receipt)
        # Keep protected access/billing fields out of prompts; receipt remains in the ledger.
        return {"evidence_id":evidence_id,"answer":result["answer"],
                "scope":result.get("scope",{"release_id":plan["release_id"]}),
                "source_manifest_sha256":result["receipt"]["release_manifest_sha256"]}

    async def query(self, plan):
        return await self.execute("query",plan)

    async def expression(self, plan):
        return await self.execute("expression",plan)
