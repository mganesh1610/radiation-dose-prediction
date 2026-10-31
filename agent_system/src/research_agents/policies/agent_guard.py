"""Shared ledger guards across both reasoning roles; opaque traces only."""
import asyncio
import re
from google.genai import types
from google.adk.models.llm_response import LlmResponse
from .run_scope import current_run
from .response_contract import clarify_response
from .model_context import fit_protocol_context, request_bytes
from .protocol_disclosure import verified_protocol_disclosure


class AgentGuard:
    def __init__(self, ledger, access, release_id, *, allowed_studies=None):
        self.ledger,self.access,self.release_id=ledger,access,release_id
        self.allowed_studies=None if allowed_studies is None else frozenset(allowed_studies)

    async def authorize(self):
        s=current_run.get()
        await asyncio.to_thread(self.access.authorize,s.principal,s.workspace,self.release_id)
        await asyncio.to_thread(self.ledger.check,s.principal.user_id,s.run_id,s.fence)
        return s

    async def before_model(self, callback_context, llm_request):
        s=await self.authorize()
        # ADK A2A annotates parts for transport provenance. The pinned Enterprise
        # Gemini converter rejects part_metadata; retain it in Sessions, remove
        # only from a copied outbound model request (not from scientific payloads).
        llm_request.contents=[content.model_copy(deep=True) for content in llm_request.contents]
        for content in llm_request.contents:
            for part in content.parts or []:
                part.part_metadata=None
        if request_bytes(llm_request)>16000:
            run=await asyncio.to_thread(self.ledger.check,s.principal.user_id,s.run_id,s.fence)
            fit_protocol_context(llm_request,{r['evidence_id'] for r in run['receipts']
                                             if r.get('tool')=='protocol'})
        size=request_bytes(llm_request)
        await asyncio.to_thread(self.ledger.reserve,s.principal.user_id,s.run_id,s.fence,"model",input_bytes=size)
        # Bounds also apply after an agent/remote call modifies the request config.
        llm_request.config.max_output_tokens=1500
        return None

    async def after_model(self, callback_context, llm_response):
        s=await self.authorize()
        usage=llm_response.usage_metadata
        if usage:
            await asyncio.to_thread(self.ledger.usage,s.principal.user_id,s.run_id,s.fence,
                usage.prompt_token_count,(usage.candidates_token_count or 0)+(usage.thoughts_token_count or 0))
        parts=llm_response.content.parts if llm_response.content else []
        if parts and not llm_response.partial and not any(p.function_call for p in parts):
            text='\n'.join(p.text for p in parts if p.text and not p.thought)
            if text:
                run=await asyncio.to_thread(self.ledger.check,s.principal.user_id,s.run_id,s.fence)
                evidence={r['evidence_id'] for r in run['receipts']}
                passages={p['chunk_id'] for r in run['receipts'] if r.get('tool')=='protocol'
                          for p in r['verified_result']['passages']}
                cited_passages=set(re.findall(r'\bpc-[a-zA-Z0-9_-]+',text))
                cited=set(re.findall(r'\bev-[a-zA-Z0-9_-]+',text))
                if cited-evidence or cited_passages-passages:
                    # One bounded recovery for the observed repeated-question
                    # failure. Never accept old receipts as current evidence.
                    if (getattr(callback_context,'agent_name',None)=='research_coordinator'
                            and run['model_calls']==1 and run['tool_calls']==0 and not evidence):
                        state=await asyncio.to_thread(self.ledger.read)
                        previous_receipts=[r for old in state['runs'].values()
                                  if old['run_id']!=s.run_id and old['owner']==s.principal.user_id
                                  and old['chat_id']==s.chat_id and old['status']=='completed'
                                  for r in old['receipts']]
                        previous={r['evidence_id'] for r in previous_receipts}
                        previous_passages={p['chunk_id'] for r in previous_receipts if r.get('tool')=='protocol'
                                           for p in r['verified_result']['passages']}
                        if ((cited or cited_passages) and cited.issubset(previous)
                                and cited_passages.issubset(previous_passages)):
                            return LlmResponse(content=types.Content(role='model',parts=[
                                types.Part(function_call=types.FunctionCall(name='transfer_to_agent',
                                    args={'agent_name':'evidence_service'}))]))
                    if cited_passages-passages:
                        raise RuntimeError('Answer contains an unverified protocol passage identifier')
                    raise RuntimeError('Answer contains an unverified evidence identifier')
                revised=clarify_response(text,s.contract,has_evidence=bool(evidence),
                    allowed_studies=self.allowed_studies)
                revised=verified_protocol_disclosure(revised,run['receipts'])
                # A receipt citation alone does not locate the source excerpt.
                # Label these as retrieved excerpts, not proof of every claim.
                shown_passages=sorted({p for r in run['receipts'] if r.get('tool')=='protocol'
                                       for p in r.get('model_passage_ids',[]) if p in passages})
                if shown_passages and not re.search(r'\bpc-[a-zA-Z0-9_-]+',revised):
                    revised+='\n\nRetrieved protocol excerpts: '+', '.join(shown_passages)
                if not evidence:
                    # Scope/clarification results have no scientific receipt.
                    # Do not present status placeholders as source identifiers.
                    label=re.compile(r'(?i)(?:\*\*)?Evidence(?:\s+IDs?)?(?:\*\*)?\s*:(?:\*\*)?[^\n]*')
                    revised=label.sub('No source receipt was generated for this request.',revised)
                missing=sorted(evidence-set(re.findall(r'\bev-[a-zA-Z0-9_-]+',revised)))
                if missing:
                    # Persist citations with the answer in managed Sessions,
                    # rather than trusting the model or a UI-only decoration.
                    revised+='\n\nEvidence: '+', '.join(missing)
                if revised!=text:
                    result=llm_response.model_copy(deep=True)
                    result.content.parts=[types.Part(text=revised)]
                    return result
        return None
