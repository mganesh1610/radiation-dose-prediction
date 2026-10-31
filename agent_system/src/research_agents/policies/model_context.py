"""Fit verified protocol responses without truncating passages or source receipts."""


def request_bytes(request):
    return len(request.model_dump_json(exclude_none=True, exclude={'tools_dict'}).encode())


def fit_protocol_context(request, evidence_ids, max_bytes=16000):
    """Mutate only the caller's outbound copy; never change stored Session events.

    The ledger still enforces its unchanged per-call and total byte limits. If
    protocol excerpts cannot free enough room, normal admission fails closed.
    Non-protocol evidence, instructions, selectors and tool schemas stay intact.
    """
    for content in reversed(request.contents):
        for part in reversed(content.parts or []):
            response=part.function_response
            if not response or response.name!='search_study_protocols':
                continue
            payload=response.response
            if not isinstance(payload,dict) or payload.get('evidence_id') not in evidence_ids:
                continue
            answer=payload.get('answer')
            if not isinstance(answer,dict) or answer.get('source_scope')!='original_study':
                continue
            passages=answer.get('passages')
            if not isinstance(passages,list):continue
            while passages and request_bytes(request)>max_bytes:
                # Retrieval order is relevance order; preserve the highest ranked
                # whole excerpt. No partial text, invented summaries or unit edits.
                passages.pop()
                answer['omitted_passage_count']=answer.get('omitted_passage_count',0)+1
            if not passages:answer['status']='insufficient_context_budget'
            if request_bytes(request)<=max_bytes:return
