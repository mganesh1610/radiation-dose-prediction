"""Small, conservative selector contract compiled from the current user turn.

This constrains tools; it is neither a general natural-language SQL parser nor
an authorization grant. Ambiguous or implicit selectors require clarification.
"""
import re


def compile_contract(text):
    counting = bool(re.search(r"\b(how many|number of|count|counts|total)\b", text, re.I))
    profiles = bool(re.search(r"\b(profiles?|samples?)\b", text, re.I))
    groups = bool(re.search(r"\b(donors?|groups?|people|participants?)\b", text, re.I))
    metric = None
    if counting and profiles != groups:
        metric = 'sample_count' if profiles else 'study_group_count'
    # This explicit conjunction requests two distinct units, not a choice
    # between ambiguous "samples or donors" or globally unique people.
    paired_counts = counting and bool(re.search(
        r'\b(?:profiles?|samples?)\s+and\s+study[- ]local\s+(?:donor\s+)?groups?\b', text, re.I))
    count_metrics = ['sample_count', 'study_group_count'] if paired_counts else ([metric] if metric else [])
    prediction_requested = bool(re.search(r'\b(predict|prediction|estimate)\b',text,re.I)
        or re.search(r'\b(calculate|compute)\b.{0,50}\bdose\b',text,re.I))
    if re.search(r"\b(do not|don't|never|without)\b.{0,40}\b(predict|prediction|estimate|calculate|compute)\b",text,re.I):
        prediction_requested = False
    genes = []
    for match in re.finditer(r"\b(?:(?:NCBI|Entrez)\s+)?Gene\s*IDs?\s*[:=]?\s*"
                             r"([1-9][0-9]*(?:(?:\s*,\s*|\s+and\s+)[1-9][0-9]*)*)\b", text, re.I):
        genes.extend(re.findall(r"\d+", match[1]))
    return {'version': 1, 'count_metric': metric, 'count_metrics': count_metrics,
            'expression_lookup_requested': bool(re.search(r'\bexpression\b', text, re.I)
                and (re.search(r'\b(show|retrieve|fetch|get|give)\b|\blook up\b', text, re.I)
                     or re.search(r'\bGSM[0-9]+\b', text))
                and not counting and not prediction_requested
                and not re.search(r'\b(plot|chart|visuali[sz]e|graph|protocol|normaliz\w*|normalis\w*|method\w*|why|how)\b', text, re.I)),
            'training_requested': bool(re.search(r'\b(train\w*|engineering job)\b|\bweb_[0-9]{6}_[a-f0-9]{32}\b', text, re.I)),
            'training_job_ids': sorted(set(re.findall(r'\bweb_[0-9]{6}_[a-f0-9]{32}\b',text))),
            'upload_ids': sorted(set(x.lower() for x in re.findall(r'\bupload(?:\s+ID)?\s*[:=]?\s*([a-f0-9]{64})\b',text,re.I))),
            'prediction_requested': prediction_requested,
            'prediction_models': sorted(set(x.upper() if len(x)==1 else x.lower() for x in re.findall(r'\b(?:model|frozen(?:\s+model)?)\s*[:=]?\s*(A|B|C|D|elastic_net|pls|gradient_boosting|median_null)\b',text,re.I))),
            'analysis_requested': bool(re.search(r'\b(plot|chart|visuali[sz]e|graph)\b', text, re.I)),
            'analysis_restricted': bool(re.search(r'\b(normaliz\w*|normalis\w*|log2?|transform\w*|predict\w*|clinical|differential|correlation|filter\w*|subset|only|exclu\w*|pool\w*|PCA|cluster\w*|heatmap|z.score|above|below|greater|less|controls?\s+only)\b|[<>≤≥]|\b\d+(?:\.\d+)?\s*Gy\b', text, re.I)),
            'sample_ids': sorted(set(re.findall(r'\bGSM[0-9]{1,14}\b', text))),
            'study_ids': sorted(set(re.findall(r'\bGSE[0-9]{1,12}\b', text))),
            'gene_ids': sorted(set(genes))}


def bind_plan(kind, plan, contract, allowed_studies):
    """Return a server-bound plan or a safe clarification without executing data tools."""
    def clarification(reason):
        return None, {'status': 'clarification_required', 'reason': reason}
    if contract.get('version') != 1:
        return clarification('Repeat the question with explicit sample IDs and count units or NCBI Gene IDs.')
    outside = set(contract['study_ids']) - set(allowed_studies)
    if outside:
        return None, {'status': 'unsupported', 'reason':
            'The requested study is outside the connected core release. This does not mean it has zero samples.',
            'requested_study_ids': sorted(outside), 'available_study_ids': sorted(allowed_studies)}
    if kind == 'expression':
        if not set(plan['gene_ids']).issubset(contract['gene_ids']):
            return clarification('Provide explicit numeric NCBI Gene IDs in this question. Gene-symbol conversion is not connected; I cannot infer an ID from TP53 or another symbol.')
        if not set(plan['sample_ids']).issubset(contract['sample_ids']):
            return clarification('Provide the GSM sample IDs explicitly in this question.')
    elif plan['metric'] == 'sample_details':
        if plan['sample_id'] not in contract['sample_ids']:
            return clarification('Provide the GSM sample ID explicitly in this question.')
    else:
        metric = contract['count_metric']
        if metric is None and contract.get('count_metrics') == ['sample_count', 'study_group_count']:
            if plan['metric'] not in contract['count_metrics']:
                return clarification('Use only the two explicitly requested count units.')
            metric = plan['metric']
        if metric is None:
            return clarification('Specify whether to count expression profiles or study-local donor groups, one metric at a time.')
        # The model cannot substitute donor counts for profile counts, even when
        # the numeric answers coincide. The SQL and receipt use this bound plan.
        plan = {**plan, 'metric': metric}
        requested = contract['study_ids']
        study = plan['filters'].get('study_id')
        if len(requested) > 1 or (requested and study != requested[0]) or (study and study not in requested):
            return clarification('Select one explicit GSE study or ask for the whole connected core release.')
    return plan, None
