"""One bounded within-study descriptive workflow; no arbitrary code or SQL."""
from typing import Literal
from pydantic import Field
from .queries import StrictRecord


class AnalysisPlan(StrictRecord):
    version: Literal[1] = 1
    study_id: str = Field(pattern=r'^GSE[0-9]{1,12}$')
    gene_id: str = Field(pattern=r'^[1-9][0-9]{0,11}$')


def bind_analysis(plan, contract, studies):
    if plan.study_id not in studies:
        return {'status':'unsupported', 'reason':'This study is outside the connected core release; no plot was produced.'}
    if (contract.get('study_ids') != [plan.study_id] or contract.get('gene_ids') != [plan.gene_id]
            or contract.get('sample_ids') or contract.get('count_metric')):
        return {'status':'clarification_required', 'reason':'For this whole-study plot, specify one GSE accession and one numeric NCBI Gene ID, with no sample subset or counting request.'}
    if not contract.get('analysis_requested') or contract.get('analysis_restricted'):
        return {'status':'clarification_required', 'reason':'This workflow plots all approved profiles of one study by recorded dose on the source linear intensity scale. Filtered, normalized, transformed, pooled and predictive analyses are not connected. Please request this supported scope explicitly.'}
    return None
