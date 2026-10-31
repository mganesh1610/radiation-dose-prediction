"""Bounded wording controls before an answer is persisted in managed Sessions.

These controls clarify selectors; they do not resolve gene symbols, validate
scientific claims, or grant permission to retrieve an upload.
"""
import re


def clarify_response(text, contract, *, has_evidence, allowed_studies=None):
    # A named study can be outside the admitted release. Do not mask that
    # scope explanation with the standalone sample-selector template.
    if contract.get('expression_lookup_requested') and not has_evidence:
        studies=set(contract.get('study_ids', []))
        if studies and allowed_studies is None:
            return text  # No authoritative catalog supplied to this guard.
        outside=studies-set(allowed_studies or ())
        if outside:
            return (', '.join(sorted(outside)) + ' is outside the connected approved core release.'
                    ' No expression values were retrieved. This is a coverage limitation,'
                    ' not a zero result or evidence that the original study lacks data.')
        missing = []
        if not contract.get('sample_ids'):
            missing.append('the GSM sample accession(s)')
        if not contract.get('gene_ids'):
            missing.append('the numeric NCBI Gene ID(s)')
        if missing:
            answer = 'Please provide ' + ' and '.join(missing) + ' in your question.'
            if not contract.get('gene_ids'):
                answer += (' Use the label "NCBI Gene ID" followed by the numeric identifier'
                           ' from your verified gene annotation. Gene-symbol conversion is not'
                           ' connected, so I cannot verify a symbol-to-ID mapping here.')
            return answer + ' No expression value has been retrieved.'
    # Repair prediction-specific wording while preserving training status requests.
    if contract.get('prediction_requested') and not contract.get('training_requested'):
        text = re.sub(r'(?<!training )\bjob\s+IDs?\b',
                      'upload ID and explicitly selected frozen model', text, flags=re.I)
    return text
