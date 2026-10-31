"""Render corpus availability from current structured receipts, never model counts.

This repairs source-coverage wording; it is not a general scientific claim verifier.
Original excerpts and stored receipts are not modified.
"""
import re

MARKER = 'Source coverage (verified):'
_COVERAGE = re.compile(
    r'\bwithh(?:eld|old\w*)\b|\bquarantin\w*\b|'
    r'\b(?:section\w*|protocol\w*|passage\w*)\b[^.!?\n]{0,120}'
    r'\b(?:unavailable|excluded|removed|omitted|not available)\b|'
    r'\b(?:excluded|removed|omitted)\b[^.!?\n]{0,120}\b(?:corpus|retrieval|GEO)\b', re.I)


def verified_protocol_disclosure(text, receipts):
    protocols=[r for r in receipts if r.get('tool')=='protocol']
    if not protocols:return text
    # Replace the designated footer on a second agent pass, making this idempotent.
    body=text.split(MARKER,1)[0].rstrip()
    # Remove generated source-availability claims. Protect literal verified
    # excerpts so source methods (e.g. excluded array features) stay unedited.
    quoted={p['text'] for r in protocols for p in r['verified_result']['passages']}
    placeholders={}
    for i,quote in enumerate(sorted(quoted,key=len,reverse=True)):
        if quote in body:
            placeholder=f'\x00VERIFIED_EXCERPT_{i}\x00';placeholders[placeholder]=quote
            body=body.replace(quote,placeholder)
    sentences=re.split(r'(?<=[.!?])(?=\s)|\n',body)
    body='\n'.join(part for part in sentences if not _COVERAGE.search(part)).strip()
    for placeholder,quote in placeholders.items():body=body.replace(placeholder,quote)
    # A removed italic limitation sentence can leave a formatting delimiter.
    body=re.sub(r'(?m)^\s*[*_]+\s*$','',body).strip()
    lines=[]
    for receipt in protocols:
        result=receipt['verified_result'];sections=sorted(set(result.get('withheld_sections',[])))
        study=result['study_id'];count=len(sections)
        if count:
            label='section' if count==1 else 'sections'
            lines.append(f"{study}: {count} {label} withheld from this application's approved retrieval corpus: "
                + ', '.join(sections) + '. Pending verification of ambiguous source units; no corrected value is verified. '
                'These sections remain in the original GEO source.')
        else:
            lines.append(f'{study}: 0 sections withheld in the approved corpus for this study. '
                         'This does not establish exhaustive coverage of the original study.')
    lines.append('Retrieved passages describe the original study, not the current project pipeline. '
                 'A missing retrieved detail does not establish that the original study lacks it.')
    return body+'\n\n'+MARKER+'\n'+'\n'.join(lines)
