"""Freeze independent references, collect authorized answers, and prepare review reports."""
import argparse
import json
import os
from pathlib import Path
from ..evaluation.references import freeze, load_frozen, source_snapshot
from ..evaluation.runner import collect, http_client, check_allowance, pending_requirements
from ..evaluation.scoring import report, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--export',type=Path,required=True);f.add_argument('--admission',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    f.add_argument('--suite-id',default='core-chat-acceptance-v2');f.add_argument('--exclude-frozen',type=Path,action='append',default=[])
    s=sub.add_parser('score');s.add_argument('--frozen',type=Path,required=True);s.add_argument('--results',type=Path,required=True);s.add_argument('--reviews',type=Path);s.add_argument('--output',type=Path,required=True)
    s.add_argument('--human-reviewers', type=Path, help='Operator-controlled JSON list of authorized human reviewer identifiers; never supplied by an AI judge')
    r=sub.add_parser('run');r.add_argument('--frozen',type=Path,required=True);r.add_argument('--preflight',type=Path,required=True);r.add_argument('--output',type=Path,required=True);r.add_argument('--base-url',required=True);r.add_argument('--headers-env',default='COPILOT_EVAL_HEADERS_JSON');r.add_argument('--execute',action='store_true');r.add_argument('--resume',action='store_true')
    a=parser.parse_args();source=Path(__file__).resolve().parents[1]
    if a.command=='freeze':
        b=freeze(a.export,a.admission,a.output,source,a.suite_id,a.exclude_frozen);print(json.dumps({'cases':len(b['cases']),'factual':30,'boundary':8,'cloud_calls':0}))
    elif a.command=='score':
        b,h=load_frozen(a.frozen)
        records=[json.loads(p.read_text()) for p in sorted(a.results.glob('*.json')) if not p.name.endswith('.journal.json') and p.name[0:1] in ('F','B')]
        reviews=json.loads(a.reviews.read_text()) if a.reviews else {}
        human_reviewers=json.loads(a.human_reviewers.read_text()) if a.human_reviewers else []
        result,template=report(b,h,records,reviews,human_reviewers=human_reviewers)
        result['reviews_sha256']=digest(reviews)
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        template_path=a.output/'review-template.json'
        if not template_path.exists():template_path.write_text(json.dumps(template,indent=2)+'\n')
        (a.output/'report.md').write_text(f'# Chat acceptance evaluation\n\nStatus: **{result["status"]}**\n\nExecuted: {result["executed"]}/38; human-reviewed: {result["human_reviewed"]}/38. Factual passes: {result["factual_passed"]}/30; boundary passes: {result["boundary_passed"]}/8.\n\nProvisional AI reviews: {result["ai_reviewed"]}; engineering reviews: {result["engineering_reviewed"]}. These do not count toward acceptance.\n\n'+('\n'.join('- '+x for x in result['limits']))+'\n\n| Case | Category | State | Automated findings | Review findings |\n|---|---|---|---|---|\n'+'\n'.join(f'| {x["case_id"]} | {x["category"]} | {x["state"]} | {", ".join(x["automated_errors"])} | {", ".join(x["review_errors"])} |' for x in result['cases'])+'\n')
        print(json.dumps({k:result[k] for k in ('status','executed','reviewed','factual_passed','boundary_passed')}))
    else:
        b,h=load_frozen(a.frozen)
        if source_snapshot(source)!=b['runtime_source_sha256']:raise ValueError('Runtime changed after freeze; report a new evaluation version')
        questions=json.loads((a.frozen/'questions.json').read_text())
        expected={'suite_id':b['suite_id'],'reference_sha256':h,'cases':[{'id':c['id'],'question':c['question']} for c in b['cases']]}
        if questions!=expected:raise ValueError('Question file differs from frozen reference set')
        needed=pending_requirements(questions,a.output,resume=a.resume)
        preflight=json.loads(a.preflight.read_text());check_allowance(preflight,needed['turns'],needed['chats'])
        if not a.execute:print('Preflight passed; use --execute only when ready for live model/cloud usage.');return
        headers=json.loads(os.environ.get(a.headers_env,'{}'))
        if not headers:raise ValueError('Authenticated request headers must be supplied through the selected environment variable')
        with http_client(a.base_url,headers) as client:
            result=collect(questions,preflight,client,a.output,resume=a.resume)
            print(json.dumps(result))
            if result['status'] != 'collected': raise SystemExit(1)


if __name__=='__main__':main()
