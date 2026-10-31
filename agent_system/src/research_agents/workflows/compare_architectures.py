"""Freeze, run and review the bounded architecture comparison."""
import argparse
import asyncio
import json
import logging
from pathlib import Path

from ..evaluation.comparison import digest, summarize, save_new, validate_questions, check_preflight
from ..evaluation.comparison_freeze import freeze
from ..evaluation.comparison_runtime import collect, verify_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze')
    for name in ('export','admission','protocol','passages','output'):f.add_argument('--'+name,type=Path,required=True)
    f.add_argument('--exclude',type=Path,action='append',default=[])
    r=sub.add_parser('run')
    for name in ('frozen','admission','protocol','preflight','output'):r.add_argument('--'+name,type=Path,required=True)
    r.add_argument('--execute',action='store_true');r.add_argument('--continue-after-failure',action='store_true')
    s=sub.add_parser('score')
    for name in ('frozen','results','output'):s.add_argument('--'+name,type=Path,required=True)
    s.add_argument('--reviews',type=Path)
    args=p.parse_args();source=Path(__file__).resolve().parents[1]
    if args.command=='freeze':
        b=freeze(args.export,args.admission,args.protocol,args.passages,args.exclude,source,args.output)
        print(json.dumps({'frozen':len(b['cases']),'attempts':54,'reference_sha256':digest(b),'cloud_calls':0}));return
    b=json.loads((args.frozen/'references.json').read_text())
    if (args.frozen/'references.sha256').read_text().strip()!=digest(b):raise ValueError('Frozen reference checksum mismatch')
    if args.command=='score':
        records=[json.loads(p.read_text()) for p in sorted(args.results.glob('C??-*.json')) if not p.name.endswith('.journal.json')]
        reviews=json.loads(args.reviews.read_text()) if args.reviews else {}
        result=summarize(b,records,reviews)
        args.output.mkdir(parents=True,exist_ok=True)
        save_new(args.output/'report.json',result)
        save_new(args.output/'review-template.json',{r['case_id']+':'+r['arm']:{'result_sha256':digest(r),
            'review_type':'engineering','pass':None,'notes':''} for r in records})
        print(json.dumps(result['arms'],indent=2));return
    q=json.loads((args.frozen/'questions.json').read_text());preflight=json.loads(args.preflight.read_text())
    verify_inputs(b,q,args.admission,args.protocol,source);check_preflight(preflight,digest(b))
    if not args.execute:
        print('Preflight passed. No cloud model/retrieval calls made.');return
    # Library debug/HTTP logs can include prompt content; keep those out of console logs.
    logging.getLogger().setLevel(logging.CRITICAL)
    asyncio.run(collect(b,q,preflight,args.admission,args.protocol,args.output,continue_after_failure=args.continue_after_failure))


if __name__=='__main__':main()
