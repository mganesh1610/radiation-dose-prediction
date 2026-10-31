"""Operator CLI: build, embed and publish an immutable public GEO corpus.

Preparation is local. Embedding and publication require separate explicit flags;
no chat tool can call this workflow or alter the approved corpus.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from google.cloud import storage
from ..persistence.managed_sessions import cloud_credentials
from ..tools.protocol_corpus import build_corpus
from ..tools.protocol_search import ProtocolEmbeddings, validate_vector
from ..audit_cloud import gcloud_token

STUDIES = ('GSE102971', 'GSE8917', 'GSE65292')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project');p.add_argument('--bucket')
    p.add_argument('--dataset',default='radiation_protocols_v1');p.add_argument('--region',default='us-west1')
    p.add_argument('--embed',action='store_true');p.add_argument('--publish',action='store_true')
    a=p.parse_args()
    paths={s:a.source_root/s/(s+'_family.soft.gz') for s in STUDIES}
    manifest,rows=build_corpus(paths)
    if len(rows)>40 or sum(len(r['text']) for r in rows)>100_000:raise ValueError('Corpus exceeds first-pilot bounds')
    root=a.output/manifest['version'];root.mkdir(parents=True,exist_ok=True)
    (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    (root/'passages.json').write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n')
    embeddings=root/'embedded.jsonl'
    credentials=cloud_credentials()
    if a.embed:
        if not a.project:raise ValueError('Project required for paid embedding calls')
        existing={}
        if embeddings.exists():
            existing={r['chunk_id']:r for r in map(json.loads,embeddings.read_text().splitlines())}
        embedder=ProtocolEmbeddings(a.project,credentials)
        with embeddings.open('a') as out:
            for row in rows:
                if row['chunk_id'] in existing:
                    old=existing[row['chunk_id']]
                    if {k:v for k,v in old.items() if k!='embedding'}!=row:raise ValueError('Existing embedding source changed')
                    validate_vector(old['embedding']);continue
                row=dict(row,embedding=embedder.embed(row['study_id']+' '+row['section']+'\n'+row['text'],'RETRIEVAL_DOCUMENT'))
                out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush()
    if a.publish:
        if not a.project or not a.bucket:raise ValueError('Project and bucket required for publication')
        data=[json.loads(line) for line in embeddings.read_text().splitlines()]
        if len(data)!=len(rows) or {r['chunk_id'] for r in data}!={r['chunk_id'] for r in rows}:raise ValueError('Incomplete embeddings')
        by_id={r['chunk_id']:r for r in rows}
        for row in data:
            validate_vector(row['embedding'])
            if {k:v for k,v in row.items() if k!='embedding'}!=by_id[row['chunk_id']]:raise ValueError('Embedding source mismatch')
        bucket=storage.Client(project=a.project,credentials=credentials).bucket(a.bucket)
        prefix='protocols/v1/'+manifest['version']+'/'
        objects={}
        for relative,path in {**{f'originals/{s}_family.soft.gz':f for s,f in paths.items()},
                              'manifest.json':root/'manifest.json','embedded.jsonl':embeddings}.items():
            sha=hashlib.sha256(path.read_bytes()).hexdigest();blob=bucket.blob(prefix+relative)
            if blob.exists():
                blob.reload()
                if (blob.metadata or {}).get('sha256')!=sha:raise ValueError('Immutable corpus object already differs')
            else:
                blob.metadata={'sha256':sha};blob.upload_from_filename(str(path),if_generation_match=0,timeout=120)
            objects[relative]={'uri':f'gs://{a.bucket}/{blob.name}','generation':str(blob.generation),'sha256':sha}
        token=gcloud_token()
        def request(path,body=None):
            req=Request('https://bigquery.googleapis.com/bigquery/v2/'+path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            with urlopen(req,timeout=30) as response:return json.load(response)
        try:dataset=request(f'projects/{a.project}/datasets/{a.dataset}')
        except HTTPError as exc:
            if exc.code!=404:raise
            dataset=request(f'projects/{a.project}/datasets',{'datasetReference':{'projectId':a.project,'datasetId':a.dataset},'location':a.region})
        if dataset['location'].lower()!=a.region.lower():raise ValueError('Dataset region differs')
        fields=[]
        for key in rows[0]:
            fields.append({'name':key,'type':'BOOL' if key=='approved' else 'INT64' if key=='embedding_dimensions' else 'STRING',
                           'mode':'REPEATED' if key=='sample_ids' else 'REQUIRED'})
        fields.append({'name':'embedding','type':'FLOAT64','mode':'REPEATED'})
        job_id='protocol_load_'+manifest['version'].replace('-','_')
        body={'jobReference':{'projectId':a.project,'location':a.region,'jobId':job_id},'configuration':{'load':{
            'sourceUris':[objects['embedded.jsonl']['uri']], 'sourceFormat':'NEWLINE_DELIMITED_JSON',
            'destinationTable':{'projectId':a.project,'datasetId':a.dataset,'tableId':'protocol_passages'},
            'schema':{'fields':fields},'writeDisposition':'WRITE_APPEND','createDisposition':'CREATE_IF_NEEDED'}}}
        try:job=request(f'projects/{a.project}/jobs',body)
        except HTTPError as exc:
            if exc.code!=409:raise
            job=request(f'projects/{a.project}/jobs/{job_id}?location={a.region}')
        publication={'corpus_version':manifest['version'],'objects':objects,'load_job':job['jobReference'],
                     'table':f'{a.project}.{a.dataset}.protocol_passages','passages':len(rows)}
        (root/'publication.json').write_text(json.dumps(publication,indent=2)+'\n')
        print(json.dumps(publication))
    else:print(json.dumps({'corpus_version':manifest['version'],'passages':len(rows),'output':str(root)}))


if __name__=='__main__':main()
