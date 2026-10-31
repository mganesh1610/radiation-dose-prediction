"""Study-filtered hybrid retrieval. Query SQL and source URLs are server controlled."""
import asyncio
import json
import math
import re
import time
from google import genai
from google.genai import types
from .protocol_corpus import MODEL, DIMENSIONS, sha
from .query_sql import CompiledQuery, scalar, digest


# Server-owned provenance: source text cannot redefine its own applicability.
SOURCE_APPLICABILITY = (
    'All returned passages describe the original GEO source study, including '
    'Sample_data_processing. They do not document or verify the current project '
    'pipeline, software, normalization or implementation. Attribute methods to '
    'the source study; current-project processing is not established by this tool.'
)


class ProtocolEmbeddings:
    def __init__(self, project, credentials, location='us-central1'):
        self.client = genai.Client(enterprise=True, project=project, location=location, credentials=credentials,
            http_options=types.HttpOptions(timeout=20000, retry_options=types.HttpRetryOptions(attempts=1)))

    def embed(self, text, task='RETRIEVAL_QUERY'):
        if task not in {'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT'} or not 1 <= len(text) <= 7000:
            raise ValueError('Embedding input outside pilot bounds')
        result = self.client.models.embed_content(model=MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=DIMENSIONS, auto_truncate=False))
        if len(result.embeddings or []) != 1: raise RuntimeError('Incomplete embedding response')
        vector = result.embeddings[0].values
        validate_vector(vector)
        return vector


def validate_vector(vector):
    if (not isinstance(vector, list) or len(vector) != DIMENSIONS
            or any(type(x) not in {int, float} or not math.isfinite(x) for x in vector)
            or not any(vector)):
        raise ValueError('Embedding dimensions or values invalid')


def model_evidence(answer, limit=3000):
    """Whole, exact passages within the model budget; full metadata stays in receipts."""
    compact={k:v for k,v in answer.items() if k!='passages'}
    compact['passages']=[]
    compact['source_scope']='original_study'
    compact['applicability']=SOURCE_APPLICABILITY
    compact['omitted_passage_count']=len(answer['passages'])
    for passage in answer['passages']:
        item={k:passage[k] for k in ('chunk_id','study_id','section','text','source_url','document_version')}
        item['source_scope']='original_study'
        trial={**compact,'passages':compact['passages']+[item],
               'omitted_passage_count':len(answer['passages'])-len(compact['passages'])-1}
        if len(json.dumps(trial,ensure_ascii=False).encode())<=limit:compact=trial
    if not compact['passages']:compact['status']='insufficient_context_budget'
    return compact


def compile_search(table, corpus, study, question, vector, blocked_passages=()):
    if not re.fullmatch(r'[a-z][a-z0-9-]+\.[A-Za-z_][A-Za-z_0-9]*\.protocol_passages', table):
        raise ValueError('Invalid protected protocol table')
    validate_vector(vector)
    tokens = sorted(set(re.findall(r'[a-z0-9]{3,}', question.lower())))[:40]
    # Both candidate paths filter before ranking. No unrestricted search followed
    # by an LLM instruction to ignore disallowed documents.
    base = f'''SELECT * FROM `{table}` WHERE corpus_version=@corpus AND study_id=@study
      AND approved=TRUE AND access_class='public' AND embedding_model=@model
      AND embedding_dimensions={DIMENSIONS}
      AND chunk_id NOT IN UNNEST(JSON_VALUE_ARRAY(@blocked))'''
    sql = f'''WITH filtered AS ({base}),
    semantic AS (
      SELECT base.chunk_id, ROW_NUMBER() OVER(ORDER BY distance, base.chunk_id) AS rank
      FROM VECTOR_SEARCH(({base}), 'embedding',
        (SELECT ARRAY(SELECT CAST(x AS FLOAT64) FROM UNNEST(JSON_VALUE_ARRAY(@vector)) x) AS embedding),
        top_k=>8, distance_type=>'COSINE', options=>'{{"use_brute_force":true}}')),
    lexical_scores AS (
      SELECT chunk_id, (SELECT COUNT(*) FROM UNNEST(JSON_VALUE_ARRAY(@tokens)) word
        WHERE STRPOS(LOWER(CONCAT(section, ' ', text)), word)>0) AS score FROM filtered),
    lexical AS (SELECT chunk_id, ROW_NUMBER() OVER(ORDER BY score DESC,chunk_id) AS rank
      FROM lexical_scores WHERE score>0 QUALIFY rank<=8),
    fused AS (SELECT chunk_id, SUM(1.0/(60+rank)) AS score FROM
      (SELECT * FROM semantic UNION ALL SELECT * FROM lexical) GROUP BY chunk_id),
    selected AS (SELECT f.* EXCEPT(embedding), r.score FROM filtered f JOIN fused r USING(chunk_id)
      ORDER BY r.score DESC,chunk_id LIMIT 4)
    SELECT TO_JSON_STRING(ARRAY(SELECT AS STRUCT * FROM selected ORDER BY score DESC,chunk_id)) AS passages'''
    parameters = [scalar('corpus', 'STRING', corpus), scalar('study', 'STRING', study), scalar('model', 'STRING', MODEL),
                  scalar('vector', 'STRING', json.dumps(vector)), scalar('tokens', 'STRING', json.dumps(tokens)),
                  scalar('blocked', 'STRING', json.dumps(list(blocked_passages)))]
    return CompiledQuery(sql, json.dumps(parameters), 'protocol-hybrid-rrf-v1')


class ProtocolSearch:
    def __init__(self, manifest, table, reader_factory, embedder):
        if manifest['embedding_model'] != MODEL or manifest['embedding_dimensions'] != DIMENSIONS:
            raise ValueError('Corpus embedding configuration mismatch')
        self.manifest, self.table, self.reader_factory, self.embedder = manifest, table, reader_factory, embedder
        self.studies = {d['study_id'] for d in manifest['documents']}
        self.blocked = manifest.get('blocked_passages', {})
        if not set(self.blocked).issubset(manifest['passages']):raise ValueError('Unknown quarantined passage')

    def search(self, study, question, before_execute=lambda: None):
        if study not in self.studies or not 1 <= len(question) <= 2000: raise ValueError('Protocol scope unsupported')
        start = time.monotonic()
        before_execute()
        vector = self.embedder.embed(question)
        query = compile_search(self.table, self.manifest['version'], study, question, vector, self.blocked)
        row, receipt = self.reader_factory().execute(query, before_execute=before_execute)
        passages = json.loads(row['passages'])
        if not isinstance(passages, list) or len(passages) > 4: raise RuntimeError('Invalid passage result')
        seen = set()
        for p in passages:
            registered = self.manifest['passages'].get(p['chunk_id'])
            if (not registered or p['chunk_id'] in seen or p['chunk_id'] in self.blocked or p['study_id'] != study
                    or p['corpus_version'] != self.manifest['version'] or not p['approved']
                    or p['access_class'] != 'public' or sha(p['text']) != p['text_sha256']
                    or any(p.get(k) != v for k, v in registered.items())):
                raise RuntimeError('Retrieved source failed corpus integrity checks')
            seen.add(p['chunk_id'])
        return {'status': 'evidence_found' if passages else 'no_evidence', 'study_id': study,
                'source_scope': 'original_study', 'applicability': SOURCE_APPLICABILITY,
                'corpus_version': self.manifest['version'], 'passages': passages,
                'withheld_sections': sorted({self.manifest['passages'][c]['section'] for c in self.blocked
                                            if self.manifest['passages'][c]['study_id']==study}),
                'coverage_note': 'Withheld sections exist in the original GEO source but are excluded from this application\'s approved retrieval corpus pending verification of ambiguous source units. GEO has not removed these sections. A missing retrieved detail does not prove it is absent from the original study. Do not infer or correct its units.',
                'limitations': 'Top-ranked public GEO metadata excerpts, not an exhaustive search or full publication methods. Source-study processing may differ from this project. Retrieval rank does not establish claim support.'}, {
                **receipt, 'template_version': query.template_version, 'embedding_model': MODEL,
                'embedding_dimensions': DIMENSIONS, 'question_sha256': sha(question),
                'corpus_version': self.manifest['version'], 'latency_ms': round(1000*(time.monotonic()-start), 2)}


class ProtocolChatTool:
    def __init__(self, search, guard, allowed_users=None):
        self.search, self.guard = search, guard
        self.allowed_users = None if allowed_users is None else frozenset(allowed_users)

    async def retrieve(self, study_id, question):
        s = await self.guard.authorize()
        if self.allowed_users is not None and s.principal.user_id not in self.allowed_users:
            return {'status':'unsupported','reason':'Protocol search is not enabled for this pilot account.'}
        await asyncio.to_thread(self.guard.ledger.reserve, s.principal.user_id, s.run_id, s.fence, 'tool')
        if s.contract.get('study_ids') != [study_id]:
            return {'status': 'clarification_required', 'reason': 'Specify one GSE study accession in the current question.'}
        if s.contract.get('count_metric') or s.contract.get('count_metrics') or s.contract.get('sample_ids') or s.contract.get('gene_ids'):
            return {'status': 'unsupported', 'reason': 'Use exact-data tools for sample values or cohort counts. Ask study-method questions separately in this pilot.'}
        if study_id not in self.search.studies:
            return {'status': 'unsupported', 'reason': 'No approved protocol document for this study in the connected corpus.'}
        if not isinstance(question,str) or not 1<=len(question)<=2000:
            return {'status':'clarification_required','reason':'Use a study-method question of at most 2000 characters.'}
        def check():
            self.guard.access.authorize(s.principal, s.workspace, self.guard.release_id)
            self.guard.ledger.check(s.principal.user_id, s.run_id, s.fence)
        answer, receipt = await asyncio.to_thread(self.search.search, study_id, question, check)
        await self.guard.authorize()
        compact=model_evidence(answer)
        receipt['model_passage_ids']=[p['chunk_id'] for p in compact['passages']]
        evidence_id = 'ev-' + digest({'answer': answer, 'receipt': receipt})[:24]
        await asyncio.to_thread(self.guard.ledger.progress, s.principal.user_id, s.run_id, s.fence,
            'verified_protocol', {**receipt, 'tool': 'protocol', 'evidence_id': evidence_id, 'verified_result': answer})
        return {'evidence_id': evidence_id, 'answer': compact}
