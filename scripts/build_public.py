"""Export the reviewed, immutable public viewer. Never upload the repo root.

Reads one hash-pinned local core release and one reviewed engineering result.
Also verifies and exports curated aggregate frozen evaluation results. No fitting or inference.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import date
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from analysis_service import Explorer
from app.server import DATASET
from build_reviewer_results import build_payload
from build_development_explorer import build_payload as build_development

DEST = ROOT / 'data/public_web'
EVIDENCE = ROOT / 'context/cloud/vercel-v1'
CORE = ROOT / 'data/cloud_exports' / ('core_' + DATASET['sha256'][:16]) / 'core.npz'
RESULT = Path(os.environ.get('DEFENDER_PUBLIC_RESULT', ROOT / 'data/reviewed_result/result.json'))
FILES = {
    'index.html': 'app/public/index.html',
    'vercel.json': 'app/public/vercel.json',
    **{'static/' + n: 'app/public/' + n for n in
       ('public-data.js', 'public-results.js', 'public.css')},
    **{'static/' + n: 'app/static/' + n for n in
       ('analysis-workspace.js','analysis-workspace.css','analysis-scene.mjs','spatial-model.mjs',
        'vendor/three/three.module.js','vendor/three/three.core.js','vendor/three/OrbitControls.js','vendor/three/LICENSE',
        'statistical-explorer.js','research-workflow.css','assistant-engine.js','research-assistant.js','style.css', 'workspace-design.css', 'explorer.css', 'explorer.js', 'heatmap-clustering.js', 'reviewer-results.js', 'reviewer-results.css', 'development-explorer.js', 'development-explorer.css', 'assets/cell-hero.png',
        'fonts/geist.woff2', 'fonts/geist.woff2.LICENSE',
        'fonts/ibm-plex-mono.woff2', 'fonts/ibm-plex-mono.woff2.LICENSE')},
}


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


class LocalRelease:
    def read(self, name, generation, expected_sha, limit):
        if (name, generation, expected_sha) != (DATASET['object'], DATASET['generation'], DATASET['sha256']):
            raise ValueError('Only the reviewed core release may be exported')
        payload = CORE.read_bytes()
        if len(payload) > limit or sha(payload) != expected_sha:
            raise ValueError('Local release integrity mismatch')
        return payload


def public_source(value):
    # Analysis IDs/values/methods stay intact; internal object paths and generations do not.
    return dict(release_id='core_' + value['sha256'][:16], sha256=value['sha256'])


def public_catalog(c):
    fields = ('release_id', 'studies', 'samples', 'profiles', 'groups', 'probes',
              'annotation_genes', 'representation', 'group_identity', 'locked_test_included')
    return {**{k: c[k] for k in fields}, 'source': public_source(c['source'])}


def public_analysis(a):
    fields = ('schema_version', 'analysis_version', 'analysis_id', 'release_id', 'options',
              'purpose', 'locked_test_included', 'samples', 'genes_available', 'genes_used',
              'selected_gene_ids', 'pca', 'heatmap', 'correlation', 'qc', 'provenance')
    return {**{k: a[k] for k in fields}, 'source': public_source(a['source'])}


def public_experiments(result):
    if (result['method'], result['holdout_study'], result['purpose'], result['train_samples'],
            result['validation_samples']) != ('elastic_net', 'GSE8917', 'engineering_smoke', 120, 25):
        raise ValueError('The reviewed engineering run changed')
    fields = ('method', 'holdout_study', 'purpose', 'train_samples', 'validation_samples', 'metrics')
    rows = [{k: row[k] for k in ('sample_id', 'group_id', 'observed_gy', 'predicted_gy')}
            for row in result['predictions']]
    return dict(schema_version=1, runs=[dict(
        run_id='elastic-net-gse8917-engineering', label='Elastic Net',
        release_id='core_' + DATASET['sha256'][:16],
        result={**{k: result[k] for k in fields}, 'predictions': rows},
        methods='Preprocessing fitted on the two training studies; GSE8917 held out. Fixed engineering settings. Donor-balanced errors.',
        limitations='One engineering check, not final model selection or locked-test evaluation. No clinical prediction model is published.')])


def build(published_date, evidence_dir=EVIDENCE):
    date.fromisoformat(published_date)
    store = LocalRelease()
    explorer = Explorer(DATASET)
    catalogue = explorer.catalog(store)
    if catalogue['profiles'] != 145 or catalogue['locked_test_included']:
        raise ValueError('Unexpected public cohort')
    exports = {'catalog.json': encode(public_catalog(catalogue))}
    for study, representation, genes in product(('all', 'GSE102971', 'GSE8917', 'GSE65292'),
                                                 ('log2', 'quantile'), (100, 500, 2000)):
        options = dict(study=study, representation=representation, genes=genes)
        output = public_analysis(explorer.analyze(store, options))
        exports[f'analysis-{study}-{representation}-{genes}.json'] = encode(output)
    exports['experiments.json'] = encode(public_experiments(json.loads(RESULT.read_text())))
    final_results, _ = build_payload()
    exports['research-results.json'] = encode(final_results)
    development, _ = build_development()
    exports['development-results.json'] = encode(development)
    for name in ('statistical-results.json','assistant-records.json'):
        exports[name] = (ROOT / 'app/static' / name).read_bytes()
    version = sha(encode({k: sha(v) for k, v in exports.items()}))[:20]
    # Fixed generated destination; unknown files fail closed rather than being uploaded.
    if DEST.is_symlink():
        raise ValueError('Build directory must not be a symlink')
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    bundle = dict(exports)
    bundle = {'data/' + version + '/' + k: v for k, v in bundle.items()}
    bundle['data/publication.json'] = encode(dict(
        schema_version=1, published_date=published_date, data_path='/data/' + version,
        profiles=145, analysis_combinations=24, public_runs=1,
        release_id=catalogue['release_id'], locked_test_included=False,
        final_evaluation=dict(release_id=final_results['release_id'],profiles=52,donors=12,status='completed_consumed',model_cards=8)))
    bundle['robots.txt'] = b'User-agent: *\nDisallow: /\n'
    for target, source in FILES.items():
        path = ROOT / source
        if path.is_symlink():
            raise ValueError('Symlinks cannot enter the public bundle')
        bundle[target] = path.read_bytes()
    for name, payload in bundle.items():
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    manifest = {k: dict(bytes=len(v), sha256=sha(v)) for k, v in bundle.items()}
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / 'build-allowlist.json').write_bytes(encode(manifest))
    # Current staging evidence stays outside the public directory; historic
    # evidence directories remain immutable when a different release is built.
    (ROOT / 'data/public-build-manifest.json').write_bytes(encode(manifest))
    summary = dict(destination=str(DEST), files=len(bundle), bytes=sum(map(len, bundle.values())),
                   version=version, published_date=published_date, profiles=145,
                   analysis_combinations=24, public_runs=1, locked_test_included=False,
                   original_release_sha256=sha(CORE.read_bytes()), original_result_sha256=sha(RESULT.read_bytes()))
    (evidence_dir / 'build-summary.json').write_bytes(encode(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--published-date', required=True)
    parser.add_argument('--evidence-dir', default=str(EVIDENCE))
    args = parser.parse_args()
    print(json.dumps(build(args.published_date, args.evidence_dir)))
