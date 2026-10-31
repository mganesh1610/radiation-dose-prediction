"""Build an explicit web-only upload directory. Never package the repository root."""
from pathlib import Path
import hashlib
import json
import shutil
import argparse
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/web_build'
FILES=['scripts/frozen_inference.py','scripts/portable_model.py','scripts/prediction_service.py',
       'app/static/prediction-workflow.js','app/static/research-workflow.css','app/static/statistical-explorer.js','app/static/statistical-results.json',
       'app/static/assistant-engine.js','app/static/research-assistant.js','app/static/assistant-records.json','app/server.py','app/requirements.txt','app/Dockerfile','app/static/index.html',
       'app/static/development-explorer.js','app/static/development-explorer.css','app/static/development-results.json',
       'app/static/reviewer-results.js','app/static/reviewer-results.css','app/static/research-results.json',
       'app/static/style.css','app/static/workspace-design.css','app/static/app.js','scripts/cloud_io.py','scripts/job_queue.py',
       'scripts/training_contract.py','scripts/application_service.py','scripts/dispatch_control.py',
       'app/static/fonts/geist.woff2','app/static/fonts/geist.woff2.LICENSE',
       'app/static/fonts/ibm-plex-mono.woff2','app/static/fonts/ibm-plex-mono.woff2.LICENSE',
       'app/static/explorer.js','app/static/heatmap-clustering.js','app/static/explorer.css','app/static/assets/cell-hero.png',
       'scripts/analysis_service.py','scripts/core_transfer.py','scripts/assay_qc.py','scripts/masked_quantile.py',
       'app/copilot_proxy.py','app/request_admission.py','app/startup_dependency.py',
       'app/static/copilot-chat.js','app/static/copilot-chat.css']
# Explicit source icons used by the current chat interface; no generated data.
FILES += ['app/static/icons/phosphor/'+name+'.svg' for name in (
    'arrow-left','arrow-right','arrow-up','arrows-clockwise','book-open',
    'caret-down','chat-circle-text','check','clock','copy','database',
    'dots-three','flask','info','magnifying-glass','note-pencil','shield-check',
    'sidebar-simple','squares-four','stop','trash','x')]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence-dir',default='context/cloud/app-v2');args=parser.parse_args()
    # This directory is generated exclusively by this script.
    if DEST.exists():shutil.rmtree(DEST)
    for name in FILES:
        source=ROOT/name
        if source.is_symlink():raise ValueError('Symlink cannot enter deployment bundle')
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    frozen=ROOT/'data/cloud_prediction/frozen-v1'
    model_manifest=json.loads((frozen/'manifest.json').read_text())
    from frozen_inference import FrozenInference
    FrozenInference(frozen)
    runtime=[]
    for filename in list(model_manifest['files'])+['manifest.json']:
        target=DEST/'runtime/frozen'/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(frozen/filename,target);runtime.append('runtime/frozen/'+filename)
    files=FILES+runtime
    manifest={name:{'bytes':(DEST/name).stat().st_size,'sha256':hashlib.sha256((DEST/name).read_bytes()).hexdigest()} for name in files}
    evidence=ROOT/args.evidence_dir;evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'build-allowlist.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'files':len(files),'bytes':sum(v['bytes'] for v in manifest.values()),'destination':str(DEST)}))

if __name__=='__main__':main()
