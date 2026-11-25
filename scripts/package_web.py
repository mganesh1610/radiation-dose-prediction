"""Build an explicit web-only upload directory. Never package the repository root."""
from pathlib import Path
import hashlib
import json
import shutil
import argparse
from public_site_assets import TEAM_EXTRAS
from overview_ui_assets import OVERVIEW_UI_FILES
from ui_sdk_assets import UI_SDK_FILES
from experiments_ui_assets import EXPERIMENT_UI_FILES
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/web_build'
FILES=['scripts/frozen_inference.py','scripts/portable_model.py','scripts/prediction_service.py',
       'app/static/pca-outliers.js','app/static/analysis-workspace.js','app/static/analysis-workspace.css','app/static/analysis-scene.mjs','app/static/spatial-model.mjs',
       'app/static/vendor/three/three.module.js','app/static/vendor/three/three.core.js','app/static/vendor/three/OrbitControls.js','app/static/vendor/three/LICENSE',
       'app/static/prediction-workflow.js','app/static/research-workflow.css','app/static/statistical-explorer.js','app/static/statistical-results.json',
       'app/static/assistant-engine.js','app/static/research-assistant.js','app/static/assistant-records.json','app/server.py','app/requirements.txt','app/Dockerfile','app/static/index.html',
       'app/static/development-explorer.js','app/static/development-explorer.css','app/static/development-results.json',
       'app/static/research-catalog.js','app/static/reviewer-results.js','app/static/reviewer-results.css','app/static/research-results.json',
       'app/static/style.css','app/static/workspace-design.css','app/static/app.js','scripts/cloud_io.py','scripts/job_queue.py',
       'scripts/training_contract.py','scripts/application_service.py','scripts/dispatch_control.py',
       'app/static/fonts/geist.woff2','app/static/fonts/geist.woff2.LICENSE',
       'app/static/fonts/ibm-plex-mono.woff2','app/static/fonts/ibm-plex-mono.woff2.LICENSE',
       'app/static/explorer.js','app/static/heatmap-clustering.js','app/static/explorer.css','app/static/assets/cell-hero.png',
       'scripts/analysis_service.py','scripts/core_transfer.py','scripts/assay_qc.py','scripts/masked_quantile.py']
FILES += ['app/static/' + name for name in (*UI_SDK_FILES, *EXPERIMENT_UI_FILES, *OVERVIEW_UI_FILES, 'ui-workspace.mjs', 'ui-workspace.css', 'workspace-system.css')]
FILES += ['app/static/' + name for name in TEAM_EXTRAS]
FILES += ['app/copilot_proxy.py', 'app/request_admission.py', 'app/startup_dependency.py',
          'app/static/copilot-chat.js', 'app/static/copilot-chat.css', 'scripts/training_bridge.py',
          'agent_system/src/research_agents/__init__.py', 'agent_system/src/research_agents/policies/__init__.py',
          'agent_system/src/research_agents/policies/demo_quota.py', 'agent_system/src/research_agents/policies/demo_quota.json']

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
    from frozen_inference import FrozenInference, HELD_OUT_FOLDER, WBC_FILE
    engine = FrozenInference(frozen)
    engine.catalog(prediction_picker=True)
    target=DEST/'runtime'/WBC_FILE
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(frozen.parent/WBC_FILE,target)
    runtime=[target.relative_to(DEST).as_posix()]
    for filename in list(model_manifest['files'])+['manifest.json']:
        target=DEST/'runtime/frozen'/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(frozen/filename,target);runtime.append('runtime/frozen/'+filename)
    extra = frozen.parent / HELD_OUT_FOLDER
    if extra.is_dir():
        names = ['manifest.json'] + [row['file'] for directory, row in engine.example_index.values() if directory == extra]
        for name in names:
            target = DEST / 'runtime' / HELD_OUT_FOLDER / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(extra / name, target)
            runtime.append(target.relative_to(DEST).as_posix())
    files=FILES+runtime
    manifest={name:{'bytes':(DEST/name).stat().st_size,'sha256':hashlib.sha256((DEST/name).read_bytes()).hexdigest()} for name in files}
    evidence=ROOT/args.evidence_dir;evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'build-allowlist.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'files':len(files),'bytes':sum(v['bytes'] for v in manifest.values()),'destination':str(DEST)}))

if __name__=='__main__':main()
