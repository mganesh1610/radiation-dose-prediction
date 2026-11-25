"""Shared file map and template composition for the released dashboard variants."""
import json
import re
from itertools import product

from overview_ui_assets import OVERVIEW_UI_FILES
from ui_sdk_assets import UI_SDK_FILES
from experiments_ui_assets import EXPERIMENT_UI_FILES

FILES = {
    **{'static/' + name: 'app/static/' + name for name in (*UI_SDK_FILES, *EXPERIMENT_UI_FILES, *OVERVIEW_UI_FILES)},
    **{'static/' + name: 'app/static/' + name for name in ('ui-workspace.mjs', 'ui-workspace.css', 'workspace-system.css')},
    'index.html': 'app/public/index.html',
    'vercel.json': 'app/public/vercel.json',
    **{'static/' + n: 'app/public/' + n for n in
       ('public-data.js', 'public-results.js', 'public.css')},
    **{'static/' + n: 'app/static/' + n for n in
       ('pca-outliers.js','analysis-workspace.js','analysis-workspace.css','analysis-scene.mjs','spatial-model.mjs',
        'vendor/three/three.module.js','vendor/three/three.core.js','vendor/three/OrbitControls.js','vendor/three/LICENSE',
        'statistical-explorer.js','research-workflow.css','assistant-engine.js','research-assistant.js','style.css', 'workspace-design.css', 'explorer.css', 'explorer.js', 'heatmap-clustering.js', 'research-catalog.js','reviewer-results.js', 'reviewer-results.css', 'development-explorer.js', 'development-explorer.css', 'assets/cell-hero.png',
        'fonts/geist.woff2', 'fonts/geist.woff2.LICENSE',
        'fonts/ibm-plex-mono.woff2', 'fonts/ibm-plex-mono.woff2.LICENSE')},
}


PREDICTION_ASSETS = ('awaiting-profile.svg', 'browse-arrow.svg', 'cell-contours.svg', 'compatibility-approved.svg',
                     'download.svg', 'geo-link.svg', 'material-icons.LICENSE', 'predict-arrow-disabled.svg',
                     'predict-arrow.svg', 'selected-profile.svg', 'upload.svg')
CHAT_ICONS = ('arrow-right', 'arrow-up', 'arrows-clockwise', 'book-open', 'check', 'copy', 'database',
              'dots-three', 'flask', 'info', 'magnifying-glass', 'note-pencil', 'shield-check',
              'sidebar-simple', 'squares-four', 'stop', 'trash', 'x')
GUEST_VARIANTS = ('copilot-chat.js', 'copilot-chat.css', 'prediction-workflow.js', 'predictions.css')
GUEST_EXTRAS = {
    **{'static/' + name: 'app/public/' + name for name in
       (*GUEST_VARIANTS, 'guest-chat.js', 'guest-workspace.js', 'guest-workspace.css')},
    **{'static/' + name: 'app/static/' + name for name in (
        'fonts/geist-mono.woff2', 'fonts/geist-mono.woff2.LICENSE', 'icons/phosphor/caret-down.svg',
        *('assets/predictions/' + name for name in PREDICTION_ASSETS),
        *('icons/phosphor/' + name + '.svg' for name in CHAT_ICONS))},
}
TEAM_EXTRAS = (
    'predictions.css', 'fonts/geist-mono.woff2', 'fonts/geist-mono.woff2.LICENSE',
    *('assets/predictions/' + name for name in PREDICTION_ASSETS),
    'icons/phosphor/LICENSE',
    *('icons/phosphor/' + name + '.svg' for name in
      (*CHAT_ICONS, 'caret-down', 'arrow-left', 'chat-circle-text', 'clock')),
)
RECORDS = ('catalog', 'experiments', 'research-results', 'development-results',
           'statistical-results', 'assistant-records',
           *('analysis-{}-{}-{}'.format(*choice) for choice in product(
               ('all', 'GSE102971', 'GSE8917', 'GSE65292'), ('log2', 'quantile'), (100, 500, 2000))))


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def share_template(public, workspace):
    start = '<div id="page-predictions"'
    end = '<div id="page-assistant"'
    if any(page.count(marker) != 1 for page in (public, workspace) for marker in (start, end)):
        raise ValueError('Cannot locate the Predictions section')
    prediction = workspace[workspace.index(start):workspace.index(end)]
    public = public[:public.index(start)] + prediction + public[public.index(end):]
    public = re.sub(r'<a class="exp-button" href="[^"]+">Open live training →</a>', '', public)
    public = public.replace('Live training · sign in ↗', 'Team sign-in ↗')
    public = public.replace('<section class="exp-card exp-published"', '<section id="guest-training" class="guest-work" aria-label="Guest training"></section><section class="exp-card exp-published"')
    if public.count('id="guest-training"') != 1 or public.count('id="research-assistant"') != 1:
        raise ValueError('Missing guest workspace mount')
    script = '<script src="/static/research-assistant.js" defer></script>'
    style = '<link rel="stylesheet" href="/static/workspace-design.css">'
    if public.count(script) != 1 or public.count(style) != 1:
        raise ValueError('Cannot locate the shared copilot dependencies')
    public = public.replace(script, '<script defer src="/static/guest-chat.js"></script>'
                            '<script defer src="/static/copilot-chat.js"></script>'+script)
    public = public.replace(style, style+'<link rel="stylesheet" href="/static/copilot-chat.css">')
    public = public.replace('</head>', '<link rel="stylesheet" href="/static/guest-workspace.css">'
                            '<script defer src="/static/guest-workspace.js"></script></head>')
    return public.replace('</head>', '<link rel="icon" href="/static/assets/experiments/shield.svg">'
                           '<link rel="stylesheet" href="/static/predictions.css">'
                           '<script defer src="/static/prediction-workflow.js"></script></head>')
