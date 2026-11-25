# Approved Experiments artwork is also used by the standalone research explorer.
EXPERIMENT_UI_FILES = (
    'experiments-workspace.js',
    *('assets/experiments/' + name + '.svg' for name in (
        'arrow', 'back', 'books', 'chevron', 'clock', 'download', 'refresh',
        'right', 'search', 'shield', 'sliders', 'sort',
    )),
)
