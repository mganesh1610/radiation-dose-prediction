"""Source publication must reject local/private inputs and retain web dependencies."""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_source(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def staged_audit(files):
    module = load_source('check_submission')
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        subprocess.run(['git', 'init', '-q', str(folder)], check=True)
        for relative, text in files.items():
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        subprocess.run(['git', '-C', str(folder), 'add', '--all'], check=True)
        with patch.object(module, 'git', lambda *args: subprocess.check_output(
                ['git', '-C', str(folder), *args])):
            with contextlib.redirect_stdout(io.StringIO()):
                return module.audit()


def test_reviewed_runtime_and_documentation_paths_are_allowed():
    assert staged_audit({
        'README.md': '# Source',
        'requirements-test.txt': 'pytest',
        'vercel.json': '{}',
        'agent_system/pyproject.toml': '[project]',
        'agent_system/requirements-example.lock': 'pytest',
        'app/static/icons/example.svg': '<svg/>',
        'docs/examples/copilot-config.example.json': '{}',
    })


def test_private_context_and_agent_instructions_are_rejected():
    for path in ('context/notes.md', 'docs/context/copy.md', 'AGENTS.md',
                 'docs/AGENTS.md', 'data/records.json', 'config/policy.json',
                 'runtime/model.json', 'agent_system/.venv-agents/private.py'):
        assert not staged_audit({path: 'local only'}), path


def test_credentials_are_rejected_even_in_an_allowed_source_path():
    key = '-----BEGIN ' + 'PRIVATE KEY-----'
    assert not staged_audit({'scripts/accidental_key.py': key})


def test_web_bundle_contains_imported_adapters_and_chat_assets():
    packager = load_source('package_web')
    for path in ('app/copilot_proxy.py', 'app/request_admission.py',
                 'app/startup_dependency.py', 'app/static/copilot-chat.js',
                 'app/static/copilot-chat.css'):
        assert path in packager.FILES
        assert (ROOT / path).is_file()
    tree = ast.parse((ROOT / 'app/server.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            candidate = 'app/' + node.module.replace('.', '/') + '.py'
            if (ROOT / candidate).is_file():
                assert candidate in packager.FILES, candidate
    for path in packager.FILES:
        if not path.endswith('-results.json') and not path.endswith('assistant-records.json'):
            assert (ROOT / path).is_file(), path
        assert not any(p in Path(path).parts for p in ('context', 'data', 'config'))
