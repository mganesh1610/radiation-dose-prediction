"""Check staged files or an entire commit tree before publication.

This is a bounded pattern/path check, not proof that arbitrary secrets cannot
exist. Inspect the file list and diff as well. Never prints matched secret values.
"""
import argparse
from pathlib import PurePosixPath
import re
import subprocess
import sys

ROOT_FILES = {'.gitignore', '.env.example', 'README.md', 'requirements.txt',
              'requirements-mac-training.txt', 'requirements-test.txt', 'vercel.json'}
ROOT_DIRS = {'app', 'scripts', 'schemas', 'tests', 'docs', 'agent_system'}
SUFFIXES = {'.py', '.R', '.md', '.txt', '.json', '.jsonc', '.js', '.mjs', '.cjs',
            '.html', '.css', '.svg', '.LICENSE', '.toml', '.lock'}
BINARY = {'app/static/assets/cell-hero.png', 'app/static/fonts/geist.woff2', 'app/static/fonts/ibm-plex-mono.woff2'}
RULES = {
    'private-key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'google-api-key': rb'AIza[0-9A-Za-z_-]{35}',
    'github-token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'google-access-token': rb'ya29\.[A-Za-z0-9_-]{20,}',
    'oauth-code': rb'4/0A[A-Za-z0-9_-]{35,}',
    'aws-access-key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'jwt': rb'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
    'private-email': rb'[A-Za-z0-9._%+-]+@(?:asu\.edu|gmail\.com)\b',
    'absolute-user-path': rb'(?:/Users/[A-Za-z0-9_-]+/|C:\\Users\\[A-Za-z0-9_-]+\\)',
}


def git(*args):
    return subprocess.check_output(['git', *args])


def audit(tree=None):
    rows = git('ls-tree', '-rz', '--full-tree', tree) if tree else git('ls-files', '--stage', '-z')
    files = []
    problems = []
    for row in rows.split(b'\0'):
        if not row:
            continue
        info, path_bytes = row.split(b'\t', 1)
        path = path_bytes.decode()
        fields = info.decode().split()
        mode, sha = (fields[0], fields[2]) if tree else (fields[0], fields[1])
        p = PurePosixPath(path)
        if mode not in ('100644', '100755'):
            problems.append((path, 'non-regular-file'))
        if (path not in ROOT_FILES and (p.parts[0] not in ROOT_DIRS or
            (p.suffix not in SUFFIXES and path not in BINARY and p.name != 'Dockerfile'))):
            problems.append((path, 'outside-submission-allowlist'))
        if any(x in ('context', 'data', 'output', 'config', 'runtime', '.git', '.venv',
                     'node_modules', '__pycache__', '.vercel', '.wrangler')
               or x.startswith('.venv-') for x in p.parts) or p.name == 'AGENTS.md':
            problems.append((path, 'local-only-path'))
        if p.name.startswith('.env') and path != '.env.example':
            problems.append((path, 'environment-file'))
        payload = git('cat-file', 'blob', sha)
        if len(payload) > 5_000_000:
            problems.append((path, 'unexpected-file-size'))
        for label, pattern in RULES.items():
            if re.search(pattern, payload):
                problems.append((path, label))
        files.append((path, len(payload)))
    if not files:
        problems.append(('(tree)', 'empty-submission'))
    for path, label in problems:
        print(f'BLOCKED {path}: {label}')
    print(f'Checked {len(files)} files, {sum(n for _, n in files):,} bytes; {len(problems)} findings')
    return not problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree', help='Commit/tree to inspect; default is staged index')
    args = parser.parse_args()
    sys.exit(0 if audit(args.tree) else 1)
