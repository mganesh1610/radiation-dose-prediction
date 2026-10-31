"""Run each Python test module in a fresh interpreter.

PyTorch and XGBoost can load conflicting OpenMP runtimes in one macOS process.
Process isolation avoids that collision without suppressing runtime safeguards.
Pytest executes both unittest classes and pytest functions. Artifact-dependent
tests still skip explicitly (or fail in strict mode).
"""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    modules = sorted((root / 'tests').glob('test_*.py'))
    if not modules:
        raise SystemExit('No test modules found')
    failed = []
    for module in modules:
        print(f'\nRunning {module.name}', flush=True)
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', str(module), '-q'], cwd=root)
        if result.returncode:
            failed.append(module.name)
    print(f'\n{len(modules)} test modules; {len(failed)} failed.', flush=True)
    if failed:
        print('Failed: ' + ', '.join(failed), file=sys.stderr)
    return bool(failed)


if __name__ == '__main__':
    sys.exit(main())
