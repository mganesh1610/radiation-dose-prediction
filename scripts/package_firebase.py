"""Stage the reviewed public file allowlist for Spark-compatible static Hosting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--project', default=os.environ.get('DEFENDER_FIREBASE_PROJECT'))
    args = parser.parse_args()
    if not args.project:
        parser.error('--project or DEFENDER_FIREBASE_PROJECT is required')
    manifest = json.loads(Path(args.manifest).read_text())
    destination = ROOT / 'data/firebase_deploy'
    public = destination / 'public'
    # Validate everything before replacing our generated public directory.
    for name, metadata in manifest.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid public path')
        source = ROOT / 'data/public_web' / name
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError('Public build has changed')
    if public.is_symlink():
        raise ValueError('Generated destination cannot be a symlink')
    if public.exists():
        shutil.rmtree(public)
    public.mkdir(parents=True)
    for name in manifest:
        if name == 'vercel.json':
            continue
        target = public / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'data/public_web' / name, target)
    headers = json.loads((ROOT / 'app/public/vercel.json').read_text())['headers']
    headers[0]['source'] = '**'
    headers[0]['headers'].append({'key':'Cache-Control','value':'public, max-age=0, must-revalidate'})
    headers[-1]['source'] = '/data/*/*.json'
    config = {'hosting': {'site':args.project, 'public':'public',
                         'ignore':['**/.*','**/node_modules/**'], 'headers':headers}}
    (destination / 'firebase.json').write_text(json.dumps(config, indent=2))
    (destination / '.firebaserc').write_text(json.dumps({'projects':{'default':args.project}}))
    print(json.dumps({'destination':str(destination),'static_files':len(manifest)-1}))


if __name__ == '__main__':
    main()
