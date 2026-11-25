"""Build the guest dashboard from this checkout and a verified public release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError
from urllib.request import build_opener, HTTPRedirectHandler, Request

from public_site_assets import FILES, GUEST_EXTRAS, RECORDS, encode, share_template

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 'https://prod.defender-asu.workers.dev'
PUBLICATION = '648e43ad846b70c1d40ab106a30ec6ba72315dbe88c06bf81cc5dfca96a255ff'
RECORD_INDEX = 'c2016adbd32633a44c98a2e77976f1050d309314b9883897d440cf741032c235'
CATALOG = 'be8c7667f5da1a63b3790ea42e55764cd8405c147cb654bf06da5167b83aaf56'
ESTIMATES = '05c83a14c00faee9cb402ae83fc59ba3033b5ef3fbbedf8658ba3a2c33562450'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verified(raw, expected, name):
    if digest(raw) != expected:
        raise ValueError(f'{name} differs from the public release used by this checkout')
    return raw


class SameOriginOnly(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download(path, body=None):
    # The prediction endpoint only retrieves published estimates for curated pairs.
    request = Request(ORIGIN + path, data=encode(body) if body is not None else None,
                      headers={'User-Agent': 'Defender-Public-Preview/1.0',
                               'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            with build_opener(SameOriginOnly()).open(request, timeout=25) as response:
                raw = response.read(4_000_001)
                if len(raw) > 4_000_000:
                    raise ValueError(f'Unexpected response size at {path}')
                return raw
        except HTTPError as error:
            if error.code not in (429, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(1 + attempt)


def write_file(directory, name, raw):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def prepare(destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Choose a new output directory. {destination} already exists.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix='.defender-preview-') as temporary:
        site = Path(temporary) / 'site'
        publication = verified(download('/data/publication.json'), PUBLICATION, 'Publication')
        release = 'data/' + RECORD_INDEX[:20]
        if json.loads(publication)['data_path'] != '/' + release:
            raise ValueError('Unexpected public release directory')
        write_file(site, 'data/publication.json', publication)

        def record(name):
            filename = name + '.json'
            raw = download(f'/{release}/{filename}')
            write_file(site, f'{release}/{filename}', raw)
            return filename, digest(raw)

        def estimate(pair):
            sample, model = pair
            response = download('/api/predict', {'sample_id': sample, 'model': model})
            return sample, model, json.loads(response)

        with ThreadPoolExecutor(max_workers=4) as pool:
            verified(encode(dict(pool.map(record, RECORDS))), RECORD_INDEX, 'Research records')
            catalog = json.loads(download('/api/prediction-catalog'))
            verified(encode(catalog), CATALOG, 'Published profile catalog')
            profiles = {example['id']: {} for example in catalog['examples']}
            for sample, model, result in pool.map(estimate, product(profiles, catalog['methods'])):
                profiles[sample][model] = result
        archive = encode({'catalog': catalog, 'profiles': profiles})
        write_file(site, 'published/estimates.json', verified(archive, ESTIMATES, 'Published estimates'))
        for target, source in {**FILES, **GUEST_EXTRAS}.items():
            if target == 'vercel.json':
                continue
            if target == 'index.html':
                html = share_template((ROOT / source).read_text(), (ROOT / 'app/static/index.html').read_text())
                write_file(site, target, html.encode())
            else:
                path = site / target
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / source, path)
        write_file(site, 'robots.txt', b'User-agent: *\nDisallow: /\n')
        site.rename(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/shared_site')
    args = parser.parse_args()
    prepare(args.output)
    print(f'Guest dashboard prepared in {args.output}')
