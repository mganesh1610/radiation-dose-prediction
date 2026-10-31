"""Acquire frozen eligible raw files; checksum/CRC/header checks only, no expression parsing."""
import gzip
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
from audit_locked_metadata import ROOT, LOG, sha
from locked_agilent_adapter import inspect_header


def acquire(row):
    p = ROOT / row['raw_local_path']; p.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(3):
        try:
            if not p.exists():
                tmp = p.with_suffix(p.suffix + '.partial')
                with urllib.request.urlopen(row['source_url'], timeout=60) as response, tmp.open('wb') as out:
                    while block := response.read(1024 * 1024): out.write(block)
                if tmp.stat().st_size != row['publisher_bytes']: raise ValueError('Publisher byte mismatch')
                tmp.replace(p)
            if p.stat().st_size != row['publisher_bytes']: raise ValueError('Publisher byte mismatch')
            expanded = 0
            with gzip.open(p, 'rb') as stream:
                while block := stream.read(1024 * 1024): expanded += len(block)
            header = inspect_header(p)
            return dict(gsm=row['gsm'], path=row['raw_local_path'], url=row['source_url'],
                        bytes=p.stat().st_size, expanded_bytes=expanded, sha256=sha(p),
                        gzip_crc_verified=True, header=header, expression_values_parsed=False,
                        interpretation='decompressed for checksum/format only; no test measurements normalized or scored')
        except Exception as exc:
            last = exc
            if attempt < 2: time.sleep(1 + attempt)
    raise last


def main():
    freeze = json.loads((LOG / 'eligibility-freeze.json').read_text())
    for n,h in freeze['sha256'].items():
        if sha(LOG / n) != h: raise ValueError('Eligibility freeze changed')
    rows = json.loads((LOG / 'eligible-samples.json').read_text()); records=[]; errors=[]
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(acquire,r):r for r in rows}
        for task in as_completed(jobs):
            try: records.append(task.result()); print('VERIFIED',len(records),jobs[task]['gsm'],flush=True)
            except Exception as e: errors.append(dict(gsm=jobs[task]['gsm'],error=str(e)))
            (LOG/'raw-integrity.json').write_text(json.dumps(dict(complete=len(records)==52 and not errors,
                    files=sorted(records,key=lambda r:r['gsm']),errors=errors,utc=datetime.now(timezone.utc).isoformat(),
                    expression_values_parsed=False,wall_seconds=time.perf_counter()-started),indent=2)+'\n')
    if errors: raise RuntimeError('Raw acquisition incomplete; see retained error records')
    print('COMPLETE raw integrity and headers only:',sum(r['bytes'] for r in records),'compressed bytes',flush=True)


if __name__=='__main__': main()
