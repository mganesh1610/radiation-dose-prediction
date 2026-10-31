"""Inventory original matrices without fitting anything to locked-test data."""
import csv,gzip,json,collections,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/dataset-acquisition'
samples=[]; inventory=[]
for path in sorted((ROOT/'data/public_radiation').glob('*/*/*matrix*txt.gz')):
    if 'series_matrix' not in path.name: continue
    meta=collections.defaultdict(list); rows=0; missing=0; nonfinite=0; minimum=None; maximum=None; columns=0
    with gzip.open(path,'rt') as stream:
        for line in stream:
            if line.startswith('!Sample_'):
                fields=next(csv.reader([line],delimiter='\t'));meta[fields[0]].append(fields[1:])
            elif line.startswith('"ID_REF"') or line.startswith('ID_REF\t'):
                columns=len(next(csv.reader([line],delimiter='\t')))-1
            elif line and not line.startswith('!') and columns:
                values=next(csv.reader([line],delimiter='\t'))[1:]
                if len(values)!=columns: raise ValueError(f'Ragged matrix {path}: {rows}')
                rows+=1
                for v in values:
                    try: x=float(v)
                    except ValueError: missing+=1;continue
                    if not math.isfinite(x):nonfinite+=1;continue
                    minimum=x if minimum is None else min(x,minimum)
                    maximum=x if maximum is None else max(x,maximum)
    ids=meta.get('!Sample_geo_accession',[[]])[0]
    for i,gsm in enumerate(ids):
        record={'study':path.parent.name,'role':path.parent.parent.name,'gsm':gsm,'matrix':path.name}
        for key,values in meta.items():
            record[key.removeprefix('!Sample_')]=' | '.join(v[i] for v in values if i<len(v))
        samples.append(record)
    inventory.append({'study':path.parent.name,'role':path.parent.parent.name,'file':path.name,
                      'samples_metadata':len(ids),'expression_columns':columns,'features':rows,
                      'missing_cells':missing,'nonfinite_cells':nonfinite,'min':minimum,'max':maximum,
                      'platforms':sorted(set(meta.get('!Sample_platform_id',[[]])[0])),
                      'species':dict(collections.Counter(meta.get('!Sample_organism_ch1',[[]])[0]))})
(LOG/'matrix-inventory.json').write_text(json.dumps(inventory,indent=2))
keys=sorted(set().union(*(r.keys() for r in samples)))
with (LOG/'sample-metadata.tsv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=keys,delimiter='\t');writer.writeheader();writer.writerows(samples)
sizes=collections.Counter();expanded=collections.Counter()
for p in (ROOT/'data/public_radiation').rglob('*'):
    if p.is_file():
        sizes[p.relative_to(ROOT/'data/public_radiation').parts[0]]+=p.stat().st_size
        if p.suffix=='.gz':
            with gzip.open(p,'rb') as s:
                while chunk:=s.read(1024*1024):expanded[p.relative_to(ROOT/'data/public_radiation').parts[0]]+=len(chunk)
        else:expanded[p.relative_to(ROOT/'data/public_radiation').parts[0]]+=p.stat().st_size
(LOG/'storage.json').write_text(json.dumps({'compressed_bytes_by_role':dict(sizes),'expanded_bytes_by_role':dict(expanded),'total_local_bytes':sum(sizes.values())},indent=2))
print(json.dumps(inventory,indent=2));print('STORAGE',dict(sizes),sum(sizes.values()))
