"""Create auditable candidate selections. Donor groups from titles are provisional."""
import csv,json,re,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'context/dataset-acquisition'
records=list(csv.DictReader((LOG/'sample-metadata.tsv').open(),delimiter='\t'))
selected=[]
for r in records:
    study=r['study'];title=r['title'];species=r.get('organism_ch1','')
    keep=(study=='GSE102971' and species=='Homo sapiens') or (study=='GSE8917' and '_24hr_' in title) or (study=='GSE65292' and ('Acute_' in title or title.startswith('0Gy_')))
    photon=study=='GSE90909' and ('X-ray' in title or 'Control' in title)
    if not (keep or photon):continue
    chars=r.get('characteristics_ch1','')
    dose=re.search(r'(?:^| \| )dose: ([\d.]+)\s*Gy',chars,re.I)
    if dose: numerical=float(dose.group(1));evidence='sample characteristics'
    elif 'Control' in title:numerical=0.;evidence='control title'
    else:
        dose=re.search(r'([\d.]+)\s*Gy',title)
        numerical=float(dose.group(1));evidence='sample title'
    donor=re.search(r'donor: (Donor \d+)',chars)
    group=donor.group(1) if donor else (title.rsplit('_',1)[1] if study=='GSE102971' else re.search(r'rep\d+',title).group())
    selected.append({'study':study,'gsm':r['gsm'],'title':title,'role':r['role'],
                     'dose_gy':numerical,'dose_evidence':evidence,'donor_group':study+':'+group,
                     'donor_status':'explicit metadata' if donor else 'title-derived; verify before training',
                     'eligibility':'candidate; protocol and donor audit required',
                     'issue':'2.2 vs 2.25 Gy discrepancy in series summary' if study=='GSE65292' and numerical==2.2 else ''})
with (LOG/'candidate-selection.tsv').open('w') as s:
    w=csv.DictWriter(s,fieldnames=list(selected[0]),delimiter='\t');w.writeheader();w.writerows(selected)
gene_sets={}
for p in (ROOT/'data/public_radiation/core').glob('*/*series_matrix*gz'):
    ids=set()
    with gzip.open(p,'rt') as s:
        table=False
        for line in s:
            if line.startswith('!series_matrix_table_begin'):table=True;continue
            if line.startswith('!series_matrix_table_end'):break
            if table and not line.startswith('"ID_REF"'):ids.add(line.split('\t')[0].strip('"'))
    genes=set();ambiguous=0;unmapped=0
    soft=next(p.parent.glob('*soft.gz'))
    with gzip.open(soft,'rt') as s:
        for line in s:
            if line.startswith('!platform_table_begin'):break
        reader=csv.DictReader(s,delimiter='\t')
        for row in reader:
            if row['ID'].startswith('!platform_table_end'):break
            if row['ID'] not in ids:continue
            symbol=row.get('GENE_SYMBOL','').strip()
            if not symbol:unmapped+=1
            elif any(x in symbol for x in ['///',';','//',',']):ambiguous+=1
            else:genes.add(symbol)
    gene_sets[p.parent.name]=genes
    print(p.parent.name,'candidate symbols',len(genes),'unmapped',unmapped,'ambiguous',ambiguous)
summary={'candidate_samples':{k:sum(r['study']==k for r in selected) for k in sorted({r['study'] for r in selected})},
         'candidate_core_common_symbols':len(set.intersection(*gene_sets.values())),
         'warning':'Symbols are original platform annotations, not current validated gene IDs. No expression-value filtering or preprocessing fitted.'}
(LOG/'selection-summary.json').write_text(json.dumps(summary,indent=2));print(summary)
