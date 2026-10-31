"""Bounded, read-only exploratory analysis of the pinned core release.

No models, training jobs, sample exclusions, or fitted training artifacts are
written here. All-cohort normalization is explicitly for descriptive plots.
"""
import io
import threading
from collections import OrderedDict

from cloud_io import digest, encode
from training_contract import CORE, validate_data

VERSION = 'core-explorer-v1'
ORDER = tuple(CORE)
DETAILS = {
    'GSE102971': ('Human whole blood', 'Start with human whole blood.', 'GPL10332'),
    'GSE8917': ('24-hour subset', 'Follow the 24-hour response.', 'GPL1708'),
    'GSE65292': ('Acute photon exposure', 'Explore acute photon exposure.', 'GPL13497'),
}


def validate_options(options):
    if (not isinstance(options, dict) or set(options) != {'study', 'representation', 'genes'}
            or options['study'] not in ('all', *ORDER)
            or options['representation'] not in ('log2', 'quantile')
            or type(options['genes']) is not int or options['genes'] not in (100, 500, 2000)):
        raise ValueError('Choose a core study, log2 or quantile representation, and 100, 500 or 2000 genes')
    return options


def pca(matrix):
    """Samples × genes. Median-impute, center genes, no variance scaling."""
    import numpy as np
    medians = np.nanmedian(matrix, axis=0)
    x = np.where(np.isfinite(matrix), matrix, medians)
    x = x - x.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(x @ x.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.)
    scores = vectors[:, order[:2]] * np.sqrt(eigenvalues[:2])
    # Fix sign for repeatable export; ties/degeneracies may still rotate axes.
    for i in range(2):
        if scores[np.argmax(np.abs(scores[:, i])), i] < 0: scores[:, i] *= -1
    return scores, eigenvalues[:2] / eigenvalues.sum()


def pairwise_correlation(x):
    import numpy as np
    n = x.shape[0]
    values, counts = [], []
    for i in range(n):
        vr, nr = [], []
        for j in range(n):
            keep = np.isfinite(x[i]) & np.isfinite(x[j]); count = int(keep.sum())
            value = None
            if count >= 3 and np.std(x[i, keep]) > 1e-12 and np.std(x[j, keep]) > 1e-12:
                value = float(np.clip(np.corrcoef(x[i, keep], x[j, keep])[0, 1], -1, 1))
            vr.append(value); nr.append(count)
        values.append(vr); counts.append(nr)
    return values, counts


class Explorer:
    """One serialized computation per instance, immutable inputs, small LRU."""
    def __init__(self, reference):
        self.reference = dict(reference)
        self.lock = threading.RLock()
        self.data = None
        self.catalogue = None
        self.views = None
        self.cache = OrderedDict()

    def load(self, cloud):
        import numpy as np
        if self.data is not None: return
        ref = self.reference
        payload = cloud.read(ref['object'], ref['generation'], ref['sha256'], limit=12_000_000)
        if len(payload) != ref['bytes'] or digest(payload) != ref['sha256']:
            raise ValueError('Core release integrity mismatch')
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        validate_data(data)
        for value in data.values(): value.setflags(write=False)
        self.data = data

    def catalog(self, cloud):
        import numpy as np
        with self.lock:
            self.load(cloud)
            if self.catalogue is not None: return self.catalogue
            d = self.data
            samples = []
            for i, sample in enumerate(d['sample_ids']):
                samples.append(dict(sample_id=str(sample), study_id=str(d['study_ids'][i]),
                    group_id=str(d['split_group_ids'][i]), dose_gy=float(d['y'][i]),
                    masked_common_probes=int((~np.isfinite(d['values'][:, i])).sum())))
            studies = []
            for study in ORDER:
                rows = [r for r in samples if r['study_id'] == study]
                subtitle, heading, platform = DETAILS[study]
                studies.append(dict(study_id=study, subtitle=subtitle, heading=heading,
                    platform=platform, profiles=len(rows), groups=len({r['group_id'] for r in rows}),
                    doses_gy=sorted({r['dose_gy'] for r in rows}), role='Core development',
                    source_url='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+study))
            self.catalogue = dict(release_id='core_'+self.reference['sha256'][:16],
                source=self.reference, studies=studies, samples=samples,
                profiles=len(samples), groups=len(set(d['split_group_ids'])),
                probes=len(d['probe_ids']), annotation_genes=len(set(d['gene_keys'])),
                representation='Per-array background-corrected linear intensities; technical failures masked.',
                group_identity='Study-local groups; global donor identity is not independently verified.',
                locked_test_included=False)
            return self.catalogue

    def prepare_views(self):
        import numpy as np
        from core_transfer import CoreTransfer
        from assay_qc import aggregate_median
        if self.views is not None: return
        d = self.data; valid = np.isfinite(d['values'])
        # All-core fitting is permitted for descriptive views only. The Mac
        # training worker continues to refit independently within each fold.
        q = CoreTransfer.fit(d['values'], valid, d['detected'], d['probe_ids'], d['gene_keys'],
            list(d['sample_ids']), d['study_ids'], list(d['sample_ids']))
        genes, normalized, _ = q.transform(d['values'], valid, d['probe_ids'])
        raw_genes, raw, _ = aggregate_median(np.log2(d['values'][q.keep]), d['gene_keys'][q.keep])
        if not np.array_equal(genes, raw_genes): raise ValueError('Exploration gene axes disagree')
        self.views = dict(genes=genes, log2=raw, quantile=normalized, probes=int(q.keep.sum()))
        for value in (genes, raw, normalized): value.setflags(write=False)

    def analyze(self, cloud, options):
        import numpy as np
        validate_options(options)
        key = encode(options)
        with self.lock:
            self.load(cloud)
            if key in self.cache:
                self.cache.move_to_end(key); return self.cache[key]
            self.prepare_views(); d = self.data; view = self.views
            take = np.ones(145, bool) if options['study'] == 'all' else d['study_ids'] == options['study']
            indices = np.flatnonzero(take)
            # Stable, interpretable column order shared by all output plots.
            indices = np.array(sorted(indices, key=lambda i:(ORDER.index(d['study_ids'][i]), d['y'][i], d['sample_ids'][i])))
            x = view[options['representation']][:, indices]
            eligible = (np.isfinite(x).mean(axis=1) >= .95) & (np.nanvar(x, axis=1) > 1e-12)
            available = np.flatnonzero(eligible)
            ranked = available[np.argsort(-np.nanvar(x[available], axis=1), kind='stable')]
            chosen = ranked[:options['genes']]
            if len(chosen) < 2: raise ValueError('Too few variable genes for this view')
            scores, variance = pca(x[chosen].T)
            catalogue = self.catalog(cloud)
            lookup = {r['sample_id']:r for r in catalogue['samples']}
            profiles = [lookup[str(d['sample_ids'][i])] for i in indices]
            heat = x[chosen[:20]]
            z = (heat - np.nanmean(heat, axis=1, keepdims=True)) / np.nanstd(heat, axis=1, keepdims=True)
            nullable = lambda a: [[None if not np.isfinite(v) else float(v) for v in row] for row in a]
            correlations, pairs = pairwise_correlation(heat[:12])
            result = dict(schema_version=1, analysis_version=VERSION,
                analysis_id=digest(encode(dict(version=VERSION,source=self.reference,options=options)))[:24],
                release_id=catalogue['release_id'], source=self.reference, options=options,
                purpose='exploratory_only_not_training_or_validation', locked_test_included=False,
                samples=profiles, genes_available=int(eligible.sum()), genes_used=len(chosen),
                selected_gene_ids=view['genes'][chosen].tolist(),
                pca=dict(scores=scores.tolist(), explained_variance_ratio=variance.tolist(),
                    method='Gene-median imputation, gene centering, no variance scaling; PCA refitted to displayed profiles.'),
                heatmap=dict(gene_ids=view['genes'][chosen[:20]].tolist(), log2=nullable(heat), zscore=nullable(z)),
                correlation=dict(gene_ids=view['genes'][chosen[:12]].tolist(), pearson=correlations, paired_counts=pairs),
                qc=dict(missing_selected_values=int((~np.isfinite(x[chosen])).sum()),
                    missing_by_sample=(~np.isfinite(x[chosen])).sum(axis=0).tolist(),
                    quantiles=np.nanquantile(x[chosen], [.05,.25,.5,.75,.95], axis=0).T.tolist(),
                    automatic_exclusions=0),
                provenance=dict(background='Per-array limma normexp saddle, offset 0, after technical masking.',
                    probe_filter='All 145 core profiles: >=95% usable and >=20% detected in each study.',
                    eligible_common_probes=view['probes'],
                    normalization=('Equal-study masked quantile reference fitted to all 145 core profiles, then log2.'
                        if options['representation']=='quantile' else 'Log2 per-array corrected intensities; no between-array normalization.'),
                    gene_aggregation='Median log2 intensity of eligible probes mapped to each NCBI GeneID.',
                    feature_selection='Highest variance genes within the displayed cohort; >=95% observed.',
                    heatmap='Top 20 genes; optional row z-score (population SD), missing cells retained.',
                    correlation='Top 12 genes; pairwise-complete Pearson r, at least 3 profiles; no hypothesis tests.',
                    caution='All-cohort exploration cannot be reused as fold-fitted training input. Study separation can reflect assay/batch differences. No samples are automatically removed.'))
            # Disallow non-JSON floats and bound the number of retained results.
            encode(result)
            self.cache[key] = result
            if len(self.cache) > 4: self.cache.popitem(last=False)
            return result
