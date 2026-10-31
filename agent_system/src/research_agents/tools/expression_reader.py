"""Authorized exact matrix slices; no fitting, imputation, normalization or pooling."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import re
import sys
import threading
from types import MappingProxyType
import zipfile

import numpy as np

from ..audit_cloud import CloudScope, ReadOnlyCloud
from ..contracts.expression import ExpressionPlan
from .query_sql import ApprovedRelease, digest
from .sample_queries import IntegrityError

READER_VERSION = "core-expression-v1"
MAX_COMPRESSED = 16 * 1024**2
MAX_EXPANDED = 32 * 1024**2
LOAD_RESERVATION = 40 * 1024**2
MAX_CELLS = 5000
MAX_RESPONSE = 512_000


@dataclass(frozen=True)
class MatrixAdmission:
    scope: CloudScope
    core: ApprovedRelease
    expression_set_id: str
    object_name: str
    generation: str
    sha256: str
    compressed_bytes: int
    features: tuple[tuple[str, str], ...]  # (probe ID, NCBI Gene ID), no averaging
    mapping_sha256: str
    table_etags: tuple[tuple[str, str], ...]
    representation: str = "core_common_probes_background_corrected"
    scale: str = "linear_intensity"
    processing_scope: str = "per_sample"
    normalization_status: str = "between_array_normalization_deferred_to_training_fold"
    fingerprint: str = field(init=False)

    def __post_init__(self):
        if (self.object_name != self.scope.prefix + self.core.release_id + "/core.npz"
                or not re.fullmatch(r"[0-9]+", self.generation) or int(self.generation) <= 0
                or not re.fullmatch(r"[a-f0-9]{64}", self.sha256)
                or not re.fullmatch(r"[a-f0-9]{64}", self.mapping_sha256)
                or not 0 < self.compressed_bytes <= MAX_COMPRESSED
                or not 0 < len(self.features) <= 30_000
                or len({p for p, g in self.features}) != len(self.features)
                or len(self.table_etags) != 3
                or set(dict(self.table_etags)) != {"expression_sets", "samples", "features"}
                or not all(v for _, v in self.table_etags)):
            raise ValueError("Invalid admitted matrix configuration")
        for probe, gene in self.features:
            if not probe or not re.fullmatch(r"[1-9][0-9]{0,11}", gene):
                raise ValueError("Invalid admitted gene/probe mapping")
        if (self.representation, self.scale, self.processing_scope, self.normalization_status) != (
                "core_common_probes_background_corrected", "linear_intensity", "per_sample",
                "between_array_normalization_deferred_to_training_fold"):
            raise ValueError("Unsupported expression representation")
        token = digest([READER_VERSION, self.scope.__dict__, self.core.__dict__, self.expression_set_id,
                        self.object_name, self.generation, self.sha256, self.features, self.mapping_sha256,
                        self.table_etags, self.representation, self.scale, self.normalization_status])
        object.__setattr__(self, "fingerprint", token)


@dataclass(frozen=True)
class MatrixFrame:
    values: np.ndarray
    detected: np.ndarray
    sample_index: object
    probe_index: object
    resident_bytes: int


def restore_admission(data):
    """Decode an operator-approved deployment record without benchmark dependencies."""
    data=dict(data);data.pop('fingerprint',None)
    data['scope']=CloudScope(**data['scope']);core=dict(data['core'])
    core['members']=tuple(tuple(v) for v in core['members'])
    core['table_etags']=tuple(tuple(v) for v in core['table_etags'])
    data['core']=ApprovedRelease(**core)
    data['features']=tuple(tuple(v) for v in data['features'])
    data['table_etags']=tuple(tuple(v) for v in data['table_etags'])
    return MatrixAdmission(**data)


def decode_matrix(payload, admitted):
    if len(payload) != admitted.compressed_bytes or len(payload) > MAX_COMPRESSED or hashlib.sha256(payload).hexdigest() != admitted.sha256:
        raise IntegrityError("Matrix size or SHA-256 mismatch")
    p, n = len(admitted.features), len(admitted.core.members)
    shapes = {"values": (p, n), "detected": (p, n), "probe_ids": (p,), "gene_keys": (p,),
              "sample_ids": (n,), "study_ids": (n,), "split_group_ids": (n,), "y": (n,)}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            members = z.infolist()
            if len(members) != 8 or {m.filename for m in members} != {k + ".npy" for k in shapes}:
                raise ValueError("Unexpected NPZ members")
            if sum(m.file_size for m in members) > MAX_EXPANDED:
                raise ValueError("Expanded archive too large")
            # Validate declared array shape/dtype/bytes BEFORE NumPy can allocate it.
            for m in members:
                name = m.filename[:-4]
                with z.open(m) as f:
                    version = np.lib.format.read_magic(f)
                    if version == (1, 0): reader = np.lib.format.read_array_header_1_0
                    elif version == (2, 0): reader = np.lib.format.read_array_header_2_0
                    else: raise ValueError("Unsupported NPY version")
                    shape, _, dtype = reader(f, max_header_size=10_000)
                    if shape != shapes[name] or dtype.hasobject:
                        raise ValueError("Unsafe shape or object dtype")
                    if name in ("values", "y") and dtype != np.dtype("float64"):
                        raise ValueError("Expected frozen float64 representation")
                    if name == "detected" and dtype != np.dtype(bool):
                        raise ValueError("Invalid detection mask")
                    if name not in ("values", "y", "detected") and (dtype.kind != "U" or not 0 < dtype.itemsize <= 256):
                        raise ValueError("Invalid ID dtype")
                    if f.tell() + math.prod(shape) * dtype.itemsize != m.file_size:
                        raise ValueError("NPY allocation size mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False, max_header_size=10_000) as z:
            arrays = {k: z[k] for k in shapes}
        expected_samples = {s: (study, group) for s, study, group in admitted.core.members}
        sample_ids = arrays["sample_ids"].tolist()
        probe_ids = arrays["probe_ids"].tolist()
        if len(set(sample_ids)) != n or set(sample_ids) != set(expected_samples):
            raise ValueError("Sample membership mismatch")
        for s, study, group in zip(sample_ids, arrays["study_ids"], arrays["split_group_ids"], strict=True):
            if expected_samples[s] != (str(study), str(group)):
                raise ValueError("Sample axis metadata mismatch")
        actual_features = list(zip(probe_ids, arrays["gene_keys"].tolist()))
        if len(set(probe_ids)) != p or dict(actual_features) != dict(admitted.features):
            raise ValueError("Probe/gene mapping mismatch")
        values = arrays["values"]
        if np.isinf(values).any() or np.any(values <= 0):
            raise ValueError("Invalid linear intensity")
        if not np.isfinite(arrays["y"]).all():
            raise ValueError("Invalid archived label")
        # Keep only required arrays/indices; archived dose labels are not predictions.
        detected = arrays["detected"]
        values.flags.writeable = detected.flags.writeable = False
        sample_index = {s: i for i, s in enumerate(sample_ids)}
        probe_index = {s: i for i, s in enumerate(probe_ids)}
        size = sys.getsizeof(values) + sys.getsizeof(detected)
        for index in (sample_index, probe_index):
            size += sys.getsizeof(index) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in index.items())
        return MatrixFrame(values, detected, MappingProxyType(sample_index), MappingProxyType(probe_index), size + 256)
    except (ValueError, KeyError, TypeError, OSError, EOFError, zipfile.BadZipFile):
        raise IntegrityError("Unsafe or incompatible matrix archive") from None


class CloudMatrixSource:
    """One bounded GET-only client per invocation; no shared request-counter races."""
    def __init__(self, scope, **kwargs):
        self.cloud = ReadOnlyCloud(scope, max_requests=12, max_bytes=MAX_COMPRESSED + 1_000_000, **kwargs)
        self.matrix_downloads = 0

    def check(self, admitted):
        if self.cloud.scope != admitted.scope:
            raise IntegrityError("Unexpected source scope")
        for table, etag in admitted.table_etags:
            if self.cloud.table_metadata(table).get("etag") != etag:
                raise IntegrityError("Registry or mapping changed; re-audit required")
        meta = self.cloud.object_metadata(admitted.object_name, admitted.generation)
        if (meta.get("name") != admitted.object_name or str(meta.get("generation")) != admitted.generation
                or int(meta["size"]) != admitted.compressed_bytes):
            raise IntegrityError("Matrix object identity changed")

    def load(self, admitted):
        self.matrix_downloads += 1
        return self.cloud.object_bytes(admitted.object_name, admitted.generation, admitted.sha256, limit=MAX_COMPRESSED)

    def stats(self):
        return {"cloud_requests": self.cloud.requests, "downloaded_response_bytes": self.cloud.downloaded_bytes,
                "matrix_downloads": self.matrix_downloads}


class ExpressionReader:
    def __init__(self, cache, access, admission_lookup, source_factory, *, max_concurrency=2):
        if not 1 <= max_concurrency <= 2: raise ValueError("Unsupported concurrency")
        self.cache, self.access = cache, access
        self.lookup, self.source_factory = admission_lookup, source_factory
        self.slots = threading.BoundedSemaphore(max_concurrency)

    def run(self, principal, workspace, payload):
        plan = ExpressionPlan.model_validate(payload)
        revision = self.access.authorize(principal, workspace, plan.release_id)
        admitted = self.lookup(workspace, plan.release_id, plan.expression_set_id)
        if admitted is None or admitted.core.release_id != plan.release_id or admitted.expression_set_id != plan.expression_set_id:
            self.cache.invalidate(workspace, plan.release_id)
            raise PermissionError("Expression set is not admitted")
        member_map = {s: (study, group) for s, study, group in admitted.core.members}
        if not set(plan.sample_ids) <= set(member_map):
            raise ValueError("Sample is outside the admitted release")
        feature_map = {}
        for probe, gene in admitted.features:
            feature_map.setdefault(gene, []).append(probe)
        if not set(plan.gene_ids) <= set(feature_map):
            raise ValueError("Gene ID is not mapped in this expression set; no guessed aliases")
        selected = [(probe, gene) for gene in plan.gene_ids for probe in feature_map[gene]]
        if len(selected) > 200 or len(selected) * len(plan.sample_ids) > MAX_CELLS:
            raise ValueError("Resolved probe-by-profile result exceeds bounds")
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("Expression reader busy; retry later")
        try:
            def authorize():
                if self.access.authorize(principal, workspace, plan.release_id) != revision or self.lookup(workspace, plan.release_id, plan.expression_set_id) != admitted:
                    raise PermissionError("Access or expression admission changed")
            source = self.source_factory(admitted)
            source.check(admitted)
            def load():
                authorize()
                return decode_matrix(source.load(admitted), admitted)
            def slice_matrix(frame):
                authorize()  # Every cache hit is authorized, not merely every download.
                row_ix = [frame.probe_index[p] for p, g in selected]
                col_ix = [frame.sample_index[s] for s in plan.sample_ids]
                vals = frame.values[np.ix_(row_ix, col_ix)]
                detected = frame.detected[np.ix_(row_ix, col_ix)]
                return {"values": [[None if np.isnan(x) else float(x) for x in row] for row in vals],
                        "detected": detected.tolist(), "missing_values": int(np.isnan(vals).sum())}
            key = (workspace, plan.release_id, admitted.fingerprint)
            answer, cache_stats = self.cache.read(key, LOAD_RESERVATION, load, slice_matrix)
            source.check(admitted)
            authorize()
            result = {"status": "ok", "release_id": plan.release_id, "expression_set_id": plan.expression_set_id,
                      "samples": [{"sample_id": s, "study_id": member_map[s][0], "group_id": member_map[s][1]} for s in plan.sample_ids],
                      "features": [{"probe_id": p, "gene_id": g} for p, g in selected],
                      "gene_namespace": "NCBI_GeneID", "aggregation": "none", "axis_order": "probe_by_sample",
                      "representation": admitted.representation, "scale": admitted.scale,
                      "normalization_status": admitted.normalization_status, "processing_scope": admitted.processing_scope,
                      "comparison_note": "Between-array normalization is deferred; these values do not establish comparable biological differences across studies.", **answer}
            receipt = {"reader_version": READER_VERSION, "release_manifest_sha256": admitted.core.manifest_sha256,
                       "matrix_sha256": admitted.sha256, "object_generation": admitted.generation,
                       "mapping_sha256": admitted.mapping_sha256, "admission_sha256": admitted.fingerprint,
                       "parameters_sha256": digest(plan.model_dump()), "result_sha256": digest(result),
                       "principal_scope_sha256": digest([principal.user_id, workspace]), "access_revision": revision,
                       "result_cells": len(selected) * len(plan.sample_ids), "cache": cache_stats,
                       "completed_at_utc": datetime.now(timezone.utc).isoformat(), **source.stats()}
            output = {"answer": result, "receipt": receipt}
            if len(json.dumps(output, allow_nan=False).encode()) > MAX_RESPONSE:
                raise ValueError("Expression response exceeds byte limit")
            return output
        except Exception:
            self.cache.invalidate(workspace, plan.release_id)
            raise
        finally:
            self.slots.release()

    def bind(self, principal, workspace):
        def get_expression_values(plan: dict) -> dict:
            """Return bounded approved expression values with separate probes and provenance."""
            return self.run(principal, workspace, plan)
        return get_expression_values
