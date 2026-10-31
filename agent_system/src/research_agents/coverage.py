"""Verify an existing core release without changing cloud or scientific state."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from .audit_cloud import AuditFailure, CloudScope, ReadOnlyCloud, TABLES


KEYS = {"studies": ("study_id",), "samples": ("sample_id",),
        "sample_annotations": ("sample_id",), "sample_groups": ("group_id",),
        "features": ("probe_id",), "split_members": ("split_id", "sample_id")}


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def keyed(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(None not in key and key not in result, "Missing or duplicate record key")
        result[key] = row
    return result


def validate_matrix(payload, registry, tables):
    """Shape/ID/value parity only; does not re-establish biological eligibility."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            require(len(members) == 8 and len({m.filename for m in members}) == 8,
                    "Unexpected or duplicate NPZ members")
            require(sum(m.file_size for m in members) <= 128_000_000,
                    "NPZ expanded size exceeds limit")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            expected = {"values", "detected", "probe_ids", "gene_keys", "sample_ids",
                        "study_ids", "split_group_ids", "y"}
            require(set(archive.files) == expected, "Unknown NPZ representation")
            arrays = {name: archive[name] for name in expected}
    except AuditFailure:
        raise
    except (ValueError, OSError, zipfile.BadZipFile):
        raise AuditFailure("Invalid or unsafe NPZ archive") from None
    n, p = int(registry["samples"]), int(registry["probes"])
    require(arrays["values"].shape == (p, n), "Matrix dimensions disagree with registry")
    require(arrays["detected"].shape == (p, n) and arrays["detected"].dtype == np.bool_,
            "Detection mask shape/type mismatch")
    for name in ("sample_ids", "study_ids", "split_group_ids", "y"):
        require(arrays[name].shape == (n,), "Sample-axis annotation mismatch")
    for name in ("probe_ids", "gene_keys"):
        require(arrays[name].shape == (p,), "Feature-axis annotation mismatch")
    samples = keyed(tables["samples"], ("sample_id",))
    annotations = keyed(tables["sample_annotations"], ("sample_id",))
    features = keyed(tables["features"], ("probe_id",))
    groups = keyed(tables["sample_groups"], ("group_id",))
    studies = keyed(tables["studies"], ("study_id",))
    sample_ids, probes = arrays["sample_ids"].tolist(), arrays["probe_ids"].tolist()
    require(len(set(sample_ids)) == n and set(samples) == {(s,) for s in sample_ids}
            and set(annotations) == set(samples), "Sample membership mismatch")
    require(len(set(probes)) == p and set(features) == {(s,) for s in probes},
            "Probe membership mismatch")
    for i, sample_id in enumerate(sample_ids):
        row, label = samples[(sample_id,)], annotations[(sample_id,)]
        require(row["study_id"] == str(arrays["study_ids"][i])
                and row["group_id"] == str(arrays["split_group_ids"][i]), "Sample axis misalignment")
        require((row["group_id"],) in groups and (row["study_id"],) in studies,
                "Missing study/group reference")
        require(groups[(row["group_id"],)]["study_id"] == row["study_id"], "Group crosses studies")
        require(label["dose_gy"] is not None and np.isfinite(label["dose_gy"])
                and float(arrays["y"][i]) == label["dose_gy"], "Recorded dose mismatch or missing label")
        require(label["dose_status"] and label["organism"] and label["tissue"], "Missing label metadata")
    for i, probe in enumerate(probes):
        require(features[(probe,)]["gene_id"] == str(arrays["gene_keys"][i]), "Probe/gene mapping mismatch")
    require(set(groups) == {(r["group_id"],) for r in samples.values()}, "Unused/missing group records")
    require(set(studies) == {(r["study_id"],) for r in samples.values()}, "Study inventory mismatch")
    for split in {r["split_id"] for r in tables["split_members"]}:
        partitions = {"train": set(), "validation": set()}
        for row in (r for r in tables["split_members"] if r["split_id"] == split):
            require(row["partition"] in partitions and (row["sample_id"],) in samples,
                    "Unknown split sample/partition")
            partitions[row["partition"]].add(samples[(row["sample_id"],)]["group_id"])
        require(all(partitions.values()) and not partitions["train"] & partitions["validation"],
                "Missing split partition or shared donor group")
    values = arrays["values"]
    require(np.issubdtype(values.dtype, np.number), "Non-numeric matrix")
    require(not np.isinf(values).any(), "Infinite matrix values")
    require(registry["scale"] == "linear_intensity"
            and np.all(values[np.isfinite(values)] > 0), "Unsupported scale or invalid intensity")
    return {"profiles": n, "probes": p, "measurement_cells": n * p,
            "study_local_groups": len(groups),
            "studies": dict(sorted(Counter(str(s) for s in arrays["study_ids"]).items())),
            "missing_expression_values": int(np.isnan(values).sum()),
            "zero_dose_profiles": int(np.sum(arrays["y"] == 0)),
            "dose_label_statuses": dict(Counter(r["dose_status"] for r in annotations.values())),
            "gene_namespaces": sorted({r["gene_namespace"] for r in features.values()}),
            "time_after_exposure_hours": sorted({r["time_after_exposure_hours"] for r in annotations.values()}),
            "numeric_tolerance": 0, "split_group_overlap": False}


def audit_release(cloud, release_id, *, expected_manifest_sha=None):
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", release_id), "Invalid release ID")
    dataset = cloud.dataset_metadata()
    require(dataset["location"].lower() == cloud.scope.region.lower(), "Unexpected dataset location")
    snapshots, selected = {}, {}
    for table in sorted(TABLES):
        meta = cloud.table_metadata(table)
        all_rows = cloud.table_rows(table, meta)
        snapshots[table] = {"etag": meta.get("etag"), "last_modified": meta.get("lastModifiedTime"),
                            "total_rows": int(meta["numRows"])}
        selected[table] = [r for r in all_rows if r["release_id"] == release_id]
        if table == "expression_sets":
            catalog_release_ids = sorted({r["release_id"] for r in all_rows})
    require(len(selected["expression_sets"]) == 1, "Expected exactly one expression representation for pilot")
    registry = selected["expression_sets"][0]
    prefix = cloud.scope.prefix + release_id + "/"
    require(registry["object"] == prefix + "core.npz", "Unexpected registered matrix object")
    manifest_name = prefix + "release.json"
    meta = cloud.object_metadata(manifest_name)
    raw = cloud.object_bytes(manifest_name, meta["generation"], expected_manifest_sha, limit=100_000)
    manifest = json.loads(raw)
    require(manifest["release_id"] == release_id and manifest.get("locked_test_included") is False,
            "Release identity/test declaration mismatch")
    require(manifest.get("purpose") == "core_development", "Unsupported release purpose")
    require(manifest["samples"] == registry["samples"] and manifest["probes"] == registry["probes"],
            "Manifest/registry dimensions disagree")
    for name in ("object", "generation", "sha256"):
        require(str(manifest["dataset"][name]) == str(registry[name]), "Registry/manifest object mismatch")
    declarations = keyed(manifest["tables"], ("table",))
    require(set(declarations) == {(t,) for t in KEYS}, "Missing/unknown release tables")
    for table, keys in KEYS.items():
        declaration = declarations[(table,)]
        name = prefix + table + ".jsonl"
        obj = cloud.object_metadata(name)
        source = cloud.object_bytes(name, obj["generation"], declaration["sha256"], limit=5_000_000)
        require(len(source) == declaration["bytes"], "Declared table bytes mismatch")
        rows = [json.loads(line) for line in source.splitlines()]
        require(len(rows) == declaration["rows"] and all(r["release_id"] == release_id for r in rows),
                "Source table count/release mismatch")
        require(keyed(rows, keys) == keyed(selected[table], keys), "BigQuery/source rows differ: " + table)
    payload = cloud.object_bytes(registry["object"], registry["generation"], registry["sha256"])
    require(len(payload) == manifest["core_bytes"] == manifest["dataset"]["bytes"], "Matrix byte count mismatch")
    result = validate_matrix(payload, registry, selected)
    require(result["study_local_groups"] == manifest["study_local_groups"], "Group count mismatch")
    for table, snapshot in snapshots.items():
        latest = cloud.table_metadata(table)
        require(latest.get("etag") == snapshot["etag"]
                and latest.get("lastModifiedTime") == snapshot["last_modified"],
                "Cross-table audit changed during reads")
    return {"schema_version": 1, "status": "core_release_integrity_verified",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(), "release_id": release_id,
            "scope": "Existing core release; cloud/export/matrix parity, not new scientific eligibility",
            "catalog_release_ids": catalog_release_ids, "matrix": result,
            "representation": {k: registry[k] for k in ("representation", "scale", "processing_scope", "normalization_status")},
            "matrix_reference": {k: registry[k] for k in ("object", "generation", "sha256")},
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest_matches_expected_hash": expected_manifest_sha is not None,
            "table_snapshots": snapshots,
            "requests": cloud.requests, "downloaded_bytes": cloud.downloaded_bytes,
            "query_jobs_created": 0, "cloud_writes": 0,
            "limitations": ["No fresh biological/protocol eligibility review", "No new cohort admission",
                            "Other cloud prefixes and unregistered studies are outside this audit",
                            "Metadata change checks are not a transactional BigQuery snapshot",
                            "No production end-user authorization or retrieval latency benchmark"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("project", "dataset", "bucket", "region", "release-id", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--expected-manifest-sha")
    args = parser.parse_args()
    scope = CloudScope(args.project, args.dataset, args.bucket, args.region)
    result = audit_release(ReadOnlyCloud(scope), args.release_id,
                           expected_manifest_sha=args.expected_manifest_sha)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "release_id", "matrix", "requests", "downloaded_bytes")}, indent=2))


if __name__ == "__main__":
    main()
