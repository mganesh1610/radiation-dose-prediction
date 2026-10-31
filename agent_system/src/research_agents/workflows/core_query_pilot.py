"""Operator-only real-record acceptance pilot; no HTTP endpoint or LLM invocation.

Requires a trusted prior audit receipt and matching local source export. Expected
counts are computed directly from the NPZ arrays, independently of query SQL.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..audit_cloud import CloudScope, ReadOnlyCloud, gcloud_token
from ..policies.access import AccessPolicy, Grant, Principal
from ..tools.bigquery_reader import BigQueryReader
from ..tools.query_sql import ApprovedRelease, compile_query, digest
from ..contracts.queries import QueryPlan
from ..tools.sample_queries import SampleQueries


def prepare(scope, audit, export, token_provider):
    if audit["status"] != "core_release_integrity_verified":
        raise ValueError("A verified core audit is required")
    cloud = ReadOnlyCloud(scope, token_provider=token_provider, max_requests=3, max_bytes=500_000)
    object_name = scope.prefix + audit["release_id"] + "/release.json"
    meta = cloud.object_metadata(object_name)
    raw = cloud.object_bytes(object_name, meta["generation"], audit["manifest_sha256"], limit=100_000)
    manifest = json.loads(raw)
    if manifest["release_id"] != audit["release_id"] or manifest.get("purpose") != "core_development" or manifest.get("locked_test_included") is not False:
        raise ValueError("Unexpected release purpose or identity")
    tables = {}
    for name in ("samples", "sample_annotations"):
        data = (export / (name + ".jsonl")).read_bytes()
        declaration = next(t for t in manifest["tables"] if t["table"] == name)
        if len(data) != declaration["bytes"] or hashlib.sha256(data).hexdigest() != declaration["sha256"]:
            raise ValueError("Local reference is not the approved cloud export")
        rows = [json.loads(line) for line in data.splitlines()]
        if len(rows) != declaration["rows"] or len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate or missing reference rows")
        if any(r["release_id"] != audit["release_id"] for r in rows):
            raise ValueError("Reference release mismatch")
        tables[name] = {r["sample_id"]: r for r in rows}
    matrix_file = export / "core.npz"
    if matrix_file.stat().st_size != manifest["core_bytes"] or hashlib.sha256(matrix_file.read_bytes()).hexdigest() != manifest["dataset"]["sha256"]:
        raise ValueError("Matrix reference does not match audited generation")
    with np.load(matrix_file, allow_pickle=False) as matrix:
        ids, y, studies, groups = (matrix[k] for k in ("sample_ids", "y", "study_ids", "split_group_ids"))
    if len(ids) != manifest["samples"] or set(ids) != set(tables["samples"]) or set(ids) != set(tables["sample_annotations"]):
        raise ValueError("Reference membership mismatch")
    members = tuple((str(s), tables["samples"][str(s)]["study_id"], tables["samples"][str(s)]["group_id"]) for s in ids)
    approved = ApprovedRelease(audit["release_id"], audit["manifest_sha256"], members,
        tuple((t, audit["table_snapshots"][t]["etag"]) for t in ("samples", "sample_annotations")))
    base = {"version": 1, "population": "approved_core_release_v1", "release_id": approved.release_id, "dose_kind": "recorded"}
    cases = []
    def count_case(name, metric, mask, **filters):
        count = int(np.sum(mask)) if metric == "sample_count" else len(set(zip(studies[mask].tolist(), groups[mask].tolist())))
        cases.append({"name": name, "plan": dict(base, metric=metric, filters=filters),
                      "expected": {"status": "ok", "count": count,
                                   "unit": "profiles" if metric == "sample_count" else "study_local_donor_groups"}})
    all_samples = np.ones(len(ids), dtype=bool)
    count_case("all_profiles", "sample_count", all_samples)
    count_case("study_local_groups", "study_group_count", all_samples)
    count_case("zero_dose_profiles", "sample_count", y == 0, dose_gy=0.0)
    count_case("two_gy_profiles", "sample_count", y == 2, dose_gy=2.0)
    selected_study = str(studies[0])
    count_case("study_specific_groups", "study_group_count", studies == selected_study, study_id=selected_study)
    count_case("dose_without_profiles", "sample_count", y == 0.7, dose_gy=0.7)
    selected_ids = [str(ids[np.flatnonzero(y == 0)[0]]), str(ids[np.flatnonzero(y == 2)[0]]),
                   next(s for s, r in tables["sample_annotations"].items() if r["dose_status"] != "source_label")]
    for name, sample_id in zip(("control_details", "exposed_details", "label_conflict_provenance"), selected_ids):
        record = dict(tables["samples"][sample_id], **tables["sample_annotations"][sample_id])
        cases.append({"name": name, "plan": dict(base, metric="sample_details", sample_id=sample_id),
                      "expected": {"status": "ok", "dose_kind": "recorded", "dose_unit": "Gy", "record": record}})
    missing = "GSM99999999999999"
    if missing in tables["samples"]: raise ValueError("Missing-sample fixture unexpectedly exists")
    cases.append({"name": "unknown_sample", "plan": dict(base, metric="sample_details", sample_id=missing),
                  "expected": {"status": "not_found", "sample_id": missing}})
    return approved, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("project", "dataset", "bucket", "region", "audit", "source-release", "output-dir"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--execute", action="store_true", help="Run at most 10 capped SELECT jobs after reference preparation")
    args = parser.parse_args()
    scope = CloudScope(args.project, args.dataset, args.bucket, args.region)
    audit = json.loads(Path(args.audit).read_text())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cached_token = [None, 0.0]
    def token():
        if cached_token[0] is None or time.monotonic() - cached_token[1] > 2400:
            cached_token[:] = [gcloud_token(), time.monotonic()]
        return cached_token[0]
    release, cases = prepare(scope, audit, Path(args.source_release), token)
    (out / "reference-cases.json").write_text(json.dumps({"release_manifest_sha256": release.manifest_sha256,
        "reference_method": "NPZ dose/study/group arrays for counts; original hash-verified JSONL for details",
        "cases": cases}, indent=2) + "\n")
    queries = [compile_query(QueryPlan.model_validate(c["plan"]), release, scope.project, scope.dataset) for c in cases]
    (out / "compiled-queries.json").write_text(json.dumps([{"name": c["name"], "sql": q.sql,
        "parameters": json.loads(q.parameters_json)} for c, q in zip(cases, queries)], indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"status": "references_prepared", "cases": len(cases), "sql_jobs": 0})); return
    # Operator verification only: this principal is NOT a real reviewer/IAP session.
    operator = Principal("operator_verification", time.time() + 900)
    workspace = "core-pilot"
    grant = Grant(operator.user_id, workspace, frozenset({release.release_id}), "operator-pilot-v1")
    access = AccessPolicy(lambda u, w: grant if (u, w) == (operator.user_id, workspace) else None)
    backend = BigQueryReader(scope, token_provider=token, max_queries=10)
    facade = SampleQueries(backend, access, lambda w, r: release if (w, r) == (workspace, release.release_id) else None)
    results = []
    for case in cases:
        result = facade.run(operator, workspace, case["plan"])
        passed = result["answer"] == case["expected"]
        results.append({"case": case["name"], "passed": passed, "result": result})
        (out / "partial-results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"case": case["name"], "passed": passed}), flush=True)
        if not passed: raise RuntimeError("Independent numerical parity failed; pilot stopped")
    receipt = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "core_exact_query_pilot_passed",
        "cases": len(results), "maximum_queries": 10, "maximum_bytes_billed_per_query": backend.cap,
        "requests": backend.requests, "response_bytes": backend.response_bytes,
        "bytes_billed": sum(r["result"]["receipt"]["bytes_billed"] for r in results),
        "source_writes": 0, "llm_calls": 0, "end_user_login_tested": False,
        "reference_sha256": digest(cases), "results": results}
    (out / "pilot-result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "cases", "requests", "bytes_billed")}))


if __name__ == "__main__":
    main()
