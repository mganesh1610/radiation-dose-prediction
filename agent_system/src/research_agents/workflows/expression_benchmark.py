"""Bounded operator benchmark with independent references and fresh worker processes."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

from ..audit_cloud import CloudScope, ReadOnlyCloud, gcloud_token
from ..policies.access import AccessPolicy, Grant, Principal
from ..tools.expression_reader import MatrixAdmission, ExpressionReader, CloudMatrixSource
from ..tools.matrix_cache import MatrixCache
from ..tools.query_sql import ApprovedRelease, digest


def restore(data):
    data = dict(data); data.pop("fingerprint", None)
    data["scope"] = CloudScope(**data["scope"])
    core = dict(data["core"])
    core["members"] = tuple(tuple(v) for v in core["members"])
    core["table_etags"] = tuple(tuple(v) for v in core["table_etags"])
    data["core"] = ApprovedRelease(**core)
    data["features"] = tuple(tuple(v) for v in data["features"])
    data["table_etags"] = tuple(tuple(v) for v in data["table_etags"])
    return MatrixAdmission(**data)


def prepare(scope, audit, export):
    if audit.get("status") != "core_release_integrity_verified": raise ValueError("Verified core audit required")
    token = gcloud_token()
    cloud = ReadOnlyCloud(scope, token_provider=lambda: token, max_requests=12, max_bytes=1_000_000)
    name = scope.prefix + audit["release_id"] + "/release.json"
    meta = cloud.object_metadata(name)
    raw = cloud.object_bytes(name, meta["generation"], audit["manifest_sha256"], limit=100_000)
    manifest = json.loads(raw)
    if manifest["release_id"] != audit["release_id"] or manifest["purpose"] != "core_development" or manifest["locked_test_included"] is not False:
        raise ValueError("Unsupported manifest")
    local = {}
    for table in ("samples", "features"):
        raw = (export / (table + ".jsonl")).read_bytes()
        decl = next(x for x in manifest["tables"] if x["table"] == table)
        if len(raw) != decl["bytes"] or hashlib.sha256(raw).hexdigest() != decl["sha256"]:
            raise ValueError("Local reference checksum mismatch")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != decl["rows"] or any(r["release_id"] != audit["release_id"] for r in rows):
            raise ValueError("Reference membership mismatch")
        local[table] = rows
    meta = cloud.table_metadata("expression_sets")
    if meta["etag"] != audit["table_snapshots"]["expression_sets"]["etag"]:
        raise ValueError("Registry changed since audit")
    registry = [r for r in cloud.table_rows("expression_sets", meta) if r["release_id"] == audit["release_id"]]
    if len(registry) != 1: raise ValueError("Ambiguous expression representation")
    reg = registry[0]
    for key in ("object", "generation", "sha256"):
        if str(reg[key]) != str(manifest["dataset"][key]): raise ValueError("Registry/manifest mismatch")
    if (reg["samples"], reg["probes"]) != (len(local["samples"]), len(local["features"])):
        raise ValueError("Registry dimensions differ")
    if any(r["gene_namespace"] != "NCBI_GeneID" for r in local["features"]):
        raise ValueError("Unexpected mapping namespace")
    core = ApprovedRelease(audit["release_id"], audit["manifest_sha256"],
        tuple((r["sample_id"], r["study_id"], r["group_id"]) for r in local["samples"]),
        tuple((t, audit["table_snapshots"][t]["etag"]) for t in ("samples", "sample_annotations")))
    admitted = MatrixAdmission(scope, core, reg["expression_set_id"], reg["object"], str(reg["generation"]),
        reg["sha256"], manifest["core_bytes"], tuple((r["probe_id"], r["gene_id"]) for r in local["features"]),
        next(x["sha256"] for x in manifest["tables"] if x["table"] == "features"),
        tuple((t, audit["table_snapshots"][t]["etag"]) for t in ("samples", "features", "expression_sets")),
        **{k: reg[k] for k in ("representation", "scale", "processing_scope", "normalization_status")})
    matrix = export / "core.npz"
    if matrix.stat().st_size != admitted.compressed_bytes or hashlib.sha256(matrix.read_bytes()).hexdigest() != admitted.sha256:
        raise ValueError("Reference matrix is not the approved source")
    with np.load(matrix, allow_pickle=False) as z:
        arrays = {k: z[k] for k in ("values", "detected", "sample_ids", "probe_ids", "gene_keys")}
    # Reference selection directly from source arrays; never call production decode/slice.
    genes = arrays["gene_keys"].tolist(); ids = arrays["sample_ids"].tolist()
    frequency = Counter(genes); ordered = list(dict.fromkeys(genes))
    missing_row, missing_col = np.argwhere(np.isnan(arrays["values"]))[0]
    small_genes = list(dict.fromkeys([genes[missing_row], max(frequency, key=frequency.get)]))
    large_genes = []; probes = 0
    for gene, count in sorted(frequency.items(), key=lambda pair: (-pair[1], pair[0])):
        if probes + count <= 100 and len(large_genes) < 50:
            large_genes.append(gene); probes += count
        if probes == 100: break
    workloads = [("small_missing_and_multi_probe", [ids[int(missing_col)]], small_genes),
                 ("medium_panel", ids[:10], ordered[:10]),
                 ("maximum_cells", ids[:50], large_genes)]
    cases = []
    for label, selected_ids, selected_genes in workloads:
        ri = [i for gene in selected_genes for i, actual in enumerate(genes) if actual == gene]
        ci = [ids.index(s) for s in selected_ids]
        values = np.take(np.take(arrays["values"], ri, axis=0), ci, axis=1)
        detection = np.take(np.take(arrays["detected"], ri, axis=0), ci, axis=1)
        expected = {"values": [[float(x) if np.isfinite(x) else None for x in row] for row in values],
                    "detected": detection.tolist(), "missing_values": int(np.isnan(values).sum()),
                    "features": [{"probe_id": str(arrays["probe_ids"][i]), "gene_id": genes[i]} for i in ri],
                    "sample_ids": selected_ids}
        request = {"version": 1, "release_id": core.release_id, "expression_set_id": admitted.expression_set_id,
                   "sample_ids": selected_ids, "gene_ids": selected_genes, "gene_namespace": "NCBI_GeneID", "aggregation": "none"}
        cases.append({"name": label, "plan": request, "expected": expected, "result_cells": values.size})
    return {"admission": asdict(admitted), "cases": cases, "reference_method": "Independent NPZ np.take axis extraction",
            "preparation_cloud_requests": cloud.requests, "numeric_tolerance": 0}


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def measure(call):
    process = psutil.Process()
    baseline = process.memory_info().rss
    high_water_before = peak_rss()
    samples = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(.005): samples.append(process.memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True); monitor.start()
    start = time.perf_counter()
    try:
        result = call()
        elapsed = time.perf_counter() - start
    finally:
        stop.set(); monitor.join()
    after = process.memory_info().rss; samples.append(after)
    return result, {"elapsed_seconds": elapsed, "rss_before_bytes": baseline, "rss_after_bytes": after,
                    "sampled_peak_rss_bytes": max(samples), "sampled_incremental_peak_bytes": max(samples) - baseline,
                    "process_high_water_bytes": peak_rss(), "high_water_before_bytes": high_water_before,
                    "rss_sample_interval_ms": 5}


def worker(config, case_index):
    admitted = restore(config["admission"]); case = config["cases"][case_index]
    auth_start = time.perf_counter(); token = gcloud_token(); credential_seconds = time.perf_counter() - auth_start
    principal = Principal("operator_expression_benchmark", time.time() + 600)
    grant = Grant(principal.user_id, "core-pilot", frozenset({admitted.core.release_id}), "benchmark-v1")
    access = AccessPolicy(lambda u, w: grant if (u, w) == (principal.user_id, "core-pilot") else None)
    cache = MatrixCache()
    reader = ExpressionReader(cache, access,
        lambda w, r, e: admitted if (w, r, e) == ("core-pilot", admitted.core.release_id, admitted.expression_set_id) else None,
        lambda a: CloudMatrixSource(a.scope, token_provider=lambda: token))
    results = []
    for phase in ("cold", "warm", "warm"):
        output, metrics = measure(lambda: reader.run(principal, "core-pilot", case["plan"]))
        answer = output["answer"]; expected = case["expected"]
        actual = {k: answer[k] for k in ("values", "detected", "missing_values", "features")}
        actual["sample_ids"] = [r["sample_id"] for r in answer["samples"]]
        if actual != expected: raise RuntimeError("Independent expression parity failed")
        receipt = output["receipt"]
        if receipt["cache"]["hit"] != (phase == "warm") or receipt["matrix_downloads"] != (1 if phase == "cold" else 0):
            raise RuntimeError("Cache/download behavior differed from benchmark plan")
        if metrics["sampled_peak_rss_bytes"] > 256 * 1024**2:
            raise RuntimeError("Pilot process RSS exceeded predeclared 256 MiB gate")
        results.append({"phase": phase, "parity_passed": True, "metrics": metrics, "receipt": receipt})
        del output, answer, actual
    return {"pid": os.getpid(), "workload": case["name"], "credential_setup_seconds": credential_seconds,
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--config")
    parser.add_argument("--execute", action="store_true")
    for flag in ("project", "dataset", "bucket", "region", "audit", "source-release", "output-dir"):
        parser.add_argument("--" + flag)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(json.loads(Path(args.config).read_text()), args.worker), allow_nan=False)); return
    if not all(getattr(args, k) for k in ("project","dataset","bucket","region","audit","source_release","output_dir")):
        parser.error("Preparation requires scope, audit, source release and output directory")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    config = prepare(CloudScope(args.project,args.dataset,args.bucket,args.region),
                     json.loads(Path(args.audit).read_text()), Path(args.source_release))
    # Freeze workload, repetitions, bounds and tolerance before any timed retrieval.
    config["benchmark_plan"] = {"cold_runs_per_workload": 5, "warm_runs_per_workload": 10,
        "fresh_processes": 15, "max_matrix_downloads": 15, "max_timed_cloud_requests": 390,
        "process_rss_gate_bytes": 256 * 1024**2, "percentile_method": "numpy linear interpolation",
        "timing_scope": "tool run including source/version checks and serialization bounds; excludes startup, credential acquisition, IAP, LLM and frontend",
        "cold_definition": "fresh Python worker, empty application cache; upstream provider/network caches not controlled",
        "memory_scope": "worker only; sampled RSS at 5 ms plus process lifetime high-water; excludes gcloud credential subprocess",
        "deployment": "local Mac operator pilot; no Cloud Run measurement"}
    config_path = out / "benchmark-plan.json"
    config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    if not args.execute:
        print(json.dumps({"status":"prepared","workloads":[{"name":c["name"],"cells":c["result_cells"]} for c in config["cases"]]})); return
    runs = []
    for index, case in enumerate(config["cases"]):
        for repetition in range(5):
            command = [sys.executable,"-B","-m","research_agents.workflows.expression_benchmark","--worker",str(index),"--config",str(config_path.resolve())]
            completed = subprocess.run(command,capture_output=True,text=True,timeout=120)
            if completed.returncode:
                (out/"worker-error.log").write_text(completed.stderr)
                raise RuntimeError("Benchmark worker failed; no partial acceptance claim")
            run = json.loads(completed.stdout);run["repetition"]=repetition+1;runs.append(run)
            (out/"partial-runs.json").write_text(json.dumps(runs,indent=2)+"\n")
            print(json.dumps({"workload":case["name"],"repetition":repetition+1,"parity_passed":True}),flush=True)
    summaries = []
    for case in config["cases"]:
        group = [r for r in runs if r["workload"] == case["name"]]
        for phase in ("cold","warm"):
            observations = [r for g in group for r in g["results"] if r["phase"]==phase]
            latency = [r["metrics"]["elapsed_seconds"] for r in observations]
            summaries.append({"workload":case["name"],"phase":phase,"n":len(observations),"cells":case["result_cells"],
                "p50_seconds":float(np.percentile(latency,50)),"p95_seconds":float(np.percentile(latency,95)),
                "sampled_peak_rss_bytes":max(r["metrics"]["sampled_peak_rss_bytes"] for r in observations),
                "sampled_incremental_peak_bytes":max(r["metrics"]["sampled_incremental_peak_bytes"] for r in observations),
                "cache_resident_bytes":max(r["receipt"]["cache"]["accounted_resident_bytes"] for r in observations),
                "downloaded_response_bytes":sum(r["receipt"]["downloaded_response_bytes"] for r in observations),
                "matrix_downloads":sum(r["receipt"]["matrix_downloads"] for r in observations)})
    observations = [r for g in runs for r in g["results"]]
    receipt = {"status":"core_expression_benchmark_passed","completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "environment":{"platform":platform.platform(),"machine":platform.machine(),"python":platform.python_version(),
                       "numpy":np.__version__,"psutil":psutil.__version__},
        "plan_sha256":digest(config),"timed_requests":len(observations),"all_parity_passed":True,
        "cloud_requests":sum(r["receipt"]["cloud_requests"] for r in observations),
        "matrix_downloads":sum(r["receipt"]["matrix_downloads"] for r in observations),
        "downloaded_response_bytes":sum(r["receipt"]["downloaded_response_bytes"] for r in observations),
        "process_peak_rss_bytes":max(r["metrics"]["process_high_water_bytes"] for r in observations),
        "queries_created":0,"source_writes":0,"llm_calls":0,"end_user_login_tested":False,
        "summaries":summaries,"runs":runs}
    if receipt["cloud_requests"]>390 or receipt["matrix_downloads"]>15:raise RuntimeError("Benchmark request budget exceeded")
    (out/"benchmark-result.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:receipt[k] for k in ("status","timed_requests","cloud_requests","matrix_downloads","process_peak_rss_bytes","summaries")}))


if __name__ == "__main__":
    main()
