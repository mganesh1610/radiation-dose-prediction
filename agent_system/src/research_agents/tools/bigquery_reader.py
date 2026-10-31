"""Bounded execution of server-compiled SELECTs, using the BigQuery REST API.

Query jobs and their temporary result storage are created; source tables are not
modified. This is an internal adapter, never a model-visible execute_sql tool.
"""
import json
import re
import threading
import uuid
from urllib.parse import quote, urlencode
from urllib.request import Request, HTTPRedirectHandler, build_opener

from ..audit_cloud import CloudScope, gcloud_token
from .query_sql import CompiledQuery


class QueryUnavailable(RuntimeError):
    def __init__(self, message, *, job_id=None, request_id=None):
        super().__init__(message)
        self.job_id, self.request_id = job_id, request_id


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class BigQueryReader:
    def __init__(self, scope: CloudScope, *, token_provider=gcloud_token, opener=None,
                 maximum_bytes_billed=32 * 1024 * 1024, max_queries=12,
                 max_requests=180, max_response_bytes=4_000_000):
        if not 1 <= maximum_bytes_billed <= 32 * 1024 * 1024 or not 1 <= max_queries <= 12:
            raise ValueError("Query budget outside pilot limits")
        self.scope, self.token_provider = scope, token_provider
        self.opener = opener or build_opener(NoRedirect()).open
        self.cap, self.max_queries = maximum_bytes_billed, max_queries
        self.max_requests, self.max_response_bytes = max_requests, max_response_bytes
        self.requests = self.response_bytes = self.queries = self.dry_runs = 0
        self._lock = threading.Lock()

    def _request(self, path, body=None):
        # Paths are built only by the methods below, never from tool arguments.
        with self._lock:
            if self.requests >= self.max_requests or self.response_bytes >= self.max_response_bytes:
                raise QueryUnavailable("Pilot cloud-read budget exhausted")
            self.requests += 1
            limit = min(250_000, self.max_response_bytes - self.response_bytes)
            request = Request("https://bigquery.googleapis.com/bigquery/v2/" + path,
                              data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                              method="GET" if body is None else "POST",
                              headers={"Authorization": "Bearer " + self.token_provider(),
                                       "Content-Type": "application/json"})
            try:
                with self.opener(request, timeout=25) as response:
                    raw = response.read(limit + 1)
                self.response_bytes += len(raw)
                if len(raw) > limit:
                    raise ValueError("Response bound exceeded")
                value = json.loads(raw)
                if not isinstance(value, dict) or value.get("error") or value.get("errors"):
                    raise ValueError("Provider error")
                return value
            except Exception:
                raise QueryUnavailable("BigQuery request failed or exceeded response bounds") from None

    def table_etags(self):
        result = {}
        for table in ("samples", "sample_annotations"):
            value = self._request(f"projects/{self.scope.project}/datasets/{self.scope.dataset}/tables/{table}")
            if not value.get("etag"):
                raise QueryUnavailable("Missing table version")
            result[table] = value["etag"]
        return result

    def _job_path(self, job):
        if (job.get("projectId") != self.scope.project
                or job.get("location", "").lower() != self.scope.region.lower()
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,1024}", job.get("jobId", ""))):
            raise QueryUnavailable("Unexpected BigQuery job identity")
        return f"projects/{self.scope.project}/jobs/{quote(job['jobId'], safe='')}"

    def execute(self, query: CompiledQuery, *, before_execute=lambda: None):
        if not isinstance(query, CompiledQuery):
            raise TypeError("Server-compiled query required")
        with self._lock:
            if self.queries >= self.max_queries:
                raise QueryUnavailable("Pilot query reservation exhausted")
            # Conservative reservation: failures consume a slot too; no automatic resubmission.
            self.queries += 1
        path = f"projects/{self.scope.project}/queries"
        request_id = str(uuid.uuid4())
        body = {"query": query.sql, "queryParameters": json.loads(query.parameters_json),
                "parameterMode": "NAMED", "useLegacySql": False, "useQueryCache": True,
                "location": self.scope.region, "maxResults": 2, "timeoutMs": 10_000,
                "jobTimeoutMs": "20000", "maximumBytesBilled": str(self.cap),
                "jobCreationMode": "JOB_CREATION_REQUIRED"}
        self.dry_runs += 1
        dry = self._request(path, dict(body, dryRun=True))
        estimate = int(dry.get("totalBytesProcessed", -1))
        if estimate < 0 or estimate > self.cap:
            raise QueryUnavailable("Dry-run estimate unavailable or over budget")
        before_execute()
        job = None
        try:
            result = self._request(path, dict(body, dryRun=False, requestId=request_id))
            job = result.get("jobReference", {})
            job_path = self._job_path(job)
            for _ in range(4):
                if result.get("jobComplete"):
                    break
                result = self._request(f"projects/{self.scope.project}/queries/{quote(job['jobId'], safe='')}?" +
                           urlencode({"location": self.scope.region, "maxResults": 2, "timeoutMs": 3000}))
                if result.get("jobReference") != job:
                    raise QueryUnavailable("Query result identity changed")
            if not result.get("jobComplete"):
                # Best effort only: Google does not guarantee immediate cancellation.
                try:
                    self._request(job_path + "/cancel?" + urlencode({"location": self.scope.region}), {})
                except QueryUnavailable:
                    pass
                raise QueryUnavailable("Query incomplete; no partial result returned")
            if result.get("pageToken") or int(result.get("totalRows", -1)) != 1 or len(result.get("rows", [])) != 1:
                raise QueryUnavailable("Unexpected result size or incomplete pagination")
            fields = result["schema"]["fields"]
            cells = result["rows"][0]["f"]
            if (len(fields) != len(cells) or len({f["name"] for f in fields}) != len(fields)
                    or any(f["type"] not in ("INTEGER", "INT64", "STRING") or f.get("mode") == "REPEATED" for f in fields)):
                raise QueryUnavailable("Unexpected result schema")
            row = {f["name"]: (int(c["v"]) if f["type"] in ("INTEGER", "INT64") else c["v"])
                   for f, c in zip(fields, cells, strict=True)}
            details = self._request(job_path + "?" + urlencode({"location": self.scope.region}))
            stats = details.get("statistics", {}).get("query", {})
            if (details.get("status", {}).get("state") != "DONE" or details["status"].get("errorResult")
                    or stats.get("statementType") != "SELECT"):
                raise QueryUnavailable("Query completion could not be verified")
            return row, {"job_id": job["jobId"], "request_id": request_id,
                         "location": job["location"], "dry_run_bytes": estimate,
                         "bytes_processed": int(stats["totalBytesProcessed"]),
                         "bytes_billed": int(stats["totalBytesBilled"]),
                         "cache_hit": bool(stats.get("cacheHit", False)),
                         "maximum_bytes_billed": self.cap}
        except (QueryUnavailable, ValueError, KeyError, TypeError):
            raise QueryUnavailable("Query could not be verified; no answer returned",
                                   job_id=(job or {}).get("jobId"), request_id=request_id) from None
