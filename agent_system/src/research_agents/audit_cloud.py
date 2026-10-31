"""Bounded GET-only cloud adapter for an operator-run coverage audit.

Not a public chatbot tool: end-user/workspace authorization belongs to the later
service facade. No SQL jobs, writes, IAM changes or API enablement are supported.
"""
from dataclasses import dataclass
import hashlib
import json
import re
import subprocess
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


TABLES = frozenset({"studies", "sample_groups", "samples", "sample_annotations",
                    "features", "expression_sets", "split_members"})


class AuditFailure(RuntimeError):
    """A bounded audit failed without exposing credential or HTTP response bodies."""


@dataclass(frozen=True)
class CloudScope:
    project: str
    dataset: str
    bucket: str
    region: str
    prefix: str = "training/v1/releases/"

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", self.project):
            raise ValueError("Invalid project ID")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,1023}", self.dataset):
            raise ValueError("Invalid dataset ID")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", self.bucket):
            raise ValueError("Invalid bucket name")
        if not re.fullmatch(r"[A-Za-z0-9-]{2,30}", self.region):
            raise ValueError("Invalid region")
        if not self.prefix.endswith("/") or not re.fullmatch(r"[A-Za-z0-9_/-]+", self.prefix):
            raise ValueError("Invalid object prefix")
        if any(p in ("", ".", "..") for p in self.prefix.rstrip("/").split("/")):
            raise ValueError("Invalid object prefix")


def gcloud_token() -> str:
    result = subprocess.run(["gcloud", "auth", "print-access-token"],
                            capture_output=True, text=True, timeout=35)
    if result.returncode or not result.stdout.strip():
        raise AuditFailure("gcloud credentials unavailable; sign-in may be required")
    return result.stdout.strip()


class ReadOnlyCloud:
    def __init__(self, scope: CloudScope, *, token_provider=gcloud_token,
                 opener=urlopen, max_requests=120, max_bytes=64_000_000,
                 max_rows=25_000):
        self.scope = scope
        self.token_provider, self.opener = token_provider, opener
        self.max_requests, self.max_bytes, self.max_rows = max_requests, max_bytes, max_rows
        self.requests = self.downloaded_bytes = 0
        self._token = None

    def _get(self, service: str, path: str, *, limit=2_000_000) -> bytes:
        hosts = {"storage": "https://storage.googleapis.com/storage/v1/",
                 "bigquery": "https://bigquery.googleapis.com/bigquery/v2/"}
        if service not in hosts or path.startswith(("/", "http")) or ".." in path.split("/"):
            raise ValueError("Unexpected API endpoint")
        allowed = min(limit, self.max_bytes - self.downloaded_bytes)
        if allowed <= 0 or self.requests >= self.max_requests:
            raise AuditFailure("Audit request/byte limit exhausted")
        if self._token is None:
            self._token = self.token_provider()
        self.requests += 1
        request = Request(hosts[service] + path, method="GET",
                          headers={"Authorization": "Bearer " + self._token})
        try:
            with self.opener(request, timeout=20) as response:
                payload = response.read(allowed + 1)
        except Exception as exc:
            raise AuditFailure(f"Cloud read failed ({type(exc).__name__})") from None
        self.downloaded_bytes += len(payload)
        if len(payload) > allowed:
            raise AuditFailure("Cloud response exceeds bounded audit size")
        return payload

    def _json(self, service, path):
        try:
            return json.loads(self._get(service, path))
        except (ValueError, UnicodeError):
            raise AuditFailure("Invalid cloud JSON") from None

    def dataset_metadata(self):
        s = self.scope
        return self._json("bigquery", f"projects/{s.project}/datasets/{s.dataset}")

    def table_metadata(self, table):
        if table not in TABLES:
            raise ValueError("Table is outside the audit allowlist")
        s = self.scope
        return self._json("bigquery", f"projects/{s.project}/datasets/{s.dataset}/tables/{table}")

    def table_rows(self, table, metadata):
        if table not in TABLES:
            raise ValueError("Table is outside the audit allowlist")
        if int(metadata["numRows"]) > self.max_rows:
            raise AuditFailure("Table exceeds audit row limit")
        fields = metadata["schema"]["fields"]
        if any(f.get("mode") == "REPEATED" or f["type"] in ("RECORD", "STRUCT") for f in fields):
            raise AuditFailure("Nested schema is unsupported by this audit")
        rows, page, seen = [], None, set()
        s = self.scope
        while True:
            params = {"maxResults": 1000}
            if page:
                params["pageToken"] = page
            path = f"projects/{s.project}/datasets/{s.dataset}/tables/{table}/data?" + urlencode(params)
            result = self._json("bigquery", path)
            for row in result.get("rows", []):
                if len(row["f"]) != len(fields):
                    raise AuditFailure("Table row/schema width mismatch")
                values = {}
                for field, cell in zip(fields, row["f"], strict=True):
                    value = cell.get("v")
                    if value is not None and field["type"] in ("INTEGER", "INT64"):
                        value = int(value)
                    elif value is not None and field["type"] in ("FLOAT", "FLOAT64"):
                        value = float(value)
                    values[field["name"]] = value
                rows.append(values)
                if len(rows) > self.max_rows:
                    raise AuditFailure("Paginated table exceeds row limit")
            page = result.get("pageToken")
            if not page:
                break
            if page in seen:
                raise AuditFailure("Repeated pagination token")
            seen.add(page)
        after = self.table_metadata(table)
        for key in ("etag", "lastModifiedTime", "numRows", "schema"):
            if after.get(key) != metadata.get(key):
                raise AuditFailure("Table changed during audit; retry a fresh snapshot")
        if len(rows) != int(metadata["numRows"]):
            raise AuditFailure("Table row count changed or pagination incomplete")
        return rows

    def _object_path(self, name):
        if (not name.startswith(self.scope.prefix) or "\\" in name
                or any(p in ("", ".", "..") for p in name.split("/"))):
            raise ValueError("Object outside allowed release prefix")
        return f"b/{quote(self.scope.bucket, safe='')}/o/{quote(name, safe='')}"

    def object_metadata(self, name, generation=None):
        path = self._object_path(name)
        if generation is not None:
            path += "?" + urlencode({"generation": self._generation(generation)})
        return self._json("storage", path)

    @staticmethod
    def _generation(value):
        if not re.fullmatch(r"[0-9]+", str(value)) or int(value) <= 0:
            raise ValueError("Invalid object generation")
        return str(value)

    def object_bytes(self, name, generation, sha256=None, *, limit=32_000_000):
        generation = self._generation(generation)
        meta = self.object_metadata(name, generation)
        if str(meta.get("generation")) != generation or meta.get("name") != name:
            raise AuditFailure("Object identity or generation mismatch")
        if int(meta["size"]) > limit:
            raise AuditFailure("Object exceeds declared audit limit")
        path = self._object_path(name) + "?" + urlencode({"alt": "media", "generation": generation})
        payload = self._get("storage", path, limit=limit)
        if len(payload) != int(meta["size"]):
            raise AuditFailure("Object byte count mismatch")
        if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
            raise AuditFailure("Object SHA-256 mismatch")
        return payload
