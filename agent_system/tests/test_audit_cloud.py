"""Mechanical transport fixtures only; these are not scientific observations."""
import hashlib
import io
import json

import pytest

from research_agents.audit_cloud import AuditFailure, CloudScope, ReadOnlyCloud


SCOPE = CloudScope("example-project", "research", "example-bucket", "us-west1")


def client(responses, **kwargs):
    requests = []
    def opener(request, timeout):
        requests.append(request)
        assert request.get_method() == "GET"
        assert timeout == 20
        payload = responses.pop(0)
        return io.BytesIO(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    return ReadOnlyCloud(SCOPE, opener=opener, token_provider=lambda: "unit-test-token", **kwargs), requests


@pytest.mark.parametrize("name", ["other/file.npz", "training/v1/releases/../secret",
                                 "training/v1/releases//bad", "training/v1/releases/a\\b"])
def test_rejects_object_outside_scope_before_request(name):
    cloud, calls = client([])
    with pytest.raises(ValueError):
        cloud.object_metadata(name)
    assert not calls


def test_no_unapproved_table_or_service_request():
    cloud, calls = client([])
    with pytest.raises(ValueError):
        cloud.table_metadata("other_table")
    with pytest.raises(ValueError):
        cloud._get("external", "anything")
    assert not calls


def test_cumulative_byte_and_request_limits():
    cloud, calls = client([b"1234", b"5678"], max_bytes=5)
    assert cloud._get("storage", "test") == b"1234"
    with pytest.raises(AuditFailure, match="bounded"):
        cloud._get("storage", "test")
    assert len(calls) == 2
    cloud, calls = client([b"{}"], max_requests=1)
    cloud._get("storage", "test")
    with pytest.raises(AuditFailure, match="limit"):
        cloud._get("storage", "test")
    assert len(calls) == 1


def test_pinned_generation_and_hash():
    name = SCOPE.prefix + "a/core.npz"
    cloud, calls = client([dict(name=name, generation="12", size="3"), b"abc"])
    assert cloud.object_bytes(name, "12", hashlib.sha256(b"abc").hexdigest()) == b"abc"
    assert all("generation=12" in c.full_url for c in calls)
    cloud, _ = client([dict(name=name, generation="13", size="3")])
    with pytest.raises(AuditFailure, match="generation"):
        cloud.object_bytes(name, "12")
    cloud, _ = client([dict(name=name, generation="12", size="3"), b"bad"])
    with pytest.raises(AuditFailure, match="SHA-256"):
        cloud.object_bytes(name, "12", hashlib.sha256(b"abc").hexdigest())


def test_null_dose_remains_null():
    meta = dict(numRows="1", etag="a", lastModifiedTime="1",
                schema={"fields": [{"name": "dose_gy", "type": "FLOAT"}]})
    cloud, _ = client([{"rows": [{"f": [{"v": None}]}]}, meta])
    assert cloud.table_rows("sample_annotations", meta) == [{"dose_gy": None}]


def test_table_change_rejects_inconsistent_read():
    meta = dict(numRows="0", etag="before", schema={"fields": []})
    cloud, _ = client([{}, {**meta, "etag": "after"}])
    with pytest.raises(AuditFailure, match="changed"):
        cloud.table_rows("samples", meta)


def test_repeated_page_token_and_row_limit_fail():
    meta = dict(numRows="0", schema={"fields": []})
    cloud, _ = client([{"pageToken": "same"}, {"pageToken": "same"}])
    with pytest.raises(AuditFailure, match="pagination"):
        cloud.table_rows("samples", meta)
    cloud, calls = client([], max_rows=10)
    with pytest.raises(AuditFailure, match="row limit"):
        cloud.table_rows("samples", {**meta, "numRows": "11"})
    assert not calls
