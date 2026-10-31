import io
import json

import pytest

from research_agents.audit_cloud import CloudScope
from research_agents.tools.bigquery_reader import BigQueryReader, QueryUnavailable
from research_agents.tools.query_sql import CompiledQuery

SCOPE = CloudScope("sample-project", "research", "sample-bucket", "us-west1")
JOB = {"projectId": SCOPE.project, "location": SCOPE.region, "jobId": "fixture_job"}
QUERY = CompiledQuery("SELECT @test", '[{"name":"test","parameterType":{"type":"INT64"},"parameterValue":{"value":"1"}}]')


def completed(**changes):
    return dict({"jobComplete": True, "jobReference": JOB, "totalRows": "1",
        "schema": {"fields": [{"name": "number", "type": "INTEGER"}]},
        "rows": [{"f": [{"v": "1"}]}]}, **changes)


def details(**changes):
    return dict({"status": {"state": "DONE"}, "statistics": {"query": {
        "statementType": "SELECT", "totalBytesProcessed": "100", "totalBytesBilled": "100",
        "cacheHit": False}}}, **changes)


def reader(responses, **limits):
    calls = []
    def opener(request, timeout):
        assert request.full_url.startswith("https://bigquery.googleapis.com/bigquery/v2/")
        calls.append((request.get_method(), request.full_url,
                      json.loads(request.data) if request.data else None))
        response = responses.pop(0)
        if isinstance(response, Exception): raise response
        return io.BytesIO(json.dumps(response).encode())
    return BigQueryReader(SCOPE, token_provider=lambda: "never-log-this-token", opener=opener, **limits), calls


def test_dryrun_precedes_bounded_execution_and_receipt():
    r, calls = reader([{"totalBytesProcessed": "100"}, completed(), details()])
    stages = []
    def authorization(): stages.append(len(calls))
    row, receipt = r.execute(QUERY, before_execute=authorization)
    assert row == {"number": 1} and stages == [1]
    assert calls[0][2]["dryRun"] is True and calls[1][2]["dryRun"] is False
    assert calls[1][2]["maximumBytesBilled"] == str(32 * 1024 * 1024)
    assert calls[1][2]["jobTimeoutMs"] == "20000" and calls[1][2]["useLegacySql"] is False
    assert receipt["job_id"] == "fixture_job" and receipt["bytes_billed"] == 100
    assert "never-log" not in json.dumps(receipt)


@pytest.mark.parametrize("dry", [{}, {"totalBytesProcessed": "33554433"}, {"errors": [{"message": "secret"}]}])
def test_bad_or_excess_dry_run_never_executes(dry):
    r, calls = reader([dry])
    with pytest.raises(QueryUnavailable): r.execute(QUERY)
    assert len(calls) == 1


def test_revoked_after_dryrun_never_executes():
    r, calls = reader([{"totalBytesProcessed": "100"}])
    def revoke(): raise PermissionError()
    with pytest.raises(PermissionError): r.execute(QUERY, before_execute=revoke)
    assert len(calls) == 1


@pytest.mark.parametrize("response", [completed(pageToken="more"), completed(totalRows="2"),
    completed(jobReference=dict(JOB, projectId="another-project")),
    completed(jobReference=dict(JOB, location="us-east1")),
    completed(schema={"fields": [{"name": "number", "type": "STRUCT"}]}),
    completed(rows=[{"f": [{"v": None}]}])])
def test_malformed_cross_scope_or_partial_result_rejected(response):
    r, _ = reader([{"totalBytesProcessed": "100"}, response])
    with pytest.raises(QueryUnavailable): r.execute(QUERY)


def test_incomplete_job_is_polled_boundedly_cancelled_and_never_answered():
    pending = {"jobComplete": False, "jobReference": JOB}
    r, calls = reader([{"totalBytesProcessed": "100"}] + [pending] * 5 + [{}])
    with pytest.raises(QueryUnavailable) as error: r.execute(QUERY)
    assert len(calls) == 7 and "/cancel?" in calls[-1][1]
    assert error.value.job_id == JOB["jobId"]


def test_resource_reservations_and_redacted_errors():
    r, calls = reader([{"totalBytesProcessed": "100"}, TimeoutError("token=secret")], max_queries=1)
    with pytest.raises(QueryUnavailable) as error: r.execute(QUERY)
    assert error.value.request_id and "secret" not in str(error.value)
    with pytest.raises(QueryUnavailable, match="reservation"): r.execute(QUERY)
    assert len(calls) == 2
    r, _ = reader([{"large": "x" * 1000}], max_response_bytes=20)
    with pytest.raises(QueryUnavailable): r.table_etags()


def test_failed_or_non_select_job_is_not_an_answer():
    r, _ = reader([{"totalBytesProcessed": "100"}, completed(),
                   details(status={"state": "DONE", "errorResult": {"message": "secret"}})])
    with pytest.raises(QueryUnavailable): r.execute(QUERY)
