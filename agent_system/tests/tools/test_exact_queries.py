"""Fabricated contract/security fixtures; real-record parity has a separate pilot."""
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from research_agents.contracts.queries import QueryPlan
from research_agents.policies.access import AccessPolicy, Grant, IapAuthenticator, Principal, IAP_ISSUER
from research_agents.tools.query_sql import ApprovedRelease, compile_query
from research_agents.tools.sample_queries import SampleQueries, IntegrityError


RELEASE = ApprovedRelease("core_fixture", "a" * 64,
    (("GSM1", "GSE1", "donor1"), ("GSM2", "GSE1", "donor1"), ("GSM3", "GSE2", "donor1")),
    (("samples", "s1"), ("sample_annotations", "a1")))
PRINCIPAL = Principal("usr_test", 200)


def payload(metric="sample_count", **kwargs):
    return {"version": 1, "metric": metric, "population": "approved_core_release_v1",
            "release_id": RELEASE.release_id, "dose_kind": "recorded", **kwargs}


def envelope(answer):
    return {"sample_rows": 3, "unique_samples": 3, "unknown_samples": 0,
            "annotation_rows": 3, "unique_annotations": 3, "unknown_annotations": 0,
            "joined_rows": 3, "answer_json": json.dumps(answer)}


def sample(dose=0.0):
    return {"release_id": RELEASE.release_id, "sample_id": "GSM1", "study_id": "GSE1",
            "group_id": "donor1", "dose_gy": dose, "dose_status": "source_label",
            "time_after_exposure_hours": 24.0, "organism": "Homo sapiens", "tissue": "whole_blood"}


class Backend:
    scope = SimpleNamespace(project="sample-project", dataset="research")
    def __init__(self, answer):
        self.row, self.calls = envelope(answer), []
        self.after = lambda: None
        self.etags = dict(RELEASE.table_etags)
    def table_etags(self):
        self.calls.append("metadata")
        return self.etags
    def execute(self, query, before_execute):
        before_execute()
        self.calls.append("execute")
        self.after()
        return self.row, {"job_id": "fixture_job", "bytes_processed": 1, "bytes_billed": 1}


def service(answer):
    grant = Grant(PRINCIPAL.user_id, "workspace", frozenset({RELEASE.release_id}), "revision1")
    grants = {("usr_test", "workspace"): grant}
    releases = {("workspace", RELEASE.release_id): RELEASE}
    b = Backend(answer)
    f = SampleQueries(b, AccessPolicy(lambda u, w: grants.get((u, w)), clock=lambda: 100),
                      lambda w, r: releases.get((w, r)))
    return f, b, grants, releases


@pytest.mark.parametrize("change", [
    {"sql": "DROP TABLE samples"}, {"user_id": "admin"}, {"workspace_id": "other"},
    {"metric": "prediction"}, {"dose_kind": "predicted"}, {"population": "all_downloaded"},
    {"metric": "global_donor_count"}, {"version": "1"}, {"sample_id": "GSM1; SELECT 1"},
    {"filters": {"dose_gy": "0"}}, {"filters": {"dose_gy": True}},
    {"filters": {"dose_gy": float("nan")}}, {"filters": {"dose_gy": -1}},
    {"filters": {"gene": "TP53"}}, {"filters": {"study_id": "GSE1' OR 1=1"}},
    {"metric": "sample_details"}, {"sample_id": "GSM1"}])
def test_invalid_or_injected_plans_never_touch_backend(change):
    f, b, _, _ = service([])
    with pytest.raises((ValidationError, ValueError)):
        f.run(PRINCIPAL, "workspace", payload(**change))
    assert not b.calls


def test_parameterization_and_release_join():
    p = QueryPlan.model_validate(payload(filters={"study_id": "GSE1", "dose_gy": 0.0}))
    q = compile_query(p, RELEASE, "sample-project", "research")
    assert "GSE1" not in q.sql and RELEASE.release_id not in q.sql
    assert "s.release_id = a.release_id AND s.sample_id = a.sample_id" in q.sql
    params = {x["name"]: x["parameterValue"] for x in json.loads(q.parameters_json)}
    assert params["dose_gy"] == {"value": "0.0"}
    assert params["study_id"] == {"value": "GSE1"}
    assert len(params["approved_ids"]["arrayValues"]) == 3


@pytest.mark.parametrize("metric,answer,unit", [("sample_count", 3, "profiles"),
            ("study_group_count", 2, "study_local_donor_groups")])
def test_count_units_and_provenance(metric, answer, unit):
    f, _, _, _ = service([{"count": answer}])
    result = f.run(PRINCIPAL, "workspace", payload(metric))
    assert result["answer"] == {"status": "ok", "count": answer, "unit": unit}
    assert result["receipt"]["release_manifest_sha256"] == RELEASE.manifest_sha256
    assert "usr_test" not in json.dumps(result)


@pytest.mark.parametrize("dose,status", [(0.0, "ok"), (None, "not_recorded"), (2.0, "ok")])
def test_missing_dose_is_not_zero(dose, status):
    f, _, _, _ = service([sample(dose)])
    answer = f.run(PRINCIPAL, "workspace", payload("sample_details", sample_id="GSM1"))["answer"]
    assert answer["status"] == status and answer["record"]["dose_gy"] == dose
    assert (answer["record"]["dose_gy"] is None) == (dose is None)
    assert answer["dose_kind"] == "recorded"


def test_unknown_sample_and_zero_count():
    f, b, _, _ = service([])
    assert f.run(PRINCIPAL, "workspace", payload("sample_details", sample_id="GSM999"))["answer"]["status"] == "not_found"
    b.row = envelope([{"count": 0}])
    assert f.run(PRINCIPAL, "workspace", payload(filters={"dose_gy": 0.7}))["answer"]["count"] == 0


@pytest.mark.parametrize("field,value", [("sample_rows", 4), ("unique_samples", 2),
    ("annotation_rows", 2), ("unique_annotations", 2), ("unknown_samples", 1),
    ("unknown_annotations", 1), ("joined_rows", 4)])
def test_duplicate_missing_and_expanded_join_rejected(field, value):
    f, b, _, _ = service([{"count": 3}]); b.row[field] = value
    with pytest.raises(IntegrityError): f.run(PRINCIPAL, "workspace", payload())


@pytest.mark.parametrize("which", ["wrong_user", "wrong_workspace", "expired", "wrong_release"])
def test_auth_denials_before_reads(which):
    f, b, _, _ = service([])
    principal = replace(PRINCIPAL, user_id="stranger") if which == "wrong_user" else PRINCIPAL
    if which == "expired": principal = replace(principal, expires_at=50)
    workspace = "other" if which == "wrong_workspace" else "workspace"
    p = payload(); p["release_id"] = "not_allowed" if which == "wrong_release" else RELEASE.release_id
    with pytest.raises(PermissionError): f.run(principal, workspace, p)
    assert not b.calls


@pytest.mark.parametrize("kind", ["revocation", "release_revocation", "grant_change", "table_change"])
def test_mid_request_changes_do_not_return_answer(kind):
    f, b, grants, releases = service([{"count": 3}])
    def mutate():
        if kind == "revocation": grants.clear()
        elif kind == "release_revocation": releases.clear()
        elif kind == "grant_change":
            grants[("usr_test", "workspace")] = replace(grants[("usr_test", "workspace")], revision="revision2")
        else: b.etags = dict(b.etags, samples="new")
    b.after = mutate
    with pytest.raises((PermissionError, IntegrityError)): f.run(PRINCIPAL, "workspace", payload())


def test_changed_table_fails_before_sql_and_bound_tool_has_no_identity_argument():
    f, b, _, _ = service([]); b.etags = {}
    bound = f.bind(PRINCIPAL, "workspace")
    assert list(inspect.signature(bound).parameters) == ["plan"]
    with pytest.raises(IntegrityError): bound(payload())
    assert "execute" not in b.calls


@pytest.mark.parametrize("answer", [[{"count": 4}], [{"count": True}], [{"count": "3"}],
    [dict(sample(), dose_gy=-1)], [dict(sample(), release_id="wrong")],
    [dict(sample(), study_id="GSE9")], [dict(sample(), predicted_gy=1.0)]])
def test_wrong_numeric_or_record_scope_fails(answer):
    f, _, _, _ = service(answer)
    p = payload() if "count" in answer[0] else payload("sample_details", sample_id="GSM1")
    with pytest.raises(IntegrityError): f.run(PRINCIPAL, "workspace", p)


def claims(**changes):
    return dict({"iss": IAP_ISSUER, "aud": "/projects/test/service", "sub": "subject",
                 "email": "reviewer@example.test", "exp": 200, "iat": 90}, **changes)


@pytest.mark.parametrize("changes", [{"iss": "attacker"}, {"aud": "/projects/wrong"},
    {"sub": ""}, {"exp": 50}, {"iat": 300}, {"email": "other@example.test"}])
def test_iap_claim_checks(changes):
    auth = IapAuthenticator("/projects/test/service", lambda: {"reviewer@example.test"},
                             verifier=lambda t, a: claims(**changes), clock=lambda: 100)
    with pytest.raises(PermissionError): auth.authenticate({"X-Goog-IAP-JWT-Assertion": "test-fixture"})


def test_unsigned_header_signature_failure_and_stable_subject_identity():
    seen = []
    def verifier(token, audience):
        seen.append((token, audience)); return claims()
    auth = IapAuthenticator("/projects/test/service", lambda: {"reviewer@example.test"}, verifier=verifier, clock=lambda: 100)
    with pytest.raises(PermissionError): auth.authenticate({"X-Goog-Authenticated-User-Email": "reviewer@example.test"})
    principal = auth.authenticate({"X-Goog-IAP-JWT-Assertion": "signed-fixture"})
    assert principal.user_id.startswith("usr_") and "reviewer" not in principal.user_id
    assert seen == [("signed-fixture", "/projects/test/service")]
    auth.verifier = lambda t, a: (_ for _ in ()).throw(ValueError("bad signature secret"))
    with pytest.raises(PermissionError, match="invalid or not authorized"):
        auth.authenticate({"X-Goog-IAP-JWT-Assertion": "bad-fixture"})
