"""Authorized, version-bound facade shared by future agent tools and HTTP routes."""
from datetime import datetime, timezone
import json

from ..contracts.queries import QueryPlan, SampleRecord
from .query_sql import compile_query, digest

PROCEDURE_ID = "exact-core-query"
PROCEDURE_VERSION = 1
PROCEDURE_STEPS = ("authorize", "validate_plan", "resolve_approved_release", "check_table_versions",
                   "compile_parameterized_select", "dry_run", "reauthorize", "execute",
                   "validate_cardinality_and_answer", "check_table_versions", "reauthorize", "return_receipt")
PROCEDURE_HASH = digest({"id": PROCEDURE_ID, "version": PROCEDURE_VERSION, "steps": PROCEDURE_STEPS})


class IntegrityError(RuntimeError):
    pass


def validate_answer(row, plan, release):
    expected = {"sample_rows", "unique_samples", "unknown_samples", "annotation_rows",
                "unique_annotations", "unknown_annotations", "joined_rows", "answer_json"}
    n = len(release.members)
    if set(row) != expected:
        raise IntegrityError("Unexpected query envelope")
    for key in expected - {"answer_json"}:
        value = row[key]
        if type(value) is not int or value != (0 if key.startswith("unknown_") else n):
            raise IntegrityError("Release membership, uniqueness or join cardinality failed")
    try:
        answer = json.loads(row["answer_json"])
        if not isinstance(answer, list):
            raise ValueError()
        if plan.metric == 'study_records':
            members = sorted(m for m in release.members if m[1] == plan.filters.study_id)
            records = [SampleRecord.model_validate(r) for r in answer]
            if (len(records) > 145 or [(r.sample_id,r.study_id,r.group_id) for r in records] != members
                    or any(r.release_id != plan.release_id for r in records)):
                raise ValueError()
            return {'status':'ok', 'dose_kind':'recorded', 'dose_unit':'Gy',
                    'records':[r.model_dump() for r in records]}
        if plan.metric == "sample_details":
            member = next((m for m in release.members if m[0] == plan.sample_id), None)
            if member is None:
                if answer:
                    raise ValueError()
                return {"status": "not_found", "sample_id": plan.sample_id}
            if len(answer) != 1:
                raise ValueError()
            record = SampleRecord.model_validate(answer[0])
            if (record.release_id != plan.release_id or
                    (record.sample_id, record.study_id, record.group_id) != member):
                raise ValueError()
            return {"status": "not_recorded" if record.dose_gy is None else "ok",
                    "dose_kind": "recorded", "dose_unit": "Gy", "record": record.model_dump()}
        if len(answer) != 1 or set(answer[0]) != {"count"} or type(answer[0]["count"]) is not int:
            raise ValueError()
        count = answer[0]["count"]
        max_count = n if plan.metric == "sample_count" else len({(m[1], m[2]) for m in release.members})
        if not 0 <= count <= max_count:
            raise ValueError()
        return {"status": "ok", "count": count,
                "unit": "profiles" if plan.metric == "sample_count" else "study_local_donor_groups"}
    except (ValueError, KeyError, TypeError):
        raise IntegrityError("Malformed or inconsistent scientific answer") from None


class SampleQueries:
    def __init__(self, backend, access, release_lookup):
        self.backend, self.access, self.release_lookup = backend, access, release_lookup

    def run(self, principal, workspace_id, payload):
        # These first two arguments come from verified backend context, never from the LLM.
        plan = QueryPlan.model_validate(payload)
        revision = self.access.authorize(principal, workspace_id, plan.release_id)
        release = self.release_lookup(workspace_id, plan.release_id)
        if release is None:
            raise PermissionError("Release is not admitted in this workspace")

        def reauthorize():
            if self.access.authorize(principal, workspace_id, plan.release_id) != revision:
                raise PermissionError("Access grant changed; start a new request")
            if self.release_lookup(workspace_id, plan.release_id) != release:
                raise PermissionError("Release admission changed")

        def versions():
            if self.backend.table_etags() != dict(release.table_etags):
                raise IntegrityError("Cloud metadata changed since admission; audit the release again")

        compiled = compile_query(plan, release, self.backend.scope.project, self.backend.scope.dataset)
        versions()
        row, execution = self.backend.execute(compiled, before_execute=reauthorize)
        answer = validate_answer(row, plan, release)
        versions()
        reauthorize()
        return {"answer": answer, "scope": plan.model_dump(),
                "receipt": {"procedure_id": PROCEDURE_ID, "procedure_version": PROCEDURE_VERSION,
                            "procedure_sha256": PROCEDURE_HASH, "template_version": compiled.template_version,
                            "sql_sha256": digest(compiled.sql), "parameters_sha256": digest(json.loads(compiled.parameters_json)),
                            "release_manifest_sha256": release.manifest_sha256,
                            "membership_sha256": digest(release.members), "access_revision": revision,
                            "principal_scope_sha256": digest([principal.user_id, workspace_id]),
                            "result_sha256": digest(answer), "row_count": len(answer['records']) if plan.metric=='study_records' else 1,
                            "completed_at_utc": datetime.now(timezone.utc).isoformat(), **execution}}

    def bind(self, principal, workspace_id):
        """The future ADK tool accepts a plan only; protected context stays in a closure."""
        def query_core_records(plan: dict) -> dict:
            """Read approved core sample details or explicit profile/study-group counts."""
            return self.run(principal, workspace_id, plan)
        return query_core_records
