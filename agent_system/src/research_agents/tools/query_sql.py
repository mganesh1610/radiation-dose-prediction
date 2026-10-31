"""Compile three reviewed read-only templates; no model-authored SQL execution."""
from dataclasses import dataclass
import hashlib
import json
import re

from ..contracts.queries import QueryPlan

TEMPLATE_VERSION = "core-exact-queries-v2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ApprovedRelease:
    release_id: str
    manifest_sha256: str
    # Entries are (sample, study, study-local group), from the verified manifest.
    members: tuple[tuple[str, str, str], ...]
    # Table etags bind reads to the audited source. A change requires re-admission.
    table_etags: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if (not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.release_id)
                or not re.fullmatch(r"[a-f0-9]{64}", self.manifest_sha256)
                or not 1 <= len(self.members) <= 1000
                or len({r[0] for r in self.members}) != len(self.members)
                or len(self.table_etags) != 2
                or set(dict(self.table_etags)) != {"samples", "sample_annotations"}
                or not all(v for _, v in self.table_etags)):
            raise ValueError("Invalid approved release configuration")
        for sample, study, group in self.members:
            if not re.fullmatch(r"GSM[0-9]{1,14}", sample) or not re.fullmatch(r"GSE[0-9]{1,12}", study) or not group:
                raise ValueError("Invalid approved membership")


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    parameters_json: str
    template_version: str = TEMPLATE_VERSION


def scalar(name, kind, value):
    return {"name": name, "parameterType": {"type": kind},
            "parameterValue": {"value": str(value)}}


def compile_query(plan: QueryPlan, release: ApprovedRelease, project: str, dataset: str):
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,1023}", dataset):
        raise ValueError("Invalid server-controlled table identifiers")
    if plan.release_id != release.release_id:
        raise ValueError("Release is not admitted")
    if plan.filters.study_id and plan.filters.study_id not in {r[1] for r in release.members}:
        raise ValueError("Study is outside the selected release")
    params = [scalar("release_id", "STRING", release.release_id),
              {"name": "approved_ids", "parameterType": {"type": "ARRAY", "arrayType": {"type": "STRING"}},
               "parameterValue": {"arrayValues": [{"value": r[0]} for r in release.members]}}]
    where = []
    for name, value, kind in (("sample_id", plan.sample_id, "STRING"),
                              ("study_id", plan.filters.study_id, "STRING"),
                              ("dose_gy", plan.filters.dose_gy, "FLOAT64")):
        if value is not None:
            where.append(name + " = @" + name)
            params.append(scalar(name, kind, value))
    predicate = " AND ".join(where) or "TRUE"
    if plan.metric == "sample_details":
        answer = "SELECT * FROM population WHERE " + predicate + " ORDER BY sample_id LIMIT 2"
    elif plan.metric == 'study_records':
        answer = 'SELECT * FROM population WHERE ' + predicate + ' ORDER BY sample_id LIMIT 146'
    else:
        count = "COUNT(*)" if plan.metric == "sample_count" else "COUNT(DISTINCT TO_JSON_STRING(STRUCT(study_id, group_id)))"
        answer = f"SELECT {count} AS count FROM population WHERE {predicate}"
    # Integrity checks precede use of any count. DISTINCT is not used to hide bad joins.
    sql = f"""WITH
s AS (SELECT release_id, sample_id, study_id, group_id FROM `{project}.{dataset}.samples` WHERE release_id = @release_id),
a AS (SELECT release_id, sample_id, dose_gy, dose_status, time_after_exposure_hours, organism, tissue
      FROM `{project}.{dataset}.sample_annotations` WHERE release_id = @release_id),
population AS (
  SELECT s.*, a.dose_gy, a.dose_status, a.time_after_exposure_hours, a.organism, a.tissue
  FROM s LEFT JOIN a ON s.release_id = a.release_id AND s.sample_id = a.sample_id
),
quality AS (SELECT
  (SELECT COUNT(*) FROM s) AS sample_rows,
  (SELECT COUNT(DISTINCT sample_id) FROM s) AS unique_samples,
  (SELECT COUNTIF(sample_id NOT IN UNNEST(@approved_ids) OR sample_id IS NULL) FROM s) AS unknown_samples,
  (SELECT COUNT(*) FROM a) AS annotation_rows,
  (SELECT COUNT(DISTINCT sample_id) FROM a) AS unique_annotations,
  (SELECT COUNTIF(sample_id NOT IN UNNEST(@approved_ids) OR sample_id IS NULL) FROM a) AS unknown_annotations,
  (SELECT COUNT(*) FROM population) AS joined_rows),
answer AS ({answer})
SELECT quality.*, TO_JSON_STRING(ARRAY(SELECT AS STRUCT * FROM answer)) AS answer_json FROM quality"""
    return CompiledQuery(sql, json.dumps(params, sort_keys=True))
