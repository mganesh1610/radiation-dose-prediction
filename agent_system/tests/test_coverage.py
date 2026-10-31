"""Small fabricated unit fixtures exercise integrity failures, not model accuracy."""
import copy
import io

import numpy as np
import pytest

from research_agents.audit_cloud import AuditFailure
from research_agents.coverage import keyed, validate_matrix


def fixture():
    arrays = dict(values=np.array([[1., np.nan, 3.], [4., 5., 6.]]),
                  detected=np.ones((2, 3), dtype=bool), probe_ids=np.array(["p1", "p2"]),
                  gene_keys=np.array(["g1", "g1"]), sample_ids=np.array(["s1", "s2", "s3"]),
                  study_ids=np.array(["study_a", "study_a", "study_b"]),
                  split_group_ids=np.array(["a1", "a2", "b1"]), y=np.array([0., 2., 4.]))
    tables = dict(samples=[], sample_annotations=[], features=[], sample_groups=[],
                  studies=[{"study_id": "study_a"}, {"study_id": "study_b"}], split_members=[])
    for i, sample in enumerate(arrays["sample_ids"]):
        tables["samples"].append(dict(sample_id=str(sample), study_id=str(arrays["study_ids"][i]), group_id=str(arrays["split_group_ids"][i])))
        tables["sample_annotations"].append(dict(sample_id=str(sample), dose_gy=float(arrays["y"][i]),
                                                dose_status="recorded", organism="fixture", tissue="fixture", time_after_exposure_hours=24.))
        tables["sample_groups"].append(dict(group_id=str(arrays["split_group_ids"][i]), study_id=str(arrays["study_ids"][i])))
        tables["split_members"].append(dict(split_id="split", sample_id=str(sample), partition="train" if i < 2 else "validation"))
    for probe in arrays["probe_ids"]:
        tables["features"].append(dict(probe_id=str(probe), gene_id="g1", gene_namespace="fixture"))
    return arrays, dict(samples=3, probes=2, scale="linear_intensity"), tables


def pack(arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def test_valid_alignment_preserves_missingness_and_multiple_probes():
    arrays, registry, tables = fixture()
    result = validate_matrix(pack(arrays), registry, tables)
    assert result["profiles"] == 3 and result["measurement_cells"] == 6
    assert result["missing_expression_values"] == 1 and result["zero_dose_profiles"] == 1


@pytest.mark.parametrize("mutation", ["duplicate_samples", "misaligned_study", "wrong_dose",
                                     "missing_dose", "wrong_gene", "wrong_shape", "infinity",
                                     "wrong_scale", "pickle_array", "cross_study_group"])
def test_corrupt_or_unsupported_input_fails(mutation):
    arrays, registry, tables = fixture()
    if mutation == "duplicate_samples": arrays["sample_ids"][1] = "s1"
    elif mutation == "misaligned_study": arrays["study_ids"][0] = "study_b"
    elif mutation == "wrong_dose": arrays["y"][0] = .5
    elif mutation == "missing_dose": tables["sample_annotations"][0]["dose_gy"] = None
    elif mutation == "wrong_gene": tables["features"][0]["gene_id"] = "wrong"
    elif mutation == "wrong_shape": arrays["values"] = arrays["values"].T
    elif mutation == "infinity": arrays["values"][0, 0] = np.inf
    elif mutation == "wrong_scale": registry["scale"] = "log2"
    elif mutation == "pickle_array": arrays["sample_ids"] = np.array([{}, {}, {}], dtype=object)
    elif mutation == "cross_study_group": tables["sample_groups"][0]["study_id"] = "study_b"
    with pytest.raises(AuditFailure):
        validate_matrix(pack(arrays), registry, tables)


def test_duplicate_relational_keys_rejected():
    with pytest.raises(AuditFailure, match="duplicate"):
        keyed([{"id": "same"}, {"id": "same"}], ("id",))
