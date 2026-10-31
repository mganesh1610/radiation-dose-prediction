"""Versioned model-visible requests. Identity and SQL are never request fields."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Filters(StrictRecord):
    study_id: str | None = Field(default=None, pattern=r"^GSE[0-9]{1,12}$")
    dose_gy: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)


class QueryPlan(StrictRecord):
    version: Literal[1]
    metric: Literal["sample_details", "sample_count", "study_group_count", "study_records"]
    population: Literal["approved_core_release_v1"]
    release_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    dose_kind: Literal["recorded"]
    sample_id: str | None = Field(default=None, pattern=r"^GSM[0-9]{1,14}$")
    filters: Filters = Field(default_factory=Filters)

    @model_validator(mode="after")
    def validate_shape(self):
        if self.metric == "sample_details":
            if self.sample_id is None or self.filters != Filters():
                raise ValueError("Sample lookup requires a sample ID and no count filters")
        elif self.sample_id is not None:
            raise ValueError("Count queries use cohort filters, not a sample ID")
        if self.metric == 'study_records' and (self.filters.study_id is None or self.filters.dose_gy is not None):
            raise ValueError('Study records require one study and no dose filter')
        return self


class SampleRecord(StrictRecord):
    release_id: str
    sample_id: str
    study_id: str
    group_id: str
    dose_gy: float | None = Field(ge=0, le=100, allow_inf_nan=False)
    dose_status: str = Field(min_length=1, max_length=180)
    time_after_exposure_hours: float | None = Field(ge=0, allow_inf_nan=False)
    organism: str = Field(min_length=1, max_length=100)
    tissue: str = Field(min_length=1, max_length=100)
