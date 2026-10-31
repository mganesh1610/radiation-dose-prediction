"""Bounded requests for one explicitly selected, approved expression set."""
from typing import Literal, Annotated
from pydantic import Field, model_validator
from .queries import StrictRecord


class ExpressionPlan(StrictRecord):
    version: Literal[1]
    release_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    expression_set_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    sample_ids: list[Annotated[str, Field(pattern=r"^GSM[0-9]{1,14}$")]] = Field(min_length=1, max_length=50)
    gene_namespace: Literal["NCBI_GeneID"]
    gene_ids: list[Annotated[str, Field(pattern=r"^[1-9][0-9]{0,11}$")]] = Field(min_length=1, max_length=50)
    aggregation: Literal["none"]

    @model_validator(mode="after")
    def unique_selectors(self):
        if len(set(self.sample_ids)) != len(self.sample_ids) or len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("Duplicate sample or gene selectors")
        return self
