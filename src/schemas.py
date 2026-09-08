from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QualityOptions(BaseModel):
    seed: int = 42
    inference_timesteps: int = Field(30, ge=1, le=100)
    cfg_value: float = Field(2.0, ge=2.0, le=10.0)
    warmup_patches: int = Field(0, ge=0, le=10)
    max_tokens: int = Field(2000, ge=2, le=10000)
    candidates: int = Field(1, ge=1, le=5)


class DesignRequest(QualityOptions):
    text: str = Field(min_length=1)
    instruct: str = Field(min_length=1)


class CloneRequest(QualityOptions):
    character_id: str
    style: str | None = None
    text: str = Field(min_length=1)
    instruct: str | None = None


class UltimateRequest(CloneRequest):
    pass


class BatchLine(BaseModel):
    id: str
    character_id: str | None = None
    style: str | None = None
    text: str
    mode: Literal["design", "controllable", "ultimate"] = "ultimate"
    instruct: str | None = None
    seed: int | None = None


class BatchRequest(QualityOptions):
    project: str
    lines: list[BatchLine]

    @model_validator(mode="after")
    def validate_ids(self):
        for value in [self.project, *(line.id for line in self.lines)]:
            if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
                raise ValueError("project and line IDs may contain only letters, digits, _ and -")
        return self

