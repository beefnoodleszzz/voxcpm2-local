from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QualityOptions(BaseModel):
    seed: int = Field(42, ge=0, le=2**32 - 1)
    inference_timesteps: int = Field(30, ge=1, le=100)
    cfg_value: float = Field(2.0, ge=2.0, le=10.0)
    warmup_patches: int = Field(0, ge=0, le=10)
    max_tokens: int = Field(2000, ge=2, le=10000)
    candidates: int = Field(1, ge=1, le=5)
    profile: Literal["production", "fast", "balanced", "quality", "ultimate"] | None = None
    normalization_profile: Literal["none", "dialogue", "narration"] = "none"
    project_lexicon: dict = Field(default_factory=dict)
    episode_lexicon: dict = Field(default_factory=dict)


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
    text: str = Field(min_length=1, max_length=5000)
    mode: Literal["design", "controllable", "ultimate"] = "controllable"
    instruct: str | None = None
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    preparation: dict | None = None

    @model_validator(mode="after")
    def validate_mode(self):
        if not self.text.strip():
            raise ValueError("Text cannot be whitespace")
        if self.mode == "design" and not (self.instruct and self.instruct.strip()):
            raise ValueError("Design mode requires instruct")
        if self.mode != "design" and not self.character_id:
            raise ValueError("Clone modes require character_id")
        if self.preparation is not None:
            required = {
                "text_original",
                "text_normalized",
                "text_spoken",
                "text_expected",
                "normalization",
                "pronunciation",
            }
            if (
                not required <= self.preparation.keys()
                or self.preparation["text_spoken"] != self.text
            ):
                raise ValueError(
                    "Prepared text provenance is incomplete or does not match line text"
                )
        return self


class BatchRequest(QualityOptions):
    project: str
    lines: list[BatchLine] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ids(self):
        if len({line.id for line in self.lines}) != len(self.lines):
            raise ValueError("Duplicate line IDs")
        for value in [self.project, *(line.id for line in self.lines)]:
            if not value or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in value
            ):
                raise ValueError("project and line IDs may contain only letters, digits, _ and -")
        return self
