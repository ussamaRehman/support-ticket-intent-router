from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must be non-empty")
        return cleaned


class PredictBatchItem(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("id", "text")
    @classmethod
    def strip_value(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must be non-empty")
        return cleaned


class PredictBatchRequest(BaseModel):
    items: List[PredictBatchItem] = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

class AlternativePrediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    alternatives: List[AlternativePrediction]


class PredictBatchItemResponse(BaseModel):
    id: str
    label: str
    confidence: float


class PredictBatchResponse(BaseModel):
    items: List[PredictBatchItemResponse]
    model_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_dir: Optional[str] = None
    model_version: Optional[str] = None
