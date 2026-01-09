from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"text": "Reset my password", "top_k": 3, "min_confidence": 0.55}]
        }
    )

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
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "items": [
                        {"id": "1", "text": "Reset my password"},
                        {"id": "2", "text": "Refund this charge"},
                    ],
                    "top_k": 3,
                    "min_confidence": 0.55,
                }
            ]
        }
    )


class AlternativePrediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    alternatives: List[AlternativePrediction]
    needs_human: bool

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "label": "account",
                    "confidence": 0.91,
                    "alternatives": [
                        {"label": "account", "confidence": 0.91},
                        {"label": "billing", "confidence": 0.06},
                        {"label": "technical", "confidence": 0.03},
                    ],
                    "needs_human": False,
                }
            ]
        }
    )


class PredictBatchItemResponse(BaseModel):
    id: str
    label: str
    confidence: float
    needs_human: bool


class PredictBatchResponse(BaseModel):
    items: List[PredictBatchItemResponse]
    model_version: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "items": [
                        {
                            "id": "1",
                            "label": "account",
                            "confidence": 0.88,
                            "needs_human": False,
                        },
                        {
                            "id": "2",
                            "label": "billing",
                            "confidence": 0.81,
                            "needs_human": False,
                        },
                    ],
                    "model_version": "0.1.0",
                }
            ]
        }
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_dir: Optional[str] = None
    model_version: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "model_loaded": False,
                    "model_dir": None,
                    "model_version": None,
                }
            ]
        }
    )


class ReadyResponse(BaseModel):
    status: str
    ready: bool
    model_loaded: bool
    model_dir: Optional[str] = None
    model_version: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "ready": True,
                    "model_loaded": False,
                    "model_dir": None,
                    "model_version": None,
                }
            ]
        }
    )
