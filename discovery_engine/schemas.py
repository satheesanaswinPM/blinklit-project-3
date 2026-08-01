from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class FeedbackDocument(BaseModel):
    id: str
    source: str
    text: str
    url: Optional[str] = None
    created_at: Optional[str] = None
    scraped_at: Optional[str] = None
    channel_meta: dict[str, Any] = Field(default_factory=dict)


class Annotation(BaseModel):
    doc_id: str
    sentiment: str
    categories: list[str] = Field(default_factory=list)
    barriers: list[str] = Field(default_factory=list)
    insight_types: list[str] = Field(default_factory=list)
    discovery_paths: list[str] = Field(default_factory=list)
    info_needs: list[str] = Field(default_factory=list)
    theme_hint: Optional[str] = None
    evidence_spans: list[str] = Field(default_factory=list)
    is_spam: bool = False
    is_off_topic_discovery: bool = False
    embedding_id: Optional[str] = None
    model_ver: str
    pipeline_run_id: str
    confidence: float = 0.0


class EvidenceQuote(BaseModel):
    doc_id: str
    source: str
    text: str
    url: Optional[str] = None
    span: str
    sentiment: Optional[str] = None


class ThemeCard(BaseModel):
    id: str
    title: str
    volume: int
    sentiment_mix: dict[str, int]
    barriers: list[str]
    categories: list[str]
    insight_types: list[str]
    discovery_paths: list[str]
    info_needs: list[str]
    evidence: list[EvidenceQuote]
    pipeline_run_id: str
    model_ver: str
    coherence: float = 0.0


class PipelineRun(BaseModel):
    id: str
    started_at: str
    finished_at: Optional[str] = None
    status: str
    n_docs: int = 0
    n_themes: int = 0
    model_ver: str
    notes: Optional[str] = None
