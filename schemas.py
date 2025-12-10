"""
Schemas for the ScientificData transcript pipeline.
Only keep structures needed for preprocess -> ASR -> reconcile -> role -> multiscale transcript.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class SrcLoc(BaseModel):
    """Source location for a token span"""
    start_tok: Optional[int] = Field(None, description="Start token index")
    end_tok: Optional[int] = Field(None, description="End token index (exclusive)")


class EvidenceMapEntry(BaseModel):
    """Evidence mapping entry"""
    span: str = Field(..., description="Text span in final_text")
    support: Literal["A", "B", "AB"] = Field(..., description="Source support")
    src_locs: Dict[str, SrcLoc] = Field(..., description="Source locations in ASR A/B")
    notes: Optional[str] = Field(None, description="Optional notes")


class UncertainSpan(BaseModel):
    """Uncertain span entry"""
    text: str = Field(..., description="Uncertain text span")
    reason: str = Field(..., description="Reason for uncertainty")
    severity: Literal["low", "medium", "high"] = Field(..., description="Severity level")


class ChangelogEntry(BaseModel):
    """Changelog entry"""
    from_: str = Field(..., alias="from", description="Original text")
    to: str = Field(..., description="Changed text")
    why: Literal["grammar", "punctuation", "normalization", "entity_disambiguation"] = Field(
        ..., description="Reason for change"
    )

    class Config:
        populate_by_name = True


class QATag(BaseModel):
    """QA tag entry"""
    offset_start: int = Field(..., description="Start offset in final_text")
    offset_end: int = Field(..., description="End offset in final_text")
    tag: Literal["Q", "A", "null"] = Field(..., description="Tag type")


class TextSpan(BaseModel):
    """Text span with token/time alignment"""
    text: str = Field("", description="Span text")
    start_tok: Optional[int] = Field(None, description="Start token index")
    end_tok: Optional[int] = Field(None, description="End token index (exclusive)")
    start_time: Optional[float] = Field(None, description="Start time in seconds")
    end_time: Optional[float] = Field(None, description="End time in seconds")


class FinalJSON(BaseModel):
    """Final reconciled transcript with evidence mapping"""
    final_text: str = Field(..., description="Unified transcript text")
    changed: bool = Field(..., description="Whether text was changed from originals")
    evidence_map: List[EvidenceMapEntry] = Field(..., description="Evidence mapping")
    uncertain_spans: List[UncertainSpan] = Field(default_factory=list, description="Uncertain spans")
    changelog: List[ChangelogEntry] = Field(default_factory=list, description="Change log")
    qa_tags: List[QATag] = Field(..., description="QA tags")


class InterviewerSegment(BaseModel):
    """Interviewer segment"""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    source: Literal["qa", "ab", "a", "b", "diar_check_pass", "diar_check_inconclusive"] = Field(
        ..., description="Source of attribution"
    )


class InterviewerSegments(BaseModel):
    """Interviewer segments output"""
    interviewer_segments: List[InterviewerSegment] = Field(..., description="List of interviewer segments")


class Defect(BaseModel):
    """Defect entry from Critic"""
    type: str = Field(..., description="Defect type")
    severity: Literal["low", "medium", "high"] = Field(..., description="Severity")
    description: str = Field(..., description="Description")
    location: Optional[str] = Field(None, description="Location in text")
    suggested_fix: Optional[str] = Field(None, description="Suggested fix")


class CriticOutput(BaseModel):
    """Critic output"""
    defects: List[Defect] = Field(default_factory=list, description="List of defects")
    status: Literal["pass", "revise"] = Field(..., description="Status")


class WordToken(BaseModel):
    """Word-level transcript item"""
    id: int = Field(..., description="Sequential word id")
    text: str = Field(..., description="Word text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    confidence: float = Field(..., description="ASR confidence")
    speaker: Literal["interviewer", "interviewee", "unknown"] = Field(..., description="Speaker label")
    source: str = Field(..., description="ASR source alias (e.g., A or model name)")


class PhraseSpan(BaseModel):
    """Phrase-level segment"""
    id: int = Field(..., description="Sequential phrase id")
    start_word: int = Field(..., description="Inclusive start word id")
    end_word: int = Field(..., description="Inclusive end word id")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Phrase text")
    speaker: Literal["interviewer", "interviewee", "unknown"] = Field(..., description="Speaker label")
    confidence: float = Field(..., description="Confidence (LLM/self-estimated)")
    source: Literal["llm", "heuristic"] = Field(..., description="How the phrase was derived")
    reason: Optional[str] = Field(None, description="Optional rationale from LLM/heuristic")


class SentenceSpan(BaseModel):
    """Sentence-level segment"""
    id: int = Field(..., description="Sequential sentence id")
    start_word: int = Field(..., description="Inclusive start word id")
    end_word: int = Field(..., description="Inclusive end word id")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Sentence text")
    speaker: Literal["interviewer", "interviewee", "unknown"] = Field(..., description="Speaker label")
    confidence: float = Field(..., description="Confidence (LLM/self-estimated)")
    source: Literal["llm", "heuristic"] = Field(..., description="How the sentence was derived")
    reason: Optional[str] = Field(None, description="Optional rationale from LLM/heuristic")


class MultiScaleTranscript(BaseModel):
    """Combined multiscale transcript"""
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the run")
    words: List[WordToken] = Field(default_factory=list, description="Word-level transcript")
    phrases: List[PhraseSpan] = Field(default_factory=list, description="Phrase-level transcript")
    sentences: List[SentenceSpan] = Field(default_factory=list, description="Sentence-level transcript")
