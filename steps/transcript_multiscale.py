"""
Build multiscale transcripts (word, phrase, sentence) from ASR tokens and roles.
Primary segmentation uses LLM; fallback relies on simple heuristics.
"""

import json
import logging
import re
from typing import List, Optional, Sequence, Tuple, Union

from ScientificData.schemas import (
    WordToken,
    PhraseSpan,
    SentenceSpan,
    MultiScaleTranscript,
    InterviewerSegments,
)
from ScientificData.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", word.lower()).strip()


class TranscriptMultiscaleBuilder:
    """Build word/phrase/sentence-level transcripts."""

    def __init__(self, llm_client: Optional[LLMClient] = None, max_words_for_llm: int = 1200):
        self.llm_client = llm_client
        self.max_words_for_llm = max_words_for_llm

    def build(
        self,
        final_text: str,
        asr_tokens: List[dict],
        interviewer_segments: InterviewerSegments,
        source_alias: str = "A",
        metadata: Optional[dict] = None,
        allowed_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> MultiScaleTranscript:
        tokens_used = self._filter_tokens(asr_tokens, allowed_intervals)
        words = self._build_words(tokens_used, interviewer_segments, source_alias)
        phrases = self._build_level(level="phrase", words=words, final_text=final_text)
        sentences = self._build_level(level="sentence", words=words, final_text=final_text)
        if allowed_intervals is not None:
            (metadata or {}).setdefault("preprocess", {})["allowed_intervals"] = allowed_intervals
        return MultiScaleTranscript(
            metadata=metadata or {},
            words=words,
            phrases=phrases,
            sentences=sentences,
        )

    def _build_words(
        self,
        asr_tokens: List[dict],
        interviewer_segments: InterviewerSegments,
        source_alias: str,
    ) -> List[WordToken]:
        words: List[WordToken] = []
        for idx, tok in enumerate(asr_tokens):
            start = float(tok.get("start", 0.0))
            end = float(tok.get("end", start))
            speaker = self._infer_speaker(start, end, interviewer_segments)
            words.append(
                WordToken(
                    id=idx,
                    text=str(tok.get("word", "")),
                    start=start,
                    end=end,
                    confidence=float(tok.get("confidence", 0.9)),
                    speaker=speaker,
                    source=source_alias,
                )
            )
        return words

    def _infer_speaker(self, start: float, end: float, interviewer_segments: InterviewerSegments) -> str:
        mid = (start + end) / 2.0
        for seg in interviewer_segments.interviewer_segments:
            if seg.start <= mid <= seg.end:
                return "interviewer"
        return "interviewee"

    def _build_level(
        self,
        level: str,
        words: List[WordToken],
        final_text: str,
    ) -> List[Union[PhraseSpan, SentenceSpan]]:
        if not words:
            return []

        chosen = self._llm_segments(level, words, final_text)
        if not chosen:
            # Fallback: use rule-based segmentation instead of single full-range segment
            chosen = self._rule_based_segments(level, words)

        spans: List[Union[PhraseSpan, SentenceSpan]] = []
        for idx, seg in enumerate(chosen):
            start_word, end_word, source, confidence, reason = seg
            if start_word < 0 or end_word >= len(words) or end_word < start_word:
                continue
            slice_words = words[start_word : end_word + 1]
            speaker = self._majority_speaker(slice_words)
            text = " ".join(w.text for w in slice_words).strip()
            start_time = slice_words[0].start
            end_time = slice_words[-1].end
            if level == "phrase":
                spans.append(
                    PhraseSpan(
                        id=idx,
                        start_word=start_word,
                        end_word=end_word,
                        start=start_time,
                        end=end_time,
                        text=text,
                        speaker=speaker,
                        confidence=confidence,
                        source=source,
                        reason=reason,
                    )
                )
            else:
                spans.append(
                    SentenceSpan(
                        id=idx,
                        start_word=start_word,
                        end_word=end_word,
                        start=start_time,
                        end=end_time,
                        text=text,
                        speaker=speaker,
                        confidence=confidence,
                        source=source,
                        reason=reason,
                    )
                )
        return spans

    def _rule_based_segments(
        self,
        level: str,
        words: List[WordToken],
    ) -> List[Tuple[int, int, str, float, str]]:
        """Rule-based segmentation fallback using punctuation and pause gaps."""
        if not words:
            return []
        # Sentence: split on sentence-ending punctuation or long pauses
        # Phrase: split on commas, short pauses, or speaker changes
        sentence_enders = {'.', '?', '!', '。', '？', '！'}
        phrase_enders = {',', ';', ':', '，', '；', '：', '-', '—'}
        pause_threshold_sent = 1.5  # seconds
        pause_threshold_phrase = 0.8

        segments: List[Tuple[int, int, str, float, str]] = []
        seg_start = 0
        for i, w in enumerate(words):
            is_boundary = False
            reason = ""
            # Check punctuation at end of word
            text = w.text.strip()
            if level == "sentence":
                if text and text[-1] in sentence_enders:
                    is_boundary = True
                    reason = f"punct:{text[-1]}"
                elif i < len(words) - 1:
                    gap = words[i + 1].start - w.end
                    if gap >= pause_threshold_sent:
                        is_boundary = True
                        reason = f"pause:{gap:.2f}s"
            else:  # phrase
                if text and text[-1] in (sentence_enders | phrase_enders):
                    is_boundary = True
                    reason = f"punct:{text[-1]}"
                elif i < len(words) - 1:
                    gap = words[i + 1].start - w.end
                    if gap >= pause_threshold_phrase:
                        is_boundary = True
                        reason = f"pause:{gap:.2f}s"
                    # Also split on speaker change for phrases
                    if words[i + 1].speaker != w.speaker:
                        is_boundary = True
                        reason = "speaker_change"
            if is_boundary:
                segments.append((seg_start, i, "heuristic", 0.6, reason))
                seg_start = i + 1
        # Add final segment
        if seg_start < len(words):
            segments.append((seg_start, len(words) - 1, "heuristic", 0.6, "end_of_text"))
        return segments if segments else [(0, len(words) - 1, "heuristic", 0.5, "fallback_full_range")]

    def _llm_segments(
        self,
        level: str,
        words: List[WordToken],
        final_text: str,
    ) -> Optional[List[Tuple[int, int, str, float, str]]]:
        if not self.llm_client:
            return None
        if len(words) > self.max_words_for_llm:
            return None
        try:
            # Compact word outline: group consecutive words by speaker
            # Format: [speaker] id-id: word1 word2 ...
            word_groups = self._compact_word_outline(words)

            # Different hints for phrase vs sentence
            level_hint = "clause/phrase boundaries" if level == "phrase" else "sentence boundaries (., ?, !)"

            prompt = f"""Split into {level}s. Words 0-{len(words)-1}. Use {level_hint}.
WORDS: {word_groups}
TEXT: {final_text[:600]}
Return segments covering all words [0,{len(words)-1}] without gaps."""

            schema_description = """{"segments": [{"start_word": int, "end_word": int}]}"""

            result = self.llm_client.chat_completion_with_schema_description(
                system_prompt=f"Segment transcript into {level}s. Return word index ranges only.",
                user_message=prompt,
                schema_description=schema_description,
                temperature=0.2,
                max_tokens=1500,
            )
            segments_raw = result.get("segments", []) if isinstance(result, dict) else []
            parsed: List[Tuple[int, int, str, float, str]] = []
            for item in segments_raw:
                try:
                    s = int(item.get("start_word"))
                    e = int(item.get("end_word"))
                except Exception:
                    continue
                if s < 0 or e >= len(words) or e < s:
                    continue
                parsed.append((s, e, "llm", 0.75, "llm_segmentation"))
            parsed_sorted = sorted(parsed, key=lambda x: x[0])
            contiguous: List[Tuple[int, int, str, float, str]] = []
            current = 0
            for seg in parsed_sorted:
                s, e, src, conf, reason = seg
                if s > current:
                    return None
                s = max(s, current)
                if e < s:
                    continue
                contiguous.append((s, e, src, conf, reason))
                current = e + 1
            if current < len(words):
                return None
            if contiguous and contiguous[0][0] == 0 and contiguous[-1][1] == len(words) - 1:
                return contiguous
            return None
        except Exception as exc:
            logger.warning("LLM segmentation for %s failed: %s", level, exc)
            return None

    def _compact_word_outline(self, words: List[WordToken]) -> str:
        """Create compact word outline grouped by speaker."""
        if not words:
            return ""
        parts = []
        cur_speaker = words[0].speaker
        cur_start = 0
        cur_words = [words[0].text]
        for i in range(1, len(words)):
            if words[i].speaker != cur_speaker:
                # Flush current group
                spk_abbr = "I" if cur_speaker == "interviewer" else "E"
                parts.append(f"[{spk_abbr}]{cur_start}-{i-1}:{' '.join(cur_words)}")
                cur_speaker = words[i].speaker
                cur_start = i
                cur_words = [words[i].text]
            else:
                cur_words.append(words[i].text)
        # Flush last group
        spk_abbr = "I" if cur_speaker == "interviewer" else "E"
        parts.append(f"[{spk_abbr}]{cur_start}-{len(words)-1}:{' '.join(cur_words)}")
        return " | ".join(parts)

    def _segment_speaker(self, words: List[WordToken], start: int, end: int) -> str:
        slice_words = words[start : end + 1]
        return self._majority_speaker(slice_words)

    def _majority_speaker(self, words: Sequence[WordToken]) -> str:
        if not words:
            return "interviewee"
        counts = {"interviewer": 0, "interviewee": 0}
        for w in words:
            counts[w.speaker] = counts.get(w.speaker, 0) + 1
        speaker = max(counts, key=lambda k: counts[k])
        return speaker or "interviewee"

    def _filter_tokens(self, tokens: List[dict], allowed: Optional[List[Tuple[float, float]]]) -> List[dict]:
        """Keep tokens whose mid-time falls inside allowed intervals."""
        if not allowed:
            return tokens

        def in_allowed(mid: float) -> bool:
            return any(start <= mid <= end for start, end in allowed)

        filtered = []
        for tok in tokens:
            start = float(tok.get("start", 0.0))
            end = float(tok.get("end", start))
            mid = (start + end) / 2.0
            if in_allowed(mid):
                filtered.append(tok)
        return filtered
