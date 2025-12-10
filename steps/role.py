"""
M2: Role Attribution Module
Maps Q segments from qa_tags to interviewer time intervals
"""

from typing import Dict, Any, List, Optional
from ScientificData.schemas import FinalJSON, InterviewerSegments, InterviewerSegment


class RoleAttributionAgent:
    """Agent for attributing roles to time segments"""
    
    def __init__(self):
        """Initialize Role Attribution agent"""
        pass
    
    def attribute_role(
        self,
        final_json: FinalJSON,
        asr_a: Dict[str, Any],
        asr_b: Dict[str, Any],
        diar_segments: Optional[List[Dict[str, Any]]] = None
    ) -> InterviewerSegments:
        """
        Map Q segments to interviewer time intervals
        
        Args:
            final_json: FinalJSON with qa_tags and evidence_map
            asr_a: ASR A result with tokens
            asr_b: ASR B result with tokens
            diar_segments: Optional diarization segments for consistency check
        
        Returns:
            InterviewerSegments with interviewer time intervals
        """
        interviewer_segments = []
        
        # Process Q tags from qa_tags
        for qa_tag in final_json.qa_tags:
            if qa_tag.tag == "Q":
                # Map Q segment to time interval
                time_segment = self._map_qa_tag_to_time(
                    qa_tag,
                    final_json,
                    asr_a,
                    asr_b
                )
                if time_segment:
                    interviewer_segments.append(time_segment)
        
        # Optional: Check consistency with diarization
        if diar_segments:
            self._check_diar_consistency(interviewer_segments, diar_segments)
        
        merged_segments = self._merge_interviewer_segments(interviewer_segments)
        return InterviewerSegments(interviewer_segments=merged_segments)
    
    def _map_qa_tag_to_time(
        self,
        qa_tag,
        final_json: FinalJSON,
        asr_a: Dict[str, Any],
        asr_b: Dict[str, Any]
    ) -> Optional[InterviewerSegment]:
        """
        Map a Q tag to a time interval using evidence_map
        
        Args:
            qa_tag: QATag with offset_start and offset_end
            final_json: FinalJSON with evidence_map
            asr_a: ASR A tokens
            asr_b: ASR B tokens
        
        Returns:
            InterviewerSegment with start/end times, or None if cannot map
        """
        # Get text span for this Q tag
        span_text = final_json.final_text[qa_tag.offset_start:qa_tag.offset_end]
        
        # Find corresponding evidence_map entries that actually overlap with Q tag
        matching_entries = []
        for ev_entry in final_json.evidence_map:
            # Check if this evidence entry actually overlaps with Q tag span
            ev_start = final_json.final_text.find(ev_entry.span)
            ev_end = ev_start + len(ev_entry.span) if ev_start != -1 else -1
            
            # Check if evidence entry overlaps with Q tag span
            if ev_start != -1 and not (ev_end <= qa_tag.offset_start or ev_start >= qa_tag.offset_end):
                matching_entries.append(ev_entry)
        
        if not matching_entries:
            return None
        
        # Determine time interval from src_locs
        # Use only the first matching entry to avoid spanning entire audio
        ev_entry = matching_entries[0]
        
        all_starts = []
        all_ends = []
        source_types = []
        
        # Use ASR_A if available
        if "A" in ev_entry.src_locs:
            src_loc_a = ev_entry.src_locs["A"]
            time_result = self._get_time_from_tokens(
                src_loc_a.start_tok,
                src_loc_a.end_tok,
                asr_a["tokens"]
            )
            if time_result:
                start, end = time_result
                all_starts.append(start)
                all_ends.append(end)
                source_types.append("a")
        
        # Use ASR_B if available
        if "B" in ev_entry.src_locs:
            src_loc_b = ev_entry.src_locs["B"]
            time_result = self._get_time_from_tokens(
                src_loc_b.start_tok,
                src_loc_b.end_tok,
                asr_b["tokens"]
            )
            if time_result:
                start, end = time_result
                all_starts.append(start)
                all_ends.append(end)
                source_types.append("b")
        
        if not all_starts:
            return None
        
        # Use earliest start and latest end from the single matching entry
        segment_start = min(all_starts)
        segment_end = max(all_ends)
        
        # Sanity check: segment should not span entire audio
        # If segment is too long (>90% of audio), skip it
        if len(asr_a["tokens"]) > 0:
            last_token_time = asr_a["tokens"][-1]["end"]
            if segment_end - segment_start > last_token_time * 0.9:
                return None
        
        # Determine source
        if len(source_types) > 1 and ("a" in source_types and "b" in source_types):
            source = "ab"
        elif "a" in source_types:
            source = "a"
        elif "b" in source_types:
            source = "b"
        else:
            source = "qa"  # Default to qa tag source
        
        return InterviewerSegment(
            start=segment_start,
            end=segment_end,
            source=source
        )
    
    def _spans_overlap(self, span1: str, span2: str, full_text: str) -> bool:
        """Check if two spans overlap in the full text"""
        # Simple check: if spans share common words
        words1 = set(span1.lower().split())
        words2 = set(span2.lower().split())
        return len(words1.intersection(words2)) > 0
    
    def _get_time_from_tokens(
        self,
        start_tok: Optional[int],
        end_tok: Optional[int],
        tokens: List[Dict[str, Any]]
    ) -> Optional[tuple]:
        """
        Get time interval from token indices
        
        Args:
            start_tok: Start token index (can be None)
            end_tok: End token index (exclusive, can be None)
            tokens: List of token dictionaries
        
        Returns:
            (start_time, end_time) tuple or None
        """
        # Check if start_tok or end_tok is None
        if start_tok is None or end_tok is None:
            return None
        
        if not tokens or start_tok < 0 or end_tok > len(tokens):
            return None
        
        if start_tok >= end_tok:
            return None
        
        start_time = tokens[start_tok]["start"]
        end_time = tokens[end_tok - 1]["end"]
        
        return (start_time, end_time)
    
    def _check_diar_consistency(
        self,
        interviewer_segments: List[InterviewerSegment],
        diar_segments: List[Dict[str, Any]]
    ):
        """
        Check consistency with diarization segments (for audit only)
        
        Args:
            interviewer_segments: Segments from qa_tags
            diar_segments: Diarization segments
        """
        # Mark segments with diar consistency
        for seg in interviewer_segments:
            # Check if this segment overlaps with any diar segment marked as interviewer
            for diar_seg in diar_segments:
                if self._time_overlap(
                    seg.start,
                    seg.end,
                    diar_seg.get("start", 0),
                    diar_seg.get("end", 0)
                ):
                    if diar_seg.get("speaker", "").lower() in ["interviewer", "q", "questioner"]:
                        seg.source = "diar_check_pass"
                    else:
                        seg.source = "diar_check_inconclusive"
                    break
    
    def _time_overlap(
        self,
        start1: float,
        end1: float,
        start2: float,
        end2: float
    ) -> bool:
        """Check if two time intervals overlap"""
        return not (end1 < start2 or end2 < start1)

    def _merge_interviewer_segments(
        self,
        segments: List[InterviewerSegment]
    ) -> List[InterviewerSegment]:
        """Deduplicate and merge overlapping interviewer segments"""
        if not segments:
            return []
        segments = sorted(
            segments,
            key=lambda s: (round(s.start, 3), round(s.end, 3), s.source)
        )
        merged: List[InterviewerSegment] = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue
            last = merged[-1]
            if abs(seg.start - last.start) < 1e-3 and abs(seg.end - last.end) < 1e-3:
                last.source = self._choose_source(last.source, seg.source)
                continue
            if seg.start <= last.end + 1e-3:
                last.end = max(last.end, seg.end)
                last.source = self._choose_source(last.source, seg.source)
            else:
                merged.append(seg)
        return merged

    def _choose_source(self, src_a: str, src_b: str) -> str:
        """Select higher priority source label"""
        priority = {
            "diar_check_pass": 5,
            "ab": 4,
            "a": 3,
            "b": 3,
            "qa": 2,
            "diar_check_inconclusive": 1
        }
        return src_a if priority.get(src_a, 0) >= priority.get(src_b, 0) else src_b

