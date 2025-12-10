"""
M1: Proposer Agent
Unifies ASR A/B transcripts, generates evidence_map and qa_tags
"""

import json
import logging
from typing import Dict, Any, List, Optional
from ScientificData.schemas import FinalJSON, EvidenceMapEntry, QATag, UncertainSpan, ChangelogEntry
from ScientificData.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ProposerAgent:
    """Agent for reconciling ASR transcripts"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        chunk_settings: Optional[Dict[str, Any]] = None,
        prompt_limits: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Proposer agent
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        chunk_settings = chunk_settings or {}
        prompt_limits = prompt_limits or {}
        self.chunk_size = chunk_settings.get("chunk_size", 200)
        self.overlap = chunk_settings.get("overlap", 30)
        self.input_token_soft_limit = chunk_settings.get("input_token_soft_limit", 500)
        self.estimated_output_soft_limit = chunk_settings.get("estimated_output_soft_limit", 4000)
        self.max_asr_text_chars = prompt_limits.get("max_asr_text_chars", 12000)
    
    def reconcile(
        self,
        asr_a: Dict[str, Any],
        asr_b: Dict[str, Any],
        glossary: Dict[str, Any],
        scenario_hint: str
    ) -> FinalJSON:
        """
        Reconcile ASR A and B into unified transcript
        
        Args:
            asr_a: Normalized ASR A result
            asr_b: Normalized ASR B result
            glossary: Glossary dictionary
            scenario_hint: Scenario type
        
        Returns:
            FinalJSON with unified text, evidence_map, qa_tags
        """
        # Prepare prompt
        system_prompt = self._get_system_prompt()
        user_message = self._build_user_message(asr_a, asr_b, glossary, scenario_hint)
        
        # Get schema description
        schema_description = self._get_schema_description()
        
        # Estimate output size and decide if we need chunking
        # Rough estimate: each token in input might generate 2-3 tokens in output
        # Be conservative: use 4x multiplier and lower threshold to ensure we don't exceed API limits
        estimated_output_tokens = (len(asr_a["tokens"]) + len(asr_b["tokens"])) * 4
        
        total_input_tokens = len(asr_a["tokens"]) + len(asr_b["tokens"])
        if (
            estimated_output_tokens > self.estimated_output_soft_limit
            or total_input_tokens > self.input_token_soft_limit
        ):
            logger.info(
                "Large transcript detected (%s + %s tokens, est output %s) using chunking",
                len(asr_a["tokens"]),
                len(asr_b["tokens"]),
                estimated_output_tokens,
            )
            result = self._reconcile_with_chunking(asr_a, asr_b, glossary, scenario_hint)
        else:
            # Call LLM with max_tokens (API limit is 8192)
            result = self.llm_client.chat_completion_with_schema_description(
                system_prompt=system_prompt,
                user_message=user_message,
                schema_description=schema_description,
                temperature=0.2,  # Low temperature for consistency
                max_tokens=8192  # API maximum
            )
        
        # Clean up result: ensure src_locs match support
        result = self._clean_evidence_map(result)
        
        # Clean up changelog: remove invalid entries
        result = self._clean_changelog(result)
        
        # Clean up qa_tags: convert None to 'null'
        result = self._clean_qa_tags(result)
        
        # Validate and return
        try:
            final_json = FinalJSON(**result)
            return final_json
        except Exception as e:
            raise ValueError(f"Failed to validate FinalJSON: {e}\nResult: {result}")
    
    def _clean_evidence_map(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean evidence_map to ensure src_locs match support"""
        if "evidence_map" not in result:
            return result
        
        for entry in result["evidence_map"]:
            support = entry.get("support", "")
            src_locs = entry.get("src_locs", {})
            
            # If support is "A", ensure B is None or missing
            if support == "A":
                if "B" in src_locs:
                    src_locs["B"] = {"start_tok": None, "end_tok": None}
                else:
                    src_locs["B"] = {"start_tok": None, "end_tok": None}
            # If support is "B", ensure A is None or missing
            elif support == "B":
                if "A" in src_locs:
                    src_locs["A"] = {"start_tok": None, "end_tok": None}
                else:
                    src_locs["A"] = {"start_tok": None, "end_tok": None}
            # If support is "AB", ensure both are present and not None
            elif support == "AB":
                if "A" not in src_locs or src_locs["A"].get("start_tok") is None:
                    # Try to infer from context or set to 0
                    src_locs["A"] = {"start_tok": 0, "end_tok": 0}
                if not src_locs or "B" not in src_locs or not src_locs["B"] or src_locs["B"].get("start_tok") is None:
                    src_locs["B"] = {"start_tok": 0, "end_tok": 0}
        
        return result
    
    def _clean_changelog(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean changelog to remove invalid entries and grammar corrections"""
        if "changelog" not in result:
            return result
        
        # Valid why values according to schema
        valid_why_values = {"grammar", "punctuation", "normalization", "entity_disambiguation"}
        
        cleaned_changelog = []
        for entry in result["changelog"]:
            # Skip if from and to are the same (no actual change)
            if entry.get("from") == entry.get("to"):
                continue
            
            # Skip grammar and punctuation corrections - we want to preserve original speech
            why = entry.get("why")
            if why in ["grammar", "punctuation"]:
                # These should not exist according to new prompt, but filter them out anyway
                continue
            
            # Skip if why is not in valid values
            if why not in valid_why_values:
                # Try to map common invalid values to valid ones
                if why == "preservation":
                    continue  # Skip preservation entries
                elif why in ["grammar_fix", "grammar_correction"]:
                    continue  # Skip grammar corrections
                elif why in ["punctuation_fix", "punctuation_correction"]:
                    continue  # Skip punctuation corrections
                elif why in ["normalize", "normalized"]:
                    entry["why"] = "normalization"
                elif why in ["entity", "disambiguation"]:
                    entry["why"] = "entity_disambiguation"
                else:
                    # Unknown value, skip this entry
                    continue
            
            cleaned_changelog.append(entry)
        
        result["changelog"] = cleaned_changelog
        return result
    
    def _clean_qa_tags(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean qa_tags: convert None to 'null' string for schema compliance"""
        if "qa_tags" not in result:
            return result
        
        cleaned_qa_tags = []
        for tag in result["qa_tags"]:
            tag_value = tag.get("tag")
            # Convert None to 'null' string (schema requires literal 'null', not Python None)
            if tag_value is None:
                tag_value = "null"
            cleaned_qa_tags.append({
                "offset_start": tag.get("offset_start"),
                "offset_end": tag.get("offset_end"),
                "tag": tag_value
            })
        
        result["qa_tags"] = cleaned_qa_tags
        return result
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Proposer"""
        return """You are a Transcript Reconciler. Unify ASR_A and ASR_B into one transcript.

RULES:
1. SELECT best version per segment from A or B - do NOT correct grammar/punctuation/words
2. Preserve original speech exactly (errors included)
3. Every span must trace to A, B, or both via evidence_map

OUTPUT:
- final_text: merged transcript (verbatim from A/B)
- evidence_map: [{span, support(A|B|AB), src_locs}] - non-overlapping spans covering full text
- qa_tags: [{offset_start, offset_end, tag(Q|A|null)}] - Q=interviewer, A=interviewee
- uncertain_spans: only if ASR quality is questionable
- changelog: only when choosing A over B or vice versa (not for grammar)

Return valid JSON only."""
    
    def _build_user_message(
        self,
        asr_a: Dict[str, Any],
        asr_b: Dict[str, Any],
        glossary: Dict[str, Any],
        scenario_hint: str
    ) -> str:
        """Build user message for LLM - concise format"""
        asr_a_text = self._truncate_text(asr_a['text'])
        asr_b_text = self._truncate_text(asr_b['text'])
        n_a, n_b = len(asr_a["tokens"]), len(asr_b["tokens"])

        # Only include glossary if it has meaningful content
        glossary_str = ""
        if glossary and (glossary.get("entities") or glossary.get("medical_terms")):
            glossary_str = f"\nGLOSSARY: {json.dumps(glossary, ensure_ascii=False)}"

        # Compact token snippet for alignment (first 40 tokens only)
        snippet_a = self._format_tokens_compact(asr_a["tokens"][:40])
        snippet_b = self._format_tokens_compact(asr_b["tokens"][:40])

        return f"""SCENARIO: {scenario_hint}

ASR_A ({n_a} tokens): {asr_a_text}
Tokens[0:40]: {snippet_a}

ASR_B ({n_b} tokens): {asr_b_text}
Tokens[0:40]: {snippet_b}
{glossary_str}
Merge A and B. Output evidence_map spans must cover final_text completely without overlap. Use 0-based token indices for src_locs."""
    
    def _get_schema_description(self) -> str:
        """Get JSON schema description for FinalJSON - minimal format"""
        return """{
  "final_text": "string",
  "changed": true|false,
  "evidence_map": [{"span": "text", "support": "A|B|AB", "src_locs": {"A": {"start_tok": int, "end_tok": int}, "B": {...}}}],
  "qa_tags": [{"offset_start": int, "offset_end": int, "tag": "Q|A|null"}],
  "uncertain_spans": [{"text": "str", "reason": "str", "severity": "low|medium|high"}],
  "changelog": [{"from": "str", "to": "str", "why": "normalization|entity_disambiguation"}]
}"""

    def _truncate_text(self, text: str) -> str:
        """Trim ASR text to avoid overloading the prompt"""
        limit = self.max_asr_text_chars
        if len(text) <= limit:
            return text
        return text[:limit] + "...[truncated]"

    def _format_tokens_compact(self, tokens: List[Dict[str, Any]]) -> str:
        """Compact token format: idx:word:start-end (no JSON overhead)"""
        parts = []
        for idx, t in enumerate(tokens):
            w = t.get("word", "").replace(":", "")  # escape colon
            s = round(float(t.get("start", 0.0)), 2)
            e = round(float(t.get("end", 0.0)), 2)
            parts.append(f"{idx}:{w}:{s}-{e}")
        return " ".join(parts)

    def _reconcile_with_chunking(
        self,
        asr_a: Dict[str, Any],
        asr_b: Dict[str, Any],
        glossary: Dict[str, Any],
        scenario_hint: str
    ) -> Dict[str, Any]:
        """
        Reconcile with chunking strategy for long transcripts
        
        Args:
            asr_a: Normalized ASR A result
            asr_b: Normalized ASR B result
            glossary: Glossary dictionary
            scenario_hint: Scenario type
        
        Returns:
            Merged result dictionary
        """
        # Chunk size: process ~200 tokens at a time with 30 token overlap
        # Smaller chunks reduce output size per chunk and avoid truncation
        chunk_size = self.chunk_size
        overlap = self.overlap
        
        tokens_a = asr_a["tokens"]
        tokens_b = asr_b["tokens"]
        
        logger.info(f"Starting chunk creation: A has {len(tokens_a)} tokens, B has {len(tokens_b)} tokens")
        
        # Create chunks
        chunks = []
        i = 0
        max_tokens = max(len(tokens_a), len(tokens_b))
        logger.info(f"Max tokens: {max_tokens}, chunk_size: {chunk_size}, overlap: {overlap}")
        
        while i < max_tokens:
            end_i = min(i + chunk_size, max_tokens)
            
            # Prevent infinite loop: if end_i hasn't advanced, break
            if end_i <= i:
                break
            
            # Extract chunks
            chunk_a_tokens = tokens_a[i:end_i] if i < len(tokens_a) else []
            chunk_b_tokens = tokens_b[i:end_i] if i < len(tokens_b) else []
            
            if not chunk_a_tokens and not chunk_b_tokens:
                break
            
            # Create chunk ASR dicts
            chunk_asr_a = {
                "text": " ".join(t["word"] for t in chunk_a_tokens),
                "tokens": chunk_a_tokens
            }
            chunk_asr_b = {
                "text": " ".join(t["word"] for t in chunk_b_tokens),
                "tokens": chunk_b_tokens
            }
            
            chunks.append({
                "start_idx_a": i if i < len(tokens_a) else len(tokens_a),
                "start_idx_b": i if i < len(tokens_b) else len(tokens_b),
                "end_idx": end_i,
                "asr_a": chunk_asr_a,
                "asr_b": chunk_asr_b
            })
            
            # Move to next chunk with overlap
            next_i = end_i - overlap
            # Prevent infinite loop: ensure we advance
            if next_i <= i:
                break
            i = next_i
            if i >= max_tokens:
                break
        
        logger.info(f"Created {len(chunks)} chunks for processing")
        if chunks:
            logger.info(
                f"Chunk ranges: "
                f"{[(c['start_idx_a'], c['start_idx_b'], c['end_idx']) for c in chunks[:5]]}"
            )
        
        # Process each chunk
        chunk_results = []
        import time
        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)} (tokens: A={len(chunk['asr_a']['tokens'])}, B={len(chunk['asr_b']['tokens'])})")
            start_time = time.time()
            try:
                system_prompt = self._get_system_prompt()
                user_message = self._build_user_message(
                    chunk["asr_a"], chunk["asr_b"], glossary, scenario_hint
                )
                schema_description = self._get_schema_description()
                
                logger.info(f"Calling LLM for chunk {idx + 1}...")
                result = self.llm_client.chat_completion_with_schema_description(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    schema_description=schema_description,
                    temperature=0.2,
                    max_tokens=6000  # Smaller max_tokens per chunk to avoid truncation
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Chunk {idx + 1} LLM call completed in {elapsed:.2f}s")
                
                # Adjust token indices to global indices
                result = self._adjust_chunk_indices(
                    result,
                    chunk["start_idx_a"],
                    chunk["start_idx_b"]
                )
                chunk_results.append(result)
                logger.info(f"Chunk {idx + 1} processed successfully")
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Error processing chunk {idx + 1} after {elapsed:.2f}s: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        
        # Merge chunk results
        merged = self._merge_chunk_results(chunk_results, asr_a, asr_b)
        return merged
    
    def _adjust_chunk_indices(
        self,
        result: Dict[str, Any],
        start_idx_a: int,
        start_idx_b: int
    ) -> Dict[str, Any]:
        """Adjust token indices in result to global indices"""
        if "evidence_map" in result:
            for entry in result["evidence_map"]:
                src_locs = entry.get("src_locs", {})
                if src_locs and "A" in src_locs and src_locs["A"] and src_locs["A"].get("start_tok") is not None:
                    src_locs["A"]["start_tok"] += start_idx_a
                    src_locs["A"]["end_tok"] += start_idx_a
                if src_locs and "B" in src_locs and src_locs["B"] and src_locs["B"].get("start_tok") is not None:
                    src_locs["B"]["start_tok"] += start_idx_b
                    src_locs["B"]["end_tok"] += start_idx_b
        
        return result
    
    def _merge_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]],
        asr_a: Dict[str, Any],
        asr_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge multiple chunk results into one"""
        if not chunk_results:
            raise ValueError("No chunk results to merge")
        
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # Merge final_text - handle overlaps by removing duplicate content
        merged_final_text = ""
        prev_text_end = ""
        for result in chunk_results:
            chunk_text = result.get("final_text", "").strip()
            if not chunk_text:
                continue
            
            # If there's overlap with previous chunk, remove it
            if prev_text_end and chunk_text.startswith(prev_text_end):
                # Remove the overlapping part
                chunk_text = chunk_text[len(prev_text_end):].strip()
            
            if merged_final_text:
                merged_final_text += " " + chunk_text
            else:
                merged_final_text = chunk_text
            
            # Keep last part of text for overlap detection (last 50 chars)
            prev_text_end = chunk_text[-50:] if len(chunk_text) > 50 else chunk_text
        
        # Merge evidence_map
        merged_evidence_map = []
        current_offset = 0
        for result in chunk_results:
            evidence_map = result.get("evidence_map", [])
            for entry in evidence_map:
                # Adjust offset_start/offset_end in qa_tags will be handled separately
                merged_evidence_map.append(entry)
            # Update offset for next chunk
            if evidence_map:
                last_span = evidence_map[-1].get("span", "")
                current_offset += len(last_span) + 1  # +1 for space
        
        # Merge qa_tags with offset adjustment
        merged_qa_tags = []
        current_offset = 0
        for result in chunk_results:
            qa_tags = result.get("qa_tags", [])
            final_text = result.get("final_text", "")
            for tag in qa_tags:
                # Convert None to 'null' string for schema compliance
                tag_value = tag.get("tag")
                if tag_value is None:
                    tag_value = "null"
                adjusted_tag = {
                    "offset_start": tag["offset_start"] + current_offset,
                    "offset_end": tag["offset_end"] + current_offset,
                    "tag": tag_value
                }
                merged_qa_tags.append(adjusted_tag)
            current_offset += len(final_text) + 1  # +1 for space
        
        # Merge uncertain_spans
        merged_uncertain_spans = []
        for result in chunk_results:
            merged_uncertain_spans.extend(result.get("uncertain_spans", []))
        
        # Merge changelog
        merged_changelog = []
        for result in chunk_results:
            merged_changelog.extend(result.get("changelog", []))
        
        # Determine if changed
        merged_changed = any(r.get("changed", False) for r in chunk_results)
        
        return {
            "final_text": merged_final_text,
            "changed": merged_changed,
            "evidence_map": merged_evidence_map,
            "uncertain_spans": merged_uncertain_spans,
            "changelog": merged_changelog,
            "qa_tags": merged_qa_tags
        }
