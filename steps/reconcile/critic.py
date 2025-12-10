"""
M1: Critic Agent
Checks final.json for defects and violations
"""

import json
from typing import Dict, Any, Optional
from ScientificData.schemas import CriticOutput, Defect, FinalJSON
from ScientificData.utils.llm_client import LLMClient


class CriticAgent:
    """Agent for checking transcript quality and defects"""
    
    def __init__(self, llm_client: LLMClient, prompt_limits: Optional[Dict[str, Any]] = None):
        """
        Initialize Critic agent
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        self.prompt_limits = prompt_limits or {}
    
    def check(
        self,
        final_json: FinalJSON,
        asr_a: Dict[str, Any],
        asr_b: Dict[str, Any]
    ) -> CriticOutput:
        """
        Check final.json for defects
        
        Args:
            final_json: FinalJSON to check
            asr_a: Original ASR A result
            asr_b: Original ASR B result
        
        Returns:
            CriticOutput with defects and status
        """
        # Prepare prompt
        system_prompt = self._get_system_prompt()
        user_message = self._build_user_message(final_json, asr_a, asr_b)
        
        # Get schema description
        schema_description = self._get_schema_description()
        
        # Batch evidence_map/qa_tags when they are long
        batch_size = self.prompt_limits.get("critic_evidence_sample", 30)
        total_batches = max(1, (len(final_json.evidence_map) + batch_size - 1) // batch_size)
        defects_all = []
        status = "pass"
        for batch_idx in range(total_batches):
            if batch_idx > 0 and batch_idx * batch_size >= len(final_json.evidence_map):
                break
            message_batch = self._build_user_message(
                final_json,
                asr_a,
                asr_b,
                evidence_offset=batch_idx * batch_size
            )
            result = self.llm_client.chat_completion_with_schema_description(
                system_prompt=system_prompt,
                user_message=message_batch,
                schema_description=schema_description,
                temperature=0.1,
                max_tokens=4000
            )
            try:
                critic_output = CriticOutput(**result)
            except Exception as e:
                raise ValueError(f"Failed to validate CriticOutput (batch {batch_idx+1}): {e}\nResult: {result}")
            defects_all.extend(critic_output.defects)
            if critic_output.status == "revise":
                status = "revise"
        return CriticOutput(defects=defects_all, status=status)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Critic"""
        return """You are a Transcript QA Critic. Check reconciled transcript against original ASRs.

CHECK:
1. Unsupported content (not in A or B)
2. Entity/number mismatches
3. QA tag consistency (Q=question/interviewer, A=answer/interviewee)
4. Red-line: suicide/self-harm polarity errors (high severity)

OUTPUT: {"defects": [{type, severity(low|medium|high), description, location?, suggested_fix?}], "status": "pass|revise"}
Return "revise" if any high-severity defect."""
    
    def _build_user_message(
        self,
        final_json: FinalJSON,
        asr_a: Dict[str, Any],
        asr_b: Dict[str, Any],
        evidence_offset: int = 0
    ) -> str:
        """Build user message for LLM - concise format"""
        evidence_sample = self.prompt_limits.get("critic_evidence_sample", 20)
        qa_sample = self.prompt_limits.get("critic_qa_sample", 30)
        max_chars = self.prompt_limits.get("max_asr_text_chars", 8000)

        # Compact evidence format: span|support|A:s-e|B:s-e
        evidence_slice = final_json.evidence_map[evidence_offset:evidence_offset + evidence_sample]
        ev_lines = []
        for e in evidence_slice:
            a_loc = e.src_locs.get("A")
            b_loc = e.src_locs.get("B")
            a_str = f"{a_loc.start_tok}-{a_loc.end_tok}" if a_loc and a_loc.start_tok is not None else "-"
            b_str = f"{b_loc.start_tok}-{b_loc.end_tok}" if b_loc and b_loc.start_tok is not None else "-"
            span_preview = e.span[:60] + "..." if len(e.span) > 60 else e.span
            ev_lines.append(f"{span_preview}|{e.support}|A:{a_str}|B:{b_str}")

        # Compact QA tags: offset_start-end:tag
        qa_slice = final_json.qa_tags[:qa_sample]
        qa_str = " ".join(f"{t.offset_start}-{t.offset_end}:{t.tag}" for t in qa_slice)

        batch_info = ""
        if len(final_json.evidence_map) > evidence_sample:
            total_batches = (len(final_json.evidence_map) + evidence_sample - 1) // evidence_sample
            batch_info = f" [batch {evidence_offset // evidence_sample + 1}/{total_batches}]"

        return f"""ASR_A: {asr_a['text'][:max_chars]}
ASR_B: {asr_b['text'][:max_chars]}
FINAL: {final_json.final_text[:max_chars]}
EVIDENCE{batch_info}:
{chr(10).join(ev_lines)}
QA_TAGS: {qa_str}
Check for defects and return status."""

    def _get_schema_description(self) -> str:
        """Get JSON schema description for CriticOutput"""
        return """{"defects": [{"type": "str", "severity": "low|medium|high", "description": "str"}], "status": "pass|revise"}"""
