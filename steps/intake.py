"""
M0: Intake Module
Collects and aligns ASR results with word-level timestamps and confidence
"""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path


class IntakeProcessor:
    """Process and normalize ASR inputs"""

    def __init__(self):
        """Initialize intake processor"""
        pass

    def process(
        self,
        audio_path: str,
        asr_a_result: Dict[str, Any],
        asr_b_result: Dict[str, Any],
        glossary: Optional[Dict[str, Any]] = None,
        scenario_hint: str = "interview"
    ) -> Dict[str, Any]:
        """
        Process and normalize ASR inputs

        Args:
            audio_path: Path to audio file
            asr_a_result: ASR A result (from ASRClient.transcribe)
            asr_b_result: ASR B result (from ASRClient.transcribe)
            glossary: Optional glossary dictionary
            scenario_hint: Scenario type ("interview" or "monologue")

        Returns:
            Standardized input bundle:
            {
                "audio_path": str,
                "asr_a": {
                    "text": str,
                    "tokens": List[Dict],
                    "confidence": List[float]
                },
                "asr_b": {
                    "text": str,
                    "tokens": List[Dict],
                    "confidence": List[float]
                },
                "glossary": Dict,
                "scenario_hint": str,
                "metadata": Dict
            }
        """
        # Validate inputs
        self._validate_inputs(asr_a_result, asr_b_result, scenario_hint)

        # Normalize ASR results
        asr_a_normalized = self._normalize_asr_result(asr_a_result, "A")
        asr_b_normalized = self._normalize_asr_result(asr_b_result, "B")

        # Prepare glossary
        if glossary is None:
            glossary = {"entities": [], "medical_terms": [], "context": scenario_hint}

        # Prepare metadata
        metadata = {
            "audio_path": str(audio_path),
            "asr_a_token_count": len(asr_a_normalized["tokens"]),
            "asr_b_token_count": len(asr_b_normalized["tokens"]),
            "scenario": scenario_hint
        }

        return {
            "audio_path": str(audio_path),
            "asr_a": asr_a_normalized,
            "asr_b": asr_b_normalized,
            "glossary": glossary,
            "scenario_hint": scenario_hint,
            "metadata": metadata
        }

    def _validate_inputs(
        self,
        asr_a_result: Dict[str, Any],
        asr_b_result: Dict[str, Any],
        scenario_hint: str
    ):
        """Validate input ASR results"""
        required_fields = ["text", "tokens"]

        for name, result in [("ASR_A", asr_a_result), ("ASR_B", asr_b_result)]:
            if not isinstance(result, dict):
                raise ValueError(f"{name} result must be a dictionary")

            for field in required_fields:
                if field not in result:
                    raise ValueError(f"{name} result missing required field: {field}")

            if not isinstance(result["text"], str):
                raise ValueError(f"{name} text must be a string")

            if not isinstance(result["tokens"], list):
                raise ValueError(f"{name} tokens must be a list")

            # Validate tokens structure
            for i, token in enumerate(result["tokens"]):
                if not isinstance(token, dict):
                    raise ValueError(f"{name} token {i} must be a dictionary")
                if "word" not in token:
                    raise ValueError(f"{name} token {i} missing 'word' field")
                if "start" not in token or "end" not in token:
                    raise ValueError(f"{name} token {i} missing timestamp fields")

        if scenario_hint not in ["interview", "monologue"]:
            raise ValueError(f"scenario_hint must be 'interview' or 'monologue', got: {scenario_hint}")

    def _normalize_asr_result(self, asr_result: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Normalize ASR result to standard format

        Args:
            asr_result: Raw ASR result
            source: Source identifier ("A" or "B")

        Returns:
            Normalized result with text, tokens, confidence
        """
        normalized = {
            "text": asr_result["text"],
            "tokens": [],
            "confidence": []
        }

        # Normalize tokens
        for token in asr_result["tokens"]:
            normalized_token = {
                "word": str(token.get("word", "")),
                "start": float(token.get("start", 0.0)),
                "end": float(token.get("end", 0.0)),
                "confidence": float(token.get("confidence", 0.9))
            }
            normalized["tokens"].append(normalized_token)
            normalized["confidence"].append(normalized_token["confidence"])

        # Ensure tokens are sorted by start time
        normalized["tokens"].sort(key=lambda x: x["start"])

        return normalized
