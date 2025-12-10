"""
ASR Runner: Execute multiple ASR models and select best results.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

from ScientificData.utils.asr_client import ASRClient

logger = logging.getLogger(__name__)


class ASRRunner:
    """Run multiple ASR configurations and select top results."""

    def __init__(self, asr_client: ASRClient, asr_runs: List[Dict[str, Any]]):
        self.asr_client = asr_client
        self.asr_runs = asr_runs

    def run_all(self, audio_path: str, max_workers: int = 3) -> List[Dict[str, Any]]:
        """Run all configured ASR models on audio and return scored results."""
        outputs: List[Dict[str, Any]] = []
        seen_texts = set()

        def run_once(alias: str, model: str, idx: int):
            result = self.asr_client.transcribe(audio_path, model)
            text_key = (result.get("text") or "").strip()
            return alias, model, idx, result, text_key

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            tasks = []
            for run in self.asr_runs:
                model = run.get("model")
                repeat = max(1, int(run.get("repeat", 1)))
                alias = run.get("alias") or model
                for idx in range(repeat):
                    tasks.append(ex.submit(run_once, alias, model, idx))
            for fut in tasks:
                alias, model, idx, result, text_key = fut.result()
                duplicate = text_key in seen_texts
                if text_key:
                    seen_texts.add(text_key)
                outputs.append({
                    "id": f"{alias}{idx + 1}",
                    "model": model,
                    "result": result,
                    "duplicate": duplicate,
                    "score": self.score_asr_output(result),
                })
        return outputs

    @staticmethod
    def score_asr_output(asr_result: Dict[str, Any]) -> float:
        """Score ASR output by confidence, length, and text completeness."""
        tokens = asr_result.get("tokens", [])
        confs = [float(t.get("confidence", 0.0)) for t in tokens if t is not None]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        length_bonus = len(tokens) * 0.1
        text_len = len(asr_result.get("text", ""))
        return mean_conf * 10 + length_bonus + text_len * 0.001

    @staticmethod
    def select_top(asr_outputs: List[Dict[str, Any]], k: int = 2) -> List[Dict[str, Any]]:
        """Select top k ASR outputs by score, preferring non-duplicates."""
        if not asr_outputs:
            return []
        sorted_outputs = sorted(
            asr_outputs,
            key=lambda item: (item.get("duplicate", False), -item.get("score", 0.0)),
        )
        return sorted_outputs[:k]


__all__ = ["ASRRunner"]

