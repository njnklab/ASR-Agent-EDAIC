"""
Default settings for the ScientificData transcript pipeline.
This is a slimmed-down version that keeps ASR runs and prompt limits only.
"""

from pathlib import Path
from typing import Optional

ASR_RUNS = [
    {"alias": "A", "model": "FunAudioLLM/SenseVoiceSmall", "repeat": 2},
    {"alias": "B", "model": "TeleAI/TeleSpeechASR", "repeat": 2},
]

CHUNK_SETTINGS = {
    "chunk_size": 200,
    "overlap": 30,
    "input_token_soft_limit": 500,
    "estimated_output_soft_limit": 4000,
}

PROMPT_LIMITS = {
    "max_asr_text_chars": 12000,
    "critic_evidence_sample": 30,
    "critic_qa_sample": 50,
}


def load_api_keys(path: Optional[Path] = None) -> dict:
    """Load API keys from a JSON file (fallback to environment variables)."""
    import json
    import os

    cfg_path = path or (Path(__file__).parent / "api_keys.json")
    keys: dict = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                keys.update(json.load(f))
        except Exception:
            pass

    if "SILICONFLOW_API_KEY" not in keys:
        env_key = os.environ.get("SILICONFLOW_API_KEY")
        if env_key:
            keys["SILICONFLOW_API_KEY"] = env_key
    if "DEEPSEEK_API_KEY" not in keys:
        env_key = os.environ.get("DEEPSEEK_API_KEY")
        if env_key:
            keys["DEEPSEEK_API_KEY"] = env_key

    def normalize(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return []

    keys["SILICONFLOW_API_KEY"] = normalize(keys.get("SILICONFLOW_API_KEY"))
    keys["DEEPSEEK_API_KEY"] = normalize(keys.get("DEEPSEEK_API_KEY"))
    return keys
