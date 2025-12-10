"""
ASR Client for SiliconFlow API
Handles transcription with word-level timestamps
"""

import requests
import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class ASRClient:
    """Client for SiliconFlow ASR API"""
    
    def __init__(self, api_key: str):
        """
        Initialize ASR client
        
        Args:
            api_key: SiliconFlow API key
        """
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.cn/v1/audio/transcriptions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def transcribe(
        self, 
        audio_path: str, 
        model_name: str = "FunAudioLLM/SenseVoiceSmall"
    ) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            model_name: ASR model name (FunAudioLLM/SenseVoiceSmall or TeleAI/TeleSpeechASR)
        
        Returns:
            Dictionary with:
                - text: Full transcript text
                - tokens: List of tokens with word-level timestamps
                - confidence: List of confidence scores (if available)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with open(audio_path, 'rb') as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            payload = {"model": model_name}
            
            try:
                response = requests.post(
                    self.base_url,
                    data=payload,
                    files=files,
                    headers=self.headers,
                    timeout=300  # 5 minutes timeout
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"ASR API request failed: {e}")
        
        # Parse response and extract tokens
        parsed = self._parse_response(result, audio_path)
        return parsed
    
    def _parse_response(self, response: Dict[str, Any], audio_path: Path) -> Dict[str, Any]:
        """
        Parse ASR API response and extract word-level timestamps
        
        Args:
            response: Raw API response
            audio_path: Path to audio file (for reference)
        
        Returns:
            Parsed result with text, tokens, confidence
        """
        # Initialize result structure
        result = {
            "text": "",
            "tokens": [],
            "confidence": []
        }
        
        # Handle different response formats
        if isinstance(response, str):
            # Simple text response
            result["text"] = response
            # Infer word-level timestamps (evenly distributed)
            words = response.split()
            duration = self._estimate_duration(audio_path)
            if words:
                time_per_word = duration / len(words)
                for i, word in enumerate(words):
                    result["tokens"].append({
                        "word": word,
                        "start": i * time_per_word,
                        "end": (i + 1) * time_per_word,
                        "confidence": 0.9  # Default confidence
                    })
        elif isinstance(response, dict):
            # Structured response
            result["text"] = response.get("text", "")
            
            # Check for word-level timestamps
            if "words" in response:
                # Direct word-level timestamps
                for word_info in response["words"]:
                    result["tokens"].append({
                        "word": word_info.get("word", ""),
                        "start": word_info.get("start", 0.0),
                        "end": word_info.get("end", 0.0),
                        "confidence": word_info.get("confidence", 0.9)
                    })
            elif "segments" in response:
                # Segment-level timestamps, need to break down to words
                words_all = []
                for seg in response["segments"]:
                    seg_text = seg.get("text", "")
                    seg_start = seg.get("start", 0.0)
                    seg_end = seg.get("end", 0.0)
                    seg_words = seg_text.split()
                    if seg_words:
                        time_per_word = (seg_end - seg_start) / len(seg_words)
                        for i, word in enumerate(seg_words):
                            words_all.append({
                                "word": word,
                                "start": seg_start + i * time_per_word,
                                "end": seg_start + (i + 1) * time_per_word,
                                "confidence": seg.get("confidence", 0.9)
                            })
                result["tokens"] = words_all
            else:
                # Only text, infer timestamps
                text = result["text"]
                words = text.split()
                duration = self._estimate_duration(audio_path)
                if words:
                    time_per_word = duration / len(words)
                    for i, word in enumerate(words):
                        result["tokens"].append({
                            "word": word,
                            "start": i * time_per_word,
                            "end": (i + 1) * time_per_word,
                            "confidence": 0.9
                        })
            
            # Extract confidence scores
            if "confidence" in response:
                if isinstance(response["confidence"], list):
                    result["confidence"] = response["confidence"]
                else:
                    result["confidence"] = [response["confidence"]]
        else:
            raise ValueError(f"Unexpected response format: {type(response)}")
        
        return result
    
    def _estimate_duration(self, audio_path: Path) -> float:
        """
        Estimate audio duration (fallback method)
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Estimated duration in seconds
        """
        try:
            import librosa
            y, sr = librosa.load(str(audio_path), sr=None)
            return len(y) / sr
        except ImportError:
            # Fallback: use pydub if available
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(str(audio_path))
                return len(audio) / 1000.0  # Convert ms to seconds
            except ImportError:
                # Final fallback: estimate based on file size
                # Rough estimate: 1 MB ≈ 6 seconds for WAV
                size_mb = audio_path.stat().st_size / (1024 * 1024)
                return max(10.0, size_mb * 6.0)



