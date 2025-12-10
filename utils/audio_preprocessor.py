import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    audio_path: str
    original_duration: float
    processed_duration: float
    energy_threshold: float
    removed_intervals: List[Tuple[float, float]]
    kept_intervals: List[Tuple[float, float]]
    temp_file: bool = False
    gain_applied: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        def _float(v):
            try:
                return float(v)
            except Exception:
                return v
        def _intervals(arr):
            return [(_float(s), _float(e)) for s, e in arr]
        return {
            "audio_path": self.audio_path,
            "original_duration": _float(self.original_duration),
            "processed_duration": _float(self.processed_duration),
            "energy_threshold": _float(self.energy_threshold),
            "removed_intervals": _intervals(self.removed_intervals),
            "kept_intervals": _intervals(self.kept_intervals),
            "temp_file": bool(self.temp_file),
            "gain_applied": _float(self.gain_applied),
        }


class AudioPreprocessor:
    """Annotate low-energy intervals and apply light enhancement for ASR (no persistent output)."""

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        reference_percentile: float = 92.0,
        energy_ratio: float = 0.25,
        min_silence_sec: float = 0.5,  # Increased from 0.35 to avoid fragmentation
        max_silence_sec: float = 4.0,
        tail_preserve_sec: float = 0.7,
        min_keep_sec: float = 0.5,  # Increased from 0.25 to avoid tiny fragments
        merge_gap_sec: float = 0.3,  # New: merge intervals closer than this
        target_sr: int = 16000,
        target_rms_db: float = -20.0,
        preemphasis_coef: float = 0.97,
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.reference_percentile = reference_percentile
        self.energy_ratio = energy_ratio
        self.min_silence_sec = min_silence_sec
        self.max_silence_sec = max_silence_sec
        self.tail_preserve_sec = tail_preserve_sec
        self.min_keep_sec = min_keep_sec
        self.merge_gap_sec = merge_gap_sec
        self.target_sr = target_sr
        self.target_rms_db = target_rms_db
        self.preemphasis_coef = preemphasis_coef

    def preprocess(self, audio_path: str) -> PreprocessResult:
        """Detect low-energy regions, apply light enhancement, return temp file path for ASR."""
        logger.info("AudioPreprocessor (enhance+annotate): loading %s", audio_path)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        original_duration = len(y) / sr if len(y) else 0.0

        if not len(y):
            logger.warning("AudioPreprocessor: empty audio detected, annotate-only skip.")
            return PreprocessResult(
                audio_path=audio_path,
                original_duration=0.0,
                processed_duration=0.0,
                energy_threshold=0.0,
                removed_intervals=[],
                kept_intervals=[],
                temp_file=False,
                gain_applied=1.0,
            )

        # Low-energy detection for metadata (on original sample rate)
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=self.hop_length, n_fft=self.frame_length
        )

        ref_value = np.percentile(rms, self.reference_percentile)
        if ref_value <= 1e-8:
            ref_value = np.max(rms)
        if ref_value <= 1e-8:
            logger.warning("AudioPreprocessor: RMS reference extremely low, skip annotations.")
            removed_intervals: List[Tuple[float, float]] = []
            kept_intervals: List[Tuple[float, float]] = [(0.0, original_duration)]
            threshold = 0.0
        else:
            threshold = float(ref_value * self.energy_ratio)
            low_segments: List[Tuple[float, float]] = []
            in_low = False
            start_time = 0.0
            for time_point, energy in zip(times, rms):
                if energy < threshold:
                    if not in_low:
                        start_time = time_point
                        in_low = True
                else:
                    if in_low:
                        low_segments.append((start_time, time_point))
                        in_low = False
            if in_low:
                low_segments.append((start_time, times[-1] if len(times) else original_duration))

            removal_segments: List[Tuple[float, float]] = []
            for seg_start, seg_end in low_segments:
                seg_end = min(seg_end, original_duration)
                duration = max(0.0, seg_end - seg_start)
                if duration < self.min_silence_sec:
                    continue
                if duration > self.max_silence_sec and duration > self.tail_preserve_sec:
                    adjusted_end = seg_end - self.tail_preserve_sec
                else:
                    adjusted_end = seg_end
                if adjusted_end - seg_start > 1e-3:
                    removal_segments.append((seg_start, adjusted_end))

            keep_segments: List[Tuple[float, float]] = []
            cursor = 0.0
            for seg_start, seg_end in removal_segments:
                seg_start = max(seg_start, 0.0)
                seg_end = min(seg_end, original_duration)
                if seg_start - cursor > self.min_keep_sec:
                    keep_segments.append((cursor, seg_start))
                cursor = max(cursor, seg_end)
            if original_duration - cursor > self.min_keep_sec:
                keep_segments.append((cursor, original_duration))

            kept_intervals = [
                (max(0.0, start), min(original_duration, end))
                for start, end in keep_segments
                if end - start > self.min_keep_sec
            ]
            # Merge intervals that are close together to avoid fragmentation
            kept_intervals = self._merge_close_intervals(kept_intervals, self.merge_gap_sec)
            if not kept_intervals:
                kept_intervals = [(0.0, original_duration)]
                removal_segments = []
            removed_intervals = [(float(s), float(e)) for s, e in removal_segments]

        # Enhancement pipeline (mono) -> resample -> loudness normalize -> preemphasis
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        rms_current = np.sqrt(np.mean(y ** 2)) if len(y) else 0.0
        target_rms = 10 ** (self.target_rms_db / 20.0)
        gain = target_rms / (rms_current + 1e-8) if rms_current > 0 else 1.0
        gain = min(gain, 20.0)  # prevent extreme boost
        y = y * gain
        # clip to [-1,1]
        y = np.clip(y, -1.0, 1.0)
        # pre-emphasis to sharpen
        y = librosa.effects.preemphasis(y, coef=self.preemphasis_coef)

        # write to temp file for ASR, not persisted in outputs
        fd, tmp_path = tempfile.mkstemp(suffix="_enhanced.wav", prefix="sd_pre_")
        Path(tmp_path).unlink(missing_ok=True)  # ensure clean write via soundfile
        sf.write(tmp_path, y, sr)

        processed_duration = len(y) / sr if len(y) else 0.0
        logger.info(
            "AudioPreprocessor: enhanced audio (gain %.2fx, sr %s) written to temp for ASR", gain, sr
        )

        return PreprocessResult(
            audio_path=str(tmp_path),
            original_duration=original_duration,
            processed_duration=processed_duration,
            energy_threshold=threshold,
            removed_intervals=removed_intervals if 'removed_intervals' in locals() else [],
            kept_intervals=kept_intervals if 'kept_intervals' in locals() else [(0.0, original_duration)],
            temp_file=True,
            gain_applied=gain,
        )

    @staticmethod
    def _merge_close_intervals(
        intervals: List[Tuple[float, float]],
        gap_threshold: float
    ) -> List[Tuple[float, float]]:
        """Merge intervals that are closer than gap_threshold to reduce fragmentation."""
        if not intervals:
            return intervals
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        for start, end in sorted_intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= gap_threshold:
                # Merge with previous interval
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged


__all__ = ["AudioPreprocessor", "PreprocessResult"]
