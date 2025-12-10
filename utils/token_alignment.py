"""
Token Alignment: Align final text spans to ASR token indices using DP.
"""

import re
from typing import Dict, Any, List, Optional, Set, Tuple

from ScientificData.schemas import FinalJSON


def tokenize(text: str) -> List[str]:
    """Extract lowercase words from text."""
    return [w for w in re.findall(r"\w+", text.lower()) if w]


def dp_align(final_words: List[str], token_words: List[str]) -> List[Optional[int]]:
    """
    DP alignment: find best token index for each final word, handling duplicates.
    Returns a list where mapping[i] = token index for final_words[i], or None if no match.
    """
    if not final_words or not token_words:
        return [None] * len(final_words)
    n, m = len(final_words), len(token_words)
    INF = float('-inf')
    dp = [[(INF, -1) for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = (0, -1)
    for j in range(m + 1):
        dp[0][j] = (0, -1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Option 1: skip token j
            if dp[i][j - 1][0] > dp[i][j][0]:
                dp[i][j] = dp[i][j - 1]
            # Option 2: match final_word[i-1] to token[j-1]
            if final_words[i - 1] == token_words[j - 1]:
                prev_score = dp[i - 1][j - 1][0]
                if prev_score + 1 > dp[i][j][0]:
                    dp[i][j] = (prev_score + 1, j - 1)
            # Option 3: skip final word i
            if dp[i - 1][j][0] > dp[i][j][0]:
                dp[i][j] = (dp[i - 1][j][0], dp[i - 1][j][1])
    # Backtrack
    mapping = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if dp[i][j][1] == j - 1 and final_words[i - 1] == token_words[j - 1]:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i][j - 1][0] >= dp[i][j][0]:
            j -= 1
        else:
            i -= 1
    return mapping


def find_all_span_positions(span_text: str, words: List[str]) -> List[Tuple[int, int]]:
    """Find all positions where span_text appears in words list."""
    span_words = tokenize(span_text)
    if not span_words:
        return []
    positions = []
    for i in range(len(words) - len(span_words) + 1):
        if words[i:i + len(span_words)] == span_words:
            positions.append((i, i + len(span_words) - 1))
    return positions


def align_evidence_map(
    final_json: FinalJSON,
    tokens_a: List[Dict[str, Any]],
    tokens_b: List[Dict[str, Any]]
) -> FinalJSON:
    """Align evidence_map src_locs with ASR tokens using DP-based alignment."""
    if not final_json.evidence_map:
        return final_json

    final_words = tokenize(final_json.final_text)
    token_words_a = [re.sub(r"\W+", "", str(t.get("word", ""))).lower() for t in tokens_a]
    token_words_b = [re.sub(r"\W+", "", str(t.get("word", ""))).lower() for t in tokens_b]

    map_a = dp_align(final_words, token_words_a) if token_words_a else []
    map_b = dp_align(final_words, token_words_b) if token_words_b else []

    used_positions: Set[Tuple[int, int]] = set()
    for ev in final_json.evidence_map:
        positions = find_all_span_positions(ev.span, final_words)
        chosen = None
        for pos in positions:
            if pos not in used_positions:
                chosen = pos
                used_positions.add(pos)
                break
        if not chosen and positions:
            chosen = positions[0]
        if not chosen:
            continue
        start_w, end_w = chosen
        if map_a and start_w < len(map_a) and map_a[start_w] is not None and end_w < len(map_a) and map_a[end_w] is not None:
            ev.src_locs["A"].start_tok = map_a[start_w]
            ev.src_locs["A"].end_tok = map_a[end_w] + 1
        if map_b and start_w < len(map_b) and map_b[start_w] is not None and end_w < len(map_b) and map_b[end_w] is not None:
            ev.src_locs["B"].start_tok = map_b[start_w]
            ev.src_locs["B"].end_tok = map_b[end_w] + 1
    return final_json


def filter_tokens_by_intervals(
    tokens: List[Dict[str, Any]],
    kept_intervals: Optional[List[Tuple[float, float]]]
) -> List[Dict[str, Any]]:
    """Filter tokens to only include those within kept_intervals."""
    if not kept_intervals:
        return tokens
    if not tokens:
        return tokens

    def in_kept(mid: float) -> bool:
        return any(s <= mid <= e for s, e in kept_intervals)

    return [
        tok for tok in tokens
        if in_kept((float(tok.get("start", 0.0)) + float(tok.get("end", tok.get("start", 0.0)))) / 2.0)
    ]


__all__ = ["align_evidence_map", "filter_tokens_by_intervals", "dp_align", "tokenize"]

