import re
from typing import List, Dict, Any


def score_chunk_relevance(query: str, chunk: Dict[str, Any]) -> float:
    """
    Computes a composite cross-relevance score (0.0 to 1.0+) based on:
    1. Base semantic similarity (from vector/RRF distance)
    2. Query term coverage ratio
    3. Section title affinity
    4. Exact phrase presence
    """
    text = chunk.get("text", "").lower()
    section_title = chunk.get("section_title", "").lower()
    dist = chunk.get("distance", 0.5)

    # 1. Base semantic score (0.0 to 1.0)
    base_sem = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

    # 2. Query token coverage
    query_tokens = [t.lower() for t in re.findall(r'\w+', query) if len(t) > 2]
    if query_tokens:
        matches = sum(1 for t in query_tokens if t in text)
        coverage_ratio = matches / len(query_tokens)
    else:
        coverage_ratio = 0.5

    # 3. Section title affinity boost
    title_boost = 0.0
    if section_title and any(t in section_title for t in query_tokens):
        title_boost = 0.15

    # 4. Phrase match boost
    phrase_boost = 0.0
    clean_q = query.strip().lower().rstrip("?.")
    if len(clean_q) > 6 and clean_q in text:
        phrase_boost = 0.10

    composite_score = (0.45 * base_sem) + (0.40 * coverage_ratio) + title_boost + phrase_boost
    return round(composite_score, 4)


def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Re-scores and re-ranks retrieved chunks, placing the highest fidelity context first.
    """
    if not chunks:
        return []

    scored_chunks = []
    for c in chunks:
        c_copy = dict(c)
        score = score_chunk_relevance(query, c_copy)
        c_copy["rerank_score"] = score
        # Calibrate confidence score to percentage (50% - 99%)
        c_copy["confidence"] = max(50, min(99, int(score * 100)))
        scored_chunks.append(c_copy)

    # Sort descending by composite score
    sorted_chunks = sorted(scored_chunks, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_chunks[:top_k]
