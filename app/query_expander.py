import re
from typing import List, Optional


def decompose_query_heuristically(query: str) -> List[str]:
    """
    Rule-based local query decomposition fallback.
    Breaks down compound questions with conjunctions (and, vs, versus, compared to, difference between)
    into independent sub-queries.
    """
    clean_q = query.strip()
    sub_queries = [clean_q]

    # Pattern for "compare X and/vs Y" or "difference between X and Y"
    compare_match = re.search(r'(?:difference between|compare|relationship between)\s+(.+?)\s+(?:and|versus|vs\.?)\s+(.+)', clean_q, re.IGNORECASE)
    if compare_match:
        part1 = compare_match.group(1).strip()
        part2 = compare_match.group(2).strip().rstrip("?.")
        sub_queries.append(f"What is {part1}?")
        sub_queries.append(f"What is {part2}?")
        return list(dict.fromkeys(sub_queries))

    # Pattern for compound questions with "and" / "as well as"
    conjunction_split = re.split(r'\s+(?:and also|as well as|along with)\s+', clean_q, flags=re.IGNORECASE)
    if len(conjunction_split) > 1:
        for part in conjunction_split:
            part_clean = part.strip().rstrip("?.")
            if len(part_clean) > 8:
                sub_queries.append(part_clean)

    # General perspective query
    sub_queries.append(f"Key concepts in {clean_q.rstrip('?.')}")

    # Remove duplicates preserving order
    return list(dict.fromkeys(sub_queries))[:3]


def expand_query(query: str, anthropic_client: Optional[object] = None, model: str = "claude-3-5-haiku-20241022") -> List[str]:
    """
    Generates 2 to 3 alternative query formulations and sub-queries to capture
    different semantic aspects of the student's question.
    Uses Anthropic Claude if available, otherwise falls back to heuristic decomposition.
    """
    if not query or len(query.strip()) < 5:
        return [query]

    if anthropic_client:
        try:
            prompt = (
                f"You are an AI research assistant. Given the student question below, generate 2 alternative, "
                f"closely related search queries or decomposed sub-queries to retrieve all necessary context from textbooks.\n"
                f"Student Question: \"{query}\"\n"
                f"Output exactly 2 queries, each on a new line. Do not number them or add markdown formatting."
            )
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text.strip()
            lines = [line.strip().lstrip("123456789.-) ") for line in raw_text.split("\n") if line.strip()]
            queries = [query] + [l for l in lines if len(l) > 5]
            return list(dict.fromkeys(queries))[:3]
        except Exception as e:
            print(f"Multi-query expansion API warning: {e}. Falling back to heuristic expansion.")

    return decompose_query_heuristically(query)
