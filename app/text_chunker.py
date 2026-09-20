import re
from typing import List, Dict, Any


def estimate_tokens(text: str) -> int:
    """Estimates the number of tokens in a text snippet (rough heuristic ~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_section_title(text: str) -> str:
    """Extracts the first heading (# or CAPS heading) found in a text snippet."""
    match = re.search(r'^(?:#{1,6}\s+(.+)$|([A-Z0-9\s]{4,}:.+)$)', text, re.MULTILINE)
    if match:
        title = match.group(1) or match.group(2)
        return title.strip()
    return "General Section"


def _find_split_point(text: str, target_end: int, min_end: int) -> int:
    """
    Finds the best split point near target_end (paragraphs > sentences > words)
    without going below min_end and avoiding cutting Markdown tables in half.
    """
    if target_end >= len(text):
        return len(text)

    window = text[min_end:target_end]

    # Priority 0: Avoid splitting inside a Markdown table
    table_match = re.search(r'(\|[^\n]+\|\n)+', text[min_end - 50:target_end + 100])
    if table_match and table_match.start() < (target_end - min_end + 50) < table_match.end():
        # Advance past the end of the table
        new_end = min_end - 50 + table_match.end()
        if new_end <= len(text):
            return new_end

    # Priority 1: Section headers or double newline (paragraph break)
    header_pos = re.search(r'\n(?=#{1,6}\s|[A-Z0-9\s]{4,}:|\n)', window)
    if header_pos:
        return min_end + header_pos.start() + 1

    pos = window.rfind("\n\n")
    if pos != -1:
        return min_end + pos + 2

    # Priority 2: Single newline
    pos = window.rfind("\n")
    if pos != -1:
        return min_end + pos + 1

    # Priority 3: Sentence terminators (. ! ?) followed by whitespace
    matches = list(re.finditer(r'[.!?]\s+', window))
    if matches:
        return min_end + matches[-1].end()

    # Priority 4: Word boundary (space)
    pos = window.rfind(" ")
    if pos != -1:
        return min_end + pos + 1

    # Fallback to target_end
    return target_end


def chunk_pages(
    pages: List[Dict[str, Any]],
    source_file: str = "unknown",
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Splits extracted pages into overlapping text chunks with source metadata,
    respecting sentence, header, table, and word boundaries.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        page_number = page.get("page", 1)
        text = page.get("text", "").strip()

        if len(text) < 20:
            continue

        start = 0
        text_len = len(text)

        while start < text_len:
            target_end = min(start + chunk_size, text_len)
            min_end = start + max(20, chunk_size - overlap)

            if target_end < text_len:
                end = _find_split_point(text, target_end, min_end)
            else:
                end = text_len

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= 20:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number,
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "section_title": extract_section_title(chunk_text),
                    "token_count": estimate_tokens(chunk_text),
                    "char_count": len(chunk_text)
                })
                chunk_index += 1

            if end >= text_len:
                break

            next_start = max(start + 1, end - overlap)
            if next_start < end and next_start > 0 and not text[next_start - 1].isspace() and not text[next_start].isspace():
                while next_start < end and not text[next_start].isspace():
                    next_start += 1

            while next_start < end and text[next_start].isspace():
                next_start += 1

            if next_start <= start or next_start >= end:
                next_start = end

            start = next_start

    return chunks

