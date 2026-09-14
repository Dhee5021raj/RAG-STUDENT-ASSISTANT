import re
from typing import List, Dict, Any

def _find_split_point(text: str, target_end: int, min_end: int) -> int:
    """
    Finds the best split point near target_end (paragraphs > sentences > words)
    without going below min_end.
    """
    if target_end >= len(text):
        return len(text)

    window = text[min_end:target_end]
    
    # Priority 1: Double newline (paragraph break)
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
    Splits extracted PDF pages into overlapping text chunks with source metadata,
    respecting sentence and word boundaries so words are not sliced in half.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        page_number = page.get("page", 1)
        text = page.get("text", "").strip()

        # Skip blank or near-empty pages
        if len(text) < 30:
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
                })
                chunk_index += 1

            if end >= text_len:
                break

            # Calculate next start position with overlap
            next_start = max(start + 1, end - overlap)
            # If next_start falls in the middle of a word, advance to the start of the next word
            if next_start < end and next_start > 0 and not text[next_start - 1].isspace() and not text[next_start].isspace():
                while next_start < end and not text[next_start].isspace():
                    next_start += 1

            # Skip any whitespace
            while next_start < end and text[next_start].isspace():
                next_start += 1

            if next_start <= start or next_start >= end:
                next_start = end

            start = next_start

    return chunks
