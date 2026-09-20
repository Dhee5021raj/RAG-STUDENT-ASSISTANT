from app.text_chunker import chunk_pages, extract_section_title

def test_extract_section_title():
    sample = "# Chapter 1: Introduction\nThis chapter discusses basic concepts."
    title = extract_section_title(sample)
    assert title == "Chapter 1: Introduction"
    print("[PASS] extract_section_title verified")

def test_markdown_table_chunking():
    table_text = """
# Data Table
| ID | Process Name | Arrival Time | Burst Time |
|----|--------------|--------------|------------|
| P1 | Process 1    | 0            | 5          |
| P2 | Process 2    | 1            | 3          |
| P3 | Process 3    | 2            | 8          |

The above table summarizes process parameters in OS CPU scheduling.
"""
    pages = [{"page": 1, "text": table_text}]
    chunks = chunk_pages(pages, source_file="table_test.md", chunk_size=250)
    assert len(chunks) > 0
    assert chunks[0]["section_title"] == "Data Table"
    assert "| P1 |" in chunks[0]["text"]
    print("[PASS] Markdown table preservation verified")

def main():
    print("=== Running Advanced Chunker Unit Tests ===")
    test_extract_section_title()
    test_markdown_table_chunking()
    print("=== All Commit 9 Tests Passed! ===")

if __name__ == "__main__":
    main()
