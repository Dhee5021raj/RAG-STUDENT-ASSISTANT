import os
import sys
from app.pdf_processor import extract_text_from_pdf
from app.text_chunker import chunk_pages

def test_synthetic_boundary_chunking():
    synthetic_page = [{
        "page": 1,
        "text": "Artificial Intelligence is transforming modern education. " * 30
    }]
    chunks = chunk_pages(synthetic_page, source_file="ai_test.txt", chunk_size=200, overlap=50)
    assert len(chunks) > 0, "No chunks produced"
    for chunk in chunks:
        # Every chunk should start and end on complete words
        words = chunk["text"].split()
        assert words[0] in ["Artificial", "Intelligence", "is", "transforming", "modern", "education."], f"Broken start word: {words[0]}"
        assert words[-1] in ["Artificial", "Intelligence", "is", "transforming", "modern", "education."], f"Broken end word: {words[-1]}"

def main():
    print("=== Running Chunker Test ===")
    test_synthetic_boundary_chunking()
    print("[PASS] Synthetic word boundary test passed!")
    
    pdf_path = "documents/test.pdf"
    if not os.path.exists(pdf_path):
        print(f"Test PDF not found at {pdf_path}")
        return

    pages = extract_text_from_pdf(pdf_path)
    print(f"Total pages extracted: {len(pages)}")
    
    chunks = chunk_pages(pages, source_file="test.pdf", chunk_size=500, overlap=100)
    print(f"Total chunks generated: {len(chunks)}")
    
    if chunks:
        print("\nSample Chunk (Index 0):")
        print(f"  Source: {chunks[0]['source_file']}")
        print(f"  Page: {chunks[0]['page']}")
        print(f"  Length: {len(chunks[0]['text'])} characters")
        print("  Snippet:")
        snippet = chunks[0]["text"][:200]
        sys.stdout.buffer.write((snippet + "\n").encode("utf-8", errors="replace"))
    
    print("\n=== Chunker Test Passed Successfully! ===")

if __name__ == "__main__":
    main()
