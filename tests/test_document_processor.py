import os
import tempfile
from app.pdf_processor import extract_text_from_file, extract_text_from_pdf, clean_text

def test_clean_text():
    raw = "Hello\r\nWorld!\x00\n"
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "Hello\nWorld!" in cleaned
    print("[PASS] clean_text helper verified")

def test_txt_and_md_extraction():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write("# Operating Systems Notes\n\nProcess management handles CPU scheduling.\nMemory management handles RAM allocation.")
        tmp_md = f.name

    try:
        pages = extract_text_from_file(tmp_md)
        assert len(pages) > 0, "Failed to extract from Markdown file"
        assert "Process management" in pages[0]["text"]
        assert pages[0]["page"] == 1
        print("[PASS] Markdown extraction verified")
    finally:
        if os.path.exists(tmp_md):
            os.remove(tmp_md)

def test_pdf_extraction_fallback():
    pdf_path = "documents/test.pdf"
    if os.path.exists(pdf_path):
        pages = extract_text_from_file(pdf_path)
        assert len(pages) > 0, "Failed to extract from PDF file"
        print(f"[PASS] PDF extraction verified ({len(pages)} pages extracted)")

def main():
    print("=== Running Document Processor Unit Tests ===")
    test_clean_text()
    test_txt_and_md_extraction()
    test_pdf_extraction_fallback()
    print("=== All Commit 1 Tests Passed! ===")

if __name__ == "__main__":
    main()
