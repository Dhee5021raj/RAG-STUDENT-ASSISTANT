import sys
from app.pdf_processor import extract_text_from_pdf


pdf_path = "documents/test.pdf"

pages = extract_text_from_pdf(pdf_path)

print(f"Total pages: {len(pages)}")

for page in pages:
    print(f"\n--- Page {page['page']} ---")
    safe_text = page["text"][:500]
    sys.stdout.buffer.write((safe_text + "\n").encode("utf-8", errors="replace"))