import os
import pymupdf


def clean_text(text: str) -> str:
    """Cleans up raw extracted text by removing null bytes and excessive inline white space."""
    if not text:
        return ""
    # Normalize null bytes and line carriage returns
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def extract_text_from_pdf(pdf_path: str):
    """Extracts text page-by-page from a PDF document using PyMuPDF."""
    document = pymupdf.open(pdf_path)
    pages = []

    for page_number, page in enumerate(document):
        raw_text = page.get_text()
        cleaned = clean_text(raw_text)
        if cleaned:
            pages.append({
                "page": page_number + 1,
                "text": cleaned
            })

    document.close()
    return pages


def extract_text_from_txt_or_md(file_path: str):
    """Extracts text from plain text (.txt) or Markdown (.md) files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    cleaned = clean_text(content)
    if not cleaned:
        return []

    # Break large text files into logical pseudo-pages (every ~3000 chars or 50 lines)
    lines = cleaned.split("\n")
    pages = []
    current_page_lines = []
    current_char_count = 0
    page_num = 1

    for line in lines:
        current_page_lines.append(line)
        current_char_count += len(line) + 1

        if current_char_count >= 3000 or len(current_page_lines) >= 60:
            pages.append({
                "page": page_num,
                "text": "\n".join(current_page_lines).strip()
            })
            page_num += 1
            current_page_lines = []
            current_char_count = 0

    if current_page_lines:
        pages.append({
            "page": page_num,
            "text": "\n".join(current_page_lines).strip()
        })

    return pages


def extract_text_from_file(file_path: str):
    """
    Main entrypoint for document extraction.
    Auto-detects file extension and extracts structured page/section content.
    Supports: .pdf, .txt, .md
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".txt", ".md"]:
        return extract_text_from_txt_or_md(file_path)
    else:
        # Fallback to text reading
        try:
            return extract_text_from_txt_or_md(file_path)
        except Exception:
            raise ValueError(f"Unsupported file format: {ext}")