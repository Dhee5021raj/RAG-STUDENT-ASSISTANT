import json
import re
from typing import List, Dict, Any, Optional


def generate_executive_summary(vector_store, rag_engine, source_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Synthesizes an executive study summary of the indexed materials or a specific source document.
    Outlines Core Themes, Key Topics, and Study Highlights with page references.
    """
    # Sample chunks from the knowledge base
    chunks = vector_store.query("overview introduction core concepts summary", n_results=6, source_filter=source_filter, use_hybrid=True)
    if not chunks:
        return {
            "summary": "No materials indexed yet to generate an executive summary.",
            "topics": [],
            "sources": []
        }

    sources_cited = list({f"{c['source']} (Page {c['page']})" for c in chunks})

    if rag_engine._anthropic_client:
        try:
            context_text = "\n\n".join([f"[{c['source']} - Page {c['page']}]: {c['text']}" for c in chunks])
            prompt = (
                "You are an academic study synthesizer. Based on the study material excerpts below, write an Executive Study Summary.\n"
                "Structure your response with:\n"
                "1. Overview & Core Objective (2-3 sentences)\n"
                "2. Major Topics Covered (bullet points with citations)\n"
                "3. Key Takeaways & Exam Tips\n\n"
                f"Excerpts:\n{context_text}"
            )
            response = rag_engine._anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                "summary": response.content[0].text.strip(),
                "sources": sorted(sources_cited),
                "mode": "claude_generative"
            }
        except Exception as e:
            print(f"Synthesis API warning: {e}. Falling back to local extractive summary.")

    # Local Extractive Mode Fallback
    summary_lines = [
        "### 📑 Executive Study Summary (Extracted)",
        "Below is an automated synthesis of the core topics found in your study material:\n",
        "**Core Excerpts & Study Themes:**"
    ]
    for idx, c in enumerate(chunks[:4], 1):
        summary_lines.append(f"- **Topic {idx} [{c['source']} | Page {c['page']}]:** {c['text'][:180]}...")

    summary_lines.append("\n💡 *Tip: Connect an Anthropic API key to generate comprehensive AI executive summaries.*")

    return {
        "summary": "\n".join(summary_lines),
        "sources": sorted(sources_cited),
        "mode": "local_extractive"
    }


def extract_concept_glossary(vector_store, rag_engine, source_filter: Optional[str] = None, top_n: int = 6) -> List[Dict[str, str]]:
    """
    Extracts key terminology and domain concepts into a structured glossary table
    with terms, definitions, and page citations.
    """
    chunks = vector_store.query("definition terminology concepts is defined as", n_results=top_n * 2, source_filter=source_filter, use_hybrid=True)
    if not chunks:
        return []

    if rag_engine._anthropic_client:
        try:
            context_text = "\n\n".join([f"[{c['source']} - Page {c['page']}]: {c['text']}" for c in chunks[:6]])
            prompt = (
                f"Extract the top {top_n} most important technical concepts/terms and their definitions from these study excerpts.\n"
                "Return ONLY a valid JSON array of objects with keys: 'term', 'definition', 'source'. Example:\n"
                '[{"term": "Process", "definition": "A program in execution.", "source": "OS.pdf Page 5"}]\n\n'
                f"Excerpts:\n{context_text}"
            )
            response = rag_engine._anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            print(f"Glossary extraction API warning: {e}. Falling back to heuristic extractor.")

    # Local Heuristic Extractor Fallback
    glossary = []
    for c in chunks:
        text = c["text"]
        # Look for patterns like "X is defined as", "X is a", or section title
        section = c.get("section_title", "").strip()
        term = section if section and section != "General Section" else text.split()[0:3]
        if isinstance(term, list):
            term = " ".join(term)

        definition = text[:150].strip().replace("\n", " ") + "..."
        glossary.append({
            "term": str(term),
            "definition": definition,
            "source": f"{c['source']} (Page {c['page']})"
        })
        if len(glossary) >= top_n:
            break

    return glossary
