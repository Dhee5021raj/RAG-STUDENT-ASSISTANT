import json
from typing import List, Dict, Any, Optional


def compare_documents_on_topic(
    vector_store,
    rag_engine,
    topic: str,
    doc1: str,
    doc2: str
) -> Dict[str, Any]:
    """
    Performs cross-document comparative analysis on a shared topic or concept
    between two uploaded study documents (e.g. Textbook vs Lecture Notes).
    Identifies shared agreements, unique perspectives, and comparative depth.
    """
    chunks_doc1 = vector_store.query(topic, n_results=3, source_filter=doc1, use_hybrid=True)
    chunks_doc2 = vector_store.query(topic, n_results=3, source_filter=doc2, use_hybrid=True)

    if not chunks_doc1 and not chunks_doc2:
        return {
            "topic": topic,
            "doc1": doc1,
            "doc2": doc2,
            "comparison": f"Neither document has indexed content regarding '{topic}'.",
            "shared_points": [],
            "doc1_unique": [],
            "doc2_unique": []
        }

    sources_cited = []
    for c in chunks_doc1:
        sources_cited.append(f"{c['source']} (Page {c['page']})")
    for c in chunks_doc2:
        sources_cited.append(f"{c['source']} (Page {c['page']})")

    if rag_engine._anthropic_client:
        try:
            c1_text = "\n".join([f"[{c['source']} Page {c['page']}]: {c['text']}" for c in chunks_doc1])
            c2_text = "\n".join([f"[{c['source']} Page {c['page']}]: {c['text']}" for c in chunks_doc2])

            prompt = (
                f"You are an academic researcher. Compare and contrast how these two study documents treat '{topic}':\n\n"
                f"Document 1 ({doc1}):\n{c1_text or 'No excerpts found.'}\n\n"
                f"Document 2 ({doc2}):\n{c2_text or 'No excerpts found.'}\n\n"
                "Return ONLY a valid JSON object with:\n"
                "{\n"
                f'  "topic": "{topic}",\n'
                f'  "doc1": "{doc1}",\n'
                f'  "doc2": "{doc2}",\n'
                '  "shared_points": ["point 1 shared by both", ...],\n'
                '  "doc1_unique": ["point or depth unique to doc1", ...],\n'
                '  "doc2_unique": ["point or depth unique to doc2", ...],\n'
                '  "comparison_summary": "2-3 sentences synthesizing the comparative takeaway"\n'
                "}"
            )
            response = rag_engine._anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=900,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                data["sources"] = sorted(list(set(sources_cited)))
                data["mode"] = "claude_generative"
                return data
        except Exception as e:
            print(f"Comparison API warning: {e}. Falling back to local comparator.")

    # Local Extractive Heuristic Comparison
    t1 = chunks_doc1[0]["text"][:140] if chunks_doc1 else "No coverage found."
    t2 = chunks_doc2[0]["text"][:140] if chunks_doc2 else "No coverage found."

    return {
        "topic": topic,
        "doc1": doc1,
        "doc2": doc2,
        "shared_points": [f"Both references explore concepts related to '{topic}'."],
        "doc1_unique": [f"Coverage in {doc1}: {t1}..."],
        "doc2_unique": [f"Coverage in {doc2}: {t2}..."],
        "comparison_summary": f"Comparative evaluation between {doc1} and {doc2} highlights complementary study context on {topic}.",
        "sources": sorted(list(set(sources_cited))),
        "mode": "local_extractive"
    }
