import json
from typing import List, Dict, Any, Optional


def generate_study_roadmap(
    vector_store,
    rag_engine,
    topic: str = "Operating Systems",
    source_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a structured, pedagogical prerequisite study roadmap and
    Mermaid.js concept dependency graph based on indexed study materials.
    """
    # Query for foundational, intermediate, and advanced sections
    chunks = vector_store.query(
        f"{topic} fundamentals principles design mechanisms architecture",
        n_results=6,
        source_filter=source_filter,
        use_hybrid=True
    )

    if not chunks:
        return {
            "topic": topic,
            "stages": [],
            "mermaid_graph": "graph TD\n    A[Upload Study Materials] --> B[Generate Roadmap]",
            "mode": "no_context"
        }

    sources_cited = list({f"{c['source']} (Page {c['page']})" for c in chunks})

    if rag_engine._anthropic_client:
        try:
            context_text = "\n\n".join([f"[{c['source']} - Page {c['page']}]: {c['text']}" for c in chunks])
            prompt = (
                f"You are an expert curriculum designer. Based on these study materials about '{topic}', "
                f"create a 3-stage prerequisite study roadmap and a valid Mermaid.js flowchart (graph TD).\n\n"
                "Return ONLY a valid JSON object with the following structure:\n"
                "{\n"
                '  "topic": "' + topic + '",\n'
                '  "stages": [\n'
                '    {"stage_num": 1, "title": "Foundational Concepts", "estimated_hours": "2-3 hrs", "concepts": ["Concept 1", "Concept 2"], "summary": "..."},\n'
                '    {"stage_num": 2, "title": "Core Mechanisms", "estimated_hours": "3-4 hrs", "concepts": ["Concept 3", "Concept 4"], "summary": "..."},\n'
                '    {"stage_num": 3, "title": "Advanced Applications", "estimated_hours": "2-3 hrs", "concepts": ["Concept 5"], "summary": "..."}\n'
                '  ],\n'
                '  "mermaid_graph": "graph TD\\n    S1[Stage 1: Foundations] --> S2[Stage 2: Core]\\n    S2 --> S3[Stage 3: Advanced]"\n'
                "}\n\n"
                f"Materials Context:\n{context_text}"
            )
            response = rag_engine._anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                data["sources"] = sorted(sources_cited)
                data["mode"] = "claude_generative"
                return data
        except Exception as e:
            print(f"Roadmap generation API warning: {e}. Falling back to local heuristic roadmap.")

    # Local Heuristic Fallback
    c1 = chunks[0].get("section_title", "Foundations") if len(chunks) > 0 else "Foundations"
    c2 = chunks[1].get("section_title", "Core Mechanisms") if len(chunks) > 1 else "Core Mechanisms"
    c3 = chunks[2].get("section_title", "Advanced Abstractions") if len(chunks) > 2 else "Advanced Abstractions"

    mermaid_diagram = (
        "graph TD\n"
        f"    A[\"Stage 1: {c1}\"] --> B[\"Stage 2: {c2}\"]\n"
        f"    B --> C[\"Stage 3: {c3}\"]"
    )

    stages = [
        {
            "stage_num": 1,
            "title": f"Stage 1: {c1}",
            "estimated_hours": "2-3 hrs",
            "concepts": [c1, "Basic Definitions", "Introductory Models"],
            "summary": chunks[0]["text"][:160] + "..." if chunks else "Initial conceptual grounding."
        },
        {
            "stage_num": 2,
            "title": f"Stage 2: {c2}",
            "estimated_hours": "3-4 hrs",
            "concepts": [c2, "Core Operations", "Policy vs Mechanism"],
            "summary": chunks[1]["text"][:160] + "..." if len(chunks) > 1 else "Core structural workflows."
        },
        {
            "stage_num": 3,
            "title": f"Stage 3: {c3}",
            "estimated_hours": "2-3 hrs",
            "concepts": [c3, "System Integration", "Case Studies"],
            "summary": chunks[2]["text"][:160] + "..." if len(chunks) > 2 else "Advanced synthesis and mastery."
        }
    ]

    return {
        "topic": topic,
        "stages": stages,
        "mermaid_graph": mermaid_diagram,
        "sources": sorted(sources_cited),
        "mode": "local_extractive"
    }
