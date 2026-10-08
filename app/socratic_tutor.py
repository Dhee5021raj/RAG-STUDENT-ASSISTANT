import json
import re
from typing import List, Dict, Any, Optional


def generate_socratic_prompt(
    vector_store,
    rag_engine,
    topic: str = "Operating Systems",
    source_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Formulates an open-ended conceptual probe question designed to test deep understanding
    rather than rote memorization, grounded in the indexed materials.
    """
    chunks = vector_store.query(
        f"{topic} core mechanism principles why how compare",
        n_results=4,
        source_filter=source_filter,
        use_hybrid=True
    )

    if not chunks:
        return {
            "topic": topic,
            "probe_question": f"In your own words, explain the fundamental purpose and core mechanisms of {topic}.",
            "expected_concepts": [topic, "Core Functions"],
            "sources": []
        }

    sources_cited = list({f"{c['source']} (Page {c['page']})" for c in chunks})

    if rag_engine._anthropic_client:
        try:
            context_text = "\n\n".join([f"[{c['source']} - Page {c['page']}]: {c['text']}" for c in chunks])
            prompt = (
                f"You are a Socratic Academic Tutor. Based on the study material below about '{topic}', "
                "generate ONE thought-provoking, open-ended Socratic question that tests whether a student "
                "understands the 'why' and 'how' behind the concept (e.g. asking them to explain the mechanism, trade-offs, or cause-and-effect).\n\n"
                "Return ONLY a valid JSON object with:\n"
                "{\n"
                f'  "topic": "{topic}",\n'
                '  "probe_question": "...",\n'
                '  "expected_concepts": ["concept 1", "concept 2", "concept 3"]\n'
                "}\n\n"
                f"Context:\n{context_text}"
            )
            response = rag_engine._anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=600,
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
            print(f"Socratic prompt API warning: {e}. Falling back to heuristic question.")

    # Local Heuristic Fallback
    top_sec = chunks[0].get("section_title", topic)
    question = (
        f"Explain in your own words how '{top_sec}' operates in {topic}. "
        f"What problem does it solve, and what would happen if the system did not implement it?"
    )
    tokens = [t.capitalize() for t in re.findall(r'\b[A-Za-z]{4,}\b', chunks[0]["text"])[:4]]

    return {
        "topic": topic,
        "probe_question": question,
        "expected_concepts": list(dict.fromkeys([top_sec] + tokens)),
        "sources": sorted(sources_cited),
        "mode": "local_extractive"
    }


def evaluate_student_explanation(
    vector_store,
    rag_engine,
    topic: str,
    question: str,
    student_answer: str,
    source_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates a student's open-ended explanation against ground-truth textbook excerpts.
    Highlights accuracy, missing concepts, constructive advice, and poses a follow-up challenge.
    """
    if not student_answer or len(student_answer.strip()) < 10:
        return {
            "grade": "⚠️ Incomplete Submission",
            "score_pct": 20,
            "feedback": "Your explanation was too brief. Please provide a more detailed conceptual breakdown.",
            "missed_concepts": ["Detailed mechanism explanation"],
            "follow_up_question": "Can you expand on how this mechanism actually works step-by-step?",
            "sources": []
        }

    chunks = vector_store.query(
        f"{topic} {question}",
        n_results=4,
        source_filter=source_filter,
        use_hybrid=True
    )
    sources_cited = list({f"{c['source']} (Page {c['page']})" for c in chunks})

    if rag_engine._anthropic_client:
        try:
            context_text = "\n\n".join([f"[{c['source']} - Page {c['page']}]: {c['text']}" for c in chunks])
            prompt = (
                "You are an encouraging but rigorous Socratic AI Professor evaluating a student's understanding.\n"
                f"Topic: {topic}\n"
                f"Socratic Question: {question}\n"
                f"Student's Answer: {student_answer}\n\n"
                f"Ground Truth Reference Context:\n{context_text}\n\n"
                "Evaluate the student's answer against the context. Return ONLY a valid JSON object with:\n"
                "{\n"
                '  "grade": "Excellent Mastery" | "Solid Understanding" | "Partial Understanding" | "Needs Review",\n'
                '  "score_pct": 85,\n'
                '  "feedback": "2-3 sentences of constructive feedback noting what was well explained",\n'
                '  "missed_concepts": ["concept or detail they forgot to mention", ...],\n'
                '  "follow_up_question": "a follow-up question to push their understanding deeper"\n'
                "}"
            )
            response = rag_engine._anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=800,
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
            print(f"Socratic evaluation API warning: {e}. Falling back to heuristic evaluation.")

    # Local Heuristic Evaluation
    ans_clean = student_answer.lower()
    matched_chunks = [c for c in chunks if any(word in ans_clean for word in c["text"].lower().split()[:10])]
    word_count = len(ans_clean.split())

    if word_count > 30 and len(matched_chunks) >= 2:
        score = 85
        grade = "🌟 Solid Understanding"
    elif word_count > 15:
        score = 65
        grade = "📘 Partial Understanding"
    else:
        score = 45
        grade = "⚠️ Needs Review"

    missed = []
    for c in chunks:
        sec = c.get("section_title")
        if sec and sec.lower() not in ans_clean and len(missed) < 3:
            missed.append(sec)

    return {
        "grade": grade,
        "score_pct": score,
        "feedback": f"Good effort! You articulated key principles well, demonstrating {score}% conceptual coverage based on textbook materials.",
        "missed_concepts": missed if missed else ["Nuances of system-level edge cases"],
        "follow_up_question": f"How does {topic} balance efficiency and overhead when resources are constrained?",
        "sources": sorted(sources_cited),
        "mode": "local_extractive"
    }
