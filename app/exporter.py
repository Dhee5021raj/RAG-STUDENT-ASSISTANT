import datetime
from typing import List, Dict, Any


def export_chat_history(messages: List[Dict[str, Any]]) -> str:
    """Exports chat message history into a clean Markdown formatted document."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 📚 Study Session Chat Export",
        f"**Generated On:** {timestamp}",
        "---",
        ""
    ]

    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"### 👤 {role}" if role == "User" else f"### 🤖 {role}")
        lines.append(content)
        lines.append("")

        sources = msg.get("sources", [])
        if sources:
            lines.append("**Cited Sources:**")
            for s in sources:
                lines.append(f"- **{s.get('source', 'unknown')}** (Page {s.get('page', 1)}): *\"{s.get('text', '')[:150]}...\"*")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def export_quiz(quiz_data: List[Dict[str, Any]], topic: str = "General Concepts") -> str:
    """Exports practice quiz questions, options, and answer key into Markdown format."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# 🎯 Practice Quiz — {topic}",
        f"**Generated On:** {timestamp}",
        "---",
        ""
    ]

    for idx, q in enumerate(quiz_data, 1):
        lines.append(f"### Q{idx}: {q.get('question', '')}")
        lines.append("")
        for opt in q.get("options", []):
            lines.append(f"- [ ] {opt}")
        lines.append("")

    lines.append("## 🔑 Answer Key & Explanations")
    lines.append("---")
    lines.append("")

    for idx, q in enumerate(quiz_data, 1):
        lines.append(f"**Q{idx} Answer:** {q.get('answer', '')}")
        lines.append(f"💡 *Explanation:* {q.get('explanation', '')}")
        lines.append("")

    return "\n".join(lines)


def export_flashcards(flashcards: List[Dict[str, str]], topic: str = "General Concepts") -> str:
    """Exports interactive flashcards into a printable Markdown flashcard deck format."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# 🎴 Flashcard Deck — {topic}",
        f"**Generated On:** {timestamp}",
        "---",
        ""
    ]

    for idx, card in enumerate(flashcards, 1):
        lines.append(f"## 🎴 Card {idx}")
        lines.append(f"**Front (Concept):** {card.get('front', '')}")
        lines.append("")
        lines.append(f"**Back (Answer):** {card.get('back', '')}")
        lines.append(f"*Source:* {card.get('source', '')}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
