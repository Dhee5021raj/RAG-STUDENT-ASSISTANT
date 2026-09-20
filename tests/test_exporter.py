from app.exporter import export_chat_history, export_quiz, export_flashcards

def test_export_chat_history():
    messages = [
        {"role": "user", "content": "What is process control block?"},
        {
            "role": "assistant",
            "content": "Process Control Block (PCB) stores state information.",
            "sources": [{"source": "os.pdf", "page": 12, "text": "PCB contains PID and registers."}]
        }
    ]
    md_output = export_chat_history(messages)
    assert "# 📚 Study Session Chat Export" in md_output
    assert "Process Control Block" in md_output
    assert "os.pdf" in md_output
    print("[PASS] Chat history export verified")

def test_export_quiz():
    quiz_data = [
        {
            "question": "What is CPU scheduling?",
            "options": ["A. Allocating CPU time", "B. Memory pagination", "C. Disk formatting"],
            "answer": "A. Allocating CPU time",
            "explanation": "CPU scheduler chooses from ready queue processes."
        }
    ]
    md_quiz = export_quiz(quiz_data, topic="OS Scheduling")
    assert "Practice Quiz — OS Scheduling" in md_quiz
    assert "Q1: What is CPU scheduling?" in md_quiz
    assert "Answer Key & Explanations" in md_quiz
    print("[PASS] Quiz export verified")

def test_export_flashcards():
    cards = [
        {"front": "Virtual Memory", "back": "Abstraction of main memory.", "source": "os.pdf (Page 55)"}
    ]
    md_cards = export_flashcards(cards, topic="OS Memory")
    assert "Flashcard Deck — OS Memory" in md_cards
    assert "Virtual Memory" in md_cards
    print("[PASS] Flashcard export verified")

def main():
    print("=== Running Exporter Unit Tests ===")
    test_export_chat_history()
    test_export_quiz()
    test_export_flashcards()
    print("=== All Commit 5 & 11 Exporter Tests Passed! ===")

if __name__ == "__main__":
    main()
