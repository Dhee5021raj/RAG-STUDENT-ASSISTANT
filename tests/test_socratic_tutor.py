from app.socratic_tutor import generate_socratic_prompt, evaluate_student_explanation
from app.rag_engine import RAGEngine

class DummyVectorStore:
    def query(self, query_text, n_results=4, **kwargs):
        return [
            {
                "source": "os.pdf",
                "page": 10,
                "text": "A Process is a program in execution with an address space, PCB, and registers.",
                "section_title": "Process Abstraction"
            },
            {
                "source": "os.pdf",
                "page": 15,
                "text": "Context switching saves CPU state of current process and restores another.",
                "section_title": "Context Switching"
            }
        ]

def test_generate_socratic_prompt():
    store = DummyVectorStore()
    rag = RAGEngine(vector_store=store)
    prompt_res = generate_socratic_prompt(store, rag, topic="Process Management")

    assert "probe_question" in prompt_res
    assert len(prompt_res["probe_question"]) > 10
    assert "expected_concepts" in prompt_res
    assert len(prompt_res["expected_concepts"]) > 0
    print(f"[PASS] Socratic probe question generated: {prompt_res['probe_question']}")

def test_evaluate_student_explanation():
    store = DummyVectorStore()
    rag = RAGEngine(vector_store=store)
    eval_res = evaluate_student_explanation(
        vector_store=store,
        rag_engine=rag,
        topic="Process Management",
        question="How does context switching work?",
        student_answer="When the CPU switches tasks, the operating system saves the PCB registers of the running process and loads the state of the new process into hardware."
    )

    assert "grade" in eval_res
    assert "score_pct" in eval_res
    assert eval_res["score_pct"] >= 50
    assert "feedback" in eval_res
    assert "follow_up_question" in eval_res
    print(f"[PASS] Student explanation evaluated: score={eval_res['score_pct']}%")

def main():
    print("=== Running Socratic Tutor Unit Tests ===")
    test_generate_socratic_prompt()
    test_evaluate_student_explanation()
    print("=== All Commit 23 Tests Passed! ===")

if __name__ == "__main__":
    main()
