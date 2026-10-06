from app.roadmap_generator import generate_study_roadmap
from app.rag_engine import RAGEngine

class DummyVectorStore:
    def query(self, query_text, n_results=4, **kwargs):
        return [
            {"source": "os.pdf", "page": 1, "text": "Hardware abstraction and system calls.", "section_title": "Foundations"},
            {"source": "os.pdf", "page": 20, "text": "Process scheduling algorithms and queues.", "section_title": "Process Scheduling"},
            {"source": "os.pdf", "page": 50, "text": "Virtual memory pagination and page replacement.", "section_title": "Memory Management"}
        ]

def test_generate_study_roadmap():
    store = DummyVectorStore()
    rag = RAGEngine(vector_store=store)
    result = generate_study_roadmap(store, rag, topic="Operating Systems")

    assert "stages" in result
    assert len(result["stages"]) == 3
    assert "mermaid_graph" in result
    assert "graph TD" in result["mermaid_graph"]
    assert "sources" in result
    print(f"[PASS] Study Roadmap generated with {len(result['stages'])} stages")
    print(f"[PASS] Mermaid graph generated:\n{result['mermaid_graph']}")

def main():
    print("=== Running Roadmap Generator Unit Tests ===")
    test_generate_study_roadmap()
    print("=== All Commit 18 Tests Passed! ===")

if __name__ == "__main__":
    main()
