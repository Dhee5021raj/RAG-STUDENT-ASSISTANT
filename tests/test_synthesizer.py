from app.synthesizer import generate_executive_summary, extract_concept_glossary
from app.rag_engine import RAGEngine

class DummyVectorStore:
    def query(self, query_text, n_results=4, **kwargs):
        return [
            {
                "source": "os_intro.pdf",
                "page": 2,
                "text": "Operating Systems manage CPU and memory resources.",
                "section_title": "Operating Systems Overview"
            },
            {
                "source": "os_intro.pdf",
                "page": 4,
                "text": "Virtual Memory provides an abstraction of physical RAM.",
                "section_title": "Virtual Memory"
            }
        ]

def test_generate_executive_summary():
    store = DummyVectorStore()
    rag = RAGEngine(vector_store=store)
    result = generate_executive_summary(store, rag)
    assert "summary" in result
    assert "sources" in result
    assert len(result["sources"]) > 0
    print("[PASS] Executive summary generator verified")

def test_extract_concept_glossary():
    store = DummyVectorStore()
    rag = RAGEngine(vector_store=store)
    glossary = extract_concept_glossary(store, rag, top_n=2)
    assert len(glossary) == 2
    assert "term" in glossary[0]
    assert "definition" in glossary[0]
    assert "source" in glossary[0]
    print(f"[PASS] Concept glossary extractor verified: {glossary[0]['term']}")

def main():
    print("=== Running Synthesizer Unit Tests ===")
    test_generate_executive_summary()
    test_extract_concept_glossary()
    print("=== All Commit 15 Tests Passed! ===")

if __name__ == "__main__":
    main()
