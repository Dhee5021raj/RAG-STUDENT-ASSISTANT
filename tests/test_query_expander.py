from app.query_expander import decompose_query_heuristically, expand_query
from app.rag_engine import RAGEngine

class DummyStore:
    def get_stats(self):
        return {"total_chunks": 5}

    def query(self, query_text, n_results=4, **kwargs):
        return [
            {"source": "os.pdf", "page": 1, "chunk_index": 0, "text": f"Context regarding {query_text}", "distance": 0.3}
        ]

def test_heuristic_query_decomposition():
    compound_query = "Compare process scheduling and memory management"
    decomposed = decompose_query_heuristically(compound_query)
    assert len(decomposed) >= 2
    assert any("process scheduling" in q.lower() for q in decomposed)
    assert any("memory management" in q.lower() for q in decomposed)
    print(f"[PASS] Heuristic decomposition: {decomposed}")

def test_rag_engine_multi_query_flag():
    engine = RAGEngine(vector_store=DummyStore())
    result = engine.answer_question(
        "Compare paging vs segmentation in operating systems",
        use_multi_query=True
    )
    assert "expanded_queries" in result
    assert len(result["expanded_queries"]) >= 2
    assert "sources" in result
    print(f"[PASS] RAG Engine multi-query integration: {result['expanded_queries']}")

def main():
    print("=== Running Query Expander Unit Tests ===")
    test_heuristic_query_decomposition()
    test_rag_engine_multi_query_flag()
    print("=== All Commit 13 Tests Passed! ===")

if __name__ == "__main__":
    main()
