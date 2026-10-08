from app.comparator import compare_documents_on_topic
from app.rag_engine import RAGEngine

class DummyVectorStore:
    def query(self, query_text, source_filter=None, **kwargs):
        if source_filter == "textbook.pdf":
            return [{"source": "textbook.pdf", "page": 40, "text": "Detailed theoretical formal definition of CPU scheduling algorithms."}]
        elif source_filter == "lecture_notes.pdf":
            return [{"source": "lecture_notes.pdf", "page": 8, "text": "Practical real-world Linux CFS scheduling trade-offs and latency."}]
        return []

def test_compare_documents_on_topic():
    store = DummyVectorStore()
    rag = RAGEngine(vector_store=store)
    result = compare_documents_on_topic(
        vector_store=store,
        rag_engine=rag,
        topic="CPU Scheduling",
        doc1="textbook.pdf",
        doc2="lecture_notes.pdf"
    )

    assert "topic" in result
    assert result["doc1"] == "textbook.pdf"
    assert result["doc2"] == "lecture_notes.pdf"
    assert "comparison_summary" in result
    assert "shared_points" in result
    assert len(result["sources"]) == 2
    print("[PASS] compare_documents_on_topic verified across two distinct document sources")

def main():
    print("=== Running Document Comparator Unit Tests ===")
    test_compare_documents_on_topic()
    print("=== All Commit 2 Tests Passed! ===")

if __name__ == "__main__":
    main()
