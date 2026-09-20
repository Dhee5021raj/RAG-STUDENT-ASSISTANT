from app.rag_engine import RAGEngine

class DummyStore:
    def get_stats(self):
        return {"total_chunks": 2}

def test_context_compression():
    engine = RAGEngine(vector_store=DummyStore())
    duplicate_chunks = [
        {"source": "doc1.txt", "page": 1, "text": "Operating systems manage memory. CPU scheduling handles process queues."},
        {"source": "doc2.txt", "page": 2, "text": "Operating systems manage memory. Device drivers control I/O hardware."}
    ]
    compressed = engine._compress_and_deduplicate_context(duplicate_chunks)
    assert len(compressed) == 2
    # The second chunk should have stripped the duplicate sentence "Operating systems manage memory."
    assert "Operating systems manage memory" in compressed[0]["text"]
    assert "Operating systems manage memory" not in compressed[1]["text"]
    assert "Device drivers control I/O hardware" in compressed[1]["text"]
    print("[PASS] Context compression and sentence deduplication verified")

def main():
    print("=== Running Context Compression Unit Tests ===")
    test_context_compression()
    print("=== All Commit 10 Tests Passed! ===")

if __name__ == "__main__":
    main()
