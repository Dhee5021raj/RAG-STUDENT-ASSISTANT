import os
import shutil
from app.text_chunker import chunk_pages, estimate_tokens
from app.vector_store import VectorStore

def test_chunker_token_estimation():
    pages = [{"page": 1, "text": "Operating systems manage hardware resources such as CPU and memory."}]
    chunks = chunk_pages(pages, source_file="os_intro.txt", chunk_size=200, overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["token_count"] > 0
    assert chunks[0]["source_file"] == "os_intro.txt"
    print("[PASS] Chunker token estimation verified")

def test_hybrid_vector_and_bm25_search():
    test_dir = "chroma_db_hybrid_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)

    try:
        store = VectorStore(persist_directory=test_dir, collection_name="hybrid_test")
        sample_pages_1 = [{"page": 1, "text": "Virtual memory allows execution of processes that are not completely in physical memory."}]
        sample_pages_2 = [{"page": 1, "text": "Deadlock prevention algorithms ensure that at least one of the four necessary conditions cannot hold."}]

        chunks_1 = chunk_pages(sample_pages_1, source_file="vm.txt")
        chunks_2 = chunk_pages(sample_pages_2, source_file="deadlock.txt")

        store.add_documents(chunks_1 + chunks_2)

        stats = store.get_stats()
        assert stats["total_chunks"] == 2
        assert len(stats["sources"]) == 2

        # Query with keyword match for Deadlock
        results = store.query("deadlock prevention conditions", n_results=2, use_hybrid=True)
        assert len(results) > 0
        assert "deadlock.txt" in results[0]["source"]
        print("[PASS] Hybrid BM25+Vector search verified")

        # Query with source filter
        filtered_results = store.query("memory", n_results=2, source_filter="vm.txt")
        assert len(filtered_results) == 1
        assert filtered_results[0]["source"] == "vm.txt"
        print("[PASS] Source filtering verified")
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)

def main():
    print("=== Running Hybrid Retrieval Unit Tests ===")
    test_chunker_token_estimation()
    test_hybrid_vector_and_bm25_search()
    print("=== All Commit 2 Tests Passed! ===")

if __name__ == "__main__":
    main()
