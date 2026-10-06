from app.reranker import score_chunk_relevance, rerank_chunks

def test_score_chunk_relevance():
    query = "What is CPU scheduling?"
    high_match = {
        "text": "CPU scheduling is the process of allocating CPU time to ready processes.",
        "section_title": "CPU Scheduling Overview",
        "distance": 0.2
    }
    low_match = {
        "text": "Disk fragmentation occurs when file systems scatter files.",
        "section_title": "File Storage",
        "distance": 0.8
    }

    score_high = score_chunk_relevance(query, high_match)
    score_low = score_chunk_relevance(query, low_match)

    assert score_high > score_low, f"Expected {score_high} > {score_low}"
    print(f"[PASS] score_chunk_relevance: high={score_high}, low={score_low}")

def test_rerank_chunks():
    query = "virtual memory paging"
    chunks = [
        {"text": "Network protocols define communication standards.", "section_title": "Networking", "distance": 0.4},
        {"text": "Virtual memory paging transfers pages between RAM and swap.", "section_title": "Virtual Memory", "distance": 0.5}
    ]

    reranked = rerank_chunks(query, chunks, top_k=2)
    assert len(reranked) == 2
    # The virtual memory chunk should be boosted to first place due to term coverage and title boost
    assert "Virtual memory paging" in reranked[0]["text"]
    assert reranked[0]["confidence"] >= 50
    print(f"[PASS] rerank_chunks successfully boosted relevant chunk to top (score={reranked[0]['rerank_score']})")

def main():
    print("=== Running Reranker Unit Tests ===")
    test_score_chunk_relevance()
    test_rerank_chunks()
    print("=== All Commit 20 Tests Passed! ===")

if __name__ == "__main__":
    main()
