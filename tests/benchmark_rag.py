import os
import time
from app.pdf_processor import extract_text_from_file
from app.text_chunker import chunk_pages
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine


def run_benchmark():
    print("=========================================================")
    print("   AI-Powered RAG Study Assistant — Evaluation Suite    ")
    print("   Metrics: MRR (Mean Reciprocal Rank), Precision@K, Latency")
    print("=========================================================")

    test_db_dir = "chroma_db_benchmark"
    store = VectorStore(persist_directory=test_db_dir, collection_name="benchmark_coll")
    store.clear()

    # Index sample document
    pdf_path = "documents/test.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found for benchmarking")
        return

    pages = extract_text_from_file(pdf_path)
    chunks = chunk_pages(pages, source_file="test.pdf")
    store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks for benchmarking.\n")

    rag = RAGEngine(vector_store=store)

    # Ground truth test cases with target keywords expected in retrieved top-k chunks
    test_cases = [
        {
            "query": "What is a process in operating system?",
            "expected_keywords": ["process", "program in execution", "state"]
        },
        {
            "query": "What are operating system design issues?",
            "expected_keywords": ["design", "specification", "policy", "mechanism"]
        },
        {
            "query": "How is memory managed in operating systems?",
            "expected_keywords": ["memory", "allocation", "virtual"]
        },
        {
            "query": "What are CPU scheduling criteria?",
            "expected_keywords": ["cpu", "scheduling", "queue", "first-come"]
        }
    ]

    reciprocal_ranks = []
    precision_scores = []
    latencies = []
    top_k = 3

    print(f"{'Query':<45} | {'Recip Rank':<10} | {'Prec@3':<8} | {'Latency (ms)':<12}")
    print("-" * 85)

    for case in test_cases:
        query = case["query"]
        expected = case["expected_keywords"]

        t0 = time.perf_counter()
        results = store.query(query, n_results=top_k, use_hybrid=True)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        rank = 0
        hits = 0

        for idx, r in enumerate(results, 1):
            text_lower = r["text"].lower()
            if any(kw in text_lower for kw in expected):
                hits += 1
                if rank == 0:
                    rank = idx

        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        prec = hits / float(top_k)
        precision_scores.append(prec)

        print(f"{query[:44]:<45} | {rr:<10.2f} | {prec:<8.2f} | {latency_ms:<12.1f}")

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_latency = sum(latencies) / len(latencies)

    print("-" * 85)
    print("[RESULTS] Benchmark Summary Results:")
    print(f"  * Mean Reciprocal Rank (MRR) : {mrr:.3f}")
    print(f"  * Average Precision@{top_k}       : {avg_precision:.3f}")
    print(f"  * Average Retrieval Latency  : {avg_latency:.1f} ms")
    print("=========================================================\n")

    # Clean up test DB
    store.clear()


if __name__ == "__main__":
    run_benchmark()
