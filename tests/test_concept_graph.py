from app.concept_graph import extract_concept_relationships
from app.rag_engine import RAGEngine

class DummyVectorStore:
    def query(self, query_text, n_results=4, **kwargs):
        return [
            {"source": "os.pdf", "page": 5, "text": "The Operating System manages hardware resources including CPU and RAM."},
            {"source": "os.pdf", "page": 12, "text": "The PCB stores register state and execution context for each process."}
        ]

def test_extract_concept_relationships():
    store = DummyVectorStore()
    rag = RAGEngine(vector_store=store)
    result = extract_concept_relationships(store, rag, max_relations=4)

    assert "relationships" in result
    assert len(result["relationships"]) > 0
    assert "mermaid_graph" in result
    assert "graph LR" in result["mermaid_graph"]
    assert "sources" in result
    rel = result["relationships"][0]
    assert "source_entity" in rel
    assert "relation" in rel
    assert "target_entity" in rel
    print(f"[PASS] Concept relationships extracted: {rel['source_entity']} -> {rel['relation']} -> {rel['target_entity']}")
    print(f"[PASS] Mermaid graph generated:\n{result['mermaid_graph']}")

def main():
    print("=== Running Concept Graph Unit Tests ===")
    test_extract_concept_relationships()
    print("=== All Commit 21 Tests Passed! ===")

if __name__ == "__main__":
    main()
