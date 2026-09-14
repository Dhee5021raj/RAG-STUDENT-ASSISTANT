import os
import sys
from app.pdf_processor import extract_text_from_pdf
from app.text_chunker import chunk_pages
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine

def main():
    print("==================================================")
    print("   AI-Powered RAG Study Assistant — Evaluation    ")
    print("==================================================")
    
    # 1. Setup store & engine
    store = VectorStore(persist_directory="chroma_db", collection_name="study_materials")
    
    # Ensure test document is indexed
    if store.get_stats()["total_chunks"] == 0:
        pdf_path = "documents/test.pdf"
        if os.path.exists(pdf_path):
            print(f"Indexing {pdf_path}...")
            pages = extract_text_from_pdf(pdf_path)
            chunks = chunk_pages(pages, source_file="test.pdf")
            store.add_documents(chunks)
            print(f"Indexed {len(chunks)} chunks.")
            
    rag = RAGEngine(vector_store=store)
    
    # Evaluation test questions
    eval_queries = [
        {"q": "What is a process in operating system?", "topic": "Process Definition"},
        {"q": "What are the course assessment details or marks distribution?", "topic": "Assessments & Marks"},
        {"q": "What are the text books or reference books for the course?", "topic": "Textbooks & Authors"},
        {"q": "What are the security design issues in OS?", "topic": "Security in OS"},
        {"q": "What is quantum computing?", "topic": "Out-of-domain / Unsupported Question"}
    ]
    
    print("\nRunning Evaluation Queries:\n")
    for i, item in enumerate(eval_queries, 1):
        query = item["q"]
        topic = item["topic"]
        print(f"[{i}/5] Testing Query: \"{query}\" (Topic: {topic})")
        
        result = rag.answer_question(query, n_results=3)
        sources = result.get("sources", [])
        
        print(f"   -> Result mode: {result['mode']}")
        print(f"   -> Chunks retrieved: {len(sources)}")
        if sources:
            pages = sorted(list(set(s['page'] for s in sources)))
            print(f"   -> Cited Pages: {pages}")
            print(f"   -> Best similarity distance: {sources[0]['distance']:.4f}")
        else:
            print("   -> No chunks retrieved.")
        print("-" * 50)
        
    print("\n[SUCCESS] All evaluation queries tested successfully!")

if __name__ == "__main__":
    main()
