import os
import sys
from app.pdf_processor import extract_text_from_pdf
from app.text_chunker import chunk_pages
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine

def main():
    print("=== Testing Full RAG Engine Pipeline ===")
    
    # 1. Setup Vector Store
    store = VectorStore(persist_directory="chroma_db", collection_name="study_materials")
    
    # Index test PDF if not already populated
    if store.get_stats()["total_chunks"] == 0:
        pdf_path = "documents/test.pdf"
        if os.path.exists(pdf_path):
            print(f"Indexing {pdf_path} into vector database...")
            pages = extract_text_from_pdf(pdf_path)
            chunks = chunk_pages(pages, source_file="test.pdf")
            store.add_documents(chunks)
            print(f"Indexed {len(chunks)} chunks.")
    else:
        print(f"Using existing database ({store.get_stats()['total_chunks']} chunks).")

    # 2. Setup RAG Engine
    rag = RAGEngine(vector_store=store)
    
    # 3. Test Question
    question = "What are the design issues or abstractions in operating systems?"
    print(f"\nStudent Question: '{question}'")
    
    result = rag.answer_question(question, n_results=3)
    
    print(f"\n--- Answer Mode: {result['mode']} ---")
    sys.stdout.buffer.write((result["answer"] + "\n\n").encode("utf-8", errors="replace"))
    
    print(f"--- Sources Cited ({len(result['sources'])} chunks) ---")
    for s in result["sources"]:
        print(f"  • {s['source']} (Page {s['page']})")
        
    print("\n=== RAG Engine Test Completed Successfully! ===")

if __name__ == "__main__":
    main()
