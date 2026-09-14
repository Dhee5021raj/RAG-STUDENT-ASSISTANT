import os
import sys
from app.pdf_processor import extract_text_from_pdf
from app.text_chunker import chunk_pages
from app.vector_store import VectorStore

def main():
    print("=== Testing Vector Store (ChromaDB) ===")
    
    # Initialize vector store
    store = VectorStore(persist_directory="chroma_db_test", collection_name="test_collection")
    store.clear()
    
    # Extract & chunk test PDF
    pdf_path = "documents/test.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return
        
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, source_file="test.pdf")
    
    print(f"Adding {len(chunks)} chunks to vector database...")
    store.add_documents(chunks)
    
    stats = store.get_stats()
    print(f"Vector Store Stats: {stats}")
    
    # Test query
    query = "What is a process in operating systems?"
    print(f"\nQuerying: '{query}'")
    results = store.query(query, n_results=3)
    
    print(f"Retrieved {len(results)} results:")
    for i, res in enumerate(results, 1):
        print(f"\n--- Result {i} (Source: {res['source']}, Page: {res['page']}, Distance: {res['distance']:.4f}) ---")
        snippet = res['text'][:250]
        sys.stdout.buffer.write((snippet + "\n").encode("utf-8", errors="replace"))
        
    print("\n=== Vector Store Test Passed Successfully! ===")

if __name__ == "__main__":
    main()
