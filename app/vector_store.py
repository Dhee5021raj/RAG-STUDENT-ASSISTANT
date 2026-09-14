import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    """
    VectorStore manages text embeddings and similarity retrieval using ChromaDB.
    Uses local on-device embeddings by default (no external API key required).
    """
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "study_materials"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Built-in lightweight sentence embedding function (runs locally)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Study materials vector index"}
        )

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embeds and stores document chunks in the vector database.
        """
        if not chunks:
            return 0

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            chunk_id = f"{chunk['source_file']}_p{chunk['page']}_c{chunk['chunk_index']}"
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk.get("source_file", "unknown"),
                "page": int(chunk.get("page", 1)),
                "chunk_index": int(chunk.get("chunk_index", 0))
            })
            ids.append(chunk_id)

        # Batch upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results: int = 4,
        distance_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search against the stored chunks.
        Optionally filters results by distance_threshold.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(n_results, count)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=actual_k
        )

        retrieved = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metas, distances):
                if distance_threshold is not None and dist > distance_threshold:
                    continue
                retrieved.append({
                    "text": doc,
                    "page": meta.get("page", 1),
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist
                })

        return retrieved

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns current database statistics.
        """
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }

    def clear(self):
        """
        Clears all indexed documents from the collection.
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
