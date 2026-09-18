import os
import re
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 keyword matching."""
    return re.findall(r'\w+', text.lower())


class VectorStore:
    """
    VectorStore manages text embeddings and hybrid similarity retrieval using ChromaDB & BM25Okapi.
    Combines dense semantic vector retrieval with sparse keyword matching using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "study_materials"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Study materials vector index"}
        )

        self.bm25 = None
        self._bm25_docs = []
        self._bm25_metas = []
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self):
        """Rebuilds in-memory BM25 index from ChromaDB documents."""
        count = self.collection.count()
        if count == 0 or BM25Okapi is None:
            self.bm25 = None
            self._bm25_docs = []
            self._bm25_metas = []
            return

        all_records = self.collection.get()
        if all_records and all_records.get("documents"):
            self._bm25_docs = all_records["documents"]
            self._bm25_metas = all_records["metadatas"] or [{}] * len(self._bm25_docs)
            tokenized_corpus = [tokenize_text(doc) for doc in self._bm25_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """Embeds and stores document chunks in ChromaDB and refreshes BM25 index."""
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

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        self._rebuild_bm25_index()
        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results: int = 4,
        distance_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Performs similarity search. If use_hybrid=True, fuses Dense Vector results
        with BM25 Sparse Keyword results using Reciprocal Rank Fusion (RRF).
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(n_results * 3, count)
        where_clause = {"source": source_filter} if source_filter else None

        # 1. Vector Search
        vector_results = self.collection.query(
            query_texts=[query_text],
            n_results=actual_k,
            where=where_clause
        )

        dense_ranks: Dict[str, float] = {}
        doc_details: Dict[str, Dict[str, Any]] = {}

        if vector_results and vector_results.get("documents") and len(vector_results["documents"]) > 0:
            docs = vector_results["documents"][0]
            metas = vector_results["metadatas"][0] if vector_results.get("metadatas") else [{}] * len(docs)
            distances = vector_results["distances"][0] if vector_results.get("distances") else [0.0] * len(docs)

            for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
                if distance_threshold is not None and dist > distance_threshold:
                    continue
                key = f"{meta.get('source')}_p{meta.get('page')}_c{meta.get('chunk_index')}"
                dense_ranks[key] = rank
                doc_details[key] = {
                    "text": doc,
                    "page": meta.get("page", 1),
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist
                }

        # If not hybrid or BM25 unavailable, return vector results sorted by distance
        if not use_hybrid or not self.bm25 or not self._bm25_docs:
            sorted_vector = sorted(doc_details.values(), key=lambda x: x["distance"])
            return sorted_vector[:n_results]

        # 2. BM25 Search
        query_tokens = tokenize_text(query_text)
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:actual_k]

        sparse_ranks: Dict[str, float] = {}
        for rank, idx in enumerate(top_bm25_indices, 1):
            if bm25_scores[idx] <= 0:
                continue
            meta = self._bm25_metas[idx]
            if source_filter and meta.get("source") != source_filter:
                continue
            doc = self._bm25_docs[idx]
            key = f"{meta.get('source')}_p{meta.get('page')}_c{meta.get('chunk_index')}"
            sparse_ranks[key] = rank
            if key not in doc_details:
                doc_details[key] = {
                    "text": doc,
                    "page": meta.get("page", 1),
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": 0.5  # placeholder distance for BM25 matches
                }

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = 1 / (60 + dense_rank) + 1 / (60 + sparse_rank)
        rrf_scores: Dict[str, float] = {}
        all_keys = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))

        for key in all_keys:
            r_dense = dense_ranks.get(key, 999)
            r_sparse = sparse_ranks.get(key, 999)
            rrf_score = (1.0 / (60 + r_dense)) + (1.0 / (60 + r_sparse))
            rrf_scores[key] = rrf_score

        sorted_keys = sorted(all_keys, key=lambda k: rrf_scores[k], reverse=True)[:n_results]
        return [doc_details[k] for k in sorted_keys]

    def get_stats(self) -> Dict[str, Any]:
        """Returns database statistics including unique source file names."""
        count = self.collection.count()
        sources = set()
        if count > 0:
            records = self.collection.get()
            if records and records.get("metadatas"):
                for m in records["metadatas"]:
                    if m and "source" in m:
                        sources.add(m["source"])

        return {
            "total_chunks": count,
            "total_documents": len(sources),
            "sources": sorted(list(sources)),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }

    def clear(self):
        """Clears all indexed documents."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        self.bm25 = None
        self._bm25_docs = []
        self._bm25_metas = []

