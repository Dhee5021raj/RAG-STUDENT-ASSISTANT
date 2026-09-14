import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    """
    RAG Engine connects Vector Retrieval with Generation.
    Supports:
      1. Anthropic Claude (when API key is provided)
      2. High-quality Local Extractive Mode (when no API key is provided)
    """
    def __init__(self, vector_store, api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._anthropic_client = None
        
        if self.api_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic client: {e}")
                self._anthropic_client = None

    def set_api_key(self, api_key: str):
        """Update API key at runtime (e.g. from UI)"""
        self.api_key = api_key.strip()
        if self.api_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                self._anthropic_client = None

    def answer_question(
        self,
        question: str,
        n_results: int = 4,
        model: str = "claude-3-5-haiku-20241022",
        distance_threshold: Optional[float] = 1.25
    ) -> Dict[str, Any]:
        """
        Retrieves relevant context with distance filtering and generates a grounded response with source citations.
        """
        # Check if knowledge base is empty
        stats = self.vector_store.get_stats()
        if stats.get("total_chunks", 0) == 0:
            return {
                "answer": "No study materials uploaded or indexed. Please upload your study documents first.",
                "sources": [],
                "mode": "no_context"
            }

        # Step 1: Retrieve relevant chunks with distance threshold
        retrieved_chunks = self.vector_store.query(
            question,
            n_results=n_results,
            distance_threshold=distance_threshold
        )
        
        if not retrieved_chunks:
            return {
                "answer": "I cannot find information about this topic in the uploaded study materials.",
                "sources": [],
                "mode": "no_context"
            }

        # Format sources summary
        sources = []
        for chunk in retrieved_chunks:
            sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "distance": chunk.get("distance", 0.0)
            })

        # Step 2: Generation via Claude (if API key available)
        if self._anthropic_client:
            try:
                context_blocks = []
                for i, c in enumerate(retrieved_chunks, 1):
                    context_blocks.append(f"[Document: {c['source']}, Page: {c['page']}]\n{c['text']}")
                context_str = "\n\n---\n\n".join(context_blocks)

                system_prompt = (
                    "You are an expert AI Study Assistant. Your task is to help students learn by answering "
                    "their questions accurately based ONLY on the provided study material context.\n\n"
                    "Rules:\n"
                    "1. Answer clearly, thoroughly, and concisely using the provided context.\n"
                    "2. Always reference the relevant page numbers and document names where applicable.\n"
                    "3. If the answer is NOT present in the provided context, clearly state: "
                    "'I cannot find information about this topic in the uploaded study materials.'\n"
                    "4. Do not hallucinate or use external knowledge that contradicts the study materials."
                )

                user_message = f"Study Material Context:\n{context_str}\n\nStudent Question:\n{question}"

                response = self._anthropic_client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}]
                )
                
                answer_text = response.content[0].text
                return {
                    "answer": answer_text,
                    "sources": sources,
                    "mode": "claude_generative"
                }
            except Exception as e:
                # Fallback to local mode if API call fails
                fallback_answer = self._generate_local_extractive_answer(question, retrieved_chunks)
                fallback_answer += f"\n\n*(Note: Claude API note: {str(e)}. Displayed in Local Retrieval Mode)*"
                return {
                    "answer": fallback_answer,
                    "sources": sources,
                    "mode": "local_fallback"
                }

        # Step 3: Local Extractive Mode (when no API key is provided)
        local_answer = self._generate_local_extractive_answer(question, retrieved_chunks)
        return {
            "answer": local_answer,
            "sources": sources,
            "mode": "local_extractive"
        }

    def _generate_local_extractive_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Extracts and formats grounded answers directly from retrieved passages without external API calls.
        """
        response_lines = [
            "### 📖 Grounded Answer (Retrieved from Study Material)",
            "",
            "The following relevant explanations were retrieved from your uploaded documents:",
            ""
        ]

        for i, chunk in enumerate(chunks, 1):
            page_info = f"**[Source: {chunk['source']} | Page {chunk['page']}]**"
            snippet = chunk["text"].strip().replace("\n", " ")
            response_lines.append(f"{i}. {page_info}")
            response_lines.append(f"   > \"{snippet}\"")
            response_lines.append("")

        response_lines.append("---")
        response_lines.append("💡 *Tip: Running in Local Retrieval Mode (No API key required). To enable full conversational AI explanations, add an `ANTHROPIC_API_KEY` in the sidebar or `.env` file.*")

        return "\n".join(response_lines)
