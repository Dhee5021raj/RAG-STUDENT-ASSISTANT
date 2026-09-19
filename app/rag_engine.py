import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    """
    RAG Engine connects Vector Retrieval with Generation.
    Supports:
      1. Multi-turn Conversational Memory awareness
      2. Multi-provider LLMs (Anthropic Claude with fallback to Local Extractive mode)
      3. Structured Practice Quiz & Flashcard generation grounded in study materials
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
        else:
            self._anthropic_client = None

    def answer_question(
        self,
        question: str,
        n_results: int = 4,
        model: str = "claude-3-5-haiku-20241022",
        distance_threshold: Optional[float] = 1.25,
        source_filter: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Retrieves relevant context with hybrid search and generates a grounded response with source citations.
        Supports multi-turn chat history context.
        """
        import time
        start_time = time.perf_counter()

        stats = self.vector_store.get_stats()
        if stats.get("total_chunks", 0) == 0:
            return {
                "answer": "No study materials uploaded or indexed. Please upload your study documents first.",
                "sources": [],
                "mode": "no_context",
                "latency_ms": 0.0
            }

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.query(
            question,
            n_results=n_results,
            distance_threshold=distance_threshold,
            source_filter=source_filter,
            use_hybrid=True
        )

        if not retrieved_chunks:
            return {
                "answer": "I cannot find information about this topic in the uploaded study materials.",
                "sources": [],
                "mode": "no_context",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 1)
            }

        sources = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "distance": chunk.get("distance", 0.0),
                "confidence": chunk.get("confidence", 85)
            }
            for chunk in retrieved_chunks
        ]

        # Step 2: Generation via Claude (if API key available)
        if self._anthropic_client:
            try:
                context_blocks = []
                for c in retrieved_chunks:
                    context_blocks.append(f"[Document: {c['source']}, Page: {c['page']}]\n{c['text']}")
                context_str = "\n\n---\n\n".join(context_blocks)

                history_context = ""
                if chat_history:
                    # Append recent 3 conversation turns
                    recent_turns = chat_history[-6:]
                    formatted_turns = [f"{m['role'].capitalize()}: {m['content']}" for m in recent_turns]
                    history_context = "\nRecent Conversation History:\n" + "\n".join(formatted_turns) + "\n\n"

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

                user_message = f"Study Material Context:\n{context_str}\n{history_context}\nStudent Question:\n{question}"

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
                    "mode": "claude_generative",
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 1)
                }
            except Exception as e:
                fallback_answer = self._generate_local_extractive_answer(question, retrieved_chunks)
                fallback_answer += f"\n\n*(Note: Claude API note: {str(e)}. Displayed in Local Retrieval Mode)*"
                return {
                    "answer": fallback_answer,
                    "sources": sources,
                    "mode": "local_fallback",
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 1)
                }

        # Step 3: Local Extractive Mode (when no API key is provided)
        local_answer = self._generate_local_extractive_answer(question, retrieved_chunks)
        return {
            "answer": local_answer,
            "sources": sources,
            "mode": "local_extractive",
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 1)
        }

    def generate_quiz(self, topic: str = "core concepts", n_questions: int = 5) -> List[Dict[str, Any]]:
        """Generates a practice quiz with questions, options, and explanations based on indexed materials."""
        chunks = self.vector_store.query(topic, n_results=5, use_hybrid=True)
        if not chunks:
            return []

        if self._anthropic_client:
            try:
                context_str = "\n".join([c["text"] for c in chunks])
                prompt = (
                    f"Based on the following study materials about '{topic}', generate a {n_questions}-question multiple choice quiz.\n"
                    "Return ONLY valid JSON array format like:\n"
                    '[{"question": "...", "options": ["A", "B", "C", "D"], "answer": "Option text", "explanation": "..."}]\n\n'
                    f"Context:\n{context_str}"
                )
                res = self._anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw_json = res.content[0].text.strip()
                # Parse JSON array out of response
                start = raw_json.find("[")
                end = raw_json.rfind("]") + 1
                if start != -1 and end > start:
                    return json.loads(raw_json[start:end])
            except Exception as e:
                print(f"Quiz generation API error: {e}")

        # Local Extractive Quiz Fallback
        quiz = []
        for i, chunk in enumerate(chunks[:n_questions], 1):
            snippet = chunk["text"][:150]
            quiz.append({
                "question": f"Question {i}: Based on {chunk['source']} (Page {chunk['page']}), which concept is highlighted?",
                "options": [
                    snippet[:60] + "...",
                    "Incorrect option distracter A",
                    "Incorrect option distracter B",
                    "None of the above"
                ],
                "answer": snippet[:60] + "...",
                "explanation": f"Extracted directly from Page {chunk['page']} of {chunk['source']}."
            })
        return quiz

    def _generate_local_extractive_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Extracts and formats grounded answers directly from retrieved passages without external API calls."""
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

