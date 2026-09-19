# 📚 AI-Powered RAG Study Assistant

A full-stack, portfolio-ready **Retrieval-Augmented Generation (RAG)** application designed to help students learn interactively from lecture notes, textbooks, and course PDFs.

The application ingests study documents, creates semantic vector embeddings, performs top-k similarity retrieval, and generates precise, grounded answers with exact source & page citations.

---

## 🌟 Key Features

* **📄 PDF Ingestion & Parsing:** Extracts structured text and preserves page numbers using PyMuPDF.
* **✂️ Intelligent Text Chunking:** Splits documents into overlapping chunks (~500 characters, 100 character overlap) to preserve semantic context across chunk boundaries.
* **🧠 Local Vector Embeddings:** Uses ChromaDB with built-in embeddings for instant offline retrieval without requiring third-party embedding API costs.
* **🔎 Semantic Similarity Retrieval:** Top-k nearest-neighbor search with cosine distance scoring.
* **🤖 Grounded AI Generation:**
  * **Claude Mode (via Anthropic API):** Conversational, hallucination-resistant answers with strict adherence to the provided context.
  * **Local Extractive Mode:** Operates out-of-the-box without an API key by extracting and formatting key passages with citations.
* **📌 Page Citations & Evidence:** Every answer links back to specific document pages and shows the exact retrieved text passages.
* **💻 Streamlit Web Interface:** Modern, responsive chat UI with multi-file PDF upload, knowledge base statistics, quick study prompts (summaries, quizzes, definitions), and conversation management.

---

## 🏗️ Architecture & Pipeline

```mermaid
flowchart TD
    A[Student PDF / Study Material] --> B[PyMuPDF Text Extraction & Page Tracking]
    B --> C[Overlapping Text Chunking]
    C --> D[Vector Embeddings]
    D --> E[(ChromaDB Vector Store)]
    
    F[Student Question] --> G[Semantic Query Search]
    E --> G
    G --> H[Top-K Relevant Chunks + Page Metadata]
    H --> I[RAG Engine]
    I --> J[Grounded Answer + Exact Page Citations]
    J --> K[Streamlit Web UI]
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* Git

### 2. Installation

Clone the repository and enter the directory:
```bash
git clone https://github.com/yourusername/rag-study-assistant.git
cd rag-study-assistant
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Anthropic Claude API Key
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Edit `.env` and add your Anthropic API Key:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
*(Note: If you don't have an API key, the app automatically runs in **Local Retrieval Mode**! You can also enter a key at any time inside the web UI).*

---

## 🖥️ Running the Web Application

Launch the Streamlit web app:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Tests & Evaluation

Run the unit, integration, and benchmark test suites:

```bash
# 1. Test Multi-Format Document Extraction (PDF, TXT, MD)
python -m tests.test_document_processor

# 2. Test Hybrid BM25 + Dense Vector Search
python -m tests.test_hybrid_retrieval

# 3. Test Session Exporter (Markdown Chat & Quiz Exporter)
python -m tests.test_exporter

# 4. Test RAG Engine Pipeline
python -m tests.test_rag_engine

# 5. Run Quantitative RAG Evaluation & Benchmarking (MRR & Precision@K)
python -m tests.benchmark_rag
```

---

## 📁 Project Structure

```
rag-study-assistant/
├── app/
│   ├── __init__.py
│   ├── pdf_processor.py      # Multi-format document extraction & text cleaning (.pdf, .txt, .md)
│   ├── text_chunker.py       # Header-aware overlapping sliding-window chunker with token counts
│   ├── vector_store.py       # ChromaDB + BM25 hybrid search index with RRF ranking
│   ├── rag_engine.py         # Grounded LLM generation, multi-turn memory & latency metrics
│   └── exporter.py           # Markdown exporter for study chat notes & practice quizzes
├── tests/
│   ├── __init__.py
│   ├── test_document_processor.py  # Multi-format extraction test
│   ├── test_hybrid_retrieval.py    # BM25 + Vector hybrid search test
│   ├── test_exporter.py            # Session export test
│   ├── test_rag_engine.py          # RAG pipeline test
│   └── benchmark_rag.py            # MRR & Precision@K benchmarking suite
├── documents/
│   └── test.pdf                    # Sample study PDF
├── app.py                          # 3-Tab Streamlit web application dashboard
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore configuration
└── README.md                       # Project documentation
```

---

## 🛠️ Technology Stack

* **Language:** Python 3.13
* **LLM Engine:** Anthropic Claude (`claude-3-5-haiku` / `claude-3-5-sonnet`)
* **Vector Database:** ChromaDB
* **Document Parsing:** PyMuPDF (`fitz`)
* **Web Framework:** Streamlit
* **Environment Management:** python-dotenv
