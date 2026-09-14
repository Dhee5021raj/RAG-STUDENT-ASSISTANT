import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from app.pdf_processor import extract_text_from_pdf
from app.text_chunker import chunk_pages
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI RAG Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(persist_directory="chroma_db", collection_name="study_materials")

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(vector_store=st.session_state.vector_store)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi there! I\'m your **AI-Powered RAG Study Assistant**.\n\nUpload your lecture notes, textbook chapters, or PDFs in the sidebar, and ask me any questions about them!"}
    ]

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ================= SIDEBAR =================
with st.sidebar:
    st.title("⚙️ Study Controls")
    
    # API Key Input
    st.subheader("🔑 Anthropic API Key")
    api_key_input = st.text_input(
        "Claude API Key (Optional)",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Optional: Add your Anthropic key for generative answers. Without a key, the app runs in Local Retrieval Mode."
    )
    if api_key_input:
        st.session_state.rag_engine.set_api_key(api_key_input)
        st.success("Claude API Key Active", icon="✅")
    else:
        st.info("Local Retrieval Mode Active (No Key Needed)", icon="ℹ️")

    st.divider()

    # Document Upload Section
    st.subheader("📂 Upload Materials")
    uploaded_files = st.file_uploader(
        "Upload study PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbook chapters, or reference PDFs."
    )

    if uploaded_files:
        if st.button("📥 Process & Index Documents", type="primary", use_container_width=True):
            with st.spinner("Processing and indexing documents..."):
                total_new_chunks = 0
                for file in uploaded_files:
                    if file.name not in st.session_state.processed_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name

                        try:
                            # 1. Extract text with page numbers
                            pages = extract_text_from_pdf(tmp_path)
                            # 2. Chunk with metadata
                            chunks = chunk_pages(pages, source_file=file.name)
                            # 3. Store in ChromaDB
                            count = st.session_state.vector_store.add_documents(chunks)
                            total_new_chunks += count
                            st.session_state.processed_files.append(file.name)
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

                st.success(f"Indexed {total_new_chunks} chunks from {len(uploaded_files)} file(s)!")

    # Database Statistics
    st.divider()
    st.subheader("📊 Knowledge Base Stats")
    stats = st.session_state.vector_store.get_stats()
    st.metric("Total Indexed Chunks", stats["total_chunks"])
    if st.session_state.processed_files:
        st.caption("Indexed files: " + ", ".join(st.session_state.processed_files))

    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.processed_files = []
        st.session_state.messages = [
            {"role": "assistant", "content": "Knowledge base reset. Upload new study documents to get started!"}
        ]
        st.rerun()

# ================= MAIN AREA =================
st.markdown('<div class="main-title">📚 AI-Powered RAG Study Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions, generate practice quizzes, and get answers strictly grounded in your study materials with page citations.</div>', unsafe_allow_html=True)

# Quick Prompts Row
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📝 Summarize Key Topics", use_container_width=True):
        st.session_state.quick_prompt = "Provide a comprehensive summary of the key topics covered in these materials."
with col2:
    if st.button("❓ Generate Practice Quiz", use_container_width=True):
        st.session_state.quick_prompt = "Generate a 5-question practice quiz based on the core concepts in these materials, with an answer key."
with col3:
    if st.button("💡 Explain Core Definitions", use_container_width=True):
        st.session_state.quick_prompt = "List and explain the top 5 most important terms and definitions from the study material."

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"🔍 Cited Sources ({len(msg['sources'])} Chunks)"):
                for s in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>📄 {s['source']} — Page {s['page']}</strong><br/>
                        <em>"{s['text'][:300]}..."</em>
                    </div>
                    """, unsafe_allow_html=True)

# Handle User Input
prompt_input = st.chat_input("Ask a question about your study documents...")
user_question = None

if prompt_input:
    user_question = prompt_input
elif "quick_prompt" in st.session_state and st.session_state.quick_prompt:
    user_question = st.session_state.quick_prompt
    st.session_state.quick_prompt = None

if user_question:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Searching study material & generating answer..."):
            result = st.session_state.rag_engine.answer_question(user_question, n_results=4)
            answer_text = result["answer"]
            sources = result.get("sources", [])

            st.markdown(answer_text)
            if sources:
                with st.expander(f"🔍 Cited Sources ({len(sources)} Chunks)"):
                    for s in sources:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>📄 {s['source']} — Page {s['page']}</strong><br/>
                            <em>"{s['text'][:300]}..."</em>
                        </div>
                        """, unsafe_allow_html=True)

    # Save to message history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources
    })
