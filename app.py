import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from app.pdf_processor import extract_text_from_file
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
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 10px 18px;
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
        {"role": "assistant", "content": "👋 Hi there! I'm your **AI-Powered RAG Study Assistant**.\n\nUpload your study PDFs, Markdown notes, or text files in the sidebar, and ask me any questions about them!"}
    ]

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Update processed files list from vector store if empty
if not st.session_state.processed_files:
    stats = st.session_state.vector_store.get_stats()
    if stats.get("sources"):
        st.session_state.processed_files = stats["sources"]

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
        "Upload study PDFs, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbook chapters, or reference documents."
    )

    if uploaded_files:
        if st.button("📥 Process & Index Documents", type="primary", use_container_width=True):
            with st.spinner("Processing and indexing multi-format documents..."):
                total_new_chunks = 0
                for file in uploaded_files:
                    if file.name not in st.session_state.processed_files:
                        ext = os.path.splitext(file.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name

                        try:
                            # 1. Multi-format text extraction
                            pages = extract_text_from_file(tmp_path)
                            # 2. Hybrid chunking
                            chunks = chunk_pages(pages, source_file=file.name)
                            # 3. Store in ChromaDB & update BM25 index
                            count = st.session_state.vector_store.add_documents(chunks)
                            total_new_chunks += count
                            if file.name not in st.session_state.processed_files:
                                st.session_state.processed_files.append(file.name)
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

                st.success(f"Indexed {total_new_chunks} chunks from {len(uploaded_files)} file(s)!")

    # Document Scope Filter
    st.divider()
    st.subheader("🎯 Query Filter Scope")
    doc_filter_options = ["All Documents"] + st.session_state.processed_files
    selected_doc_filter = st.selectbox("Search Scope", options=doc_filter_options, index=0)
    active_filter = None if selected_doc_filter == "All Documents" else selected_doc_filter

    # Database Statistics
    st.divider()
    st.subheader("📊 Knowledge Base Stats")
    stats = st.session_state.vector_store.get_stats()
    st.metric("Total Indexed Chunks", stats["total_chunks"])
    st.metric("Total Documents", stats.get("total_documents", len(st.session_state.processed_files)))

    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.processed_files = []
        st.session_state.messages = [
            {"role": "assistant", "content": "Knowledge base reset. Upload new study documents to get started!"}
        ]
        st.rerun()

# ================= MAIN AREA =================
st.markdown('<div class="main-title">📚 AI-Powered RAG Study Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions, generate practice quizzes, and get answers strictly grounded in your study materials with page citations & hybrid retrieval.</div>', unsafe_allow_html=True)

# 3-Tab Layout
tab_chat, tab_quiz, tab_kb = st.tabs(["💬 Chat & Grounded QA", "⚡ Practice Quiz", "📂 Knowledge Base Manager"])

# ----------------- TAB 1: CHAT -----------------
with tab_chat:
    # Quick Prompts Row
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 Summarize Key Topics", use_container_width=True):
            st.session_state.quick_prompt = "Provide a comprehensive summary of the key topics covered in these materials."
    with col2:
        if st.button("❓ Explain Core Definitions", use_container_width=True):
            st.session_state.quick_prompt = "List and explain the top 5 most important terms and definitions from the study material."
    with col3:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history cleared. What would you like to ask about your materials?"}
            ]
            st.rerun()

    st.divider()

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
            with st.spinner("Searching study material (Hybrid BM25 + Vector) & generating answer..."):
                result = st.session_state.rag_engine.answer_question(
                    user_question,
                    n_results=4,
                    source_filter=active_filter,
                    chat_history=st.session_state.messages[:-1]
                )
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

# ----------------- TAB 2: PRACTICE QUIZ -----------------
with tab_quiz:
    st.subheader("🎯 Generate Practice Quiz")
    quiz_topic = st.text_input("Quiz Topic / Concept Keyword", value="operating system concepts")
    n_q = st.slider("Number of Questions", min_value=1, max_value=10, value=5)

    if st.button("🚀 Generate Quiz Now", type="primary"):
        with st.spinner("Analyzing study materials & creating practice quiz..."):
            quiz_data = st.session_state.rag_engine.generate_quiz(topic=quiz_topic, n_questions=n_q)
            st.session_state.current_quiz = quiz_data

    if "current_quiz" in st.session_state and st.session_state.current_quiz:
        st.markdown("---")
        for idx, q in enumerate(st.session_state.current_quiz, 1):
            st.markdown(f"#### Q{idx}: {q['question']}")
            user_choice = st.radio(f"Select your answer for Q{idx}:", options=q["options"], key=f"q_{idx}")
            if st.button(f"Check Answer for Q{idx}", key=f"btn_{idx}"):
                if user_choice == q["answer"]:
                    st.success("🎉 Correct!")
                else:
                    st.error(f"❌ Incorrect. Correct answer: **{q['answer']}**")
                st.info(f"💡 Explanation: {q['explanation']}")
            st.markdown("---")

# ----------------- TAB 3: KNOWLEDGE BASE -----------------
with tab_kb:
    st.subheader("📋 Indexed Document Directory")
    kb_stats = st.session_state.vector_store.get_stats()

    if kb_stats["total_chunks"] == 0:
        st.info("No documents indexed yet. Use the sidebar to upload PDFs, TXT, or MD notes.")
    else:
        st.success(f"Knowledge Base Active with **{kb_stats['total_chunks']}** indexed chunks across **{kb_stats.get('total_documents', 0)}** documents.")
        if kb_stats.get("sources"):
            st.markdown("### Uploaded Files:")
            for s in kb_stats["sources"]:
                st.markdown(f"- 📄 **{s}**")

