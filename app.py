import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from app.pdf_processor import extract_text_from_file
from app.text_chunker import chunk_pages
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine
from app.exporter import export_chat_history, export_quiz, export_flashcards, export_study_roadmap
from app.history_tracker import log_query, get_history_records, clear_history
from app.synthesizer import generate_executive_summary, extract_concept_glossary
from app.quiz_evaluator import evaluate_quiz_submission
from app.roadmap_generator import generate_study_roadmap
from app.concept_graph import extract_concept_relationships
from app.socratic_tutor import generate_socratic_prompt, evaluate_student_explanation
from app.spaced_repetition import schedule_card_review, generate_deck_review_summary
from app.comparator import compare_documents_on_topic

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
    .badge-conf {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
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

    # Document Scope Filter & Hyperparameters
    st.divider()
    st.subheader("🎯 Retrieval Hyperparameters")
    doc_filter_options = ["All Documents"] + st.session_state.processed_files
    selected_doc_filter = st.selectbox("Search Scope", options=doc_filter_options, index=0)
    active_filter = None if selected_doc_filter == "All Documents" else selected_doc_filter

    top_k_val = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=4)
    dist_thresh_val = st.slider("Distance Threshold", min_value=0.5, max_value=2.0, value=1.25, step=0.05)
    use_multi_query_toggle = st.checkbox(
        "🔍 Multi-Query Expansion",
        value=False,
        help="Decomposes compound questions into sub-queries for broader semantic recall across multi-part topics."
    )
    use_reranker_toggle = st.checkbox(
        "⚡ Cross-Scoring Re-ranker",
        value=True,
        help="Applies multi-aspect lexical & semantic scoring to prioritize highest-relevance context chunks."
    )

    # Pomodoro Focus Timer
    st.divider()
    st.subheader("⏱️ Pomodoro Study Timer")
    timer_session = st.selectbox("Timer Interval", ["25 Min Study (Focus)", "5 Min Short Break", "15 Min Long Break"], index=0)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("▶️ Start Session", use_container_width=True):
            st.toast(f"Started: {timer_session}! Stay focused on your notes.", icon="🎯")
    with col_t2:
        if st.button("⏹️ Reset", use_container_width=True):
            st.toast("Timer reset.", icon="🔄")

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

# 5-Tab Layout
tab_chat, tab_quiz, tab_tutor, tab_guide, tab_kb = st.tabs(["💬 Chat & QA", "⚡ Quiz & Flashcards", "🧠 Socratic Recall", "📖 Study Guide & Roadmap", "📂 Knowledge Base & Analytics"])

# ----------------- TAB 1: CHAT -----------------
with tab_chat:
    # Quick Prompts Row
    col1, col2, col3, col4 = st.columns(4)
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
    with col4:
        chat_md = export_chat_history(st.session_state.messages)
        st.download_button(
            "📥 Export Notes (.md)",
            data=chat_md,
            file_name="study_session_notes.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.divider()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("expanded_queries") and len(msg["expanded_queries"]) > 1:
                st.caption("🔍 Decomposed Sub-Queries: " + " • ".join([f"`{q}`" for q in msg["expanded_queries"][1:]]))
            if "sources" in msg and msg["sources"]:
                latency_str = f" | ⚡ {msg.get('latency_ms', 0)}ms" if "latency_ms" in msg else ""
                with st.expander(f"🔍 Cited Sources ({len(msg['sources'])} Chunks{latency_str})"):
                    for s in msg["sources"]:
                        conf_pct = s.get("confidence", 85)
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>📄 {s['source']} — Page {s['page']}</strong> <span class="badge-conf">🟢 {conf_pct}% Match</span><br/>
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
                    n_results=top_k_val,
                    distance_threshold=dist_thresh_val,
                    source_filter=active_filter,
                    chat_history=st.session_state.messages[:-1],
                    use_multi_query=use_multi_query_toggle,
                    use_reranker=use_reranker_toggle
                )
                answer_text = result["answer"]
                sources = result.get("sources", [])
                latency = result.get("latency_ms", 0.0)
                expanded = result.get("expanded_queries", [])

                st.markdown(answer_text)
                if expanded and len(expanded) > 1:
                    st.caption("🔍 Decomposed Sub-Queries: " + " • ".join([f"`{q}`" for q in expanded[1:]]))
                if sources:
                    with st.expander(f"🔍 Cited Sources ({len(sources)} Chunks | ⚡ {latency}ms)"):
                        for s in sources:
                            conf_pct = s.get("confidence", 85)
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>📄 {s['source']} — Page {s['page']}</strong> <span class="badge-conf">🟢 {conf_pct}% Match</span><br/>
                                <em>"{s['text'][:300]}..."</em>
                            </div>
                            """, unsafe_allow_html=True)

        # Save to message history & analytics log
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "latency_ms": latency,
            "expanded_queries": expanded
        })

        top_src = sources[0]["source"] if sources else "None"
        log_query(
            question=user_question,
            mode=result.get("mode", "unknown"),
            latency_ms=latency,
            n_sources=len(sources),
            top_source=top_src
        )

# ----------------- TAB 2: PRACTICE QUIZ -----------------
with tab_quiz:
    st.subheader("🎯 Generate Practice Quiz")
    quiz_topic = st.text_input("Quiz Topic / Concept Keyword", value="operating system concepts")
    n_q = st.slider("Number of Questions", min_value=1, max_value=10, value=5)

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        if st.button("🚀 Generate Quiz Now", type="primary", use_container_width=True):
            with st.spinner("Analyzing study materials & creating practice quiz..."):
                quiz_data = st.session_state.rag_engine.generate_quiz(topic=quiz_topic, n_questions=n_q)
                st.session_state.current_quiz = quiz_data

    with col_btn2:
        if "current_quiz" in st.session_state and st.session_state.current_quiz:
            quiz_md = export_quiz(st.session_state.current_quiz, topic=quiz_topic)
            st.download_button(
                "📥 Export Quiz (.md)",
                data=quiz_md,
                file_name=f"practice_quiz_{quiz_topic.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )

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

        # Full Quiz Evaluation Mode
        st.subheader("📊 Full Quiz Evaluation & Scoring")
        if st.button("📝 Submit & Grade Entire Quiz", type="secondary", use_container_width=True):
            user_answers = {}
            for i in range(1, len(st.session_state.current_quiz) + 1):
                user_answers[i] = st.session_state.get(f"q_{i}", "")
            st.session_state.quiz_eval = evaluate_quiz_submission(st.session_state.current_quiz, user_answers)

        if "quiz_eval" in st.session_state and st.session_state.quiz_eval:
            ev = st.session_state.quiz_eval
            col_sc1, col_sc2, col_sc3 = st.columns(3)
            col_sc1.metric("Overall Score", f"{ev['score']} / {ev['total']}")
            col_sc2.metric("Accuracy", f"{ev['percentage']}%")
            col_sc3.metric("Mastery Rating", ev['mastery_level'])
            st.progress(ev['percentage'] / 100.0)

            if ev['revision_needed']:
                st.warning("⚠️ Targeted Revision Needed:")
                for rev in ev['revision_needed']:
                    st.markdown(f"- **Q{rev['question_num']} ({rev['question']})**: *{rev['study_recommendation']}*")

    st.divider()
    st.subheader("🎴 Generate Flashcard Deck")
    card_topic = st.text_input("Flashcard Topic", value="core concepts", key="fc_topic")
    n_cards_val = st.slider("Number of Cards", min_value=1, max_value=10, value=5, key="fc_count")

    col_fc1, col_fc2 = st.columns([2, 1])
    with col_fc1:
        if st.button("🎴 Generate Flashcards Now", type="primary", use_container_width=True):
            with st.spinner("Generating flashcard deck..."):
                st.session_state.current_flashcards = st.session_state.rag_engine.generate_flashcards(topic=card_topic, n_cards=n_cards_val)

    with col_fc2:
        if "current_flashcards" in st.session_state and st.session_state.current_flashcards:
            fc_md = export_flashcards(st.session_state.current_flashcards, topic=card_topic)
            st.download_button(
                "📥 Export Flashcards (.md)",
                data=fc_md,
                file_name=f"flashcards_{card_topic.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    if "current_flashcards" in st.session_state and st.session_state.current_flashcards:
        st.markdown("---")
        deck_stats = generate_deck_review_summary(st.session_state.current_flashcards)
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        col_ds1.metric("🌱 Learning (1d)", deck_stats["learning"])
        col_ds2.metric("🔄 Reviewing (2-6d)", deck_stats["reviewing"])
        col_ds3.metric("🏆 Mastered (>6d)", deck_stats["mastered"])

        for idx, card in enumerate(st.session_state.current_flashcards, 1):
            next_date = card.get("next_review_date", "Pending First Review")
            with st.expander(f"🎴 Flashcard {idx}: {card['front']} (📅 Due: {next_date})"):
                st.markdown(f"**Answer / Definition:**\n{card['back']}")
                st.caption(f"Source: {card['source']}")

                st.markdown("##### 🧠 How well did you recall this?")
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1:
                    if st.button("🔴 Again (1d)", key=f"srs_again_{idx}", use_container_width=True):
                        st.session_state.current_flashcards[idx-1] = schedule_card_review(card, "again")
                        st.rerun()
                with col_r2:
                    if st.button("🟡 Hard (2d)", key=f"srs_hard_{idx}", use_container_width=True):
                        st.session_state.current_flashcards[idx-1] = schedule_card_review(card, "hard")
                        st.rerun()
                with col_r3:
                    if st.button("🟢 Good (4d)", key=f"srs_good_{idx}", use_container_width=True):
                        st.session_state.current_flashcards[idx-1] = schedule_card_review(card, "good")
                        st.rerun()
                with col_r4:
                    if st.button("⭐ Easy (7d)", key=f"srs_easy_{idx}", use_container_width=True):
                        st.session_state.current_flashcards[idx-1] = schedule_card_review(card, "easy")
                        st.rerun()

# ----------------- TAB 3: SOCRATIC ACTIVE RECALL -----------------
with tab_tutor:
    st.subheader("🧠 Socratic Active Recall & Conceptual Drill")
    st.caption("Transform passive reading into active learning. Answer deep conceptual probes in your own words and receive grounded Socratic critique.")

    tutor_topic = st.text_input("Active Recall Topic", value="Process Synchronization", key="tutor_topic_input")
    if st.button("🎯 Generate Socratic Probe Question", type="primary", use_container_width=True):
        with st.spinner("Formulating diagnostic probe question from study materials..."):
            st.session_state.socratic_data = generate_socratic_prompt(
                vector_store=st.session_state.vector_store,
                rag_engine=st.session_state.rag_engine,
                topic=tutor_topic,
                source_filter=active_filter
            )
            st.session_state.socratic_evaluation = None

    if "socratic_data" in st.session_state and st.session_state.socratic_data:
        soc = st.session_state.socratic_data
        st.markdown("---")
        st.info(f"**🧐 Socratic Challenge Question:**\n\n### {soc['probe_question']}")
        if soc.get("expected_concepts"):
            st.caption("Key Concept Anchors: " + ", ".join([f"`{c}`" for c in soc["expected_concepts"][:4]]))

        student_ans = st.text_area(
            "Write your conceptual explanation (in your own words):",
            placeholder="Explain the mechanism, why it exists, and how the components interact...",
            height=160,
            key="student_socratic_response"
        )

        if st.button("🔍 Evaluate My Understanding", type="secondary", use_container_width=True):
            with st.spinner("Analyzing your explanation against textbook principles..."):
                eval_res = evaluate_student_explanation(
                    vector_store=st.session_state.vector_store,
                    rag_engine=st.session_state.rag_engine,
                    topic=tutor_topic,
                    question=soc["probe_question"],
                    student_answer=student_ans,
                    source_filter=active_filter
                )
                st.session_state.socratic_evaluation = eval_res

        if "socratic_evaluation" in st.session_state and st.session_state.socratic_evaluation:
            ev = st.session_state.socratic_evaluation
            st.markdown("---")
            col_m1, col_m2 = st.columns([1, 1])
            col_m1.metric("Conceptual Coverage", f"{ev['score_pct']}%")
            col_m2.metric("Tutor Evaluation", ev['grade'])
            st.progress(ev['score_pct'] / 100.0)

            st.markdown(f"**💬 Socratic Feedback:**\n\n{ev['feedback']}")

            if ev.get("missed_concepts"):
                st.warning("⚠️ Critical Concepts to Review:")
                for mc in ev["missed_concepts"]:
                    st.markdown(f"- **{mc}**")

            if ev.get("follow_up_question"):
                st.info(f"**🚀 Next Level Challenge:**\n\n*{ev['follow_up_question']}*")

# ----------------- TAB 4: STUDY GUIDE & GLOSSARY -----------------
with tab_guide:
    st.subheader("📖 Document Executive Summary & Concept Glossary")
    st.caption("Synthesize high-level study guides and extract technical concept definitions from your uploaded documents.")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if st.button("📑 Generate Executive Summary", type="primary", use_container_width=True):
            with st.spinner("Synthesizing executive summary..."):
                st.session_state.exec_summary = generate_executive_summary(
                    vector_store=st.session_state.vector_store,
                    rag_engine=st.session_state.rag_engine,
                    source_filter=active_filter
                )

    with col_g2:
        if st.button("📚 Extract Concept Glossary Table", type="secondary", use_container_width=True):
            with st.spinner("Extracting technical concepts and definitions..."):
                st.session_state.glossary_data = extract_concept_glossary(
                    vector_store=st.session_state.vector_store,
                    rag_engine=st.session_state.rag_engine,
                    source_filter=active_filter,
                    top_n=8
                )

    if "exec_summary" in st.session_state and st.session_state.exec_summary:
        st.markdown("---")
        st.markdown(st.session_state.exec_summary["summary"])
        if st.session_state.exec_summary.get("sources"):
            st.caption("Referenced Sections: " + ", ".join(st.session_state.exec_summary["sources"]))

    if "glossary_data" in st.session_state and st.session_state.glossary_data:
        st.markdown("---")
        st.subheader("📚 Key Concepts Glossary")
        st.dataframe(st.session_state.glossary_data, use_container_width=True)

    st.divider()
    st.subheader("🗺️ Prerequisite Study Roadmap & Dependency Graph")
    st.caption("Generate a pedagogical learning path with visual concept dependency graphs from your documents.")

    rm_topic = st.text_input("Roadmap Topic Focus", value="Core Concepts", key="rm_topic")
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        if st.button("🗺️ Generate Prerequisite Roadmap", type="primary", use_container_width=True):
            with st.spinner("Synthesizing learning path and dependency graph..."):
                roadmap_data = generate_study_roadmap(
                    vector_store=st.session_state.vector_store,
                    rag_engine=st.session_state.rag_engine,
                    topic=rm_topic,
                    source_filter=active_filter
                )
                st.session_state.current_roadmap = roadmap_data

    with col_r2:
        if "current_roadmap" in st.session_state and st.session_state.current_roadmap:
            rm_md = export_study_roadmap(st.session_state.current_roadmap)
            st.download_button(
                "📥 Export Roadmap (.md)",
                data=rm_md,
                file_name=f"study_roadmap_{rm_topic.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    if "current_roadmap" in st.session_state and st.session_state.current_roadmap:
        rm = st.session_state.current_roadmap
        st.markdown("#### 🧭 Concept Dependency Flowchart")
        st.markdown(f"```mermaid\n{rm.get('mermaid_graph', '')}\n```")

        st.markdown("#### 📚 Phased Learning Path")
        for stg in rm.get("stages", []):
            with st.expander(f"{stg.get('title', 'Stage')} (Est. Time: {stg.get('estimated_hours', 'N/A')})"):
                st.markdown(f"**Key Concepts:** {', '.join(stg.get('concepts', []))}")
                st.markdown(f"**Stage Overview:** {stg.get('summary', '')}")

    st.divider()
    st.subheader("🕸️ Concept Relationship Knowledge Graph")
    st.caption("Discover semantic connections and interconnected domain relationships across your study materials.")

    if st.button("🕸️ Generate Concept Knowledge Graph", type="secondary", use_container_width=True):
        with st.spinner("Analyzing semantic relationships across study materials..."):
            st.session_state.concept_graph_data = extract_concept_relationships(
                vector_store=st.session_state.vector_store,
                rag_engine=st.session_state.rag_engine,
                source_filter=active_filter
            )

    if "concept_graph_data" in st.session_state and st.session_state.concept_graph_data:
        cg = st.session_state.concept_graph_data
        st.markdown("#### 🌐 Interconnected Concept Network")
        st.markdown(f"```mermaid\n{cg.get('mermaid_graph', '')}\n```")

        if cg.get("relationships"):
            st.markdown("#### 🔗 Discovered Concept Triples")
            st.dataframe(cg["relationships"], use_container_width=True)

    st.divider()
    st.subheader("🔄 Cross-Document Topic Comparison")
    st.caption("Compare and contrast how different study materials (e.g. textbook vs lecture slides) cover a topic.")

    all_files = st.session_state.processed_files
    if len(all_files) < 2:
        st.info("💡 Upload at least two documents in the sidebar to enable cross-document comparative analysis.")
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            doc_a = st.selectbox("Document A", options=all_files, index=0, key="comp_doc_a")
        with col_c2:
            doc_b = st.selectbox("Document B", options=all_files, index=1 if len(all_files) > 1 else 0, key="comp_doc_b")

        comp_topic = st.text_input("Comparison Topic", value="CPU Scheduling", key="comp_topic_input")
        if st.button("⚖️ Compare Documents on Topic", type="secondary", use_container_width=True):
            with st.spinner(f"Comparing '{comp_topic}' across {doc_a} and {doc_b}..."):
                st.session_state.comparison_result = compare_documents_on_topic(
                    vector_store=st.session_state.vector_store,
                    rag_engine=st.session_state.rag_engine,
                    topic=comp_topic,
                    doc1=doc_a,
                    doc2=doc_b
                )

        if "comparison_result" in st.session_state and st.session_state.comparison_result:
            cmp_res = st.session_state.comparison_result
            st.markdown("---")
            st.markdown(f"#### ⚖️ Comparison Summary: *{cmp_res.get('topic', '')}*")
            st.markdown(f"**Synthesis:** {cmp_res.get('comparison_summary', '')}")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f"**Unique to {cmp_res.get('doc1')}:**")
                for u in cmp_res.get("doc1_unique", []):
                    st.markdown(f"- {u}")
            with col_s2:
                st.markdown(f"**Unique to {cmp_res.get('doc2')}:**")
                for u in cmp_res.get("doc2_unique", []):
                    st.markdown(f"- {u}")

            if cmp_res.get("shared_points"):
                st.markdown("**Shared Core Concepts:**")
                for sp in cmp_res["shared_points"]:
                    st.markdown(f"- {sp}")

# ----------------- TAB 5: KNOWLEDGE BASE -----------------
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

    st.divider()
    st.subheader("📊 Query Analytics & History Log")
    history_logs = get_history_records(limit=25)
    if not history_logs:
        st.info("No query logs recorded yet. Ask questions in Tab 1 to track analytics.")
    else:
        st.dataframe(history_logs, use_container_width=True)
        if st.button("🧹 Clear Query History"):
            clear_history()
            st.rerun()


