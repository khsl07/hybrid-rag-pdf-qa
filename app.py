import os
import time
import tempfile
import streamlit as st
import numpy as np
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

from rag.pdf_loader import PDFLoader, DocumentChunk
from rag.embeddings import GeminiEmbedder
from rag.vector_store import FAISSVectorStore
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever, RetrievalResult
from rag.generator import GroundedGenerator


st.set_page_config(
    page_title="Hybrid RAG PDF Q&A | FAISS + BM25 + Gemini",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-both {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #A5D6A7;
    }
    .badge-dense {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #90CAF9;
    }
    .badge-sparse {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #FFCC80;
    }
    .citation-card {
        border-left: 4px solid #1E88E5;
        background-color: #FAFAFA;
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "bm25_retriever" not in st.session_state:
        st.session_state.bm25_retriever = None
    if "hybrid_retriever" not in st.session_state:
        st.session_state.hybrid_retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = []


init_session_state()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key_env = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=api_key_env,
        type="password",
        help="Get a free key at https://aistudio.google.com",
    )
    api_key = api_key_input.strip()

    st.markdown("---")
    st.subheader("🔍 Retrieval Pipeline")

    search_mode = st.selectbox(
        "Search Mode",
        options=["hybrid", "dense", "sparse"],
        format_func=lambda x: {
            "hybrid": "⚡ Hybrid (FAISS + BM25 via RRF)",
            "dense": "🧠 Dense Semantic (FAISS Only)",
            "sparse": "🔤 Sparse Keyword (BM25 Only)",
        }[x],
        index=0,
    )

    top_k = st.slider("Top-K Retrieved Passages", min_value=1, max_value=12, value=4)

    with st.expander("Advanced RRF Hyperparameters"):
        rrf_k = st.slider(
            "RRF Constant (k)",
            min_value=10,
            max_value=100,
            value=60,
            help="Higher values give more balanced fusion across lower ranks.",
        )
        alpha = st.slider(
            "Dense vs. Sparse Balance (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="1.0 = Pure Dense (FAISS), 0.0 = Pure Sparse (BM25), 0.5 = Balanced.",
        )

    model_name = st.selectbox(
        "Gemini Generation Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
    )

    st.markdown("---")
    st.subheader("📄 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    chunk_size = st.number_input("Chunk Size (characters)", value=600, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (characters)", value=120, step=20)

    index_btn = st.button("🚀 Index Document(s)", use_container_width=True, type="primary")


# ----------------- INGESTION LOGIC -----------------
if index_btn:
    if not api_key:
        st.error("Please enter a valid Gemini API Key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        with st.status("Indexing documents into Hybrid Store...", expanded=True) as status:
            st.write("1️⃣ Extracting pages and applying semantic chunking...")
            loader = PDFLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            new_chunks = []
            doc_names = []

            for uploaded_file in uploaded_files:
                doc_name = uploaded_file.name
                doc_names.append(doc_name)
                chunks = loader.load_and_chunk(uploaded_file, doc_name=doc_name)
                new_chunks.extend(chunks)

            if not new_chunks:
                status.update(label="❌ No readable text found in PDF.", state="error")
            else:
                st.write(f"Extracted {len(new_chunks)} chunks across {len(doc_names)} document(s).")

                st.write("2️⃣ Generating Gemini embeddings for dense index...")
                embedder = GeminiEmbedder(api_key=api_key)
                texts = [c.content for c in new_chunks]
                embeddings = embedder.embed_texts(texts)

                st.write("3️⃣ Building FAISS FlatIP (Cosine Similarity) index...")
                vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
                vector_store.add_chunks(new_chunks, embeddings)

                st.write("4️⃣ Building BM25Okapi lexical inverted index...")
                bm25_retriever = BM25Retriever()
                bm25_retriever.index_chunks(new_chunks)

                # Initialize hybrid engine
                hybrid_retriever = HybridRetriever(
                    vector_store=vector_store,
                    bm25_retriever=bm25_retriever,
                )

                st.session_state.chunks = new_chunks
                st.session_state.vector_store = vector_store
                st.session_state.bm25_retriever = bm25_retriever
                st.session_state.hybrid_retriever = hybrid_retriever
                st.session_state.indexed_docs = doc_names

                status.update(
                    label=f"✅ Indexed {len(new_chunks)} chunks successfully!",
                    state="complete",
                )


# ----------------- MAIN UI -----------------
st.markdown('<div class="main-title">⚡ LLM-Powered Hybrid RAG PDF Q&A System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Dense Semantic (FAISS) + Sparse Lexical (BM25) fused via <b>Reciprocal Rank Fusion (RRF)</b> with grounded <b>Google Gemini API</b> synthesis.</div>',
    unsafe_allow_html=True,
)

# Metrics Strip
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Indexed Documents", len(st.session_state.indexed_docs))
with col2:
    st.metric("Total Chunks", len(st.session_state.chunks))
with col3:
    dim = st.session_state.vector_store.dimension if st.session_state.vector_store else 0
    st.metric("Embedding Dim", f"{dim}d" if dim else "N/A")
with col4:
    vocab_size = (
        len(st.session_state.bm25_retriever.bm25.doc_freqs)
        if st.session_state.bm25_retriever and st.session_state.bm25_retriever.bm25
        else 0
    )
    st.metric("BM25 Vocab Size", vocab_size if vocab_size else "N/A")

st.markdown("---")

tab_chat, tab_compare, tab_corpus = st.tabs(["💬 Grounded Q&A Chat", "⚖️ Hybrid vs Dense vs Sparse Comparator", "📚 Indexed Chunks"])


# ----------------- TAB 1: CHAT -----------------
with tab_chat:
    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "results" in message and message["results"]:
                with st.expander("🔍 Retrieved Sources & Attributions"):
                    for i, res in enumerate(message["results"], start=1):
                        c = res.chunk
                        badge_html = {
                            "both": '<span class="badge-both">⚡ FAISS + BM25 Match</span>',
                            "dense_only": '<span class="badge-dense">🧠 Dense Only</span>',
                            "sparse_only": '<span class="badge-sparse">🔤 Sparse Only</span>',
                        }.get(res.source_channel, res.source_channel)

                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <b>[{i}] {c.doc_name} — Page {c.page_number}</b> &nbsp; {badge_html}<br>
                                <small><b>RRF Score:</b> {res.rrf_score:.5f} | <b>Dense Rank:</b> {res.dense_rank or 'N/A'} (Sim: {f'{res.dense_score:.3f}' if res.dense_score is not None else 'N/A'}) | <b>BM25 Rank:</b> {res.sparse_rank or 'N/A'} (Score: {f'{res.sparse_score:.2f}' if res.sparse_score is not None else 'N/A'})</small>
                                <p style="margin-top: 8px; color: #333; font-size: 0.95rem;">{c.content}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Question Input
    user_query = st.chat_input("Ask a question about your indexed PDF...")

    if user_query:
        if not api_key:
            st.error("Please provide a Gemini API Key in the sidebar.")
        elif not st.session_state.hybrid_retriever:
            st.warning("Please upload and index a PDF first!")
        else:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Retrieving hybrid passages and generating grounded answer..."):
                    t0 = time.time()

                    # 1. Embed query
                    embedder = GeminiEmbedder(api_key=api_key)
                    query_vector = embedder.embed_query(user_query)

                    # 2. Hybrid Retrieve
                    results = st.session_state.hybrid_retriever.retrieve(
                        query=user_query,
                        query_vector=query_vector,
                        top_k=top_k,
                        search_mode=search_mode,
                        rrf_k=rrf_k,
                        alpha=alpha,
                    )

                    t_retrieve = time.time() - t0

                    # 3. Generate Answer
                    generator = GroundedGenerator(api_key=api_key, model_name=model_name)
                    answer = generator.generate_answer(user_query, results)

                    t_total = time.time() - t0

                    st.markdown(answer)
                    st.caption(f"⏱️ Retrieval: {t_retrieve*1000:.1f}ms | Total: {t_total:.2f}s | Mode: {search_mode.upper()}")

                    if results:
                        with st.expander("🔍 Retrieved Sources & Attributions"):
                            for i, res in enumerate(results, start=1):
                                c = res.chunk
                                badge_html = {
                                    "both": '<span class="badge-both">⚡ FAISS + BM25 Match</span>',
                                    "dense_only": '<span class="badge-dense">🧠 Dense Only</span>',
                                    "sparse_only": '<span class="badge-sparse">🔤 Sparse Only</span>',
                                }.get(res.source_channel, res.source_channel)

                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <b>[{i}] {c.doc_name} — Page {c.page_number}</b> &nbsp; {badge_html}<br>
                                        <small><b>RRF Score:</b> {res.rrf_score:.5f} | <b>Dense Rank:</b> {res.dense_rank or 'N/A'} (Sim: {f'{res.dense_score:.3f}' if res.dense_score is not None else 'N/A'}) | <b>BM25 Rank:</b> {res.sparse_rank or 'N/A'} (Score: {f'{res.sparse_score:.2f}' if res.sparse_score is not None else 'N/A'})</small>
                                        <p style="margin-top: 8px; color: #333; font-size: 0.95rem;">{c.content}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "results": results}
                    )


# ----------------- TAB 2: RETRIEVAL COMPARATOR -----------------
with tab_compare:
    st.subheader("⚖️ Side-by-Side Pipeline Comparison")
    st.write(
        "Evaluate how **Dense (FAISS)** and **Sparse (BM25)** rankings are fused by **Reciprocal Rank Fusion (RRF)**."
    )

    test_query = st.text_input(
        "Enter a test query for ranking breakdown:",
        value="What are the key findings or architectural components?",
    )

    if st.button("Run Multi-Retriever Comparison"):
        if not api_key:
            st.error("Please provide a Gemini API Key.")
        elif not st.session_state.hybrid_retriever:
            st.warning("Please upload and index a PDF first.")
        else:
            embedder = GeminiEmbedder(api_key=api_key)
            q_vec = embedder.embed_query(test_query)

            dense_res = st.session_state.hybrid_retriever.retrieve(
                test_query, q_vec, top_k=5, search_mode="dense"
            )
            sparse_res = st.session_state.hybrid_retriever.retrieve(
                test_query, q_vec, top_k=5, search_mode="sparse"
            )
            fused_res = st.session_state.hybrid_retriever.retrieve(
                test_query, q_vec, top_k=5, search_mode="hybrid", rrf_k=rrf_k, alpha=alpha
            )

            col_d, col_s, col_f = st.columns(3)

            with col_d:
                st.markdown("#### 🧠 FAISS (Dense)")
                for r in dense_res:
                    st.info(f"**Page {r.chunk.page_number}** (Cosine: {r.dense_score:.4f})\n\n{r.chunk.content[:160]}...")

            with col_s:
                st.markdown("#### 🔤 BM25 (Sparse)")
                for r in sparse_res:
                    st.warning(f"**Page {r.chunk.page_number}** (BM25: {r.sparse_score:.2f})\n\n{r.chunk.content[:160]}...")

            with col_f:
                st.markdown("#### ⚡ Fused RRF (Hybrid)")
                for r in fused_res:
                    st.success(f"**Page {r.chunk.page_number}** [{r.source_channel.upper()}] (RRF: {r.rrf_score:.5f})\n\n{r.chunk.content[:160]}...")


# ----------------- TAB 3: CORPUS INSPECTOR -----------------
with tab_corpus:
    st.subheader("📚 Corpus & Chunk Inspector")
    if not st.session_state.chunks:
        st.info("No documents indexed yet. Upload a PDF from the sidebar to populate.")
    else:
        st.write(f"Total chunks indexed: **{len(st.session_state.chunks)}**")
        for chunk in st.session_state.chunks[:50]:
            with st.expander(f"{chunk.chunk_id} | {chunk.doc_name} (Page {chunk.page_number}) - {chunk.char_count} chars"):
                st.write(chunk.content)
