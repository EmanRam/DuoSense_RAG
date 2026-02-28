"""
app.py  –  DouSense Multimodal RAG  •  Streamlit UI
Run:  streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from config import TOP_K
from core.embedder import CLIPEmbedder
from core.pdf_processor import PDFProcessor
from core.retriever import MultimodalRetriever
from core.vector_store import ChromaVectorStore

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DouSense",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;800;900&family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,60,180,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(20,180,160,0.12) 0%, transparent 55%);
}

[data-testid="stSidebar"] {
    background: rgba(14,14,22,0.97) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #7c3aed, #14b8a6, #7c3aed);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

.sidebar-brand { padding: 2rem 1.5rem 1rem; margin-bottom: 0.5rem; }
.sidebar-brand .logo-mark {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.4rem; font-weight: 900;
    letter-spacing: 0.06em; text-transform: uppercase;
    background: linear-gradient(135deg, #a78bfa 0%, #14b8a6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1;
}
.sidebar-brand .tagline {
    font-size: 0.72rem; color: rgba(255,255,255,0.35);
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-top: 0.35rem; font-weight: 300;
}

.sidebar-label {
    font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    padding: 0 1.5rem; margin: 1.5rem 0 0.5rem;
}

/* ── Mode info badge ── */
.mode-info {
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin: 0.4rem 0 0.8rem;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.45);
    line-height: 1.45;
}
.mode-info-icon { font-size: 0.9rem; flex-shrink: 0; }

[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(124,58,237,0.35) !important;
    border-radius: 12px !important;
    background: rgba(124,58,237,0.04) !important;
    transition: all 0.25s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(124,58,237,0.6) !important;
    background: rgba(124,58,237,0.08) !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.65rem 1.5rem !important;
    font-family: 'Syne', sans-serif !important; font-size: 0.85rem !important;
    font-weight: 600 !important; letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.4) !important;
}
.stButton > button:disabled {
    background: rgba(255,255,255,0.06) !important;
    box-shadow: none !important; transform: none !important;
}

[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #14b8a6) !important;
}

.status-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(20,184,166,0.12); border: 1px solid rgba(20,184,166,0.25);
    color: #14b8a6; font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.06em; padding: 0.3rem 0.8rem;
    border-radius: 999px; margin: 0.5rem 1.5rem;
}
.status-badge::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: #14b8a6; animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}

/* ── Hero ── */
.hero-header { padding: 3rem 0 1.5rem; text-align: center; }
.hero-header h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 2.8rem !important; font-weight: 900 !important;
    letter-spacing: 0.06em !important; color: white !important;
    line-height: 1.1 !important; margin-bottom: 0.6rem !important;
    text-transform: uppercase !important;
}
.hero-header h1 span {
    background: linear-gradient(135deg, #a78bfa 0%, #14b8a6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-header p {
    color: rgba(255,255,255,0.38) !important; font-size: 0.92rem !important;
    font-weight: 300 !important; letter-spacing: 0.06em !important;
}

/* ── Active mode banner ── */
.mode-banner {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.22);
    border-radius: 999px; padding: 0.3rem 1rem;
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.06em;
    color: rgba(167,139,250,0.85); margin-bottom: 1.5rem;
}

/* ── Uploaded image preview ── */
.img-preview-wrap {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; overflow: hidden;
    margin: 0.5rem 0 1rem;
}

/* ── Feature cards ── */
.features-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin: 2rem 0; }
.feature-card {
    background: rgba(18,18,30,0.7); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.5rem 1.4rem;
    transition: all 0.25s ease; position: relative; overflow: hidden;
}
.feature-card:hover { border-color:rgba(255,255,255,0.13); transform:translateY(-2px); background:rgba(24,24,40,0.85); }
.fc-purple::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#7c3aed,#a78bfa); }
.fc-teal::before   { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#14b8a6,#06b6d4); }
.fc-amber::before  { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#f59e0b,#ef4444); }
.feature-icon { font-size: 1.6rem; margin-bottom: 0.8rem; display: block; }
.feature-title { font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700; color:rgba(255,255,255,0.85); margin-bottom:0.4rem; letter-spacing:0.02em; }
.feature-desc { font-size:0.78rem; color:rgba(255,255,255,0.38); line-height:1.65; font-weight:300; }

/* ── Recent queries ── */
.section-title { font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:600; letter-spacing:0.14em; text-transform:uppercase; color:rgba(255,255,255,0.28); margin:2rem 0 0.75rem; }
.query-history { display:flex; flex-direction:column; gap:0.45rem; margin-bottom:2rem; }
.query-pill {
    display:flex; align-items:center; gap:0.6rem;
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
    border-radius:8px; padding:0.55rem 0.9rem;
    font-size:0.82rem; color:rgba(255,255,255,0.45); line-height:1.4;
    transition: all 0.2s ease;
}
.query-pill:hover { background:rgba(124,58,237,0.08); border-color:rgba(124,58,237,0.2); color:rgba(255,255,255,0.6); }
.query-pill::before { content:'›'; color:rgba(124,58,237,0.5); font-size:1rem; flex-shrink:0; }

/* ── Empty state ── */
.empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:2rem 2rem 3rem; gap:0.8rem; }
.empty-icon { width:64px; height:64px; border-radius:18px; background:rgba(124,58,237,0.1); border:1.5px solid rgba(124,58,237,0.2); display:flex; align-items:center; justify-content:center; font-size:1.6rem; margin-bottom:0.3rem; }
.empty-state h3 { font-family:'Syne',sans-serif; font-size:1rem; font-weight:600; color:rgba(255,255,255,0.6); margin:0; }
.empty-state p  { font-size:0.82rem; color:rgba(255,255,255,0.28); margin:0; text-align:center; max-width:260px; line-height:1.65; }

/* ── Chat ── */
[data-testid="stChatMessage"] { background:transparent !important; border:none !important; padding:0.5rem 0 !important; }
[data-testid="stChatInput"] { background:rgba(20,20,35,0.9) !important; border:1.5px solid rgba(255,255,255,0.08) !important; border-radius:14px !important; backdrop-filter:blur(12px) !important; transition:border-color 0.2s ease !important; }
[data-testid="stChatInput"]:focus-within { border-color:rgba(124,58,237,0.5) !important; box-shadow:0 0 0 3px rgba(124,58,237,0.1) !important; }
[data-testid="stChatInput"] textarea { color:rgba(255,255,255,0.85) !important; font-family:'DM Sans',sans-serif !important; font-size:0.9rem !important; }
[data-testid="stChatInput"] textarea::placeholder { color:rgba(255,255,255,0.2) !important; }

.answer-card { background:rgba(18,18,30,0.8); border:1px solid rgba(255,255,255,0.07); border-left:3px solid #7c3aed; border-radius:0 14px 14px 0; padding:1.4rem 1.6rem; color:rgba(255,255,255,0.82); font-size:0.92rem; line-height:1.75; backdrop-filter:blur(8px); margin:0.3rem 0; }
.user-card { background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(91,33,182,0.1)); border:1px solid rgba(124,58,237,0.2); border-radius:14px 14px 4px 14px; padding:0.9rem 1.3rem; color:rgba(255,255,255,0.88); font-size:0.92rem; line-height:1.65; margin-left:auto; max-width:85%; }

[data-testid="stExpander"] { background:rgba(14,14,22,0.6) !important; border:1px solid rgba(255,255,255,0.06) !important; border-radius:10px !important; margin-top:0.5rem !important; }
[data-testid="stExpander"] summary { font-size:0.75rem !important; color:rgba(255,255,255,0.35) !important; letter-spacing:0.05em !important; font-weight:500 !important; }
.chunk-item { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.5rem; font-size:0.8rem; color:rgba(255,255,255,0.5); line-height:1.6; }
.chunk-label { font-size:0.65rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:rgba(20,184,166,0.7); margin-bottom:0.35rem; }

.divider { height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent); margin:1rem 0; }
.stats-row { display:flex; gap:0.5rem; flex-wrap:wrap; margin:0.75rem 0; }
.stat-pill { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:999px; padding:0.2rem 0.7rem; font-size:0.7rem; color:rgba(255,255,255,0.4); letter-spacing:0.04em; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 2px; }

/* ── Hide menu & footer only — leave header so sidebar toggle works ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedder() -> CLIPEmbedder:
    return CLIPEmbedder()

@st.cache_resource(show_spinner=False)
def get_vector_store() -> ChromaVectorStore:
    embedder = get_embedder()
    return ChromaVectorStore(embedding_dim=embedder.embedding_dimension())


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("indexed", False),
    ("processor", None),
    ("retriever", None),
    ("chat_history", []),
    ("chunk_count", 0),
    ("doc_name", ""),
    ("recent_queries", []),
    ("input_mode", "Both"),
    ("uploaded_image_b64", None),
    ("uploaded_image_name", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo-mark">DouSense</div>
        <div class="tagline">Multimodal Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.indexed:
        st.markdown(f'<div class="status-badge">Ready · {st.session_state.doc_name[:22]}</div>', unsafe_allow_html=True)

    # ── Input Mode ────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">Input Mode</div>', unsafe_allow_html=True)

    mode_cols = st.columns(3)
    modes = [("📝", "Text"), ("🖼️", "Image"), ("✦", "Both")]
    for col, (icon, label) in zip(mode_cols, modes):
        with col:
            is_active = st.session_state.input_mode == label
            if st.button(
                f"{icon}\n{label}",
                key=f"mode_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.input_mode = label
                st.session_state.indexed = False
                st.session_state.chat_history = []
                st.session_state.uploaded_image_b64 = None
                st.rerun()

    mode = st.session_state.input_mode

    mode_descriptions = {
        "Text":  ("📄", "Upload a plain text file (.txt). Only text content will be indexed."),
        "Image": ("🖼️", "Upload a standalone image and ask questions about it directly."),
        "Both":  ("✦",  "Upload a PDF that contains both text and images — all indexed together."),
    }
    m_icon, m_desc = mode_descriptions[mode]
    st.markdown(f'<div class="mode-info"><span class="mode-info-icon">{m_icon}</span>{m_desc}</div>', unsafe_allow_html=True)

    # ── Upload — one uploader per mode ───────────────────────────────────────
    uploaded_file  = None
    uploaded_image = None
    uploaded_txt   = None

    if mode == "Text":
        st.markdown('<div class="sidebar-label">Text File</div>', unsafe_allow_html=True)
        uploaded_txt = st.file_uploader(" ", type=["txt"], key="txt_upload", label_visibility="collapsed")

    elif mode == "Image":
        st.markdown('<div class="sidebar-label">Image</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(" ", type=["png", "jpg", "jpeg", "webp"], key="img_upload", label_visibility="collapsed")

    else:  # Both
        st.markdown('<div class="sidebar-label">PDF Document</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(" ", type=["pdf"], key="pdf_upload", label_visibility="collapsed")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=TOP_K, label_visibility="collapsed")

    st.markdown("")

    btn_labels = {"Text": "✦  Index Text File", "Image": "✦  Load Image", "Both": "✦  Index PDF"}
    can_index = (
        (mode == "Text"  and uploaded_txt   is not None) or
        (mode == "Image" and uploaded_image is not None) or
        (mode == "Both"  and uploaded_file  is not None)
    )
    index_btn = st.button(btn_labels[mode], disabled=not can_index, use_container_width=True)

    if st.session_state.indexed:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        mode_pill = {"Text": "📝 Text", "Image": "🖼️ Image", "Both": "✦ Both"}[mode]
        st.markdown(f"""
        <div style="padding: 0 1rem;">
            <div class="stats-row">
                <span class="stat-pill">{mode_pill}</span>
                <span class="stat-pill">📄 {st.session_state.doc_name[:16]}</span>
                <span class="stat-pill">🗂 {st.session_state.chunk_count} chunks</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Indexing / Loading logic ───────────────────────────────────────────────────
if index_btn and can_index:
    embedder = get_embedder()
    vector_store = get_vector_store()

    ph = st.empty()
    ph.markdown('<div style="text-align:center;padding:2rem;color:rgba(255,255,255,0.4);font-size:0.85rem;">✦ Processing…</div>', unsafe_allow_html=True)

    doc_name = ""
    chunk_count = 0
    processor = PDFProcessor(embedder=embedder, vector_store=vector_store)

    # ── TEXT mode: index plain .txt file ─────────────────────────────────────
    if mode == "Text" and uploaded_txt is not None:
        from langchain_core.documents import Document as LCDocument
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_content = uploaded_txt.read().decode("utf-8", errors="ignore")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.create_documents(
            [text_content],
            metadatas=[{"page": 0, "type": "text"}],
        )
        vector_store.clear()
        embeddings = [embedder.embed_text(c.page_content) for c in chunks]
        vector_store.add_documents(chunks, embeddings)

        st.session_state.processor = processor
        doc_name = uploaded_txt.name
        chunk_count = vector_store.count()

    # ── IMAGE mode: embed standalone image ────────────────────────────────────
    elif mode == "Image" and uploaded_image is not None:
        from langchain_core.documents import Document as LCDocument

        img_bytes = uploaded_image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        st.session_state.uploaded_image_b64 = img_b64
        st.session_state.uploaded_image_name = uploaded_image.name

        img_id = "standalone_image_0"
        img_embedding = embedder.embed_image(pil_img)
        img_doc = LCDocument(
            page_content=f"[Image: {img_id}]",
            metadata={"page": 0, "type": "image", "image_id": img_id},
        )
        processor.image_data_store = {img_id: img_b64}
        vector_store.clear()
        vector_store.add_documents([img_doc], [img_embedding])

        st.session_state.processor = processor
        doc_name = uploaded_image.name
        chunk_count = vector_store.count()

    # ── BOTH mode: full PDF (text + embedded images) ──────────────────────────
    elif mode == "Both" and uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)
        processor.process(tmp_path)

        st.session_state.processor = processor
        doc_name = uploaded_file.name
        chunk_count = vector_store.count()

    ph.empty()

    st.session_state.retriever = MultimodalRetriever(
        embedder=embedder,
        vector_store=vector_store,
        image_data_store=st.session_state.processor.image_data_store,
        top_k=top_k,
    )
    st.session_state.indexed = True
    st.session_state.chat_history = []
    st.session_state.chunk_count = chunk_count
    st.session_state.doc_name = doc_name
    st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>Ask Your <span>Document</span></h1>
    <p>Upload a PDF · Index it · Ask anything — text &amp; images understood</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.indexed:

    # ── Feature cards ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card fc-purple">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">Multimodal Understanding</div>
            <div class="feature-desc">Processes both text and images using CLIP embeddings — charts, diagrams, and photos included, nothing missed.</div>
        </div>
        <div class="feature-card fc-teal">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Semantic Retrieval</div>
            <div class="feature-desc">ChromaDB-powered vector search finds the most relevant chunks by meaning, not just keywords.</div>
        </div>
        <div class="feature-card fc-amber">
            <span class="feature-icon">💬</span>
            <div class="feature-title">Vision-Language Model</div>
            <div class="feature-desc">GPT-4o reads retrieved text and images together, generating answers that reference visual content naturally.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.recent_queries:
        st.markdown('<div class="section-title">Recent Queries</div>', unsafe_allow_html=True)
        pills = "".join(
            f'<div class="query-pill">{(q[:90] + "…") if len(q) > 90 else q}</div>'
            for q in st.session_state.recent_queries[-6:][::-1]
        )
        st.markdown(f'<div class="query-history">{pills}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <h3>No content indexed yet</h3>
            <p>Choose an input mode in the sidebar, upload your content, and click Index.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Active mode banner ────────────────────────────────────────────────────
    mode_banners = {
        "Text":  "📝  Text Mode — querying document text only",
        "Image": "🖼️  Image Mode — querying uploaded image",
        "Both":  "✦  Multimodal Mode — querying text + images",
    }
    st.markdown(f'<div class="mode-banner">{mode_banners[st.session_state.input_mode]}</div>', unsafe_allow_html=True)

    # ── Show uploaded image preview ───────────────────────────────────────────
    if st.session_state.input_mode in ("Image", "Both") and st.session_state.uploaded_image_b64:
        with st.expander(f"🖼️  Uploaded image — {st.session_state.uploaded_image_name}", expanded=False):
            st.markdown('<div class="img-preview-wrap">', unsafe_allow_html=True)
            st.image(f"data:image/png;base64,{st.session_state.uploaded_image_b64}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f'<div class="user-card">{turn["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<div class="answer-card">{turn["content"]}</div>', unsafe_allow_html=True)
                if turn.get("docs"):
                    with st.expander(f"  {len(turn['docs'])} retrieved chunks"):
                        for doc in turn["docs"]:
                            doc_type = doc.metadata.get("type", "unknown")
                            page = doc.metadata.get("page", "?")
                            if doc_type == "text":
                                preview = doc.page_content[:220] + "…" if len(doc.page_content) > 220 else doc.page_content
                                st.markdown(f'<div class="chunk-item"><div class="chunk-label">Text · Page {page}</div>{preview}</div>', unsafe_allow_html=True)
                            else:
                                image_id = doc.metadata.get("image_id", "")
                                image_b64 = st.session_state.retriever.image_data_store.get(image_id)
                                st.markdown(f'<div class="chunk-label">Image · Page {page}</div>', unsafe_allow_html=True)
                                if image_b64:
                                    st.image(f"data:image/png;base64,{image_b64}", use_column_width=True)

    # ── Chat input ────────────────────────────────────────────────────────────
    placeholders = {
        "Text":  "Ask about the document text…",
        "Image": "Ask about the image…",
        "Both":  "Ask anything about the document or images…",
    }
    if query := st.chat_input(placeholders[st.session_state.input_mode]):
        if query not in st.session_state.recent_queries:
            st.session_state.recent_queries.append(query)
            if len(st.session_state.recent_queries) > 20:
                st.session_state.recent_queries.pop(0)

        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner(""):
            retriever: MultimodalRetriever = st.session_state.retriever
            answer, docs = retriever.answer(query)

        st.session_state.chat_history.append({"role": "assistant", "content": answer, "docs": docs})
        st.rerun()