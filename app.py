"""
RADAR RAG - Streamlit UI
Two tabs:
  1. Ingest  – upload / drag-drop documents → save to docs folder → run ingestion pipeline
  2. Query   – chat interface with streamed answers and source links
"""

import os
import io
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from qdrant_client.http import models as qdrant_models

import config
from qdrant_helper import get_qdrant_client
from urlBuilder import build_sharepoint_file_url
from ingest import (
    get_file_parser,
    chunk_text,
)

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RADAR Knowledge Base",
    page_icon="📡",
    layout="wide",
)

# ── Shared helpers ────────────────────────────────────────────────────────────
def format_score(score: float) -> str:
    return f"{score * 100:.1f}%"


@st.cache_resource(show_spinner=False)
def get_client():
    """Cached Qdrant client so we don't reconnect on every rerun."""
    return get_qdrant_client()


def ensure_collection(client, collection_name: str):
    """Create the Qdrant collection if it doesn't exist yet."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=client.get_fastembed_vector_params(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – INGEST
# ═══════════════════════════════════════════════════════════════════════════
def ingest_tab():
    st.header("📥 Ingest Documents")
    st.caption(
        "Upload one or more documents. They will be saved to the documents folder "
        "and indexed into the selected Qdrant knowledge base automatically."
    )

    # ── Knowledge base selector ──
    kb_label = st.selectbox(
        "Target knowledge base",
        options=list(config.KNOWLEDGE_BASES.values()),
        help="Choose which knowledge base to ingest documents into.",
    )
    # Resolve label → collection name
    collection_name = next(
        k for k, v in config.KNOWLEDGE_BASES.items() if v == kb_label
    )
    sharepoint_folder = config.SHAREPOINT_FOLDER_PATHS.get(
        collection_name, config.SHAREPOINT_FOLDER_PATH
    )

    uploaded_files = st.file_uploader(
        label="Drop files here or click Browse",
        type=["txt", "md", "pdf", "docx", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Supported formats: .txt  .md  .pdf  .docx  .xlsx  .xls",
    )

    col1, col2 = st.columns([1, 3])
    reset_db = col1.checkbox(
        "Reset collection first",
        value=False,
        help="Deletes ALL existing vectors in the selected collection before ingesting. Use with caution.",
    )

    if col2.button("⚡ Ingest selected files", disabled=not uploaded_files, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
            return

        client = get_client()
        ensure_collection(client, collection_name)

        # Optionally reset
        if reset_db and client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            ensure_collection(client, collection_name)
            st.info(f"Collection **{kb_label}** reset.")

        progress = st.progress(0, text="Starting…")
        log = st.empty()
        messages = []

        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            suffix = Path(filename).suffix.lower()

            messages.append(f"**Processing:** `{filename}`")
            log.markdown("\n\n".join(messages))

            # ── Save to docs folder ──
            dest_path = Path(config.DOCS_DIR) / filename
            dest_path.write_bytes(uploaded_file.read())

            # ── Parse ──
            parser = get_file_parser(suffix)
            if not parser:
                messages.append(f"  ⚠️ Skipped — unsupported type `{suffix}`")
                log.markdown("\n\n".join(messages))
                continue

            text = parser(dest_path)
            if not text.strip():
                messages.append(f"  ⚠️ Skipped — file appears empty or unreadable")
                log.markdown("\n\n".join(messages))
                continue

            # ── Chunk ──
            chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            if not chunks:
                messages.append(f"  ⚠️ Skipped — no text chunks extracted")
                log.markdown("\n\n".join(messages))
                continue

            # ── Delete old vectors for this file (avoid duplicates) ──
            if not reset_db:
                try:
                    client.delete(
                        collection_name=collection_name,
                        points_selector=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="document_name",
                                    match=qdrant_models.MatchValue(value=filename),
                                )
                            ]
                        ),
                    )
                except Exception:
                    pass

            # ── Build metadata & upload ──
            sharepoint_url = build_sharepoint_file_url(
                tenant=config.SHAREPOINT_TENANT,
                folder_path=sharepoint_folder,
                file_name=filename,
            )

            chunk_texts = chunks
            chunk_metadatas = [
                {
                    "document_name": filename,
                    "file_path": str(dest_path),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "sharepoint_url": sharepoint_url,
                }
                for idx, _ in enumerate(chunks)
            ]

            client.add(
                collection_name=collection_name,
                documents=chunk_texts,
                metadata=chunk_metadatas,
            )

            messages.append(
                f"  ✅ Indexed **{len(chunks)} chunks** into **{kb_label}** from `{filename}`"
            )
            log.markdown("\n\n".join(messages))
            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Processed {i + 1}/{len(uploaded_files)} files",
            )

        progress.progress(1.0, text="Done!")
        st.success(f"Ingestion into **{kb_label}** complete!")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – QUERY
# ═══════════════════════════════════════════════════════════════════════════
def query_tab():
    st.header("💬 Ask the Knowledge Base")
    st.caption("Questions are answered using only the documents in the selected knowledge base.")

    # ── Sidebar settings ──
    with st.sidebar:
        st.subheader("⚙️ Query Settings")

        kb_label = st.selectbox(
            "Knowledge base",
            options=list(config.KNOWLEDGE_BASES.values()),
            help="Choose which knowledge base to search.",
            key="query_kb",
        )
        collection_name = next(
            k for k, v in config.KNOWLEDGE_BASES.items() if v == kb_label
        )

        top_k = st.slider("Number of source chunks", min_value=1, max_value=10, value=3)
        st.markdown("---")
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    # ── Chat history ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Input ──
    query = st.chat_input(f"Ask a question about {kb_label}…")
    if not query:
        return

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── RAG pipeline ──
    client = get_client()

    if not client.collection_exists(collection_name):
        with st.chat_message("assistant"):
            st.error(f"No knowledge base found for **{kb_label}**. Please ingest some documents first.")
        return

    with st.spinner(f"Searching {kb_label}…"):
        results = client.query(
            collection_name=collection_name,
            query_text=query,
            limit=top_k,
        )

    if not results:
        with st.chat_message("assistant"):
            msg = "I couldn't find any relevant information in the knowledge base."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})
        return

    # Build context & deduplicate sources
    context_parts = []
    raw_sources = []

    for point in results:
        text = getattr(point, "document", None) or point.payload.get("document", "")
        metadata = getattr(point, "metadata", None) or point.payload or {}

        doc_name = metadata.get("document_name", "Unknown")
        chunk_idx = metadata.get("chunk_index", 0)
        sharepoint_url = metadata.get("sharepoint_url")

        context_parts.append(f"Source: {doc_name} (Chunk {chunk_idx})\nContent: {text}")
        raw_sources.append({"name": doc_name, "score": point.score, "url": sharepoint_url})

    # Deduplicate — keep highest score per document
    seen: dict[str, dict] = {}
    for s in raw_sources:
        name = s["name"]
        if name not in seen or s["score"] > seen[name]["score"]:
            seen[name] = s

    unique_sources = list(seen.values())

    context_text = "\n\n=== CONTEXT SPLIT ===\n\n".join(context_parts)
    system_prompt = (
        "You are a helpful AI assistant. You answer user queries based on the provided context documents.\n"
        "Instructions:\n"
        "1. Base your answer strictly on the retrieved context below.\n"
        "2. If the context does not contain enough information, respond with: "
        "'I am sorry, but the provided documents do not contain information to answer this question.'\n"
        "3. Keep your answers accurate, well-structured, and factual.\n\n"
        "Retrieved Context:\n"
        "--------------------------------------------------\n"
        f"{context_text}\n"
        "--------------------------------------------------\n"
    )

    # ── Stream LLM response ──
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            stream = groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ Error calling Groq API: {e}"
            response_placeholder.error(full_response)

        _render_sources(unique_sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "sources": unique_sources}
    )


def _render_sources(sources: list[dict]):
    """Renders the collapsible sources block under an answer."""
    if not sources:
        return
    with st.expander(f"📄 Sources used ({len(sources)})", expanded=False):
        for i, s in enumerate(sources, 1):
            name = s.get("name", "Unknown")
            score = s.get("score", 0)
            url = s.get("url")
            if url:
                st.markdown(f"**{i}. [{name}]({url})** — Relevance: `{format_score(score)}`")
            else:
                st.markdown(f"**{i}. {name}** — Relevance: `{format_score(score)}`")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    st.title("📡 RADAR Knowledge Base")
    st.markdown("---")

    tab_ingest, tab_query = st.tabs(["📥 Ingest Documents", "💬 Query"])

    with tab_ingest:
        ingest_tab()

    with tab_query:
        query_tab()


if __name__ == "__main__":
    main()
