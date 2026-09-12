import os
import tempfile
import time
import json
import streamlit as st

from main import answer_question, vector_store
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

st.set_page_config(
    page_title="AI Contract Summary & Risk Analyzer",
    layout="wide",
)

# Custom Styling
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .app-header {
        padding: 4px 0 18px 0;
        border-bottom: 1px solid #e0e4ea;
        margin-bottom: 20px;
    }
    .app-header h1 { color: #1F3864; margin-bottom: 2px; font-size: 28px; }
    .app-header p { color: #666; font-size: 14px; margin: 0; }

    .risk-card {
        background: #fdf2f2;
        border-left: 5px solid #d9534f;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .risk-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .risk-title {
        font-weight: 700;
        font-size: 16px;
        color: #a94442;
    }
    .risk-badge {
        background: #d9534f;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .risk-desc {
        font-size: 14px;
        color: #333333;
        margin-bottom: 6px;
        line-height: 1.5;
    }
</style>
<div class="app-header">
    <h1>AI Contract Summary & Risk Analyzer</h1>
    <p>Upload a contract or paste text to get a plain-language summary, flagged risks, and follow-up Q&A.</p>
</div>
""", unsafe_allow_html=True)

# Session State Setup
if "active_source" not in st.session_state:
    st.session_state.active_source = None
if "summary_text" not in st.session_state:
    st.session_state.summary_text = None
if "red_flags" not in st.session_state:
    st.session_state.red_flags = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


def process_and_ingest_docs(docs, batch_size: int = 20) -> int:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        try:
            vector_store.add_documents(batch)
            time.sleep(2)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.warning("Hit API rate limit. Pausing 15s before retrying batch...")
                time.sleep(15)
                vector_store.add_documents(batch)
            else:
                raise e

    return len(chunks)


def safe_answer_question(query: str, k: int = 5, filter_dict: dict = None):
    try:
        return answer_question(query, k=k, filter_dict=filter_dict)
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            st.warning("**Gemini API rate limit reached.** Wait ~15-20s and try again.")
        else:
            st.error(f"An unexpected error occurred: {error_msg}")
        return None, None


def fetch_and_parse_red_flags(filter_clause: dict):
    prompt = (
        "Identify all high-risk clauses, financial penalties, auto-renewals, short notice windows, and liability caps. "
        "Return ONLY a raw JSON array of objects. Do NOT use markdown code blocks or triple backticks. "
        "Each object must have these keys: "
        '{"title": "short risk title", "risk_level": "HIGH/MEDIUM/CRITICAL", "description": "detailed explanation"}'
    )
    
    raw_response, _ = safe_answer_question(prompt, k=8, filter_dict=filter_clause)
    
    if not raw_response:
        return []

    cleaned = raw_response.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return [{"title": "Contract Risk Alert", "risk_level": "HIGH", "description": raw_response}]


def wipe_vector_db():
    try:
        vector_store.reset_collection()
    except Exception:
        try:
            existing_ids = vector_store.get()["ids"]
            if existing_ids:
                vector_store.delete(ids=existing_ids)
        except Exception:
            pass


# 1. Input Section OR Active Status Bar
if st.session_state.active_source is None:
    st.subheader("1. Input Document")
    
    contract_title = st.text_input(
        "Contract Title / Reference Name (Optional):",
        placeholder="e.g., Commercial Lease Agreement, Freelance NDA, Service Terms",
    )

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload Document (PDF, TXT, MD)", type=["pdf", "txt", "md"])
    with col2:
        raw_text = st.text_area(
            "Or Paste Contract Text / Terms:",
            height=140,
            placeholder="Paste contract clauses, non-competes, or service terms here...",
        )

    submit_button = st.button("Analyze Contract (Summary & Risks)", type="primary", use_container_width=True)

    if submit_button:
        target_source = None
        docs_to_ingest = []

        if uploaded_file is not None:
            target_source = contract_title.strip() if contract_title.strip() else uploaded_file.name
            suffix = os.path.splitext(uploaded_file.name)[-1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = UnstructuredFileLoader(tmp_path, mode="single")
            loaded_docs = loader.load()
            os.remove(tmp_path)

            for d in loaded_docs:
                d.metadata["source"] = target_source
            docs_to_ingest = loaded_docs

        elif raw_text.strip():
            target_source = contract_title.strip() if contract_title.strip() else "Pasted Contract Document"
            docs_to_ingest = [Document(page_content=raw_text, metadata={"source": target_source})]
        else:
            st.error("Please upload a file or paste contract text before running analysis.")

        if docs_to_ingest:
            with st.status("Processing document and running analysis...", expanded=True) as status:
                st.write("Wiping previous document vectors to prevent context pollution...")
                wipe_vector_db()

                st.write("Chunking and indexing current document into ChromaDB...")
                num_chunks = process_and_ingest_docs(docs_to_ingest)

                filter_clause = {"source": target_source}

                st.write("Generating summary...")
                summary_text, _ = safe_answer_question(
                    "Provide a comprehensive executive summary of this contract, outlining the core "
                    "agreement, key obligations, and main parties involved.",
                    k=5, filter_dict=filter_clause,
                )

                st.write("Extracting structured red flags...")
                flags_data = fetch_and_parse_red_flags(filter_clause)

                status.update(label=f"Analysis complete — indexed {num_chunks} chunks for `{target_source}`.", state="complete")

            st.session_state.active_source = target_source
            st.session_state.summary_text = summary_text
            st.session_state.red_flags = flags_data
            st.session_state.chat_messages = []
            st.rerun()

else:
    st.info(f"Active Contract: **{st.session_state.active_source}**")
    if st.button("Analyze Another Contract / Reset", type="secondary"):
        wipe_vector_db()
        st.session_state.active_source = None
        st.session_state.summary_text = None
        st.session_state.red_flags = None
        st.session_state.chat_messages = []
        st.rerun()

# 2. Results Section (Full-Width Stacked Layout)
if st.session_state.summary_text or st.session_state.red_flags:
    st.divider()
    
    st.subheader("Contract Summary")
    if st.session_state.summary_text:
        st.markdown(st.session_state.summary_text)

    st.divider()

    st.subheader("High-Risk Clauses")
    if st.session_state.red_flags:
        for flag in st.session_state.red_flags:
            title = flag.get("title", "Risk Flag")
            level = flag.get("risk_level", "HIGH")
            desc = flag.get("description", "")

            st.markdown(f"""
            <div class="risk-card">
                <div class="risk-card-header">
                    <span class="risk-title">{title}</span>
                    <span class="risk-badge">{level}</span>
                </div>
                <div class="risk-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.caption("Note: This risk analysis does not constitute legal advice. Flags are intended to highlight potential clauses you may want to review closely.")
    # 3. Follow-Up Chat for Q&A Feature
    st.divider()
    st.subheader("Ask Questions")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if follow_up := st.chat_input(f"Ask about {st.session_state.active_source}..."):
        st.session_state.chat_messages.append({"role": "user", "content": follow_up})
        with st.chat_message("user"):
            st.markdown(follow_up)

        with st.chat_message("assistant"):
            with st.spinner("Checking the contract..."):
                filter_clause = {"source": st.session_state.active_source}
                answer, chunks = safe_answer_question(follow_up, k=5, filter_dict=filter_clause)
                if answer:
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})