import os
import tempfile
import streamlit as st

# Import vector_store and answer_question directly from main.py
from main import answer_question, vector_store
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit Page Setup
st.set_page_config(
    page_title="Contract QA & Risk Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Contract QA & Risk Analyzer")

# Helper function for dynamic document ingestion inside app.py
def process_and_ingest(file_path: str) -> int:
    loader = UnstructuredFileLoader(file_path, mode="single")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)
    
    # Add chunks directly to Chroma vector store loaded in main.py
    vector_store.add_documents(chunks)
    return len(chunks)

# Sidebar: Document Ingestion
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload a Contract (PDF, TXT, MD)", 
    type=["pdf", "txt", "md"]
)

# Handle Dynamic File Processing
if uploaded_file:
    if "ingested_filename" not in st.session_state or st.session_state.ingested_filename != uploaded_file.name:
        with st.sidebar.status("Ingesting document into vector store...", expanded=True) as status:
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Ingest locally without modifying main.py
            num_chunks = process_and_ingest(tmp_path)
            os.remove(tmp_path)
            
            st.session_state.ingested_filename = uploaded_file.name
            status.update(label=f"Ingested {num_chunks} chunks!", state="complete", expanded=False)

st.sidebar.info("Uploaded contracts are indexed into ChromaDB for instant Q&A.")

# Navigation Tabs
tab1, tab2 = st.tabs(["💬 Contract Chatbot", "🚨 Risk & Red Flag Scan"])

# TAB 1: Chatbot Interface
with tab1:
    st.subheader("Interactive Contract QA")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input Field
    if user_query := st.chat_input("Ask about termination fees, notice windows, liabilities..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching vector store and analyzing context..."):
                answer, chunks = answer_question(user_query, k=5)
                
                # Deduplicate and format source citations
                sources = list(set(c.metadata.get("source", "Unknown") for c in chunks))
                source_text = "\n\n**Sources Referenced:**\n" + "\n".join(f"- `{s}`" for s in sources)
                
                full_response = f"{answer}\n{source_text}"
                st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# TAB 2: Automated Risk Scanner
with tab2:
    st.subheader("Automated Legal Risk Scan")
    st.write("Scan all indexed contracts for hidden financial penalties, strict notice windows, and risky clauses.")
    
    if st.button("Generate Risk Profile", type="primary"):
        with st.spinner("Extracting risky clauses across vector database..."):
            risk_query = (
                "Identify all high-risk clauses, financial penalties, automatic renewal traps, "
                "short cancellation notice windows, and liability caps across the provided documents. "
                "Structure the result as a clear bulleted risk list with citations."
            )
            analysis_text, chunks = answer_question(risk_query, k=8)
            
            st.warning("⚠️ High Risk Items & Unfavorable Terms Identified")
            st.markdown(analysis_text)