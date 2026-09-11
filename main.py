import os
import warnings
import logging

warnings.filterwarnings("ignore")

logging.getLogger("unstructured").setLevel(logging.ERROR)
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

# STEP 1: Load contracts from files:
doc_directory = './docs'
documents = []
SUPPORTED_EXTENSIONS = ('.md', '.pdf', '.txt')
file_paths = [
    os.path.join(doc_directory, file)
    for file in os.listdir(doc_directory)
    if file.lower().endswith(SUPPORTED_EXTENSIONS)
]
assert len(file_paths) > 0, f"No supported files found in {doc_directory}!"

from langchain_community.document_loaders import UnstructuredFileLoader
for file_path in file_paths:
    loader = UnstructuredFileLoader(file_path, mode="single")
    documents.extend(loader.load())

# Ensure that all files (md, txt, and pdf) files can be read by the system:
# print(f"Successfully loaded {len(documents)} contract document(s)!")


# STEP 2: Split the contract into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 200,
)
chunks = text_splitter.split_documents(documents)

# print(f"Total chunks created: {len(chunks)}")

# Chech the first 3 chunks to verify chunk readability
#print("\nCHECKING CHUNKS:")
#for i, chunk in enumerate(chunks[:3]):
#    source = chunk.metadata.get('source', 'Unknown')
#    page = chunk.metadata.get('page', 'N/A')
#    print(f"\n--- Chunk {i+1} (Source: {source} | Page: {page}) ---")
#    print(chunk.page_content[:350] + "...")
#print('\n')


# STEP 3: Embeddings, Vector Store & Retrieval Function
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Initialize Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

db_path = "./chroma_db"

# Store chunks into Chroma:
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=db_path
)

def retrieve(query: str, k: int = 3):
    """Retrieves top-k relevant document chunks for a given search query."""
    return vector_store.similarity_search(query, k=k)

# Test of Step 3
#test_query = "What is the penalty for terminating the commercial lease early?"
#retrieved_docs = retrieve(test_query, k=2)

#print("STEP 3 RETRIEVAL TEST:")
#for idx, doc in enumerate(retrieved_docs, 1):
#    src = doc.metadata.get('source', 'Unknown')
#    print(f"\nResult {idx} (Source: {src}):")
#    print(doc.page_content[:250] + "...")
#print("\n")


# STEP 4: Generation with Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini with temperature=0 for deterministic and factual outputs
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0
)

def answer_question(query: str, k: int = 3):
    retrieved_chunks = retrieve(query, k=k)
    
    context = "\n\n---\n\n".join(
        f"[Source: {c.metadata.get('source', 'Unknown')}]\n{c.page_content}"
        for c in retrieved_chunks
    )
    
    prompt = f"""You are a contract review assistant. Answer the question using ONLY the context below.

If the answer isn't in the context, say "Not found in the provided documents."
Always cite which source document(s) you used.

Context:
{context}

Question: {query}
Answer:"""

    response = llm.invoke(prompt)
    
    # Extract string from response block
    if isinstance(response.content, list):
        answer_text = "".join(
            block["text"] if isinstance(block, dict) and "text" in block else str(block)
            for block in response.content
        )
    else:
        answer_text = str(response.content)

    return answer_text, retrieved_chunks


# Quick Test of Step 4
print("\nSTEP 4 GENERATION TEST:")
test_q = "What are the penalties or buyout fees for terminating the commercial lease early?"
answer, chunks_used = answer_question(test_q)

print(f"Question: {test_q}\n")
print(f"Gemini Answer:\n{answer}")
print("\n")