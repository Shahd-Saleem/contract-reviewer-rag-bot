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


# STEP 3: Embeddings, Vector Store & Retrieval Function
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Initialize Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

db_path = "./chroma_db"

# Load existing database if directory exists, otherwise create a new one
if os.path.exists(db_path) and os.listdir(db_path):
    print("Loading existing Chroma vector database from disk...")
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
else:
    print("Creating new Chroma vector database...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

def retrieve(query: str, k: int = 5):
    """Retrieves top-k relevant document chunks for a given search query."""
    return vector_store.max_marginal_relevance_search(query, k=k)

# Test of Step 3
#test_query = "What is the penalty for terminating the commercial lease early?"
#retrieved_docs = retrieve(test_query, k=5)

#print("STEP 3 RETRIEVAL TEST:")
#for idx, doc in enumerate(retrieved_docs, 1):
#    src = doc.metadata.get('source', 'Unknown')
#    print(f"\nResult {idx} (Source: {src}):")
#    print(doc.page_content[:250] + "...")


# STEP 4: Generation with Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini with temperature=0 for deterministic and factual outputs
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0
)

def answer_question(query: str, k: int = 5):
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
#print("\nSTEP 4 GENERATION TEST:")
#test_q = "What are the penalties or buyout fees for terminating the commercial lease early?"
#answer, chunks_used = answer_question(test_q)

#print(f"Question: {test_q}\n")
#print(f"Gemini Answer:\n{answer}")


# STEP 5: Automated Pipeline Evaluation
# Evaluation dataset matching the uploaded contract and terms documents
# eval_dataset = [
#     {
#         "question": "What is the mandatory buyout fee and security deposit forfeit for early lease termination?",
#         "expected_source": "commercial_lease_agreement"
#     },
#     {
#         "question": "Under what conditions must lease repairs be approved in writing by the Landlord?",
#         "expected_source": "commercial_lease_agreement"
#     },
#     {
#         "question": "What is the geographic radius enforced by the Non-Compete clause, and how long does it last?",
#         "expected_source": "employment_contract"
#     },
#     {
#         "question": "How is severance pay calculated if an employee is terminated without cause?",
#         "expected_source": "employment_contract"
#     },
#     {
#         "question": "How often does the automatic baseline price increase occur, and by what percentage?",
#         "expected_source": "predatory_vendor_contract"
#     },
#     {
#         "question": "What is the exact window and method required to cancel the software agreement?",
#         "expected_source": "predatory_vendor_contract"
#     },
#     {
#         "question": "How long is Customer data retained after contract termination, and what is the starting fee for data export?",
#         "expected_source": "saas_terms_of_service"
#     },
#     {
#         "question": "What is the maximum SLA credit cap if provider uptime falls below 99.5%?",
#         "expected_source": "saas_terms_of_service"
#     },
#     {
#         "question": "What are the payment due terms and late fee rates under this Master Services Agreement?",
#         "expected_source": "service_agreement"
#     },
#     {
#         "question": "What law governs the Google Terms of Service and where must disputes be resolved?",
#         "expected_source": "google_terms_of_service"
#     },
#     {
#         "question": "For business users, what is Google's total aggregate liability limit arising out of these terms?",
#         "expected_source": "google_terms_of_service"
#     },
#     {
#         "question": "What is the penalty-free cancellation window for service-only contracts vs. essential hardware contracts?",
#         "expected_source": "Consumer_Services_Agreement_du"
#     },
#     {
#         "question": "How many Spam Call complaints result in line suspension, and what happens after 1 complaint?",
#         "expected_source": "Consumer_Services_Agreement_du"
#     },
#     {
#         "question": "What is the maximum total monetary liability du will pay for all claims within a 12-month period under UAE law?",
#         "expected_source": "Consumer_Services_Agreement_du"
#     }
# ]

# import time
# def evaluate_pipeline(dataset):
#     print("STEP 5: EVALUATION REPORT:")
    
#     retrieval_hits = 0
#     total_queries = len(dataset)

#     for idx, test in enumerate(dataset, 1):
#         q = test["question"]
#         exp_src = test["expected_source"]
        
#         answer, chunks = answer_question(q, k=5)
#         retrieved_sources = [c.metadata.get("source", "") for c in chunks]
        
#         # Check if the correct contract source file was retrieved
#         hit = any(exp_src in src for src in retrieved_sources)
#         if hit:
#             retrieval_hits += 1
            
#         print(f"\n[Test {idx}/{total_queries}]")
#         print(f"Query: {q}")
#         print(f"Expected File Match ({exp_src}): {'PASS' if hit else 'FAIL'}")
#         print(f"Generated Answer:\n{answer}")
#         print("-" * 50)

#         time.sleep(6)

#     score = (retrieval_hits / total_queries) * 100
#     print(f"\nFinal Retrieval Recall Score: {score:.1f}%")

# # Execute Evaluation
# evaluate_pipeline(eval_dataset)