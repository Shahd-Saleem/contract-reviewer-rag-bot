import os
import warnings
import logging

warnings.filterwarnings("ignore")

logging.getLogger("unstructured").setLevel(logging.ERROR)
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

# STEP 1: Load contracts from PDF files:
doc_directory = './docs'
documents = []
file_paths = [os.path.join(doc_directory, file) for file in os.listdir(doc_directory) if file.endswith('.md')]

from langchain_community.document_loaders import UnstructuredFileLoader
for file_path in file_paths:
    loader = UnstructuredFileLoader(file_path, mode="single")
    documents.extend(loader.load())

print(f"Successfully loaded {len(documents)} contract document(s)!")


# STEP 2: Split the contract into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 200,
)
chunks = text_splitter.split_documents(documents)


# STEP 3: Create embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


# STEP 4: Store into vector database
from langchain_chroma import Chroma

db_path = "./chroma_db"

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=db_path
)

# STEP 5: Test Similarity Search
query = "What are the penalties or fees for early termination?"
results = vector_store.similarity_search(query, k=2)

print("\n--- RETRIEVAL TEST RESULTS ---")
for i, res in enumerate(results, 1):
    print(f"\nResult {i} (Source: {res.metadata.get('source')}):")
    print(res.page_content)