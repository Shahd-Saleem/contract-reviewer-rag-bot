import os
import warnings
import logging

warnings.filterwarnings("ignore")

logging.getLogger("unstructured").setLevel(logging.ERROR)
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

# STEP 1: Load contracts from pdf files:
doc_directory = './docs'
documents = []
file_paths = [os.path.join(doc_directory, file) for file in os.listdir(doc_directory) if file.endswith('.md')]

from langchain_community.document_loaders import UnstructuredFileLoader
for file_path in file_paths:
    loader = UnstructuredFileLoader(file_path, mode="single")
    documents.extend(loader.load())

print(f"Successfully loaded {len(documents)} contract document(s)!")