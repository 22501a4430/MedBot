# store_index.py
import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split

# LangChain + Chroma imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

load_dotenv()

# config (can override via .env)
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "medical-chatbot")

print("Loading PDFs from data/ ...")
extracted_data = load_pdf_file(data="data/")
filtered = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered)  # assume list of strings or dicts with 'text'

# Build langchain Documents
docs = []
for i, chunk in enumerate(text_chunks):
    if isinstance(chunk, dict):
        text = chunk.get("text") or chunk.get("page_text") or str(chunk)
        metadata = {k: v for k, v in chunk.items() if k != "text"}
    else:
        text = str(chunk)
        metadata = {}
    metadata["chunk_id"] = i
    docs.append(Document(page_content=text, metadata=metadata))

print(f"Using embedding model: {EMBED_MODEL} (this will load locally)...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})

print("Creating/updating Chroma collection and persisting embeddings...")
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
)

# persist to disk
vectordb.persist()
print("Done. Embeddings stored in:", PERSIST_DIR)
