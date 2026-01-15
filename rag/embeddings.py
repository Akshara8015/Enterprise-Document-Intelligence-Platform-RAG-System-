from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag.ingestion import get_all_docs
from rag.chunking import clause_aware_chunking, chunk_stats

# ==============================
# Load documents
# ==============================

pdf_folder = r"C:/Users/Akshara jain/OneDrive/Desktop/Enterprise Document Intelligence Project/data"
all_docs = get_all_docs(pdf_folder)

# ==============================
# Chunking
# ==============================

chunked_docs = clause_aware_chunking(all_docs)
chunk_stats(chunked_docs)

# ==============================
# Embeddings + FAISS
# ==============================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = FAISS.from_documents(
    documents=chunked_docs,
    embedding=embedding_model
)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

VECTOR_INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_index")

vectorstore.save_local(VECTOR_INDEX_PATH)

# Persist index
vectorstore.save_local("vector_index/")
print("Vectorstore saved")
