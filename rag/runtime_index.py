from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag.ingestion import load_pdf
from rag.chunking import clause_aware_chunking  # or your chunker

def build_vectorstore_from_pdfs(pdf_paths: list):

    all_docs = []

    for path in pdf_paths:
        docs = load_pdf(path)
        all_docs.extend(docs)

    # Chunk documents
    chunked_docs = clause_aware_chunking(all_docs)

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Build FAISS (IN-MEMORY)
    vectorstore = FAISS.from_documents(
        documents=chunked_docs,
        embedding=embedding_model
    )

    return vectorstore
