# Enterprise-Document-Intelligence-Platform-RAG-System-

Enterprise Document Intelligence Platform is a production-style Retrieval-Augmented Generation (RAG) system for answering questions from multiple PDFs with high accuracy, grounding, and safety.

Unlike basic chatbots, this system focuses on retrieval quality, explainability, evaluation, and reliability.

# What it does

- Ingests multiple legal / research PDFs
- Performs semantic search using transformer embeddings
- Generates grounded answers with page-level citations
- Detects hallucinations and low-confidence answers
- Exposes APIs via FastAPI and UI via Streamlit

# Tech Stack

- Python, LangChain
- HuggingFace Transformers
- FAISS (Vector DB)
- FastAPI, Streamlit
- Ollama / OpenAI (LLMs)
