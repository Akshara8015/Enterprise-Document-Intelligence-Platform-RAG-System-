import re
import time
from typing import List
import json
from datetime import datetime

from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.eval_scores import evaluate_rag, compute_evidence_coverage , compute_citation_accuracy  # evaluation module
from rag.metrics_eval import update_metrics
from rag.query_expand import expand_query
from rag.judge1 import rank_and_select_chunks
from rag.judge2 import compress_context

# ==============================
# CONFIG
# ==============================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../rag
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # .../Enterprise Document Intelligence Project

VECTOR_INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_index")

MIN_CONFIDENCE = 0.4
MIN_SOURCES = 1
MAX_CONTEXT_CHARS = 3000

# ==============================
# Load Vector Store + Retriever
# ==============================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = FAISS.load_local(
    VECTOR_INDEX_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 8,
        "lambda_mult": 0.6
    }
)

# ==============================
# LLM (USING OLLAMA HERE BCZ NO API NEEDED, CAN ALSO USE CHATOPENAI)
# ==============================

llm = ChatOllama(
    model="phi3",
    temperature=0
)

# ==============================
# Prompt Template
# ==============================

RAG_PROMPT = """
You are a legal document analysis assistant.

STRICT RULES:
- Answer ONLY if the information is EXPLICITLY present in the context
-If the question asks for a definition, do NOT include implementation details unless explicitly requested.
- Quote exact phrases when possible
- If information is incomplete or ambiguous, say:
  "The documents do not provide sufficient information."
- Do not infer purposes or intent. Answer ONLY using retrieved text.
- Before returning an answer:
    Each bullet/statement must map to a retrieved chunk
    If not → drop it or refuse

Context:
{context}

Question:
{question}

Answer:
- Use bullet points
- Cite sources like [Source 1]
- Do NOT add external knowledge

"""

prompt = PromptTemplate(
    template=RAG_PROMPT,
    input_variables=["context", "question"]
)

# ==============================
# Safety Utilities
# ==============================

def normalize_query(query: str) -> str:
    return query.strip().lower()

def is_prompt_injection(query: str) -> bool:
    patterns = [
        r"ignore .* instruction",
        r"act as",
        r"bypass",
        r"system prompt",
        r"jailbreak"
    ]
    return any(re.search(p, query.lower()) for p in patterns)


def is_advice_seeking(query: str) -> bool:
    return any(
        phrase in query.lower()
        for phrase in ["should i", "what should i do", "is it legal for me"]
    )


def mask_pii(text: str) -> str:
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)
    return text

def rewrite_query(query: str) -> str:
    if len(query.split()) < 4:
        return f"Explain in detail: {query}"
    return query


# ==============================
# Retrieval + Deduplication
# ==============================

def retrieve_chunks(query: str):
    query = normalize_query(query)
    expanded_queries = expand_query(query)

    all_docs = []
    for q in expanded_queries:
        retrieved = retriever.invoke(q)
        all_docs.extend(retrieved)

    ranked_docs = rank_and_select_chunks(
        all_docs,
        max_chunks=5
    )
    compressed_docs = compress_context(
        ranked_docs,
        query,
        embedding_model,
        max_sentences=3
    )

    return compressed_docs


def deduplicate_sources(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []

    for d in docs:
        key = (d.metadata["document"], d.metadata["page_start"])
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    return unique_docs


# ==============================
# Context Builder
# ==============================

def build_context(docs: List[Document]) -> str:
    blocks = []

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        content = mask_pii(doc.page_content)
        # print("content", content)

        block = (
            f"[Source {i}]\n"
            f"Document: {meta.get('document', 'Unknown')}\n"
            f"Section: {meta.get('section', 'N/A')}\n"
            f"Pages: {meta.get('page_start', 'N/A')}-{meta.get('page_end', 'N/A')}\n"
            f"Relevant Extract:\n"
            f"{content}"
        )
        blocks.append(block)

    context = "\n\n".join(blocks)
    return context


# ==============================
# LOGGING AND MONITORING
# ==============================

def log_interaction(query: str, result: dict):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "confidence_score": result.get("confidence_score"),
        "is_grounded": result.get("is_grounded"),
        "num_sources": result.get("num_sources"),
    }

    with open("rag_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")


# ==============================
# SAFE RAG PIPELINE
# ==============================

def rag_answer(query: str, vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15}
    )

    start_time = time.time()

    # 1️ HARD SAFETY BLOCKS
    if is_prompt_injection(query):
        update_metrics(blocked=True)
        return {"answer": "Unsafe query detected.", "sources": []}

    if is_advice_seeking(query):
        update_metrics(blocked=True)
        return {"answer": "I cannot provide legal or personal advice.", "sources": []}

    # 2️ RETRIEVAL
    query = rewrite_query(query)
    docs = retrieve_chunks(query)
    # docs = retriever.invoke(query)

    if len(docs) < MIN_SOURCES:
        update_metrics(blocked=True)
        result = {
            "query": query,
            "answer": "Insufficient evidence found in the documents.",
            "confidence_score": 0.0,
            "is_grounded": False,
            "num_sources": len(docs),
            "sources": []
        }
        log_interaction(query, result)
        return result

    # 3️ CONTEXT
    context = build_context(docs)

    # 4️ LLM CALL
    response = llm.invoke(
        prompt.format(context=context, question=query)
    )

    # 5️ EVALUATION
    evaluation = evaluate_rag(
        answer=response.content,
        context=context,
        docs=docs
    )

    coverage = compute_evidence_coverage(
        answer=response.content,
        docs=docs,
        embedding_model=embedding_model
    )

    citation_eval = compute_citation_accuracy(
        answer=response.content,
        docs=docs,
        embedding_model=embedding_model
    )

    # 6️ FAIL-CLOSED SAFETY
    if ( not evaluation["is_grounded"] or evaluation["confidence_score"] < MIN_CONFIDENCE
        or coverage["coverage_score"] < 0.6 or citation_eval["citation_accuracy"] < 0.7
    ):
        update_metrics(
            confidence=evaluation["confidence_score"],
            blocked=True
        )

        result = {
            "query": query,
            "answer": "Insufficient reliable evidence found to answer this question.",
            "confidence_score": evaluation["confidence_score"],
            "coverage_score": coverage["coverage_score"],
            "citation_accuracy": citation_eval["citation_accuracy"],
            "invalid_citations": citation_eval["invalid_citations"],
            "is_grounded": evaluation["is_grounded"],
            "num_sources": len(docs),
            "sources": []
        }
        log_interaction(query, result)
        return result

    # 7️ FINAL SAFE ANSWER
    final_answer = (
        response.content.strip()
    )

    latency = round(time.time() - start_time, 2)

    update_metrics(
        confidence=evaluation["confidence_score"],
        sources_used=len(docs),
        latency=latency,
        success=True
    )

    result = {
        "query": query,
        "answer": final_answer,
        "confidence_score": evaluation["confidence_score"],
        "coverage_score": coverage["coverage_score"],
        "citation_accuracy": citation_eval["citation_accuracy"],
        "invalid_citations": citation_eval["invalid_citations"],
        "supported_sentences": coverage["supported"],
        "unsupported_sentences": coverage["unsupported"],
        "is_grounded": evaluation["is_grounded"],
        "num_sources": len(docs),
        "time_taken": round(time.time() - start_time, 2),
        "sources": [
            {
                "document": d.metadata["document"],
                "section": d.metadata["section"] or "",
                "pages": f"{d.metadata['page_start']}-{d.metadata['page_end']}" or "",
                "chunk_id": d.metadata["chunk_id"]
            }
            for d in docs
        ]
    }

    log_interaction(query, result)
    return result


# ==============================
# TEST
# ==============================

if __name__ == "__main__":
    query = "What is meant by issue-level interpretation in legal analysis?"

    result = rag_answer(query)

    print("\nANSWER:\n", result["answer"])
    print("\nCONFIDENCE:", result.get("confidence_score"))
    print("\nSOURCES:")
    for s in result.get("sources", []):
        print(s)
