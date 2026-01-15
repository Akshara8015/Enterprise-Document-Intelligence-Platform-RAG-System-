import re
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

# ==============================
# Utility: Sentence split
# ==============================

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 10]


# ==============================
# Grounding Check ( USED FOR PREVENTING HALLUCINATIONS)
# ==============================

def grounding_score(answer: str, context: str) -> float:
    """
    Measures how much of the answer is supported by the context
    """
    answer_sents = split_sentences(answer)
    context_lower = context.lower()

    if not answer_sents:
        return 0.0

    grounded = 0
    for sent in answer_sents:
        if sent.lower()[:40] in context_lower:
            grounded += 1

    return grounded / len(answer_sents)


# ==============================
# Retrieval Coverage
# ==============================

def retrieval_coverage(docs: List[Document]) -> float:
    """
    Measures diversity of evidence
    """
    unique_pages = set()
    for d in docs:
        unique_pages.add((d.metadata["document"], d.metadata["page_start"]))

    return min(1.0, len(unique_pages) / 3)


# ==============================
# Final Confidence Score
# ==============================

def confidence_score(answer: str, context: str, docs: List[Document]) -> float:
    g_score = grounding_score(answer, context)
    r_score = retrieval_coverage(docs)

    # weighted confidence
    return round((0.6 * g_score + 0.4 * r_score), 2)


# ==============================
# Full Evaluation
# ==============================


eval_model = SentenceTransformer("all-mpnet-base-v2")

def evaluate_rag(answer: str, context: str, docs):
    if not answer :
        # or "not found" in answer.lower():
        return {
            "confidence_score": 0.0,
            "is_grounded": False,
            "warning": "No answer generated"
        }

    answer_emb = eval_model.encode(answer, convert_to_tensor=True)
    context_emb = eval_model.encode(context, convert_to_tensor=True)

    similarity = util.cos_sim(answer_emb, context_emb).item()

    source_bonus = min(len(docs) * 0.1, 0.3)  # reward multiple sources
    confidence = min(similarity + source_bonus, 1.0)

    is_grounded = confidence >= 0.5

    return {
        "confidence_score": round(confidence, 3),
        "is_grounded": is_grounded,
        "warning": None if is_grounded else "Low semantic grounding"
    }

# ==============================
# COVERAGE SCORE
# ==============================

def compute_evidence_coverage( answer: str, docs, embedding_model, threshold: float = 0.40 ):
    """
    Measures how much of the answer is supported by retrieved documents.
    """

    # Split answer into sentences
    answer_sentences = [
        s.strip() for s in re.split(r"[.?!]", answer) if len(s.strip()) > 10
    ]

    if not answer_sentences:
        return {
            "coverage_score": 0.0,
            "supported": [],
            "unsupported": []
        }

    # Collect all evidence text
    evidence_texts = [doc.page_content for doc in docs]

    # Embed
    answer_embeds = embedding_model.embed_documents(answer_sentences)
    evidence_embeds = embedding_model.embed_documents(evidence_texts)

    supported = []
    unsupported = []

    for sent, sent_vec in zip(answer_sentences, answer_embeds):
        sims = cosine_similarity(
            [sent_vec],
            evidence_embeds
        )[0]

        if max(sims) >= threshold:
            supported.append(sent)
        else:
            unsupported.append(sent)

    coverage_score = len(supported) / len(answer_sentences)

    return {
        "coverage_score": round(coverage_score, 2),
        "supported": supported,
        "unsupported": unsupported
    }


# ==============================
# CITATION SCORE
# ==============================

def extract_citations(answer: str):
    """
    Extracts cited source numbers like [Source 1]
    """
    return set(
        int(num) for num in re.findall(r"\[Source (\d+)\]", answer)
    )

def compute_citation_accuracy( answer: str, docs, embedding_model, threshold: float = 0.40 ):
    citations = extract_citations(answer)

    if not citations:
        return {
            "citation_accuracy": 0.0,
            "invalid_citations": ["No citations found"]
        }

    invalid = []

    # Split answer into sentences
    sentences = [
        s.strip()
        for s in re.split(r"[.?!]", answer)
        if len(s.strip()) > 10
    ]

    doc_texts = [d.page_content for d in docs]
    doc_embeddings = embedding_model.embed_documents(doc_texts)

    for cited in citations:
        if cited > len(docs):
            invalid.append(f"Source {cited} does not exist")
            continue

        cited_doc = docs[cited - 1]
        cited_embed = doc_embeddings[cited - 1]

        # Check if any sentence matches cited doc
        sentence_embeds = embedding_model.embed_documents(sentences)
        sims = cosine_similarity(
            sentence_embeds,
            [cited_embed]
        )

        if max(sims.flatten()) < threshold:
            invalid.append(f"Source {cited} weakly supports answer")

    accuracy = 1 - (len(invalid) / len(citations))

    return {
        "citation_accuracy": round(max(accuracy, 0), 2),
        "invalid_citations": invalid
    }


