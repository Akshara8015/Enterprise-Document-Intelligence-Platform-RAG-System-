from collections import Counter
import hashlib

def get_meta(doc, key, default=""):
    return doc.metadata.get(key, default)


def compute_frequencies(docs):
    doc_counter = Counter()
    section_counter = Counter()

    for d in docs:
        doc_name = get_meta(d, "document")
        section = get_meta(d, "section")

        if doc_name:
            doc_counter[doc_name] += 1
        if section:
            section_counter[section] += 1

    return doc_counter, section_counter


def score_chunks(docs):
    doc_freq, section_freq = compute_frequencies(docs)

    scored = []
    for doc in docs:
        base_score = doc.metadata.get("score", 1.0)

        doc_bonus = doc_freq[get_meta(doc, "document")] * 0.15
        section_bonus = section_freq[get_meta(doc, "section")] * 0.10

        final_score = base_score + doc_bonus + section_bonus

        doc.metadata["final_score"] = round(final_score, 3)
        scored.append(doc)

    return scored


def deduplicate_by_content(docs):
    seen = {}
    unique_docs = []

    for d in docs:
        content_hash = hashlib.md5(
            d.page_content.strip().encode("utf-8")
        ).hexdigest()

        if content_hash not in seen:
            seen[content_hash] = d
            unique_docs.append(d)

    return unique_docs


def rank_and_select_chunks(
    docs,
    max_chunks=5
):
    docs = deduplicate_by_content(docs)
    docs = score_chunks(docs)

    docs.sort(
        key=lambda d: d.metadata.get("final_score", 0),
        reverse=True
    )

    return docs[:max_chunks]
