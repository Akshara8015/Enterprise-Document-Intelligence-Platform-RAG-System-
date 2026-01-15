import re
from langchain_core.documents import Document
import numpy as np

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def compress_chunk( doc: Document, query_embedding, embedding_model, top_n = 3 ):
    sentences = split_sentences(doc.page_content)
    if not sentences:
        return doc

    sent_embeddings = embedding_model.embed_documents(sentences)

    scores = np.dot(
        sent_embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[-top_n:]
    top_sentences = [sentences[i] for i in sorted(top_indices)]

    compressed_text = " ".join(top_sentences)

    return Document(
        page_content=compressed_text,
        metadata=doc.metadata
    )

def compress_context(
    docs,
    query,
    embedding_model,
    max_sentences=3
):
    query_embedding = embedding_model.embed_query(query)

    compressed_docs = []
    for d in docs:
        compressed = compress_chunk(
            d,
            query_embedding,
            embedding_model,
            top_n=max_sentences
        )
        compressed_docs.append(compressed)

    return compressed_docs

