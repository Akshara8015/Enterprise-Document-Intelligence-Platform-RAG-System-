from collections import defaultdict
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ==============================
# Clause-aware Chunking
# ==============================

def clause_aware_chunking( docs: List[Document], min_tokens: int = 50 ) -> List[Document]:
    grouped = defaultdict(list)

    # 1. Group by document + section
    for doc in docs:
        key = (
            doc.metadata["document"],
            doc.metadata.get("section", "NO_SECTION")
        )
        grouped[key].append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size= 1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", ";"]
    )

    chunked_docs: List[Document] = []

    # 2. Chunk per section
    for (document, section), section_docs in grouped.items():

        combined_text = " ".join(d.page_content for d in section_docs).strip()
        if not combined_text:
            continue

        chunks = splitter.split_text(combined_text)
        pages = sorted(set(d.metadata["page"] for d in section_docs))

        for idx, chunk in enumerate(chunks):

            token_estimate = len(chunk.split())
            if token_estimate < min_tokens:
                continue  # drop low-quality chunks

            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "document": document,
                        "section": section,
                        "page_start": pages[0],
                        "page_end": pages[-1],
                        "chunk_type": "clause",
                        "token_count": token_estimate,
                        "chunk_id": f"{document}_{section}_chunk_{idx}"
                    }
                )
            )

    return chunked_docs


# ==============================
# Chunk Statistics
# ==============================

def chunk_stats(chunks: List[Document]):
    lengths = [c.metadata["token_count"] for c in chunks]

    print("\n CHUNK STATS")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg tokens per chunk: {sum(lengths) // len(lengths)}")
    print(f"Min tokens: {min(lengths)}")
    print(f"Max tokens: {max(lengths)}")
