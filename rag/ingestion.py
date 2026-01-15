import os
import re
import pdfplumber
from typing import List
from langchain_core.documents import Document

# ==============================
# Utility Functions
# ==============================

def clean_text(text):
    # Remove repeated headers
    text = re.sub(r'Public Law \d+–\d+.*', '', text)

    # Fix hyphenated line breaks
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove page numbers
    text = re.sub(r'\nPage \d+\n', '\n', text)

    return text.strip()


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if re.fullmatch(r"\d+", line):  # page numbers
        return True
    if len(line) < 3:
        return True
    return False


def is_section_heading(line: str) -> bool:
    return (
        len(line.split()) <= 10
        and (
            line.isupper()
            or line.istitle()
            or line.startswith("Section")
            or line.startswith("ARTICLE")
        )
    )


# ==============================
# Single PDF Loader
# ==============================

def load_pdf(file_path: str) -> List[Document]:
    documents: List[Document] = []
    current_section = None
    doc_name = os.path.basename(file_path)

    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):

            raw_text = page.extract_text()
            needs_ocr = raw_text is None

            # ----------------------
            # TEXT EXTRACTION
            # ----------------------
            if raw_text:
                lines = raw_text.split("\n")
                buffer = []

                for line in lines:
                    line = clean_text(line)

                    if is_noise_line(line):
                        continue

                    if is_section_heading(line):
                        current_section = line
                        continue

                    buffer.append(line)

                    if line.endswith("."):
                        paragraph = " ".join(buffer)
                        buffer = []

                        documents.append(
                            Document(
                                page_content=paragraph,
                                metadata={
                                    "document": doc_name,
                                    "source": file_path,
                                    "page": page_index,
                                    "element_type": "paragraph",
                                    "section": current_section,
                                    "needs_ocr": False,
                                    "chunk_id": f"{doc_name}_p{page_index}_l{line}"
                                }
                            )
                        )

            # ----------------------
            # TABLE EXTRACTION
            # ----------------------
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                table_text = clean_text(str(table))
                if not table_text:
                    continue

                documents.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "document": doc_name,
                            "source": file_path,
                            "page": page_index,
                            "element_type": "table",
                            "section": current_section,
                            "needs_ocr": False,
                            "chunk_id": f"{doc_name}_p{page_index}_t{table_idx}"
                        }
                    )
                )

            # ----------------------
            # OCR FLAGGING
            # ----------------------
            if needs_ocr:
                continue

    return documents


# ==============================
# Folder Loader
# ==============================

def load_pdfs_from_folder(folder_path: str) -> List[Document]:
    all_documents: List[Document] = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f" Processing: {file_name}")

        try:
            docs = load_pdf(file_path)
            all_documents.extend(docs)
        except Exception as e:
            print(f" Failed to process {file_name}: {e}")

    return all_documents


# ==============================
#  INGESTION STATS
# ==============================

def ingestion_stats(docs: List[Document]):
    total = len(docs)
    ocr_pages = sum(1 for d in docs if d.metadata.get("needs_ocr"))
    sections = set(d.metadata.get("section") for d in docs if d.metadata.get("section"))

    print("\n INGESTION STATS")
    print(f"Total elements extracted: {total}")
    print(f"OCR-required pages: {ocr_pages}")
    print(f"Unique sections detected: {len(sections)}")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    pdf_folder = "C:/Users/Akshara jain/OneDrive/Desktop/Enterprise Document Intelligence Project/data"
    all_docs = load_pdfs_from_folder(pdf_folder)

    ingestion_stats(all_docs)

    print("\n Sample Document:\n")
    print(all_docs[0])

def get_all_docs(folder_path: str):
    all_docs = load_pdfs_from_folder(folder_path)
    ingestion_stats(all_docs)
    return all_docs

