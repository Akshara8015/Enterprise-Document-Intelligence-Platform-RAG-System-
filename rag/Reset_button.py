def reset_rag():
    import shutil, os
    shutil.rmtree("faiss_index", ignore_errors=True)
    print("RAG memory reset complete.")

# reset_rag()