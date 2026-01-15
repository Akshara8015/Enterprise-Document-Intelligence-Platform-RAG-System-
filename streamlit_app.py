import streamlit as st
import requests
import tempfile
import os

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Enterprise Document Intelligence", layout="wide")

st.title("📄 Enterprise Document Intelligence Platform")

# ==============================
# Upload PDFs
# ==============================

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload at least one PDF.")
    st.stop()

# Save uploaded PDFs temporarily
temp_dir = tempfile.mkdtemp()
pdf_paths = []

for file in uploaded_files:
    file_path = os.path.join(temp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    pdf_paths.append(file_path)

st.success(f"{len(pdf_paths)} PDF(s) uploaded successfully.")

# ==============================
# Question Input
# ==============================

query = st.text_input("Ask a question based on the uploaded documents:")

if st.button("🔍 Get Answer"):

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Analyzing documents..."):
        response = requests.post(
            API_URL,
            json={
                "question": query,
                "pdf_paths": pdf_paths
            },
            timeout=120
        )

    if response.status_code != 200:
        st.error(response.text)
        st.stop()

    result = response.json()

    # ==============================
    # Display Answer
    # ==============================

    st.subheader("🧠 Answer")
    st.write(result["answer"])

    # ==============================
    # Metrics
    # ==============================

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence Score", result.get("confidence_score", "N/A"))
    col2.metric("Sources Used", result.get("num_sources", 0))
    col3.metric("Time Taken (s)", result.get("time_taken", 0))

    # ==============================
    # Sources
    # ==============================

    st.subheader("📚 Sources")
    for src in result.get("sources", []):
        st.markdown(
            f"""
            - **Document:** {src['document']}
            - **Section:** {src['section']}
            - **Pages:** {src['pages']}
            """
        )
