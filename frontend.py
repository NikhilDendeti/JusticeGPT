# streamlit_app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Insurance Claim Processor", layout="wide")
st.title("📄 Insurance Claim Processor")

# Sidebar file uploader
st.sidebar.header("Upload Policy Documents")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("Upload and Index") and uploaded_files:
    with st.spinner("Uploading and indexing documents..."):
        files = [("files", (file.name, file.read(), "application/pdf")) for file in uploaded_files]
        res = requests.post(f"{API_URL}/upload-docs", files=files)
        if res.status_code == 200:
            st.sidebar.success("Documents indexed successfully!")
        else:
            st.sidebar.error(f"Failed to upload documents. Error: {res.text}")

st.markdown("---")
st.subheader("Ask a Claim Question")

query = st.text_area("Enter your query (e.g., 'Can a 45-year-old male get coverage for heart surgery in Hyderabad with a 2-year-old policy?')")

if st.button("Submit Query") and query:
    with st.spinner("Processing query..."):
        res = requests.post(f"{API_URL}/query", json={"query": query})
        if res.status_code == 200:
            result = res.json()
            st.success(f"**Decision:** {result.get('decision', 'N/A').capitalize()}")
            st.write(f"**Amount:** ₹{result.get('amount', 'N/A')}")
            st.markdown("**Justification:**")
            st.write(result.get("justification", "No justification provided."))
        else:
            st.error(f"Failed to process query. Error: {res.text}")
