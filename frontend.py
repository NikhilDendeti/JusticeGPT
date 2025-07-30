import streamlit as st
import requests

API_URL = "http://0.0.0.0:8080"

st.set_page_config(page_title="Insurance Claim Processor", layout="wide")
st.title("📄 Insurance Claim Processor")

st.sidebar.header("Upload Policy Documents")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("Upload and Index") and uploaded_files:
    with st.spinner("Uploading and indexing documents..."):
        success = True
        for file in uploaded_files:
            res = requests.post(
                f"{API_URL}/upload-docs",
                files={"file": (file.name, file.read(), "application/pdf")}
            )
            if res.status_code != 200:
                success = False
                st.sidebar.error(f"Failed to upload {file.name}. Error: {res.text}")
                break
        if success:
            st.sidebar.success("All documents indexed successfully!")

st.markdown("---")
st.subheader("Ask a Claim Question")

query = st.text_area("Enter your query (e.g., 'Can a 45-year-old male get coverage for heart surgery in Hyderabad with a 2-year-old policy?')")

if st.button("Submit Query") and query:
    with st.spinner("Processing query..."):
        res = requests.post(f"{API_URL}/query", json={"query": query})
        if res.status_code == 200:
            result = res.json()
            st.success(f"**Decision:** {result.get('decision', 'N/A').capitalize()}")
            st.markdown("**Justification:**")
            st.write(result.get("justification", "No justification provided."))
        else:
            st.error(f"Failed to process query. Error: {res.text}")
