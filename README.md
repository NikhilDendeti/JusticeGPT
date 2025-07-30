# Insurance Claims RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, ChromaDB, and CrewAI to process health insurance claims and policy documents using OpenAI's LLMs.

---

## 🚀 Project Objective

Build a system that can:

1. Accept unstructured insurance policy documents (PDF).
2. Let users query policy coverage in natural language.
3. Retrieve relevant clauses using semantic search.
4. Determine claim decisions (approve/reject) and explain reasoning.
5. Validate decisions using a multi-agent system for compliance assurance.

---

## 🛠️ Tech Stack

* **FastAPI** – Backend API
* **Streamlit** – Frontend UI
* **LangChain** – Retrieval and chaining logic
* **ChromaDB** – Vector storage
* **OpenAI GPT-4** – Language model
* **CrewAI** – Multi-agent reasoning for policy validation

---

## 📁 Project Structure

```bash
├── main.py               # FastAPI backend with vectorstore and CrewAI logic
├── app.py                # Streamlit frontend for interaction
├── docs/                 # Uploaded PDF files
├── chroma_store/         # Persistent vector database
├── .env                  # API keys
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 📦 Installation

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your-key
```

---

## ▶️ Run the App

Start the FastAPI backend:

```bash
uvicorn main:app --reload --port 8080
```

Start the Streamlit frontend:

```bash
streamlit run app.py
```

---

## 🌐 API Endpoints

### `/upload-docs`

* Method: `POST`
* Description: Upload and index PDF documents.
* Input: `multipart/form-data` with `file`
* Output: Success message or error

### `/query`

* Method: `POST`
* Description: Submit a natural language claim question.
* Input: `{ "query": "Can a 45M get heart surgery in Pune?" }`
* Output:

```json
{
  "decision": "approved",
  "amount": "50,000",
  "justification": "Yes, clause 2.3 allows this (Policy, p.10)."
}
```

### `/run`

* Method: `GET`
* Description: Triggers CrewAI agents to extract and validate policies.
* Output: Multi-agent reasoning result

---

## 🧠 How It Works

### 1. **Upload Phase**

* PDF is parsed using `PyPDFLoader`
* Chunked with `RecursiveCharacterTextSplitter`
* Embedded using `OpenAIEmbeddings`
* Stored in ChromaDB with metadata

### 2. **Query Phase**

* Query sent to `/query`
* Top 5 chunks retrieved from vectorstore
* GPT-4 answers using prompt template with citation format

### 3. **Multi-Agent Reasoning (CrewAI)**

* **Agent 1:** Insurance Claims Expert extracts structured info
* **Agent 2:** Compliance Validator reviews for policy alignment
* Crew output offers interpretability & validation beyond raw LLM response

---

## 🧪 Improvements Implemented

* Deduplication by document name
* Modularized PDF processing and vectorstore logic
* Pydantic model usage for strong request validation
* Error handling for empty vectorstore, corrupted PDFs
* JSON response for easy integration into downstream systems

---

## 🏆 Built For HackRx 6.0

**Challenge:** Build an LLM-based assistant to evaluate insurance queries against policy documents.

**Problem Statement Summary:**

* Input: Query like "46M, knee surgery, Pune, 3-month policy"
* Output: Decision (approve/reject), amount, justification referencing PDF clauses

### Key Features

* Works even if query is vague or incomplete
* Full explainability via clause citation
* JSON output for audit or workflow integration
* Modular CrewAI logic for enterprise-grade rule-checking

### Sample Output

```json
{
  "decision": "approved",
  "amount": "50,000",
  "justification": "Yes, knee surgery is covered as per clause 3.2 (Policy, p.12)."
}
```

---

## 📌 Future Enhancements

* Document hash-based deduplication
* Parallel agent execution with task orchestration
* Streaming response with real-time citations
* Fine-tuned LLM for insurance-specific reasoning
* Multi-modal document ingestion (images, tables)

---

## 👥 Authors & Credits

Built by Nikhil Dendeti & Team for HackRx 6.0.

Mentorship & support from Bajaj Finserv Health and HackRx organizers.
