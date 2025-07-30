# Insurance Claims RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, ChromaDB, and CrewAI to process health insurance claims and policy documents using OpenAI's LLMs.

---

## 🚀 Project Objective

Build a system that can:

1. Accept unstructured insurance policy documents (PDF).
2. Let users query policy coverage in natural language.
3. Retrieve relevant clauses using semantic search.
4. Determine claim decisions (approve/reject) and explain reasoning.

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

### `/query`

* Method: `POST`
* Description: Submit a natural language claim question.
* Returns: JSON with decision, amount, and justification.

### `/run`

* Method: `GET`
* Description: Triggers CrewAI agents to extract and validate policies.

---

## 🧠 How It Works

1. **Upload Phase**

   * PDF is parsed into chunks.
   * Chunks are embedded and stored in ChromaDB.

2. **Query Phase**

   * User asks a natural question.
   * Top 5 relevant chunks retrieved semantically.
   * GPT-4 answers using the context with citations.

3. **Multi-Agent Validation** (Optional)

   * Claims Expert extracts claim data.
   * Compliance Agent checks rules.

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

* Document deduplication
* Multi-modal input (emails, images)
* Fine-tuned LLM for domain-specific logic
* Streaming output and real-time feedback

---

## 👥 Authors & Credits

Built by Nikhil Dendeti & Team for HackRx 6.0.

Mentorship & support from Bajaj Finserv Health and HackRx organizers.
