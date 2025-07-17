# main.py

import os
import json
import PyPDF2
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# FastAPI app instance
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangChain components
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.0, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Global retriever
retriever = None

# -------- Helper Functions -------- #

def extract_text_from_pdf(path: str) -> str:
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

def load_pdf_and_split(path: str) -> List[Document]:
    text = extract_text_from_pdf(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": path}) for chunk in chunks]

def initialize_vector_store(documents: List[Document]) -> FAISS:
    return FAISS.from_documents(documents, embedding_model)

def parse_user_query(query: str) -> dict:
    system_prompt = (
        "You are a smart query interpreter for insurance documents. "
        "Extract structured data from the user query. "
        "Respond with ONLY valid JSON containing keys: age, gender, procedure, location, policy_duration."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    response = llm.invoke(messages)
    return json.loads(response.content.strip())

def run_reasoning_engine(parsed_query: dict, relevant_chunks: List[Document]) -> dict:
    prompt_template = PromptTemplate(
        input_variables=["query", "clauses"],
        template=(
            "You are an insurance decision engine.\n"
            "Given the structured user query and the relevant insurance clauses, decide if the claim is approved.\n"
            "Respond ONLY with a JSON object with keys:\n"
            "- decision: approved/rejected\n"
            "- amount: ₹ or number if applicable\n"
            "- justification: explanation with clause references\n\n"
            "User Query:\n{query}\n\nRelevant Clauses:\n{clauses}"
        )
    )

    formatted_prompt = prompt_template.format(
        query=json.dumps(parsed_query, indent=2),
        clauses="\n\n".join([doc.page_content for doc in relevant_chunks])
    )

    result = llm.invoke(formatted_prompt)
    return json.loads(result.content.strip())

# -------- Request Schema -------- #

class QueryRequest(BaseModel):
    query: str

# -------- API Endpoints -------- #

@app.post("/upload-docs")
async def upload_docs(files: List[UploadFile] = File(...)):
    all_docs = []
    for file in files:
        path = f"/tmp/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        all_docs.extend(load_pdf_and_split(path))

    global retriever
    retriever = initialize_vector_store(all_docs)
    return JSONResponse({"message": "Documents uploaded and indexed successfully."})

@app.post("/query")
async def handle_query(req: QueryRequest):
    if not retriever:
        return JSONResponse({"error": "No documents uploaded yet."}, status_code=400)

    try:
        parsed_query = parse_user_query(req.query)
        relevant_docs = retriever.similarity_search(req.query, k=5)
        result = run_reasoning_engine(parsed_query, relevant_docs)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)
