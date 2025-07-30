import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DIR = "./chroma_store"
UPLOAD_DIR = "./docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore

def load_vectorstore():
    retriever = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings()
    ).as_retriever(search_kwargs={"k": 5})
    return retriever

def get_qa_chain():
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return None
    
    retriever = load_vectorstore()

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an insurance assistant. Based on the policy document and claims information provided below, answer the question.

Use only the context. Cite like (Policy, p.12). If not found, say "Not available in the documents."

Context:
{context}

Question: {question}

Answer:
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return chain

@app.post("/upload-docs")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    documents = process_pdf(file_location)
    create_vectorstore(documents)

    return {"message": f"{file.filename} processed and vector store created."}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def run_query(request: QueryRequest):
    chain = get_qa_chain()
    if chain is None:
        return {"error": "Vectorstore not available. Upload a document first."}

    result = chain.run(request.query)
    return {
        "decision": "approved" if "yes" in result.lower() else "rejected",
        "amount": "50,000" if "yes" in result.lower() else "0",
        "justification": result
    }

def create_agents():
    claims_agent = Agent(
        role="Insurance Claims Expert",
        goal="Extract and validate insurance policy claims",
        backstory="Expert in reading and understanding policy documents and claims structure.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    validation_agent = Agent(
        role="Policy Compliance Validator",
        goal="Cross-check extracted claims against policy compliance",
        backstory="Experienced in verifying if the extracted claims follow all insurance rules.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    return claims_agent, validation_agent

def create_crew(document_text):
    claims_agent, validation_agent = create_agents()

    task1 = Task(
        description=f"Extract and structure relevant claims from the following policy document:\n{document_text}",
        expected_output="A structured summary of the extracted claims",
        agent=claims_agent
    )

    task2 = Task(
        description="Validate the extracted claims against compliance rules and highlight any issues.",
        expected_output="Compliance status of each claim with necessary comments",
        agent=validation_agent
    )

    crew = Crew(
        agents=[claims_agent, validation_agent],
        tasks=[task1, task2],
        verbose=True
    )

    return crew

@app.get("/run")
def run_agents():
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        return {"error": "No PDF files found in docs/"}

    file_path = os.path.join(UPLOAD_DIR, pdf_files[0])
    documents = process_pdf(file_path)
    document_text = "\n".join([doc.page_content for doc in documents])

    crew = create_crew(document_text)
    result = crew.kickoff()

    return {"result": result}
