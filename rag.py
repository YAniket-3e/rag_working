from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from config import API_KEY, API_BASE, MODEL_NAME
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
input_variables=["context", "question"],
template="""
You are a QA assistant.

Answer ONLY using the context below.

If the answer is not present,
reply exactly:

"I don't know based on the provided knowledge."

Context:
{context}

Question:
{question}

Answer:
"""
)

BASE_DIR = Path(__file__).parent

knowledge_file = BASE_DIR / "knowledge" / "knowledge.txt"

with open(knowledge_file, encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.create_documents([text])

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(
    docs,
    embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k":3}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key="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",
    openai_api_base="https://models.inference.ai.azure.com"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt
    }
)

def ask_rag(question: str):

    result = qa_chain.invoke(
        {
            "query":question
        }
    )

    return result["result"]
