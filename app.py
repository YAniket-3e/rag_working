from fastapi import FastAPI
from rag import ask_rag
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI(
    title="RAG Chatbot API",
    version="1.0"
)

class Question(BaseModel):
    question: str

templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.get("/health")
def health():
    return {
        "status":"healthy working..."
    }

@app.post("/ask")
def ask(data: Question):
    try:
        answer = ask_rag(data.question)
        return {
            "success": True,
            "question": data.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
