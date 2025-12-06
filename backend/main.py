# backend/main.py
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from .rag_engine import RAGEngine

# Load .env from project ROOT (voice_assistant/.env)
# We run uvicorn from root, so this will work:
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Debug: show if key is loaded (masked)
print("DEBUG API KEY:", (OPENAI_API_KEY[:10] + "..." if OPENAI_API_KEY else "NONE"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Check your .env file in project root.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# CORS so frontend (Live Server) can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    context_chunks: List[str]


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def startup_event():
    rag_engine.load_documents()
    rag_engine.build_index()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    query = req.query

    retrieved = rag_engine.retrieve(query, k=3)
    context_texts = [c[0] for c in retrieved] if retrieved else []

    context_block = "\n\n".join(context_texts) if context_texts else "No extra context."

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant answering questions for a student. "
                    f"Here is optional context from their notes:\n{context_block}\n"
                    "Use the context only if it is relevant."
                ),
            },
            {"role": "user", "content": query},
        ],
    )

    answer = completion.choices[0].message.content

    return AskResponse(answer=answer, context_chunks=context_texts)
