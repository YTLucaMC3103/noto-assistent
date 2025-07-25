from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from embedder import suche_relevante_passagen, gpt_antwort

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # später auf deine Domain beschränken!
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    passagen = suche_relevante_passagen(req.message)
    antwort = gpt_antwort(req.message, "\n".join(passagen))
    return {"answer": antwort}
