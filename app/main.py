import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

my_token = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = my_token

repo_id = "mistralai/Mistral-Nemo-Instruct-2407"


@app.get("/chat")
async def chat(query: str):
   
    

    return {"answer": ("answer", "Sorry, I couldn't find the answer.")}
