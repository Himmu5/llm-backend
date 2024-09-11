from fastapi import FastAPI
from contextlib import asynccontextmanager
from llm.llm_config import llm

app = FastAPI()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     llm()
    

app.get("/llm/search")
def search():
    rag_chain=llm().get_rag_chain()
    res = rag_chain.invoke("What is context of the document?")
    print(res)
    return {"search": "searching for documents"}