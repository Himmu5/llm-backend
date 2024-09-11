from fastapi import FastAPI
from llm.llm_config import llm
from contextlib import asynccontextmanager
import os


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     key = "M6kNzVruUj6YfaH3VDJpF9zrPpTtxou2Z885NKNb"
#     os.environ["COHERE_API_KEY"] = key
    
myApp = FastAPI()

@myApp.get("/llm/search")
def search():
    rag_chain=llm().get_rag_chain()
    res = rag_chain.invoke({"input": "What is total work experience?"})
    print(res['answer'])
    return {"answer": res['answer']}
    