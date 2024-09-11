import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llm.llm_config import LLMConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db.db_config import index 

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
embeddings=HuggingFaceEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=100
)
text_field = "text"

@app.get("/chat")
async def chat():
    llm_class = LLMConfig()
    loader=PyPDFLoader('./docs/resume.pdf')
    text_obj=loader.load()
    # data=get_textData(text_obj)
    index.upsert_documents(text_obj)
    splits=text_splitter.split_documents(text_obj)

    return {"answer": splits}


def get_textData(rawData):
    arr=[]
    for data in rawData:
        arr.append(data.page_content)
    return arr

    
#     vectorstore = PineconeVectorStore(index_name='pdf-qa-store', embedding=embeddings)
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
#     chain = RetrievalQAWithSourcesChain.from_chain_type(
#     llm=llm_class.llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(),
#     memory=memory
# )
#     query = 'What is name of the candidate?'
#     response = chain(query)
    