from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_db(splited_data):
    vectorstore = Chroma.from_documents(documents=splited_data, embedding=HuggingFaceEmbeddings())
    return vectorstore;