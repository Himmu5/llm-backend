from db.db_config import get_db
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def get_splitted_data():    
    loader = PyPDFLoader("https://himanshu-rag.s3.eu-north-1.amazonaws.com/RTK_Query_vs_Redux_Saga_Comparison+(1).pdf",extract_images=True)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits
 

def get_retriever():
    data = get_splitted_data()
    vector_db = get_db(data)
    retriever = vector_db.as_retriever()
    return retriever
