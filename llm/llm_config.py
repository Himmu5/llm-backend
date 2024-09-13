from langchain_cohere import ChatCohere
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from prompt.llm_prompt import get_prompt_template
from llm.retriever import get_retriever
import os

class llm:
    key = "M6kNzVruUj6YfaH3VDJpF9zrPpTtxou2Z885NKNb"
    os.environ["COHERE_API_KEY"] = key
    llm = ChatCohere(model="command-r-plus")

    def get_rag_chain(self):  
        prompt = get_prompt_template();
        retriever=get_retriever()
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain
