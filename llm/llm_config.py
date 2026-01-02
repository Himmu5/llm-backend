from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage
from prompt.llm_prompt import get_prompt_template
from llm.retriever import get_retriever, get_retriever_multi
from prompt.financial_prompt import get_financial_analysis_prompt
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class llm:
    # Explicitly pass API key to avoid using gcloud application default credentials
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY
    )

    def chat(self, query: str) -> str:
        """Simple chat without document context"""
        response = self.llm.invoke([HumanMessage(content=query)])
        return response.content

    def get_rag_chain(self, fileName):  
        """RAG chain with single document context"""
        prompt = get_prompt_template()
        retriever = get_retriever(fileName)
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    def get_rag_chain_multi(self, fileNames: list):  
        """RAG chain with multiple document contexts"""
        prompt = get_prompt_template()
        retriever = get_retriever_multi(fileNames)
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    def get_financial_rag_chain(self, fileNames: list):
        """RAG chain specialized for financial report analysis"""
        prompt = get_financial_analysis_prompt()
        if len(fileNames) == 1:
            retriever = get_retriever(fileNames[0])
        else:
            retriever = get_retriever_multi(fileNames)
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain
