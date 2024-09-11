import os
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec

os.environ['PINECONE_API_KEY'] = "197321a8-987a-498d-b027-7868d682c1de"
os.environ['PINECONE_API_ENV'] = "gep-starter"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index = pc.create_index("pdf-qa-store",dimension=1024, metric="cosine",spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) )
